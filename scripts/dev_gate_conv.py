#!/usr/bin/env python3
"""dev_gate_conv: end-to-end device-execution gate for the encoder conv
lowerings new at 504a1e4 (F1 in-proj k1x1 wmaj, F2 depthwise k1x9 g1024
+bias, F3 out-proj k1x1, F4 rank-4 custom-pad padconv) plus the encoder
last-dim slice and the r3/r4 [0,2,1(,3)] transposes.

Each case is the exact MIL spelling the compiler now lowers (the
research/oracles/h13 encoder_* cases), minted with REAL random fp16
weights — uniform payloads are permutation-blind, which is the mechanism
that let the wrong oproj function ship green — executed on the ANE via
libane-strict, and compared against an fp32 numpy reference across three
input rngs. Exits 0 on PASS.

MIL pad order is [top, bottom, left, right] (mil_numpy.py _conv; the
Apple oracle output surface [1,8,375,750] for pd=[0,0,1,0] confirms the
W pad is LEFT).

Run on jw16 under /tmp/m1-gpu.lock.
"""
import ctypes, os, subprocess, sys, hashlib, struct, json
from pathlib import Path
import numpy as np

# The baseline gate ran the 504a1e4 build; a fix branch points MHWC at its
# own build so the same script re-gates the changed compiler in place.
COMPILER = os.environ.get("MHWC",
                          "/home/joshuawarren/src/mil-hwx-compiler-504a1e4/build/mil-hwxc")
LIB = "/var/tmp/jw16-oproj-place/libane-strict-fill.so"
ENV = {"LD_LIBRARY_PATH": "/home/joshuawarren/.local/mil-hwx-gnustep/lib",
       "PATH": "/usr/bin:/bin", "HOME": "/home/joshuawarren"}
WORKDIR = Path("/tmp/dev-gate-conv")
BUDGET = 0.05
SEEDS = (11, 33, 57)

lib = ctypes.CDLL(LIB)
for name, rest, args in (
    ("__ane_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
    ("__ane_free", None, [ctypes.c_void_p]),
    ("ane_exec", ctypes.c_int, [ctypes.c_void_p]),
    ("__ane_src_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_dst_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_send", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_read", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
):
    f = getattr(lib, name); f.restype = rest; f.argtypes = args

def row_bytes(width):
    return (width * 2 + 63) & ~63

def pack(x):
    """NCHW fp16 [N,C,H,W] -> ANE tile bytes (row = alignUp(width*2, 64))."""
    n, c, h, w = x.shape
    rb = row_bytes(w)
    tile = np.zeros(n * c * h * rb, dtype=np.uint8)
    flat = x.reshape(n * c, h, w).astype("<f2")
    raw = flat.tobytes()
    view = tile.reshape(n * c, h, rb)
    view[:, :, :w * 2] = np.frombuffer(raw, dtype=np.uint8).reshape(n * c, h, w * 2)
    return tile

def unpack(buf, shape):
    """Inverse of pack; buf is the raw dst bytes."""
    n, c, h, w = shape
    rb = row_bytes(w)
    view = np.frombuffer(buf, dtype=np.uint8, count=n * c * h * rb).reshape(n * c, h, rb)
    halves = view[:, :, :w * 2].copy().view("<f2")
    return halves.reshape(n, c, h, w).astype(np.float32)

def run(anec_path, x4):
    """Send packed x4 and return (raw_dst_bytes, dst_bytes). The anec tile
    may exceed the packed surface (guard tiles); the manifest strides are
    authoritative for placement, so packed data sits at offset 0."""
    nn = lib.__ane_init(str(anec_path).encode(), 0)
    assert nn
    try:
        s0, d0 = int(lib.__ane_src_size(nn, 0)), int(lib.__ane_dst_size(nn, 0))
        tile = pack(x4)
        assert s0 >= tile.nbytes, f"anec src {s0} < packed {tile.nbytes}"
        lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
        assert lib.ane_exec(nn) == 0
        out = np.zeros(d0, dtype=np.uint8)
        lib.__ane_read(nn, ctypes.c_char_p(out.ctypes.data), 0)
        return out.tobytes(), d0
    finally:
        lib.__ane_free(nn)

def blob(records):
    """BLOBFILE weights.bin body: records = [(payload_bytes,) ...] mirrors
    dev_gate_ffn — w record descriptor at 64, b record at 88."""
    data_start = (64 + 4096 * 24 + 63) & ~63
    blobs = [r.tobytes() for r in records]
    total = data_start + sum(len(b) for b in blobs)
    out = bytearray(total)
    struct.pack_into("<II", out, 0, 2, 2)
    off = data_start
    for i, (magic_off, b) in enumerate(zip((64, 88), blobs)):
        struct.pack_into("<II", out, magic_off, 0xDEADBEEF, 1)
        struct.pack_into("<QQ", out, magic_off + 8, len(b), off)
        out[off:off + len(b)] = b
        off += len(b)
    return bytes(out)

def conv_mil(x_shape, w_shape, w_off, b_shape=None, b_off=None,
             pad_type="valid", pad=(0, 0, 0, 0), groups=1):
    head = f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, {list(x_shape)}> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
"""
    bias_arg = "bias = b, " if b_shape else ""
    bias_line = ""
    if b_shape:
        bias_line = (f'    tensor<fp16, {list(b_shape)}> b = const()[name = string("b"), '
                     f'val = tensor<fp16, {list(b_shape)}>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({b_off})))];\n')
    y_shape = out_shape(x_shape, w_shape, pad_type, pad, groups)
    pd_vals = pad if pad_type != "same" else (0, 0, 0, 0)
    body = head + (
        f'    tensor<fp16, {list(w_shape)}> w = const()[name = string("w"), val = tensor<fp16, {list(w_shape)}>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({w_off})))];\n'
        + bias_line +
        f'    string pt = const()[name = string("pt"), val = string("{pad_type}")];\n'
        '    tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];\n'
        f'    tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([{pd_vals[0]}, {pd_vals[1]}, {pd_vals[2]}, {pd_vals[3]}])];\n'
        '    tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];\n'
        f'    int32 gp = const()[name = string("gp"), val = int32({groups})];\n'
        f'    tensor<fp16, {list(y_shape)}> y = conv({bias_arg}dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("y")];\n'
        '  } -> (y);\n}\n')
    return body, y_shape

def out_shape(x_shape, w_shape, pad_type, pad, groups):
    # stride 1, dilation 1; pad = [top, bottom, left, right]
    n, cin, h, w = x_shape
    cout, cin_g, kh, kw = w_shape
    if pad_type == "same":
        pt = max(0, (h - 1) + kh - h); pb = pt - pt // 2; pt = pt // 2
        pl = max(0, (w - 1) + kw - w); pr = pl - pl // 2; pl = pl // 2
    else:
        pt, pb, pl, pr = pad
    return (n, cout, (h + pt + pb - kh) + 1, (w + pl + pr - kw) + 1)

def ref_conv(x32, W32, b32, pad_type, pad, groups):
    n, cin, h, w = x32.shape
    cout, cin_g, kh, kw = W32.shape
    if pad_type == "same":
        pt = max(0, (h - 1) + kh - h); pb = pt - pt // 2; pt = pt // 2
        pl = max(0, (w - 1) + kw - w); pr = pl - pl // 2; pl = pl // 2
    else:
        pt, pb, pl, pr = pad
    xp = np.pad(x32, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    oh = (h + pt + pb - kh) + 1
    ow = (w + pl + pr - kw) + 1
    s = xp.strides
    cols = np.lib.stride_tricks.as_strided(
        xp, shape=(n, cin, kh, kw, oh, ow),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    per = cout // groups
    out = np.empty((n, cout, oh, ow), dtype=np.float32)
    if cin_g == 1 and per == 1:
        out = np.einsum("nckhij,ckh->ncij", cols, W32[:, 0], optimize=True)
    else:
        for g in range(groups):
            xg = cols[:, g * cin_g:(g + 1) * cin_g].reshape(n, cin_g * kh * kw, oh * ow)
            wg = W32[g * per:(g + 1) * per].reshape(per, cin_g * kh * kw)
            out[:, g * per:(g + 1) * per] = (wg @ xg).reshape(n, per, oh, ow)
    if b32 is not None:
        out = out + b32.reshape(1, cout, 1, 1)
    return out

def case_conv(name, x_shape, w_shape, groups, pad_type, pad, bias, seed):
    """F1/F2/F3/F4: rank-4 conv MIL spelling with real random weights."""
    wd = WORKDIR / name
    (wd / "model").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal(w_shape).astype(np.float16)
    b = (rng.standard_normal((w_shape[0],)).astype(np.float16) * 0.05
         if bias else None)
    payload = blob([W] + ([b] if b is not None else []))
    (wd / "model/weights.bin").write_bytes(payload)
    w_off, b_off = 64, 88
    mil, y_shape = conv_mil(x_shape, w_shape, w_off,
                            (w_shape[0],) if bias else None, b_off,
                            pad_type, pad, groups)
    (wd / "model/test.mil").write_text(mil)
    anec = compile_case(wd, name)

    def ref_fn(x):
        W32 = W.astype(np.float32)
        b32 = b.astype(np.float32) if b is not None else None
        return ref_conv(x.astype(np.float32), W32, b32, pad_type, pad, groups)

    return {"name": name, "anec": anec, "x_shape": x_shape,
            "y_shape": y_shape, "ref": ref_fn}

def case_struct(name, mil, x_shape, y_shape, ref_fn):
    """slice / transpose: no weights."""
    wd = WORKDIR / name
    (wd / "model").mkdir(parents=True, exist_ok=True)
    (wd / "model/test.mil").write_text(mil)
    anec = compile_case(wd, name)
    return {"name": name, "anec": anec, "x_shape": x_shape,
            "y_shape": y_shape, "ref": ref_fn}

def compile_case(wd, name):
    out = wd / "out"
    import shutil
    if out.exists():
        shutil.rmtree(out)
    r = subprocess.run([COMPILER, "--mil", str(wd / "model/test.mil"),
                        "--model-root", str(wd / "model"), "--target", "H13",
                        "--format", "anec", "--output", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, f"{name}: compile failed: {r.stderr[-800:]}"
    anec = out / "program-0.anec"
    assert anec.exists(), f"{name}: no program-0.anec"
    return anec

MIL_SLICE = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 749]> x) {
    tensor<int32, [4]> eb = const()[name = string("eb"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    tensor<int32, [4]> ee = const()[name = string("ee"), val = tensor<int32, [4]>([1, 8, 375, 375])];
    tensor<fp16, [1, 8, 375, 375]> y = slice_by_index(x = x, begin = eb, end = ee)[name = string("y")];
  } -> (y);
}
"""

def mil_transpose(x_shape, y_shape, rank):
    perm = "[0, 2, 1]" if rank == 3 else "[0, 2, 1, 3]"
    pl = f"tensor<int32, [{rank}]> perm = const()[name = string(\"perm\"), val = tensor<int32, [{rank}]>({perm})];"
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, {list(x_shape)}> x) {{
    {pl}
    tensor<fp16, {list(y_shape)}> y = transpose(perm = perm, x = x)[name = string("y")];
  }} -> (y);
}}
"""

def build_cases():
    cases = []
    # F1 in-proj: k1x1 g1 valid, 1024 -> 2048, W-major encoder spell
    cases.append(case_conv("f1-inproj-k1x1-c1024-n2048-wmaj",
                           (1, 1024, 1, 375), (2048, 1024, 1, 1),
                           1, "valid", (0, 0, 0, 0), False, 101))
    # F2 depthwise+bias: k1x9 g1024 same
    cases.append(case_conv("f2-depthwise-k1x9-c1024-g1024-bias",
                           (1, 1024, 1, 375), (1024, 1, 1, 9),
                           1024, "same", None, True, 202))
    # F3 out-proj: k1x1 g1 valid, 1024 -> 1024
    cases.append(case_conv("f3-outproj-k1x1-c1024-n1024",
                           (1, 1024, 1, 375), (1024, 1024, 1, 1),
                           1, "valid", (0, 0, 0, 0), False, 303))
    # F4 padconv: rank-4 custom pad [0,0,1,0] (W left pad), g8, 749 -> 750
    cases.append(case_conv("f4-padconv-c8-n8-k1x1-g8-p0010",
                           (1, 8, 375, 749), (8, 1, 1, 1),
                           8, "custom", (0, 0, 1, 0), False, 404))
    # slice last-dim window (rel-pos)
    cases.append(case_struct(
        "slice-lastdim-749-375", MIL_SLICE,
        (1, 8, 375, 749), (1, 8, 375, 375),
        lambda x: x.astype(np.float32)[..., :375]))
    # transposes r3 [0,2,1], both directions
    cases.append(case_struct(
        "transpose-r3-375x1024",
        mil_transpose((1, 375, 1024), (1, 1024, 375), 3),
        (1, 375, 1024), (1, 1024, 375),
        lambda x: x.astype(np.float32).transpose(0, 2, 1)))
    cases.append(case_struct(
        "transpose-r3-1024x375",
        mil_transpose((1, 1024, 375), (1, 375, 1024), 3),
        (1, 1024, 375), (1, 375, 1024),
        lambda x: x.astype(np.float32).transpose(0, 2, 1)))
    # transposes r4 [0,2,1,3], both encoder geometries
    cases.append(case_struct(
        "transpose-r4-8x375x128",
        mil_transpose((1, 8, 375, 128), (1, 375, 8, 128), 4),
        (1, 8, 375, 128), (1, 375, 8, 128),
        lambda x: x.astype(np.float32).transpose(0, 2, 1, 3)))
    cases.append(case_struct(
        "transpose-r4-375x256x16",
        mil_transpose((1, 375, 256, 16), (1, 256, 375, 16), 4),
        (1, 375, 256, 16), (1, 256, 375, 16),
        lambda x: x.astype(np.float32).transpose(0, 2, 1, 3)))
    return cases

def main():
    if not Path(COMPILER).exists():
        print(f"compiler missing: {COMPILER}", file=sys.stderr)
        return 1
    summary = {"compiler": COMPILER, "budget": BUDGET, "cases": {}}
    failures = []
    for case in build_cases():
        try:
            rels = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                x = rng.standard_normal(case["x_shape"]).astype(np.float16)
                x4 = x.reshape(1, 1, *x.shape[-2:]) if x.ndim == 3 else x
                y4_shape = (1, 1, *case["y_shape"][-2:]) \
                    if len(case["y_shape"]) == 3 else case["y_shape"]
                raw, d0 = run(case["anec"], x4)
                expected = row_bytes(y4_shape[3]) * int(np.prod(y4_shape[:3]))
                assert d0 >= expected, \
                    f"{case['name']}: dst {d0} < expected {expected}"
                y = unpack(raw[:expected], y4_shape).reshape(case["y_shape"])
                ref = case["ref"](x)
                assert y.shape == ref.shape, \
                    f"{case['name']}: y {y.shape} != ref {ref.shape}"
                rels.append(float(np.linalg.norm(y - ref)
                                  / np.linalg.norm(ref)))
        except Exception as exc:
            print(f"{case['name']}: ERROR {exc}")
            summary["cases"][case["name"]] = {"error": str(exc), "pass": False}
            failures.append(case["name"])
            continue
        worst = max(rels)
        ok = worst <= BUDGET
        if not ok:
            failures.append(case["name"])
        summary["cases"][case["name"]] = {
            "anec": str(case["anec"]),
            "sha256": hashlib.sha256(case["anec"].read_bytes()).hexdigest()[:16],
            "rngs": dict(zip(map(str, SEEDS), [round(r, 6) for r in rels])),
            "worst": round(worst, 6), "pass": ok,
        }
        print(f"{case['name']}: rels={['%.6f' % r for r in rels]} "
              f"worst={worst:.6f} pass={ok}")
    summary["pass"] = not failures
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "gate-summary.json").write_text(json.dumps(summary, indent=2))
    print("DEV_GATE_CONV:", "PASS" if summary["pass"] else "FAIL")
    return 0 if summary["pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
