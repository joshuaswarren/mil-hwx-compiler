#!/usr/bin/env python3
"""dev_gate_ffn: end-to-end device-execution gate for the two FFN rank-3
linear geometries (m375 k1024 n4096 uniform, m375 k4096 n1024 uniform).
Compiles real-weight programs with the mil-hwxc under test and executes
them on the ANE against fp32 references across three rngs. Exits 0 on PASS.
Run on t6001-test-host under /tmp/m1-gpu.lock."""
import ctypes, subprocess, sys, hashlib, struct
from pathlib import Path
import numpy as np

COMPILER = "$HOME/src/mil-hwx-compiler/build/mil-hwxc"
LIB = "/var/tmp/t6001-test-host-oproj-place/libane-strict-fill.so"
ENV = {"LD_LIBRARY_PATH": "$HOME/.local/mil-hwx-gnustep/lib",
       "PATH": "/usr/bin:/bin", "HOME": "$HOME"}
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

def run(anec_path, x16):
    nn = lib.__ane_init(str(anec_path).encode(), 0)
    assert nn
    try:
        s0, d0 = int(lib.__ane_src_size(nn, 0)), int(lib.__ane_dst_size(nn, 0))
        tile = np.zeros(s0, dtype=np.uint8)
        tile[0:x16.nbytes] = np.frombuffer(x16.tobytes(), dtype=np.uint8)
        lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
        assert lib.ane_exec(nn) == 0
        y = np.zeros(d0 // 2, dtype="<f2")
        lib.__ane_read(nn, ctypes.c_char_p(y.ctypes.data), 0)
        return y.astype(np.float32)
    finally:
        lib.__ane_free(nn)

def compile_geom(name, k, n):
    wd = Path(f"/tmp/dev-gate-ffn/{name}")
    (wd / "model").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(name) % 1000)
    W = rng.standard_normal((n, k)).astype(np.float16)
    bias = rng.standard_normal((n,)).astype(np.float16) * 0.05
    data_start = (64 + 4096 * 24 + 63) & ~63
    wb, bb = W.tobytes(), bias.tobytes()
    blob = bytearray(data_start + len(wb) + len(bb))
    struct.pack_into('<II', blob, 0, 2, 2)
    struct.pack_into('<II', blob, 64, 0xDEADBEEF, 1)
    struct.pack_into('<QQ', blob, 72, len(wb), data_start)
    struct.pack_into('<II', blob, 88, 0xDEADBEEF, 1)
    struct.pack_into('<QQ', blob, 96, len(bb), data_start + len(wb))
    blob[data_start:data_start + len(wb)] = wb
    blob[data_start + len(wb):] = bb
    (wd / "model/weights.bin").write_bytes(bytes(blob))
    np.save(wd / "W.npy", W); np.save(wd / "b.npy", bias)
    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, 375, {k}]> x) {{
    tensor<fp16, [{n}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{n}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [{n}]> b = const()[name = string("b"), val = tensor<fp16, [{n}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(88)))];
    tensor<fp16, [1, 375, {n}]> y = linear(weight = w, bias = b, x = x)[name = string("y")];
  }} -> (y);
}}
'''
    (wd / "model/test.mil").write_text(mil)
    out = wd / "out"
    import shutil
    if out.exists(): shutil.rmtree(out)
    r = subprocess.run([COMPILER, "--mil", str(wd / "model/test.mil"),
                        "--model-root", str(wd / "model"), "--target", "H13",
                        "--format", "anec", "--output", str(out)],
                       capture_output=True, text=True, env=ENV)
    assert r.returncode == 0, r.stderr[-500:]
    anec = out / "program-0.anec"
    return anec, W, bias

summary = {"geometries": {}}
for name, k, n in (("mm1", 1024, 4096), ("mm2", 4096, 1024)):
    anec, W, bias = compile_geom(name, k, n)
    rels = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((375, k)).astype(np.float16)
        y = run(anec, x)[:375 * n].reshape(375, n)
        ref = x.astype(np.float32) @ W.astype(np.float32).T + bias.astype(np.float32)
        rels.append(float(np.linalg.norm(y - ref) / np.linalg.norm(ref)))
    worst = max(rels)
    summary["geometries"][name] = {
        "anec": str(anec), "sha256": hashlib.sha256(anec.read_bytes()).hexdigest()[:16],
        "rngs": dict(zip(map(str, SEEDS), rels)), "worst": worst,
        "pass": worst <= BUDGET,
    }
    print(f"{name}: rels={['%.6f' % r for r in rels]} worst={worst:.6f} pass={worst<=BUDGET}")

ok = all(g["pass"] for g in summary["geometries"].values())
print("DEV_GATE_FFN:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
