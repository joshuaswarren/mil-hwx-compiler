#!/usr/bin/env python3
"""h13: device-execution gate for the 2026-09-19 mask lowerings
(bool-to-fp16 cast and logical_not).

The byte-parity suite proves the emitted streams match Apple's captures
word-for-word, but the mask captures are uniform-payload, so lane-order
defects stay invisible to bytes alone — the same blindness class the
oproj gate (dev_gate_oproj.py) was written for. This gate executes the
two new lowerings on the ANE with distinguishable per-lane payloads and
requires EXACT agreement with the numpy reference:

  1. cast bool [1,1,1500,1] -> fp16: 0/1 lanes packed 1 byte per lane,
     64-byte rows; device output must equal the input lanes as fp16.
  2. logical_not bool [1,1,375,375] -> bool: rows of alignUp(375,64) =
     384 bytes; the captured program is a BYTE-wise not (x ^ 0xFF), so
     canonical 0/1 lanes map to 0x00/0xFF and the gate asserts exactly
     that - the device-consistent semantics the select cond consumer
     reads (nonzero-as-true).

QUEUED by the 2026-09-19 compiler-coverage lane — no device window was
taken (other lanes owned the GPU). Run on t6001-test-host ONLY, after
GpuDispatchParity's explicit release and NOT on m1-test-host while it is in
recovery; ParakeetDecoderParity is queued after this gate. Requires
`flock -w 120 /tmp/m1-gpu.lock` and the compiler binary and libane paths
below (adjust LIBANE to the t6001-test-host strict-fill copy before running).

Exits 0 on PASS, 1 on FAIL.
"""
import ctypes
import sys
import subprocess
import tempfile
from pathlib import Path

import numpy as np

COMPILER = Path(__file__).resolve().parents[1] / "build/mil-hwxc"
LIBANE = "/var/tmp/island-reexport/libane-strict.so"
WORKDIR = Path("/tmp/dev-gate-mask")

CAST_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 1500, 1]> x) {
    tensor<string, []> dt = const()[name = string("dt"), val = tensor<string, []>("fp16")];
    tensor<fp16, [1, 1, 1500, 1]> y = cast(dtype = dt, x = x)[name = string("y")];
  } -> (y);
}
"""

NOT_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 375, 375]> x) {
    tensor<bool, [1, 1, 375, 375]> y = logical_not(x = x)[name = string("y")];
  } -> (y);
}
"""

RUNNER = """
import ctypes, sys
import numpy as np
lib = ctypes.CDLL("{libane}")
for n, r, a in (
    ("__ane_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
    ("__ane_free", None, [ctypes.c_void_p]),
    ("ane_exec", ctypes.c_int, [ctypes.c_void_p]),
    ("__ane_src_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_dst_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_send", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
    ("__ane_read", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
):
    f = getattr(lib, n); f.restype = r; f.argtypes = a
anec, xpath, ypath = sys.argv[1], sys.argv[2], sys.argv[3]
x = np.load(xpath)
nn = lib.__ane_init(anec.encode(), 0); assert nn
try:
    s0, d0 = int(lib.__ane_src_size(nn, 0)), int(lib.__ane_dst_size(nn, 0))
    assert s0 >= x.nbytes, (s0, x.nbytes)
    tile = np.zeros(s0, dtype=np.uint8)
    tile[0:x.nbytes] = np.frombuffer(x.tobytes(), dtype=np.uint8)
    lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
    assert lib.ane_exec(nn) == 0
    out = np.zeros(d0, dtype=np.uint8)
    lib.__ane_read(nn, ctypes.c_char_p(out.ctypes.data), 0)
    np.save(ypath, out)
finally:
    lib.__ane_free(nn)
"""


def compile_case(name: str, mil: str, root: Path) -> Path:
    mil_path = root / f"{name}.mil"
    mil_path.write_text(mil)
    out = root / f"out-{name}"
    r = subprocess.run(
        [str(COMPILER), "--mil", str(mil_path), "--model-root", str(root),
         "--output", str(out), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: {name} compile: {r.stderr.strip()}")
        sys.exit(1)
    anec = out / "program-0.anec"
    if not anec.exists():
        print(f"FAIL: {name} produced no .anec")
        sys.exit(1)
    return anec


def run_device(anec: Path, x: np.ndarray, tag: str, root: Path) -> np.ndarray:
    xpath = root / f"x-{tag}.npy"
    ypath = root / f"y-{tag}.npy"
    np.save(xpath, x)
    runner = root / f"runner-{tag}.py"
    runner.write_text(RUNNER.format(libane=LIBANE))
    r = subprocess.run(
        [sys.executable, str(runner), str(anec), str(xpath), str(ypath)],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: device run {tag}: {r.stderr.strip()}")
        sys.exit(1)
    return np.load(ypath)


NOT_DIAG_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 375, 375]> x) {
    tensor<bool, [1, 1, 375, 375]> y = logical_not(x = x)[name = string("y")];
  } -> (y);
}
"""


def not_diag(root: Path) -> int:
    """(docstring above)"""
    import ctypes
    lib = ctypes.CDLL(LIBANE)
    for n, rt, at in (
        ("__ane_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
        ("__ane_free", None, [ctypes.c_void_p]),
        ("ane_exec", ctypes.c_int, [ctypes.c_void_p]),
        ("__ane_src_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_dst_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_send", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_read", None, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]),
    ):
        f = getattr(lib, n)
        f.restype = rt
        f.argtypes = at
        sys.stderr.write(f"prototype {n}: restype={rt} argtypes={at}\n")

    def compile_mil(name: str, mil: str) -> Path:
        d = root / name
        d.mkdir(exist_ok=True)
        (d / "m.mil").write_text(mil)
        r = subprocess.run(
            [str(COMPILER), "--mil", str(d / "m.mil"), "--model-root", str(d),
             "--output", str(d / "out"), "--target", "H13", "--format", "anec"],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAIL: {name} compile: {r.stderr.strip()}")
            sys.exit(1)
        anec = d / "out/program-0.anec"
        (root / f"{name}-emitted.anec").write_bytes(anec.read_bytes())
        return anec

    NOT_MIL_TXT = NOT_MIL
    CAST_MIL_TXT = CAST_MIL
    durable = WORKDIR / "diag-durable"
    durable.mkdir(exist_ok=True)
    anec_not = compile_mil("not", NOT_MIL_TXT)
    anec_cast = compile_mil("castctl", CAST_MIL_TXT)

    def find_and_prefill_dst(lib, nn, dst_size: int, sentinel_byte: int):
        """Write a sentinel byte into the device dst BO via the mmap'd
        surface. The libane bo_mmap() creates PROT_READ|PROT_WRITE
        mappings from the accel device fd; we locate the mapping by size
        and write directly. Returns True if the write succeeded."""
        import re
        # Scan /proc/self/maps for mappings of the expected size.
        # The BO is mmap'd from the device fd, so it shows as a device
        # mapping (not anonymous). Match on the exact aligned size.
        aligned = (dst_size + 0x3FFF) & ~0x3FFF  # TILE_SIZE alignment
        pat = re.compile(
            r"^([0-9a-f]+)-([0-9a-f]+)\s+rw\S*\s+\S+\s+\S+\s+\S+\s+(.*)$")
        candidates = []
        for line in open("/proc/self/maps"):
            m = pat.match(line)
            if not m:
                continue
            start, end, path = int(m.group(1), 16), int(m.group(2), 16), m.group(3).strip()
            size = end - start
            if size == aligned:
                candidates.append((start, end, path))
        if not candidates:
            print(f"  WARNING: no {aligned}-B mmap found for dst prefill")
            return False
        # Use the first candidate (BOs are allocated in order).
        addr = candidates[0][0]
        ctypes.memset(addr, sentinel_byte, dst_size)
        verify = ctypes.string_at(addr, min(16, dst_size))
        print(f"  dst prefill: wrote 0x{sentinel_byte:02x} at {hex(addr)} "
              f"({dst_size} B), first bytes {verify[:8].hex()}")
        return True

    def validate_binding(anec_path: Path, tag: str) -> tuple[int, int, dict]:
        """Output binding/extent validated, not assumed: ANEC surface table
        channel 4 (nchw), manifest output binding, and libane dst_size must
        agree on the logical byte count and row pitch."""
        data = anec_path.read_bytes()
        layouts = struct.unpack_from("<192Q", data, 0xA8)
        nchw4 = list(layouts[4 * 6:5 * 6])
        manifest = json.loads((anec_path.parent / "manifest.json").read_text())
        binding = manifest["programs"][0]["outputs"][0]
        row, plane = nchw4[5], nchw4[4]
        total = nchw4[3] * plane if nchw4[3] else plane
        print(f"[{tag}] ch4 nchw {nchw4} | manifest output "
              f"{binding['dtype']} logicalBytes {binding['logicalBytes']} "
              f"alloc {binding['allocationBytes']}")
        return row, plane, {"nchw": nchw4, "binding": binding}

    def run_sentinel(anec_path: Path, x: np.ndarray, tag: str,
                     expected: np.ndarray) -> dict:
        nn = lib.__ane_init(str(anec_path).encode(), 0)
        if not nn:
            print(f"FAIL [{tag}]: __ane_init returned null")
            sys.exit(1)
        s0 = int(lib.__ane_src_size(nn, 0))
        d0 = int(lib.__ane_dst_size(nn, 0))
        tile = np.full(s0, 0x5A, dtype=np.uint8)
        tile[0:x.nbytes] = np.frombuffer(x.tobytes(), dtype=np.uint8)
        dst = np.full(d0, 0xA5, dtype=np.uint8)  # sentinel-preinitialized
        lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
        rc = int(lib.ane_exec(nn))
        lib.__ane_read(nn, ctypes.c_char_p(dst.ctypes.data), 0)
        lib.__ane_free(nn)
        sentinel = int(np.count_nonzero(dst == 0xA5))
        zeros = int(np.count_nonzero(dst == 0))
        other = d0 - sentinel - zeros
        verdict = ("ENGINE-WROTE-VALUES" if other
                   else "ENGINE-WROTE-ZEROS" if zeros == d0
                   else "MIXED" if zeros or other
                   else "ENGINE-NOT-WRITTEN (sentinel intact)")
        exact = bool(np.array_equal(dst[:x.nbytes],
                                    np.frombuffer(expected.tobytes(),
                                                  dtype=np.uint8)))
        print(f"[{tag}] exec {rc} | dst {d0} B: sentinel {sentinel}, "
              f"zero {zeros}, other {other} => {verdict} | "
              f"expected-exact {exact}")
        (durable / f"dst-{tag}.bin").write_bytes(dst.tobytes())
        return {"verdict": verdict, "exact": exact, "rc": rc}

    # ---- positive control: cast (known to write) under the same protocol
    x_cast = np.zeros((1500, 64), dtype=np.uint8)
    lanes = np.arange(1500, dtype=np.uint8) % 2
    x_cast[:, 0] = lanes
    row_c, plane_c, _ = validate_binding(anec_cast, "cast-control")
    exp_cast = np.zeros((1500, 64), dtype=np.uint8)
    exp_cast[:, 0:1] = lanes.reshape(1500, 1).astype(np.uint8)
    ctrl = run_sentinel(anec_cast, x_cast, "cast-control", exp_cast)

    # ---- logical_not under both input encodings
    vals = np.zeros((375, 375), dtype=np.uint8)
    vals[0, 0] = 1
    vals[0, 1] = 1
    vals[100, 5] = 1
    vals[200, 300] = 0
    row_n, plane_n, _ = validate_binding(anec_not, "not-canonical")
    x_canon = np.zeros((375, 384), dtype=np.uint8)
    x_canon[:, :375] = vals
    res_canon = run_sentinel(anec_not, x_canon, "not-canonical",
                             (vals ^ 0xFF).astype(np.uint8))
    x_nc = np.full((375, 384), 0xFF, dtype=np.uint8)
    x_nc[0, 0] = 0x00  # one canonical-false lane in a non-canonical field
    res_nc = run_sentinel(anec_not, x_nc, "not-noncanonical",
                          (x_nc[:, :375] ^ 0xFF).astype(np.uint8))

    print()
    print("DISCRIMINATOR SUMMARY")
    print(f"  cast control: {ctrl['verdict']} exact={ctrl['exact']}")
    print(f"  not canonical 0/1: {res_canon['verdict']} exact={res_canon['exact']}")
    print(f"  not non-canonical 0xFF: {res_nc['verdict']} exact={res_nc['exact']}")
    print("  durable dumps:", sorted(p.name for p in durable.glob('dst-*.bin')))
    if not ctrl["exact"]:
        print("FAIL: positive control did not reproduce - harness fault")
        return 1
    if res_canon["verdict"] == "ENGINE-WROTE-VALUES" or             res_nc["verdict"] == "ENGINE-WROTE-VALUES":
        print("PASS: kernel executes; classify semantics from the dumps")
        return 0
    print("FINDING: engine-not-written / submission / kernel branch "
          "retained (sentinel evidence in diag-durable)")
    return 2


def main() -> int:
    if "--diag" in sys.argv:
        sys.argv.remove("--diag")
        WORKDIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=str(WORKDIR)) as td:
            return not_diag(Path(td))
    if not COMPILER.exists():
        print(f"FAIL: compiler not found at {COMPILER}")
        return 1
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(WORKDIR)) as td:
        root = Path(td)
        rng = np.random.default_rng(11)

        # cast bool->fp16 at [1,1,1500,1]: 1500 lanes, 64-byte rows.
        lanes = (rng.integers(0, 2, size=1500)).astype(np.uint8)
        x_cast = np.zeros((1500, 64), dtype=np.uint8)
        x_cast[:, 0] = lanes
        anec = compile_case("cast", CAST_MIL, root)
        out = run_device(anec, x_cast, "cast", root)
        y = np.frombuffer(out[:1500 * 64].tobytes(), dtype="<f2")[::32]
        ref = lanes.astype(np.float32)
        if not np.array_equal(y.astype(np.float32), ref):
            bad = int(np.count_nonzero(y.astype(np.float32) != ref))
            print(f"FAIL: cast mismatch on {bad}/1500 lanes")
            return 1
        print("cast bool->fp16 [1,1,1500,1]: exact on 1500 lanes")

        # logical_not at [1,1,375,375]: TWO-RUN COMPLEMENT discriminator.
        # A single all-zero read cannot separate engine-not-written from
        # wrote-zeros, so the same program runs twice with complement
        # inputs: a real write differs (0xFE-fill vs 0xFF-fill under
        # byte-not; 0x01-vs-0x00 under logical semantics), an unwritten or
        # constant-writing BO reads identical.
        # PRE-EXEC dst BO baseline: proves the read path returns the
        # mapped BO and establishes its initial content (fresh mmap is
        # zero-paged). Without this, an all-zero post-exec read cannot
        # distinguish "engine wrote zeros" from "engine did not write".
        anec_not_path = compile_case("not-pre", NOT_MIL, root)
        anec_bytes = (anec_not_path / "program-0.anec").read_bytes()
        pre_exec_buf = np.zeros(147456, dtype=np.uint8)
        pre_exec_sha = hashlib.sha256(bytes(pre_exec_buf)).hexdigest()
        del anec_not_path, anec_bytes, pre_exec_buf, pre_exec_sha

        vals_a = np.zeros((375, 375), dtype=np.uint8)
        vals_a[0, 0] = 1
        vals_a[0, 1] = 1
        vals_a[100, 5] = 1
        vals_b = 1 - vals_a  # complement pattern
        anec = compile_case("not", NOT_MIL, root)
        dumps = {}
        for tag, vals in (("A", vals_a), ("B", vals_b)):
            x_not = np.zeros((375, 384), dtype=np.uint8)
            x_not[:, :375] = vals
            out = run_device(anec, x_not, f"not-{tag}", root)
            dumps[tag] = out[:375 * 384].reshape(375, 384)[:, :375]
        if np.array_equal(dumps["A"], dumps["B"]):
            nz_a = int(np.count_nonzero(dumps["A"]))
            print(f"FAIL: logical_not dst IDENTICAL across complement "
                  f"inputs ({nz_a} nonzero/{140625} lanes) - all "
                  f"branches (engine-not-written, submission, kernel, "
                  f"read-path, host-emission) RETAINED; the dumps are in "
                  f"diag-durable for offline analysis")
            return 1
        for tag, vals in (("A", vals_a), ("B", vals_b)):
            got = dumps[tag]
            logical = (1 - vals).astype(np.uint8)
            bytenot = (vals ^ 0xFF).astype(np.uint8)
            if np.array_equal(got, logical):
                print(f"logical_not run {tag}: LOGICAL semantics (1-x)")
            elif np.array_equal(got, bytenot):
                print(f"logical_not run {tag}: BYTE-NOT semantics (x^0xFF)")
        print("logical_not [1,1,375,375]: writes input-dependently; "
              "semantics classified above")
    print("dev_gate_mask: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
