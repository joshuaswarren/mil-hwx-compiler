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
    """2026-09-19 finding: ane_exec returns 0 but the logical_not dst reads
    all zeros for any input, while the cast program is device-exact through
    the identical send/exec/read path. Emission is verified correct offline
    (surface tables, selector slots, DMA extents all consistent with the
    working cast). This diagnostic stages three set lanes, runs once, and
    dumps the full dst allocation plus the set-lane offsets so the next
    device window can distinguish engine-write vs read-path placement.
    """
    import ctypes
    import struct as _struct
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
    diag = root / "diag"
    diag.mkdir(exist_ok=True)
    (diag / "m.mil").write_text(NOT_DIAG_MIL)
    r = subprocess.run(
        [str(COMPILER), "--mil", str(diag / "m.mil"), "--model-root", str(diag),
         "--output", str(diag / "out"), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL: diag compile: {r.stderr.strip()}")
        return 1
    anec = diag / "out/program-0.anec"
    # preserve the emitted ANEC next to this receipt for offline diffing
    (root / "logical-not-diag.anec").write_bytes(anec.read_bytes())
    x = np.zeros((375, 384), dtype=np.uint8)
    x[0, 0] = 0x01
    x[0, 1] = 0x01
    x[100, 5] = 0x01
    np.save(diag / "x.npy", x)
    nn = lib.__ane_init(str(anec).encode(), 0)
    if not nn:
        print("FAIL: __ane_init returned null")
        return 1
    s0 = int(lib.__ane_src_size(nn, 0))
    d0 = int(lib.__ane_dst_size(nn, 0))
    print(f"src {s0} dst {d0} (logical 144000)")
    tile = np.zeros(s0, dtype=np.uint8)
    tile[0:x.nbytes] = np.frombuffer(x.tobytes(), dtype=np.uint8)
    lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
    rc = lib.ane_exec(nn)
    print("ane_exec:", rc)
    out = np.zeros(d0, dtype=np.uint8)
    lib.__ane_read(nn, ctypes.c_char_p(out.ctypes.data), 0)
    (diag / "dst-dump.bin").write_bytes(out.tobytes())
    nz = np.nonzero(out)[0]
    print("nonzero dst bytes:", len(nz), "first 20 offsets:", nz[:20].tolist())
    # set-lane home offsets for reference: 0x00, 0x01 (row 0), 100*384+5
    print("set-lane home offsets: [0, 1, 38405]")
    lib.__ane_free(nn)
    return 0 if len(nz) else 2  # 2 = the documented all-zero finding


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

        # logical_not at [1,1,375,375]: 375 rows of 384 bytes.
        vals = (rng.integers(0, 2, size=(375, 375))).astype(np.uint8)
        x_not = np.zeros((375, 384), dtype=np.uint8)
        x_not[:, :375] = vals
        anec = compile_case("not", NOT_MIL, root)
        out = run_device(anec, x_not, "not", root)
        got = out[:375 * 384].reshape(375, 384)[:, :375]
        expect = vals ^ 0xFF
        if not np.array_equal(got, expect):
            bad = int(np.count_nonzero(got != expect))
            print(f"FAIL: logical_not mismatch on {bad} lanes")
            return 1
        print("logical_not [1,1,375,375]: exact byte-not on 140625 lanes")
    print("dev_gate_mask: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
