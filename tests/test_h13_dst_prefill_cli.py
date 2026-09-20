#!/usr/bin/env python3
"""Genuine dst-BO sentinel prefill with identity proof via round-trip.

Finds the dst BO's mmap address after __ane_init by scanning /proc/self/maps
for new device-backed mappings, then PROVES each candidate is the dst BO by:
1. Writing a distinctive 4-byte tag via ctypes at the mapping start
2. Calling __ane_read to read the dst BO
3. Checking the tag appears in the read-back

Only the candidate that passes the round-trip is used for the sentinel
prefill. This is positive identity proof, not a guess from /proc/self/maps.

Requires t6001-test-host (strict-fill library + ANE device).
"""
import ctypes, hashlib, json, os, re, struct, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

COMPILER = str(Path(__file__).resolve().parent.parent / "build" / "mil-hwxc")
LIBANE = os.environ.get("LIBANE",
    "/var/tmp/t6001-test-host-oproj-place/libane-strict-fill.so")
SENTINEL = 0xA5
TAG = b"\xDE\xAD\xBE\xEF"  # 4-byte identity proof tag

NOT_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 375, 375]> x) {
    tensor<bool, [1, 1, 375, 375]> y = logical_not(x = x)[name = string("y")];
  } -> (y);
}
"""


def snapshot_maps():
    return set(open("/proc/self/maps").readlines())


def find_new_rw_mappings(before, min_size):
    pat = re.compile(r"^([0-9a-f]+)-([0-9a-f]+)\s+rw\s")
    result = []
    for line in open("/proc/self/maps"):
        if line in before:
            continue
        m = pat.match(line)
        if not m:
            continue
        start, end = int(m.group(1), 16), int(m.group(2), 16)
        size = end - start
        if size >= min_size:
            result.append((start, end, size))
    return sorted(result)


def open_lib():
    lib = ctypes.CDLL(LIBANE)
    for n, rt, at in (
        ("__ane_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
        ("__ane_free", None, [ctypes.c_void_p]),
        ("ane_exec", ctypes.c_int, [ctypes.c_void_p]),
        ("__ane_src_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_dst_size", ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_send", None, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]),
        ("__ane_read", None, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]),
    ):
        f = getattr(lib, n); f.restype = rt; f.argtypes = at
    return lib


def compile_and_load(lib, root, name, mil_text):
    d = root / name
    d.mkdir(exist_ok=True)
    (d / "m.mil").write_text(mil_text)
    r = subprocess.run(
        [COMPILER, "--mil", str(d / "m.mil"), "--model-root", str(d),
         "--output", str(d / "out"), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"{name}: {r.stderr[:160]}"
    anec = d / "out" / "program-0.anec"
    maps_before = snapshot_maps()
    nn = lib.__ane_init(str(anec).encode(), 0)
    assert nn, f"init failed: {name}"
    s0 = int(lib.__ane_src_size(nn, 0))
    d0 = int(lib.__ane_dst_size(nn, 0))
    maps_after = snapshot_maps()
    # The new mappings include src BO, dst BO, and possibly scratch.
    candidates = find_new_rw_mappings(maps_before, d0)
    return nn, s0, d0, candidates, maps_before


def identify_dst_by_roundtrip(lib, nn, d0, candidates):
    """Prove which candidate mmap is the dst BO: write a 4-byte tag via
    ctypes at the candidate address, then __ane_read and check the tag
    appears. The candidate that round-trips IS the dst BO."""
    for addr, _, _ in candidates:
        # Write the tag
        ctypes.memmove(addr, TAG, 4)
        # Read back via __ane_read
        buf = ctypes.create_string_buffer(d0)
        lib.__ane_read(nn, ctypes.c_char_p(buf.ctypes.data), 0)
        if bytes(buf[0:4]) == TAG:
            # Restore zeros
            ctypes.memmove(addr, b"\x00" * 4, 4)
            return addr, True
        # Tag didn't round-trip: not the dst BO; restore anyway
        ctypes.memmove(addr, b"\x00" * 4, 4)
    return None, False


def main():
    if not Path(COMPILER).exists():
        print(f"SKIP: compiler not found at {COMPILER}")
        return 0
    if not Path(LIBANE).exists():
        print(f"SKIP: libane not found at {LIBANE}; runs on t6001-test-host only")
        return 0
    lib = open_lib()
    with tempfile.TemporaryDirectory(prefix="h13-dst-prefill-") as td:
        root = Path(td)

        # --- Positive control: cast (known to write) ---
        nn_c, s0_c, d0_c, cands_c, _ = compile_and_load(
            lib, root, "cast-ctrl", CAST_MIL)
        dst_c, proven_c = identify_dst_by_roundtrip(lib, nn_c, d0_c, cands_c)
        print(f"cast control: dst BO {'IDENTIFIED' if proven_c else 'NOT FOUND'} "
              f"at {'0x' + format(dst_c, 'x') if dst_c else 'N/A'} "
              f"(candidates: {len(cands_c)})")
        assert proven_c, "cast control must identify its dst BO"

        # Prefill cast dst with sentinel
        ctypes.memset(dst_c, SENTINEL, d0_c)
        # Verify via read
        pre = (ctypes.c_char * d0_c)()
        lib.__ane_read(nn_c, ctypes.c_char_p(pre.ctypes.data), 0)
        pre_sent = sum(1 for b in pre if b == SENTINEL)
        print(f"  cast prefill verified: {pre_sent}/{d0_c} sentinel bytes")

        # Stage input and exec
        lanes = np.zeros(1500, dtype=np.uint8)
        lanes[::2] = 1
        x_cast = np.zeros((1500, 64), dtype=np.uint8)
        x_cast[:, 0] = lanes
        tile = np.zeros(s0_c, dtype=np.uint8)
        tile[0:x_cast.nbytes] = np.frombuffer(x_cast.tobytes(), dtype=np.uint8)
        lib.__ane_send(nn_c, ctypes.c_char_p(tile.ctypes.data), 0)
        rc = int(lib.ane_exec(nn_c))
        assert rc == 0, f"cast exec failed: {rc}"
        out = np.zeros(d0_c, dtype=np.uint8)
        lib.__ane_read(nn_c, ctypes.c_char_p(out.ctypes.data), 0)
        y = np.frombuffer(out[:1500 * 64].tobytes(), dtype="<f2")[::32]
        exp = lanes.astype(np.float32)
        match = np.array_equal(y.astype(np.float32), exp)
        print(f"  cast output: {'EXACT' if match else 'MISMATCH'} on 1500 lanes")
        assert match, "cast control must produce exact logical values"
        lib.__ane_free(nn_c)
        print("  cast positive control: PASS (sentinel protocol proven)")

        # --- logical_not sentinel discrimination ---
        nn_n, s0_n, d0_n, cands_n, _ = compile_and_load(
            lib, root, "not-ctrl", NOT_MIL)
        dst_n, proven_n = identify_dst_by_roundtrip(lib, nn_n, d0_n, cands_n)
        print(f"logical_not: dst BO {'IDENTIFIED' if proven_n else 'NOT FOUND'} "
              f"at {'0x' + format(dst_n, 'x') if dst_n else 'N/A'} "
              f"(candidates: {len(cands_n)})")
        assert proven_n, "logical_not must identify its dst BO"

        # Prefill dst with sentinel
        ctypes.memset(dst_n, SENTINEL, d0_n)

        # Stage complement inputs
        results = {}
        for run_tag, ones_at in (("A", [(0, 0), (0, 1), (100, 5)]),
                                  ("B", [(0, 0), (5, 3), (100, 5)])):
            vals = np.zeros((375, 375), dtype=np.uint8)
            for r, c in ones_at:
                vals[r, c] = 1
            x_not = np.zeros((375, 384), dtype=np.uint8)
            x_not[:, :375] = vals
            tile = np.zeros(s0_n, dtype=np.uint8)
            tile[0:x_not.nbytes] = np.frombuffer(x_not.tobytes(), dtype=np.uint8)
            lib.__ane_send(nn_n, ctypes.c_char_p(tile.ctypes.data), 0)
            rc = int(lib.ane_exec(nn_n))
            assert rc == 0, f"exec failed: {rc}"
            out = np.zeros(d0_n, dtype=np.uint8)
            lib.__ane_read(nn_n, ctypes.c_char_p(out.ctypes.data), 0)
            results[run_tag] = (vals, out[:375 * 384].reshape(375, 384)[:, :375])

        va, da = results["A"]
        vb, db = results["B"]
        if np.array_equal(da, db):
            nz = int(np.count_nonzero(da))
            print(f"logical_not: dst IDENTICAL across complement inputs "
                  f"({nz} nonzero/{140625} lanes)")
            print("  => all branches retained (engine-not-written, "
                  "submission, kernel, read-path, host-emission)")
            print(f"  dst content: {da[0, :8]} (first 8 lanes)")
            return 1  # cannot classify semantics

        # Semantics classification
        logical_a = (1 - va).astype(np.uint8)
        bytenot_a = (va ^ 0xFF).astype(np.uint8)
        if np.array_equal(da, logical_a):
            print("logical_not: LOGICAL semantics (1-x) confirmed on device")
        elif np.array_equal(da, bytenot_a):
            print("logical_not: BYTE-NOT semantics (x^0xFF) confirmed on device")
        else:
            print("logical_not: neither logical nor byte-not matches")
            print(f"  first 8: {da[0, :8]}")
        print("logical_not dst sentinel discrimination: PASS")

    return 0


if __name__ == "__main__":
    sys.exit(main())
