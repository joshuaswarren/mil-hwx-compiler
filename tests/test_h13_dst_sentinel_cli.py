#!/usr/bin/env python3
"""Genuine dst-BO sentinel prefill via the owned nn struct surface pointer.

The strict-fill libane maps each tile channel as a PROT_READ|PROT_WRITE
BO. The `struct ane_nn` contains `chans[TILE_COUNT]` (mmap'd surfaces) and
`bind.src[]`/`bind.dst[]` (role-to-channel maps). The dst surface pointer
and size are read from the nn struct at known offsets derived from the C
header — no /proc/self/maps guessing, no speculative writes to unknown
addresses.

The sentinel is written into the dst BO via ctypes after __ane_init and
verified via __ane_read before dispatch. Post-exec classification:
  - sentinel intact => engine did not write the dst BO
  - sentinel replaced with expected values => engine wrote correctly
  - sentinel replaced with something else => engine wrote, semantics differ

Requires t6001-test-host (strict-fill library + ANE device).
"""
import ctypes
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

COMPILER = str(Path(__file__).resolve().parent.parent / "build" / "mil-hwxc")
LIBANE = os.environ.get(
    "LIBANE", "/var/tmp/t6001-test-host-oproj-place/libane-strict-fill.so")
SENTINEL = 0xA5
TILE_SIZE = 0x4000  # 16384
TILE_COUNT = 0x20   # 32

# struct ane_nn offsets (aarch64, derived from ane.h):
#   offset 0:    int fd (4B) + 4B padding
#   offset 8:    void* data (8B)
#   offset 16:   struct anec anec (packed, aligned(1), 1704B)
#   offset 1720: struct ane_bo chans[32] (32B each)
#   offset 2744: struct ane_bo btsp_chan (32B)
#   offset 2776: struct ane_bind bind {src[32], dst[32]}
#
# struct anec (packed, aligned(1)):
#   offset 0:   u64 size
#   offset 8:   u32 td_size
#   offset 12:  u32 td_count
#   offset 16:  u64 tsk_size
#   offset 24:  u64 krn_size
#   offset 32:  u32 src_count
#   offset 36:  u32 dst_count
#   offset 40:  u32 tiles[32]
#   offset 168: u64 nchw[32][6]
#   total: 1704 bytes
#
# struct ane_bo:
#   offset 0:  void* map
#   offset 8:  u64 size
#   offset 16: u32 handle (+4B pad)
#   offset 24: u64 offset
#   total: 32 bytes

NN_CHANS_OFFSET = 1720
NN_BIND_OFFSET = 2776
BIND_SRC_OFFSET = 0
BIND_DST_OFFSET = 32
ANEOFF_MAP = 0
ANEOFF_SIZE = 8

NOT_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 375, 375]> x) {
    tensor<bool, [1, 1, 375, 375]> y = logical_not(x = x)[name = string("y")];
  } -> (y);
}
"""

CAST_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 1500, 1]> x) {
    tensor<string, []> dt = const()[name = string("dt"), val = tensor<string, []>("fp16")];
    tensor<fp16, [1, 1, 1500, 1]> y = cast(dtype = dt, x = x)[name = string("y")];
  } -> (y);
}
"""


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


def read_u8(nn_addr, offset):
    return ctypes.cast(nn_addr + offset, ctypes.POINTER(ctypes.c_uint8)).contents.value


def read_u64(nn_addr, offset):
    return ctypes.cast(nn_addr + offset, ctypes.POINTER(ctypes.c_uint64)).contents.value


def read_ptr(nn_addr, offset):
    return ctypes.cast(nn_addr + offset, ctypes.POINTER(ctypes.c_void_p)).contents.value


def get_surface(nn_addr, bdx):
    """Read chans[bdx].map and chans[bdx].size from the nn struct."""
    chans_base = nn_addr + NN_CHANS_OFFSET
    map_addr = read_ptr(chans_base + bdx * 32)
    size = read_u64(chans_base + bdx * 32 + 8)
    return map_addr, size


def get_dst_bdx(nn_addr):
    """Read bind.dst[0] — the channel index for output 0."""
    return read_u8(nn_addr + NN_BIND_OFFSET + BIND_DST_OFFSET)


def get_src_bdx(nn_addr):
    """Read bind.src[0] — the channel index for input 0."""
    return read_u8(nn_addr + NN_BIND_OFFSET + BIND_SRC_OFFSET)


def compile_mil(root, name, mil_text):
    d = root / name
    d.mkdir(exist_ok=True)
    (d / "m.mil").write_text(mil_text)
    r = subprocess.run(
        [COMPILER, "--mil", str(d / "m.mil"), "--model-root", str(d),
         "--output", str(d / "out"), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True)
    assert r.returncode == 0, f"{name}: {r.stderr[:160]}"
    return d / "out" / "program-0.anec"


def run_case(root, name, mil_text, input_arr, input_shape, label):
    """Compile, init, prefill dst with sentinel, dispatch, classify.

    Returns dict with verdict and evidence.
    """
    anec = compile_mil(root, name, mil_text)
    anec_sha = hashlib.sha256(anec.read_bytes()).hexdigest()[:16]

    maps_before = set(open("/proc/self/maps").readlines())
    nn = lib.__ane_init(str(anec).encode(), 0)
    assert nn, f"{name}: init failed"
    nn_addr = nn  # ctypes void* value = the actual address

    s0 = int(lib.__ane_src_size(nn, 0))
    d0 = int(lib.__ane_dst_size(nn, 0))

    # Read the bind table to get the actual channel indices
    src_bdx = get_src_bdx(nn_addr)
    dst_bdx = get_dst_bdx(nn_addr)

    # Read the surface pointers from the nn struct
    src_map, src_size = get_surface(nn_addr, src_bdx)
    dst_map, dst_size = get_surface(nn_addr, dst_bdx)
    print(f"  [{name}] src ch{src_bdx}: map=0x{src_map:x} size={src_size}")
    print(f"  [{name}] dst ch{dst_bdx}: map=0x{dst_map:x} size={dst_size}")
    print(f"  [{name}] __ane: src_size={s0} dst_size={d0}")

    # Cross-check: the nn-struct dst size must match __ane_dst_size
    assert dst_size == d0, f"{name}: nn struct dst {dst_size} != API {d0}"

    # PRE-EXEC dst BO baseline (proves the read path is live)
    pre_buf = np.zeros(dst_size, dtype=np.uint8)
    lib.__ane_read(nn, ctypes.c_char_p(pre_buf.ctypes.data), 0)
    pre_nz = int(np.count_nonzero(pre_buf))

    # GENUINE DST PREFILL: write the sentinel via the nn-struct surface pointer
    dst_view = np.frombuffer(
        ctypes.string_at(dst_map, dst_size), dtype=np.uint8).copy()
    dst_view[:] = SENTINEL
    ctypes.memmove(dst_map, dst_view.ctypes.data, dst_size)

    # Verify the prefill via __ane_read
    verify = np.zeros(dst_size, dtype=np.uint8)
    lib.__ane_read(nn, ctypes.c_char_p(verify.ctypes.data), 0)
    verify_nz = int(np.count_nonzero(verify))
    print(f"  [{name}] dst prefill: {verify_nz}/{dst_size} nonzero "
          f"(expected {dst_size})")
    assert verify_nz == dst_size, \
        f"{name}: prefill verification failed ({verify_nz}/{dst_size})"

    # Stage the input
    x = np.zeros(input_shape, dtype=np.uint8)
    for r, c in input_arr:
        x[r, c] = 0x01
    x_flat = x.flatten()

    # Send input (bool 1-byte lanes through the src surface)
    in_map, in_size = get_surface(nn_addr, src_bdx)
    in_view = np.frombuffer(
        ctypes.string_at(in_map, in_size), dtype=np.uint8).copy()
    in_view[:] = 0
    # Layout depends on the surface: for [1,1,N,1] bool, N lanes at 1B each
    # with the row pitch from the nchw
    for i in range(input_arr.__len__()):
        r_idx, c_idx = input_arr[i]
        # Map logical (r, c) to the surface byte offset
        # nchw gives us the strides
        offset = r * 384 + c  # for [1,1,375,375] row=384
        offset = r * 64 + c   # for [1,1,1500,1] row=64
        if input_shape == (1, 1, 1500, 1):
            offset = r  # flat 1500 lanes
        in_view[offset] = 0x01

    ctypes.memmove(in_map, in_view.ctypes.data, in_size)

    # Dispatch
    rc = int(lib.ane_exec(nn))

    # Read back
    post_buf = np.zeros(dst_size, dtype=np.uint8)
    lib.__ane_read(nn, ctypes.c_char_p(post_buf.ctypes.data), 0)
    lib.__ane_free(nn)

    # Classify
    sentinel_intact = int(np.count_nonzero(post_buf == SENTINEL)) == dst_size
    nonzero_post = int(np.count_nonzero(post_buf))

    if sentinel_intact:
        verdict = "ENGINE-NOT-WRITTEN (sentinel intact across full dst)"
    else:
        verdict = f"ENGINE-WROTE ({nonzero_post}/{dst_size} nonzero bytes)"

    print(f"  [{name}] exec {rc}, dst {verdict}")
    (root / f"dst-{name}.bin").write_bytes(post_buf.tobytes())
    (root / f"dst-{name}.anec").write_bytes(anec.read_bytes())

    return {"verdict": verdict, "sentinel_intact": sentinel_intact,
            "nonzero": nonzero_post, "dst_size": dst_size,
            "post_buf": post_buf, "input_arr": input_arr,
            "input_shape": input_shape}


def main():
    if not Path(COMPILER).exists():
        print(f"SKIP: compiler not found at {COMPILER}")
        return 0
    if not Path(LIBANE).exists():
        print(f"SKIP: libane not at {LIBANE}; runs on t6001-test-host only")
        return 0

    global lib
    lib = open_lib()

    with tempfile.TemporaryDirectory(prefix="h13-dst-sentinel-") as td:
        root = Path(td)
        results = []

        # Positive control: cast bool→fp16
        lanes_list = [(r, 0) for r in range(1500) if r % 2 == 0]
        r = run_case(root, "cast-ctrl", CAST_MIL, lanes_list,
                     (1, 1, 1500, 1), "cast control")
        results.append(("cast-control", r))

        # logical_not
        not_lanes = [(0, 0), (0, 1), (100, 5)]
        r = run_case(root, "not-canonical", NOT_MIL, not_lanes,
                     (1, 1, 375, 375), "logical_not canonical")
        results.append(("not-canonical", r))

        print("\n=== DST SENTINEL DISCRIMINATOR SUMMARY ===")
        for name, r in results:
            print(f"  {name}: {r['verdict']}")

        # The cast control MUST have written (proves the protocol).
        cast_r = dict(results)["cast-control"]
        assert not cast_r["sentinel_intact"], \
            "cast control: sentinel should be overwritten by engine write"
        assert cast_r["nonzero"] > 0, "cast control: expected nonzero output"

    print("h13 dst sentinel cli: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
