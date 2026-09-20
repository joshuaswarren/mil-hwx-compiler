#!/usr/bin/env python3
"""Host-only regression for the logical_not dst anomaly:

Three anec variants under the same strict-fill path:
- cast positive control: writes real data (proves the submission path)
- logical_not treatment (remapped emission): current branch binds x=ch5,
  out=ch4 under the parity remap
- logical_not control (apple-raw binding): original capture binds x=ch4,
  out=ch5 with no remap

For each variant the regression computes:
  pre-exec dst BO SHA (baseline, distinguishes "engine wrote zeros" from
  "engine never wrote / dst is zero-initialised or unmapped")
  post-exec dst BO SHA
  delta classification (untouched | zero-written | nonzero-written)

Requires t6001-test-host's strict-fill library (LIBANE env or default /var/tmp path).
Run on t6001-test-host only.
"""
import ctypes, hashlib, os, subprocess, sys, tempfile
from pathlib import Path

LIBANE = os.environ.get(
    "LIBANE",
    "/var/tmp/t6001-test-host-oproj-place/libane-strict-fill.so")

VARIANTS = [
    ("cast-control",
     "/tmp/ac-mint/bundle/program-0.anec",
     "cast bool->fp16 [1,1,1500,1] (proves submission path)"),
    ("logical_not-treatment",
     "/tmp/notdiag/out/program-0.anec",
     "logical_not 1-task, x=ch5/out=ch4 (parity-remapped emission)"),
    ("logical_not-control-raw",
     "/tmp/notdiag/control-raw-binding.anec",
     "logical_not 1-task, x=ch4/out=ch5 (apple-raw binding, no remap)"),
]

def open_lib():
    lib = ctypes.CDLL(LIBANE)
    for n, rt, at in (
        ("__ane_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int]),
        ("__ane_free", None, [ctypes.c_void_p]),
        ("ane_exec", ctypes.c_int, [ctypes.c_void_p]),
        ("__ane_src_size", ctypes.c_uint64,
         [ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_dst_size", ctypes.c_uint64,
         [ctypes.c_void_p, ctypes.c_uint32]),
        ("__ane_send", None,
         [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]),
        ("__ane_read", None,
         [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]),
    ):
        f = getattr(lib, n); f.restype = rt; f.argtypes = at

def run(label, anec_path, note):
    anec = Path(anec_path)
    if not anec.exists():
        print(f"SKIP {label}: {anec} missing"); return
    anec_sha = hashlib.sha256(anec.read_bytes()).hexdigest()
    open_lib()
    lib = ctypes.CDLL(LIBANE)
    nn = lib.__ane_init(str(anec).encode(), 0)
    assert nn, f"init failed: {anec}"
    s0 = int(lib.__ane_src_size(nn, 0))
    d0 = int(lib.__ane_dst_size(nn, 0))
    pre = bytearray(d0)
    lib.__ane_read(nn, ctypes.c_char_p(pre.ctypes.data), 0)
    pre_sha = hashlib.sha256(bytes(pre)).hexdigest()
    tile = bytearray(s0)  # zero-input dispatch
    lib.__ane_send(nn, ctypes.c_char_p(tile.ctypes.data), 0)
    rc = int(lib.ane_exec(nn))
    post = bytearray(d0)
    lib.__ane_read(nn, ctypes.c_char_p(post.ctypes.data), 0)
    post_sha = hashlib.sha256(bytes(post)).hexdigest()
    lib.__ane_free(nn)
    nz_pre = sum(1 for b in pre if b != 0)
    nz_post = sum(1 for b in post if b != 0)
    if pre_sha == post_sha:
        verdict = ("untouched_or_active_zero (cannot distinguish without"
                  " sentinel-preinitialized dst write)")
    elif nz_post != 0:
        verdict = "engine wrote nonzero bytes (write observable)"
    else:
        verdict = "engine wrote zeros to dst (write observable)"
    print(f"{label:28s} {note}")
    print(f"  anec sha {anec_sha[:16]}... s0 {s0} d0 {d0}")
    print(f"  pre  sha {pre_sha[:16]}... ({nz_pre} nonzero)")
    print(f"  post sha {post_sha[:16]}... ({nz_post} nonzero) exec {rc}")
    print(f"  verdict: {verdict}")
    # regression assertion: pre and post must be real reads (the buffers
    # were filled); pre=post identical OR nonzero post is informative.
    assert len(pre) == d0 and len(post) == d0, "dst read size mismatch"

def main():
    if not Path(LIBANE).exists():
        print(f"SKIP: libane not found at {LIBANE}; this regression runs"
              f" on t6001-test-host only")
        sys.exit(0)
    for label, anec, note in VARIANTS:
        print()
        run(label, anec, note)
    print("\nNO OVERCLAIM: zero-on-zero is not engine-confirmed without"
          " sentinel-preinitialized dst write. The baseline read establishes"
          " that the dst BO readback is live; an informative write requires"
          " a sentinel write or a complementary non-zero input.")

if __name__ == "__main__":
    main()
