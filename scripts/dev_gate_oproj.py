#!/usr/bin/env python3
"""h13: device-execution gate for the encoder rank-3 linear.

The 874/874 byte-parity suite is structurally blind to plane-permutation
bugs because the captured oracles are uniform-payload (every half is
identical), so any permutation produces the same bytes. This gate
executes the emitted bundle on the device against real weights and
checks rel_l2 vs the fp32 reference.

MUST be invoked via mil-hwx-compiler/scripts/run_dev_gate.sh with a
flock-protected m1-test-host GPU/ANE refcnt check before the build that bumps
ANE_COMPILER_COMMIT. The script is gated on the (375, 1024, 1024)
uniform geometry AneChannelFix CRT-measured; smaller/different
geometries stay on the byte-parity gate until their own CRT runs.

Run on m1-test-host (GPU lock, ANE refcnt 0). Exits 0 on PASS, 1 on FAIL.
"""
import ctypes, sys, hashlib, subprocess, json, os
from pathlib import Path
import numpy as np

# These are loaded from the workspace; the script lives next to the
# mil-hwx-compiler checkout that produced the bundle under test.
m1-test-host = "m1-test-host"
LIB = "/var/tmp/island-reexport/libane-strict.so"
ISLAND_BUNDLE = Path("/var/tmp/ANE-full-battery/bundles/island-oproj-L00/program-0.anec")
SOURCE_ROOT = Path("/var/tmp/EncoderParityAne/encoder-source")

RELAY_BUDGET = 0.05  # rel_l2 ≤ 0.05; ~0.012 is the fp16 W noise floor

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

def run(anec, x16):
    nn = lib.__ane_init(str(anec).encode(), 0)
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

sys.path.insert(0, "/var/tmp/ANE-full-battery")
import mint as M

consts = M.parse_consts((SOURCE_ROOT / "model.mil").read_text())
blobs = M.Blobs(SOURCE_ROOT / "model-root")

def fp32(name, shape):
    c = consts[name]
    return np.frombuffer(blobs.read(c["path"], c["offset"], int(np.prod(shape)) * 2),
                         dtype="<f2").reshape(shape).astype(np.float32)

W = fp32("encoder_layers_0_self_attn_o_proj_weight_to_fp16_palettized", (1024, 1024))
B = fp32("linear_2_bias_0_to_fp16", (1024,))

# run on three independent rngs; pass if every rel_l2 ≤ RELAY_BUDGET
results = []
for seed in (11, 33, 57):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((375, 1024)).astype(np.float16).astype(np.float32)
    ref = x @ W.T + B
    y = run(ISLAND_BUNDLE, x.astype(np.float16))[:384000].reshape(375, 1024)
    rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
    results.append({"seed": seed, "rel_l2": rel})

worst = max(r["rel_l2"] for r in results)
summary = {
    "island": str(ISLAND_BUNDLE),
    "bundle_sha256": hashlib.sha256(ISLAND_BUNDLE.read_bytes()).hexdigest(),
    "constantBytes": 2097152,
    "rngs": results,
    "worst_rel_l2": worst,
    "budget": RELAY_BUDGET,
    "pass": worst <= RELAY_BUDGET,
}
print(json.dumps(summary, indent=2))
sys.exit(0 if summary["pass"] else 1)