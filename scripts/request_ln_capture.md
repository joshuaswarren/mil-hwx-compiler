# Apple CPU capture request: layer_norm [1,375,1024] axes=[-1]

Family: 120 callsites in the pinned Parakeet encoder MIL
(encoder-pinned.mil ac8e9526…; every layer's norm_feed_forward1 +
norm_self_att). This is the largest remaining uncovered family after the
bd-scale row.

## Current blocker (verified by failing-first test)

tests/test_h13_scalar_mul_broadcast.py::test_ln_affine_encoder_geometry_is_decoded
FAILS at c15c9a4+: the compiler rejects the real geometry with
`h13.norm-outside-envelope`. Two independent gaps:

1. No decoded row at the encoder geometry. The 118 norm oracles cover
   [1,C,1,1]-class and small rank-4 surfaces ([1,128,16,16] etc.); there
   is no row for [1,375,1024] axes=[-1] (canonical [1,1024,375] class).
2. The affine form (gamma/beta) is rejected by Apple's compiler itself
   in the capture harness: research/oracles/h13/
   norm_layer_norm_ax1_1x1024x1x1_affine.json has error
   `callback_status=1` and zero tasks. Comment at ANEH13Compiler.mm:1480
   records the same. So an affine Apple capture cannot be taken directly.

## Route (existing decoded rows, no new template family)

Apple's own planner decomposes affine LN as
`y = normalize(x) * gamma + beta` (one norm program + two per-channel
broadcasts). The per-channel rows for [1,375,1024]·[1,1,1024] mul and
add ALREADY exist in kBroadcastTasks (env_bcast_mul/add_1x375x1024_
blob_1x1x1024, byte-green). What is missing is only the NON-AFFINE
normalize program at the encoder geometry.

## Capture to request (CPU-only, studio-host ane-compile-hwx)

MIL (non-affine, real geometry; epsilon MUST be the real value read
from /var/tmp/EncoderParityAne/encoder-source/model-root/weights/
normalized.bin at offset 8391296 — do not substitute):

    program(1.3)
    [buildInfo = dict<string, string>({})]
    {
      func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
        tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>(-1)];
        fp16 epsilon = const()[name = string("epsilon"), val = fp16(<REAL_BITS>)];
        tensor<fp16, [1, 375, 1024]> y = layer_norm(axes = axes, epsilon = epsilon, x = x)[name = string("y")];
      } -> (y);
    }

Command: ane-compile-hwx CAPTURE_DIR OUTPUT_DIR h13
(weights.bin may be empty; no BLOBFILE in this form).

## Acceptance after capture

1. Decode to research/oracles/h13/norm_ln_ax-1_1x375x1024.json.
2. Extend normParityPlan/normSurface to accept the rank-3 canonical
   surface [1,1024,375] with axes mask matching [-1], keyed to the new
   row.
3. tests/test_h13_scalar_mul_broadcast.py::test_ln_affine_encoder_
   geometry_is_decoded flips to a two-part composition check: the
   compiler peels affine LN into norm + mul + add using decoded rows,
   single-program each, and the program count for the real MIL region
   is 3 (not refusal).
4. Full parity suite re-run (896 + new row cases) must stay 0 FAIL.

## Alternative (if Apple rejects the non-affine geometry too)

The geometry class [1,N,C] with C=1024 may be outside Apple's norm
envelope entirely; then the encoder lanes keep LN on GPU (certified
A/C fallback behavior) and the coverage census records LN as
device-unavailable — not silently worked around.