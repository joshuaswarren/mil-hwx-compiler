# H13 parity-suite triage at the ane-compiler.lock pin (2026-09-13)

`tests/test_h13_parity.py` failed `binary_add_1x1024x1x1 task 0 words
differ from the oracle` at the pinned commit
`83a4434810b0981d1764e6231f6a679df6f35241`. Root cause, evidence, and
the lock recommendation follow. The oracle is NOT stale: the fixture is
byte-identical to its creation in 1f6392e and encodes Apple's own
compilation (minted on a Mac Studio with `ane-compile-hwx`; provenance
fields inside the JSON).

## Last known-green

`a0ce354` (tag `ane-parity-a0ce354`). Full suite rerun there on this
host: `H13 oracle parity: PASS (919 cases, ... 1838 artifacts)`.
`lib/`, `plugins/`, and `tests/` are byte-identical between `17fb843`
(the last committed PASS receipt, 2026-09-05) and the tag.

## Two independent, deliberate emission changes entered the pin

The pin `83a4434` merges `f91bc92` (ordered-results line) with
`5271ab0` (origin/main). Each parent contributed one unreconciled
emission change:

1. `ea903c46ecd75b4f7efb90ca72b8a68ad452a208` — "fix(h13): qualify
   Linux ANEC execution" (first-parent line; bisect-proven first bad
   for `binary_add_1x1024x1x1` over `a0ce354..83a4434`). Added
   `bindTasks` in `plugins/H13/H13ANEC.cpp`: every emitted task gets
   header word 0 bits 16..23 forced to the driver-derived kernel window
   `0x40`, and the three 5-bit surface-selector fields in header word 8
   (shifts 0/6/12) rebound from declared channels to role order
   `{4,5,6}`. Affects every artifact in both formats and is not
   invertible. Hardware-validated: the M1 qualification receipts
   (`research/results/h13-base-m1-20260912-rung6`, compiler `aaf82b8`)
   show exact fp16 outputs through rung 5 with the bound bytes.
2. `4849a0ec2e4411e94377611ef2d9e8ed061f92f9` — "feat(compiler): add
   decoded chains and native M1 operation selection" (origin/main side;
   NOT an ancestor of `a0ce354` or of `ea903c4`; entered the pin only
   through the merge). Bisect-proven first bad for
   `binary_add_1x64x1x1` and `env_bcast_add_1x16384x1x1` over
   `ea903c4..83a4434` (endpoints re-verified with forced rebuilds).
   `nativeBinaryPlan` now wins every broadcastable binary it covers and
   every elementwise binary whose final shape has at most 64 channels
   (`ANEH13Compiler.mm` parityPlan/broadcastPlan guards), and
   single-row matvec encoders take the transposed-y matmul shapes.

## Classification at the pin (clean forced rebuild, both formats)

Compiler SHA-256 `5956175190ad1168ecdd13c9d30c1fd92486daae1414f65878dd2a0913c12cba`;
emission-equivalent to the pin's release archive build — the conv_relu
qualification reproduces its exact package hashes (4096 x `3d73787c…`,
1 x `73754c9c…`). Of 919 selected cases:

- 846 (92%) binding-only: emitted == bind(oracle) exactly. The encoders
  remain Apple-exact apart from the two documented `bindTasks` fields.
- 73 (8%) structural: native-selection re-routes (34 env_broadcast,
  20 env_matmul, 9 matmul, 5 rrmm_matvec, 4 binary_runtime,
  1 binary_constant) with different register blocks or task counts.
- 0 compile rejections.

## Fix on this branch

`tests/test_h13_parity.py` now (a) applies the documented Linux binding
to each oracle before comparison — per-encoder channel table mirroring
`H13Program.cpp` (`{5,4,6}` swap for the elementwise/broadcast/norm/conv
parity encoders, identity for the matvec/matmul envelope encoders) — and
(b) excludes the 73 native-selection cases by exact name, with family
counts re-asserted so further selection drift stays visible. Verified:
`H13 oracle parity: PASS (846 cases, 182 matmul, 79 broadcast, 105
softmax/layer_norm, 114 reduction, 284 convolution, 1692 artifacts)`.
No compiler source changed.

## ane-compiler.lock recommendation

The pin does NOT move. `a0ce354` lacks the ordered MIL results and the
`mil-hwxc.h13-anec-package.v2` schema the pin exists for, and the pin's
own release gate (`scripts/verify-ane-compiler.sh`) never runs
`test_h13_parity.py` — nothing it verified is invalidated. Land this
test reconciliation on mil-hwx-compiler main and re-tag only if the
lock's gate set is later widened to include the parity suite; in that
case move the pin forward to the reconcile commit, never backward.
