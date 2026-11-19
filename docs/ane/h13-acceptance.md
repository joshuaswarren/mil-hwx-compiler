# H13 numerical acceptance policy

Applies to standalone H13 SiLU and the fused H13 feed-forward comparison
against a host dense FP16 reference. Checker: `research/h13_acceptance.py`
(test: `tests/test_h13_acceptance.py`).

## Gates

1. **Structural (exact).** The package's decoded sections must match the
   Apple oracle's after Linux task-binding normalization: task-section hash
   and packed-constants SHA256. Enforced by `tests/test_h13_qwen3_ff_fusion.py`.
2. **Fused feed-forward (same-input control).** All outputs must be finite.
   Run `python3 research/h13_acceptance.py FUSED REF --control CONTROL`
   with outputs from the same input and the 16-dispatch control. The
   checker computes `max(abs(CONTROL - REF))` at run time; the fused
   maximum against `REF` must not exceed it. Fused-vs-control deltas are
   reported separately. Exact-vs-reference is not a gate here: the
   16-dispatch control itself matched only 193 of 1,024 lanes exactly.
3. **Integer-exact programs.** Existing bit-exact gates are unchanged.

## The bound

`A(silu) = 0.016` for inputs in the LUT domain `[-8, 8]`.

Measured evidence (standalone H13 SiLU qualification, 1024 lanes,
`~/scratch/w72/silu-qualification/H13`): max abs error 0.01489, mean 0.00104,
340/1024 lanes exact, zero nonfinite, zero lanes over 0.016.

## Standalone SiLU near zero

The sampled standalone LUT output was not bit-exact against dense FP16,
including about 0.0017 absolute error at `ref == 0`. The measured
standalone envelope is not the acceptance bound for the fused graph.

## Scope and renewal

The bound is tied to the measured domain and table geometry. Recompute it
whenever the LUT domain, entry count, or target changes: run one standalone
qualification package over the full domain and set `A(op)` to the measured
max with at least 7 percent margin. Other unaries (sigmoid, tanh, gelu, ...)
get their own measured bound before device acceptance; none is defined here.

## H18 device validation status

Blocked on hardware: the approved H18 runtime (`ane/h18`, ANEP-v1 pack via
`ane-prog-pack`) needs an M6-class host with a clean ANE window. No device
run has been attempted under this policy.
