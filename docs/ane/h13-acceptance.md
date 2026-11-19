# H13 numerical acceptance policy

Status: proposed 2026-11-19 (lane w72). Applies to H13 unary results checked
against a host dense FP16 reference. Checker: `research/h13_acceptance.py`
(test: `tests/test_h13_acceptance.py`).

## Gates

1. **Structural (exact).** The package's decoded sections must match the
   Apple oracle's after Linux task-binding normalization: task-section hash
   and packed-constants SHA256. Enforced by `tests/test_h13_qwen3_ff_fusion.py`.
2. **Numerical (envelope).** Every output lane must be finite and satisfy
   `|got - ref| <= A(op)`. Any inf or NaN fails outright.
3. **Oracle device run (exact, preferred).** When a same-target oracle
   package output exists for the same inputs, require bitwise equality with
   it; the dense reference gate above is the fallback for lanes without an
   oracle run.

## The bound

`A(silu) = 0.016` for inputs in the LUT domain `[-8, 8]`.

Measured evidence (standalone H13 SiLU qualification, 1024 lanes,
`~/scratch/w72/silu-qualification/H13`): max abs error 0.01489, mean 0.00104,
340/1024 lanes exact, zero nonfinite, zero lanes over 0.016.

## Why not a relative or exact gate

- Exact FP16 equality vs the dense reference is unsatisfiable for any LUT
  unary: Apple applies nonlinearity through a table in the constant section
  (docs/ane/numerics.md), and fused results inherit the table's error.
- Relative ulp bounds fail near zero: the LUT quantizes the zero bin, and the
  measured error at `ref == 0` is 0.0017 (about 28k ulp relatively). Only an
  absolute bound covers the whole domain.

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
