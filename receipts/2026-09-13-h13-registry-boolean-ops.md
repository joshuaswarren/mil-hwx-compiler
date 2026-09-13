# H13 registry ops receipt (less / floor / select / floor_div)

## Source

- Compiler base: `7fd06ac` (`feature/h13-batched-matmul`)
- Branch: `feature/h13-registry-boolean-ops`
- Ground truth: mlx-omarchy `docs/2026-09-13-h13-boolean-mask-fp16.md`
  (commit `f1147038`, DECIDED, prototype verified)
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback)

## Survey result: no decoded templates exist for any of the four

Programmatic survey of every decoded template table
(`plugins/H13/*.inc`): no comparison, floor, or lane-pick operation appears
in the elementwise, matvec, envelope, norm, conv, or chain corpora (the
only "Floor" token is `ChainSegment::ZeroFloor`, an unrelated scheduler
constant). The IR registry had no classification for `less`, `floor`,
`select`, or `floor_div`. Per the assignment rule, no task words were
invented: the four ops get registry entries and exact rejection contracts,
and the mint list below carries the encoder load.

## What shipped

- **Registry**: new `ANEOperationKindMask` classifies `less`, `floor`,
  `select`, `floor_div`, `logical_and`, `logical_not`
  (`lib/IR/ANEOperationGraph.{h,mm}`); H16G's kind switch handles it as
  unsupported. The scheduler's ALU fusion heuristics never see these ops.
- **Exact rejections** (`plugins/H13/ANEH13Compiler.mm`), one named code
  per op, each carrying the decision doc's reasoning:
  - `h13.less-needs-decoded-encoder` — a comparison is a step function;
    no finite `+,-,*,/` expression over fp16 produces `(x < y)` as 0/1
    exactly (rounding destroys sign-based constructions; `x == y` must be
    0 while `y - x = +0` is indistinguishable from tiny positives).
  - `h13.floor-needs-decoded-encoder` — floor is total and exact over fp16
    (|x| ≥ 2048 already integral; below it every integer is representable;
    NaN/±inf pass) but has no arithmetic identity.
  - `h13.select-needs-decoded-encoder` — the blend identity NaNs on the
    encoder's 24 attention-bias selects (`0 * -inf` on unmasked lanes; no
    `+,-,*` arrangement avoids it), and the `+0.0`-fill family belongs to
    the frontend mul rewrite, not a compiler encoder.
  - `h13.floor-div-needs-decoded-encoder` — composes real_div with floor;
    blocked on the floor encoder. The contract states the fp16 semantics:
    `floor_div = floor(fp16_div(x, y))`, not integer floor division
    (`3199/32 → 100.0` in fp16 vs `99` exact), matching the package's own
    cast-then-divide reference.
- **Tests** (`tests/test_h13_registry_cli.py`, wired into `make test-h13`):
  encoder-shaped rejection contracts for all four — the `[1,375]`
  arange-vs-length `less` (cast to fp16 at the return boundary like the
  graph), the `[1]`-shaped `floor` and `floor_div` by the constant 2.0,
  the `[1,8,375,375]` `-inf`-fill `select` with an fp16 0/1 cond (the
  decision doc's boundary conversion: no bool dtype reaches the device),
  and the `+0.0`-fill family — plus a positive control proving the divide
  half of `floor_div` is real today: `real_div` by constant 2.0 is
  byte-identical to `mul` by 0.5 through the decoded reciprocal path.

`logical_and`/`logical_not` are classified but intentionally left on the
generic no-encoder rejection: the decision doc assigns their lowering
(`and = mul`, `not = mul(-1) + 1`) to the frontend rewrite, and a second
compiler-side convention beside it is exactly what this repo forbids.

## Mint list for the next OracleMint round

| op | form to capture | shapes |
|---|---|---|
| `less` | fp16 compare, fp16 0/1 result | flat (375,1,1), (750,1,1), (1500,1,1) — the encoder's channel-mask extents — plus (64,1,1)/(512,1,1) generic |
| `floor` | fp16 floor (exact) | (64,1,1), (512,1,1), (1,1,1) — the encoder's instances are [1]-shaped length scalars |
| `select` | fp16 lane pick by 0/1 mask; operands a, b, cond | CHW (8,375,375) and (1024,375,1)-family plus generic; capture at least one `-inf` a-constant form and one runtime-a form |
| `floor_div` | native (Apple's own choice) or confirm the real_div+floor composition | [1]-shaped by constant 2.0; generic runtime divisor |

Capture notes: non-uniform weights/values wherever constants are involved
(the packed-const lesson from the batched round); retain raw constant
bytes; record taskCount and per-task sizes.

## Commands and observed results

```text
make -j8 build/mil-hwxc                    # 0 errors
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS
  h13 composite cli: PASS
  h13 batched cli: PASS
  h13 registry cli: PASS                   # new suite
python3 tests/test_h13_parity.py build/mil-hwxc
  AssertionError: binary_add_1x1024x1x1 ...  # pre-exists (ParityTriage), unchanged
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change,
or Mac execution was performed. No files under `ane-linux-experiments`
were edited or executed; the encoder artifact was read only.
