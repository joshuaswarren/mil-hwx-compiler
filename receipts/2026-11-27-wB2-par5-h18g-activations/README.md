# OPEN-par-5: Parakeet activation lowering probe — gelu / elu / silu at H18G (wB2, 2026-11-27)

Card (Main 17:47 CST): compile `aten.gelu`, `aten.elu`, `aten.silu` to the M6
ANE; record pass/fail per op with the error text. Compile-level probe at the
H18G (t8152 / M6) emit target — same scope the G708 emit probe used; no
silicon run.

## Inputs

Single-op fp16 `[1,512,1,1]` MIL programs (`inputs/`):

- `gelu.mil` — verbatim `research/oracles/h14/unary_gelu_c512.json` MIL
  (`gelu(mode=EXACT)`).
- `silu.mil` — verbatim `research/oracles/h14/unary_silu_c512.json` MIL.
- `elu.mil` — hand-written on the same shape (`elu(alpha=1.0)`); no elu
  oracle exists in the corpus.

Command per op:

```
./build/mil-hwxc --target H18G --mil inputs/<op>.mil --model-root . \
  --output out-<op> --input x512.fp16
```

## Results — all three FAIL at the same named gate

| op | rc | verdict | error text |
|----|----|---------|-----------|
| gelu | 65 | FAIL | `5:5: error [h18g.unsupported-program]: H18G requires one fp16 [1,256] x [256,256] constant matmul without transposes` |
| silu | 65 | FAIL | identical `h18g.unsupported-program` refusal |
| elu | 65 | FAIL | identical `h18g.unsupported-program` refusal |

## Control at H14 — the refusals are the H18G envelope, not the ops

Same three MIL programs at `--target H14`:

| op | rc | result |
|----|----|--------|
| gelu | 0 | `compiled target=H14 artifacts=1 format=anec` (program-0.anec written) |
| silu | 0 | `compiled target=H14 artifacts=1 format=anec` (program-0.anec written) |
| elu | 65 | `5:5: error [h14.outside-parity-envelope]: H14 supports only the decoded fp16 elementwise, scalar-constant, and unary parity envelope` |

## Verdict

- **gelu / silu — FAIL on M6 today, envelope-blocked only.** Both compile
  cleanly at H14 (the h14-unary receipt ran the same ops bit-exact on M2
  hardware), so the ops and their MIL are fine. The H18G emitter accepts
  exactly one program shape today — the device-proven fp16 `[1,256] x
  [256,256]` constant matvec — and names that envelope on refusal
  (`h18g.unsupported-program`). Same refusal G708 measured for `mul`.
- **elu — FAIL at two levels.** No elu oracle exists anywhere in
  `research/oracles/`; H14 refuses it as outside the decoded parity
  envelope, and H18G refuses it at the same single-program gate. elu is
  not in this compiler's ANE coverage on any target — there is no
  hardware-proven evidence ANE even has an elu primitive.
- For Parakeet specifically: the encoder's activations (gelu in the Parakeet
  stack) cannot lower through this backend until the H18G emitter grows a
  second program shape — the named gap to close is op coverage, not driver
  or packaging. `mul`, `gelu`, `silu`, `elu` all die at the same gate.

Compiler head: `60df441` (main, includes #41 H18G target + #42). Input MIL,
the fp16 input tensor, and per-op outputs preserved under `inputs/` and
`/tmp/wB2-par5` on the authoring CT.
