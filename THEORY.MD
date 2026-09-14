# Theory: Encoder Coverage Is Exact Template Match

## Problem

The encoder's remaining H13 work is not one missing rewrite. Each
family keys a decoded table, and the encoder shapes land on different
sides of those tables.

Per-head linear `[1,375,1024] × [128,1024] + bias[128] → [1,375,128]`
compiles to 1875 programs (750 matmul + 1125 add) and 412 MB. That is
the M=1 native fallback, not a missed larger tile.

Attention scores `[1,8,375,128] × [1,8,128,375] → [1,8,375,375]` compile
to one program. That form is already in `kBatchedTasks`. Combined with
runtime-a select at the same shape, the island compiles as two programs.
`--schedule=chain` does not fuse them.

GLU `silu`/`sigmoid` at `[1,1024,375]` does not compile. 64-CHW and
512-CHW do. Concat of stacked heads remains `h13.unsupported-concat`.

Encoder `layer_norm` at `[1,375,1024]` (120 ops, gamma/beta present) and
`softmax` at `[1,8,375,375]` (24 ops) stay `h13.norm-outside-envelope`.
Vector `layer_norm` `[1,1024,1,1]` and softmax `[1,8,128,128]` do compile.

All 101 encoder `conv`s stay `h13.conv-outside-envelope`. Nearby square
1×1 `{256,32,32}` compiles. Encoder 1×1 is `{256,750,32}`.

The leftover 24 attention-bias `select`s are encoder-shaped. Runtime-a
at `[1,8,375,375]` with a bool cond compiles. Constant `-inf` a does
not. That is `h13.select-needs-decoded-encoder`, not a missing CHW.

## Operating Theory

H13 emission is exact-match against decoded templates.

Matmul:

- Single-batch envelope: no M=375, no `(*, 1024, 128)`. Const-weight
  N=128 exists only as `(1,256,128)` and `(32,256,128)` (K=256).
- Native `encodeMatvec`: reduction 256 or 512, always 512 output columns.
- Batched envelope: `(375,128,375)`, `(375,128,749)`, `(375,375,128)` at
  B∈{2,4,8,16}. Rank-3 and rank-4 share an entry. B=1 is not batched.

The linear is constant-weight `(375,1024,128)`, ty=1. Bias plus
`rows > 1` lowers to one M=1 linear per row. K=1024 splits into two
512-chunks. N=128 pads to 512. Each row pays 2 matmuls + 1 runtime
`{128,1,1}` add + 2 64-lane bias adds: `375 × 5 = 1875`.

A 32-row const template cannot compose over K=1024: a 32×256 tile of
row-major `[375,1024]` is 32 strided 256-element segments, and one
binding is one contiguous slice. `(32,1024,128)` is absent.

Attention scores use the full last axis of `[1,8,375,128]` as K, with a
per-batch y. That is exactly `(B=8, 375, 128, 375, runtime)`. Flattening
to 3000 rows would multiply every batch against one shared y.

LUT unaries (`silu`, `sigmoid`, and the rest of that row in
`kElementwiseTasks`) capture only CHW `(64,1,1)` and `(512,1,1)`.
`parityPlan` flattens rank-3 `[1,1024,375]` to `{384000,1,1}`. The
64-lane split is binary-only. Relu at this size rewrites to `maximum`
against zero, then 64-lane splits. Do not compile relu on the 375×375
score surface.

Norm keys `(operation, input CHW, output CHW, axisMask, keepDims)`.
Rank-3 `[1,375,1024]` last-axis is `{1,375,1024}` `0x08`. Rank-4
`[1,8,375,375]` last-axis is `{8,375,375}` `0x08`. Neither is in
`kNormTasks`. Affine gamma/beta are a second miss. Encoder epsilon is a
`tensor<fp16,[]>` blob, not `fp32(1e-5)`.

Conv keys `(kernel, stride, groups, bias, input CHW, output CHW)`. 175
templates, spatial square 1/8/16/32/64. Encoder 1×1 is `{256,750,32}`.
`750/32` is not an integer. Rank-3 1D never reaches the table.

Select keys `(constInput, CHW)`. Runtime-a bool-cond hits
`(false, 8, 375, 375)` and `(false, 64, 1, 1)`. Constant-a is refused
before lookup. `bindTasks` order is `[y, a, b, cond]` template
channels `{7, 4, 5, 6}`. Template 6 is the 1-byte cond (375-wide DMA
row 384). Templates 4 and 5 are fp16 (row 768). MIL `select(a, b,
cond)` is `cond ? a : b`. Apple's kernel is `cond ? template4 :
template5`. Binding a to 5 inverted polarity: `{64,1,1}` was
bit-exact only after host invert-cond. Operand swap, not cond
negate. `{7, 6, 4, 5}` put 384-stride DMA on `a`. `bindTasks` remaps
selectors only. The 375-wide select blends through channel 3, and Apple
uses channel 3 as a three-stage scratch arena (cond-false half at 0,
cond expanded to fp16 at 2256000, cond-true half at 4560000), 6816000
bytes touched. `encodeBooleanOp` sized channel 3 as one output surface,
2310144 bytes, so tasks 3 and 4 ran 4.5 MB past the declared scratch.
Channel 3 is now sized from the task descriptors' own channel-3 extent:
`alignTile(scratchExtent)` = 6832128 = 417 tiles, equal to Apple's
`__DATA/__bss` gap on 1451 of 1452 captures, 0 under-estimates. On
device this took island B from 10728 wrong lanes to 0. The earlier
"first 8-row L2 tile" reading (`h13.select-first-l2-tile`) is retracted;
the L2 words `0x0480c=0x180`, `0x04810=0xc00` and const `0x0001` were
never the defect. A rebuilt anec needs its manifest `scratch_bytes`
re-stamped: `load_bundle` gates on exact equality. Receipts:
`receipts/2026-09-14-h13-bool-dma-stride.md`,
`receipts/2026-09-14-h13-select-polarity.md`,
`receipts/2026-09-14-h13-select-l2-fix.md` (supersedes
`receipts/2026-09-14-h13-select-l2-tile.md`).

Chain fusion is a third table, not a package stitch. `--schedule=chain`
accepts only two ops, at most two boundary inputs, every input on the
producer, and a final `relu`. The producer must be `apple-parity-matvec`
or `h13-oracle-parity`. `PostOperation` is only `Relu` (one clamp bit).
`composePrograms` returns that single fused program and refuses a
multi-program relink: task DMA is wired to boundary channels, with no
addressing for a declared intermediate. The attn+select island has four
boundary inputs, a batched-matmul producer, and a select consumer. The
package envelope is per-op two programs. The chain envelope is
`h13.chain-outside-envelope`.

## Strategy

Pin the batched scores path with a rank-4 compile. Keep the attn+select
island as a two-program per-op package. Name `h13.chain-outside-envelope`
for `--schedule=chain` on that island. Name `h13.linear-outside-envelope`
for `(375,1024,128)` const-W. Name `h13.unary-outside-envelope` for GLU
silu/sigmoid. Keep `h13.norm-outside-envelope` for encoder
layer_norm/softmax. Keep `h13.conv-outside-envelope` for every encoder
conv. Keep `h13.select-needs-decoded-encoder` for constant-a select.
Size channel 3 from the task descriptors, never from one surface. Do
not pad N. Do not emit mul-by-1 copies. Do not steal `(375,128,749)` for
the linear GEMM. Do not tile LUT unaries. Do not emit 375 vector
layer_norms or pad softmax 375 to 512. Do not tile `{256,750,32}` into
32×32 convs. Do not emit the `-inf` packing as a general const-a
encoder. Do not relink the 208-task score matmul onto the 5-task select.
Do not patch select L2 `0xc00` or const `0x0001`. Do not host-invert
select cond; bind a to template 4.

Concat stays unlowered. Split of one buffer stays a view. Encoder
select closes if the frontend materializes the fill as runtime-a, then
the island is the two-program package.

## Key Discoveries

The 750/1125 split is arithmetic: 375 rows × 2 K512 matvecs, plus 375
envelope adds and 750 64-lane bias adds. Native matvec rejects reduction
128, so N=128 cannot be the K tile.

`supportsBatchedMatmul({8,375,128,375,false,false,true})` is true.
`supportsMatmulParity({375,128,375,false,false,true})` is false.

`supportsElementwise(Silu, {64,1,1})` and `{512,1,1}` are true.
`supportsElementwise(Silu, {384000,1,1})` is false.

`supportsNormParity(LayerNorm, {1024,1,1}, 0x02)` is true.
`supportsNormParity(LayerNorm, {1,375,1024}, 0x08)` is false.
`supportsNormParity(Softmax, {8,128,128}, 0x08)` is true.
`supportsNormParity(Softmax, {8,375,375}, 0x08)` is false.

`supportsConvParity({1,1,1,true,{256,32,32},{256,32,32}})` is true.
`supportsConvParity({1,1,1,true,{256,750,32},{256,750,32}})` is false.

`supportsBooleanOp({Select, false, 8, 375, 375})` is true.
`supportsBooleanOp({Select, true, 8, 375, 375})` is true in the table
and refused at the MIL gate.

Attn+select per-op: 2 programs. Select anec after the polarity remap is
`d40ec023cfd5b2c5d5cfeb95f4a4673024f495224acd4448e8498bf50114780f`
(375) and `732301a3cae256dca2a8483e49d92f3831c484f9873fdd9ee35e359e85ff508d`
(64). Cond DMA stays 384 on header ch7. Chain on the same MIL:
`h13.chain-outside-envelope`, four inputs, rc 65. Add+relu
`[1,512,1,1]` still fuses. Batched-matmul+relu (2 inputs) is
`h13.chain-unrepresentable-edge`: that encoder has no post-op field.

## Open Questions

Whether Apple emits one fused matmul+bias program for `(375,1024,128)`,
one LUT unary for `[1,1024,375]`, one affine layer_norm for
`[1,375,1024]`, one softmax for `[1,8,375,375]`, one conv for
`[1,256,750,32]` / the 1D k9, or one fused score-matmul+select. Until
those captures exist, the ISA cannot express them here. Encoder select
runtime-a at `[1,8,375,375]` is already decoded and exact on device once
channel 3 is sized as the scratch arena. `{64,1,1}` polarity is the
operand remap.
