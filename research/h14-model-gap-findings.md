# H14 model-gap mint: findings per family

Minted 2026-09-29 on the Mac (M1 Ultra, macOS 26.6.2) with
`~/recurrent-mint/ane-compile-hwx` (SHA-256 `2060776c...`, per-record provenance
in every JSON), both targets, from `research/mint_h14_model_gaps.py`. 33 cases,
60 decoded + 3 rejected per target (66 compiles; the remaining rejections are
the corpus's standing `transpose_y=false` refusals, not this campaign's).
Records: `research/oracles/{h13,h14}/`, prefixes `matmul_m1_k{1536,3072,*}` /
`matmul_m8_k2048_n2048_ty1`, `garms_*`, `gasel_*`, `gabmm_*`.

## (a) Qwen matvec grid

All ten new constant-weight points decoded: `(K, N)` = 1536/2048/3072 squared
plus `(2048, 5120)` (the Qwen3.8-2B qkv projection) at M=1, and
`(2048, 2048)` at M=8, the first decoded M=8 point above K=1024. Every one is
the two-task form of the minted grid, and every constant section is exactly
`K * N * 2` bytes (largest: 20 MiB for the qkv weight). The packing model in
`mint_h14_matvec_probes.py` accepts all of them (`columns % 256 == 0` holds for
1536, 3072, 5120), so the encoder extended by data alone: `make test-h14-parity`
now compiles 46 matvec oracles against the templates.

## rms_norm: decompose, not mint

coremltools 9.0 has no native `rms_norm` MIL op; the torch frontend decomposes
it (abs → reduce_max → real_div → square → reduce_mean → add eps → sqrt →
mul → real_div → gamma mul). The campaign minted the sub-ops and the whole
chain at the Qwen decode shape `[1, 2048, 1, 1]`:

- `garms_reduce_reduce_max/mean_1x2048x1x1_kd1`: **Apple refuses** the
  standalone flat-C reduction (`callback_status=1` on both targets), unlike
  the decoded `[1, 64, 8, 8]` spatial reductions.
- `garms_chain_c2048`: the whole chain **decodes as one 7-task program** with a
  128-byte constant section; with the BLOBFILE gamma it is 8 tasks and an
  8,320-byte section (the 4-bytes-per-channel gamma block of the per-channel
  constant form, plus the 128-byte table). Apple splits the decomposition into
  tasks the way softmax does, which is why the standalone reduce refuses while
  the chain compiles.
- The scalar plumbing decodes and is now covered by the elementwise encoder:
  `unary_sqrt_c1`, `binary_add_1x1x1x1`, the runtime scalar-tensor broadcasts
  `mul`/`real_div` at `[1, 2048, 1, 1] x [1, 1, 1, 1]` (two tasks, the
  broadcast form), and `binary_mul_c2048_constant_blob` (two tasks, 4 bytes per
  channel). The runtime-broadcast `real_div` joins the classified elementwise
  families, which also uncovered the first campaign's six
  `binary_real_div_1xCx1x1` records.

The chain is now covered by the `apple-parity-rms-norm` encoder: it walks
the nine-op subgraph in `rmsNormChainPlan`, validates the resolved gamma
payload as a uniform fp16(0x1p-1) (every halfword = 0x3800), and emits the
recorded whole-program stream via `OracleRmsNormTemplate` tables generated
from the captured `garms_chain_c{128,2048}{,_gamma}` oracles. The decoder's
allow r1 (amax) to be used twice — `real_div(x, amax)` and `mul(rms, amax)`
both consume it — every other intermediate is still required to be unique.
Decoded support:

- channels = 128 (no gamma, gamma): `[1, 128, 1, 1]` x → `[1, 128, 1, 1]` y,
  7-task and 8-task programs, 128 / 640-byte constant sections.
- channels = 2048 (no gamma, gamma): `[1, 2048, 1, 1]` x → `[1, 2048, 1, 1]` y,
  7-task and 8-task programs, 128 / 8,320-byte constant sections.

Refused (out of envelope, named): rms_norm chains whose channel count is not
in {128, 2048}; rms_norm chains with a non-uniform gamma weight (every
halfword must equal 0x3800); rms_norm chains where any r2..r7 intermediate
is used more than once; rms_norm chains whose eps literal is not 0x0080
(fp16 2^-17); rms_norm chains where `x` is not a function input.

## select const-fill (ninf)

The encoder gained the `apple-parity-select` const-fill path for H14:
`selectPlan` recognizes a constant `a`, loads its BLOBFILE payload, validates
that the fill is uniform (every halfword matches `fillBits`), and lays the
fill at `(fillBlocks, fillBlockData, fillBlockStride)` derived from the
captured `gasel_ninf_1x8x375x375` section (375 blocks × 3000 halfwords at a
3008-halfword stride; selector table sits in the final 256 bytes). The
sub-header layout for the BLOBFILE weights.bin is the **aligned** form
(length at byte 72, payload offset at byte 80) — the convolution/matvec
sub-header (length at byte 68, payload offset at byte 76) makes the resolver
emit `Cannot retrieve file blob properties`. Decoded support: the
single-decoded `[1, 8, 375, 375]` ninf point on H14 (same constant section
on H13, but the H14 encoder only covers H14). The runtime-form select (a, b,
cond) on any covered shape still binds three inputs and is supported on both
targets.

## (b) Island B: select

`select` with a **bool** cond decodes on both targets as **five tasks**: sizes
`[628, 504, 628, 504, 504]` on H13 and shape-dependent word counts on H14,
with a 256-byte constant section holding a 46-nonzero-byte selector table. The
table is target- and (at most shapes) size-independent: the 256-byte sections
of `gasel_rrb_1x1024x375` hash identically on H13 and H14. H13 pads the same
table to 2,048 bytes at `[1, 64, 1, 1]` and `[1, 8, 375, 375]`; H14 keeps 256
bytes at every decoded shape. `select` with an fp16 cond remains refused on
both targets, matching the upstream H13 captures.

The const-fill form (`a` = full-size BLOBFILE fp16 -inf, the materialized mask
fill) first refused with `MILFramework error: Cannot retrieve file blob
properties` — the campaign's `om.blob` sub-header (length at byte 64). With the
aligned layout (length at 72, payload offset at 80, the layout
`tests/test_h14_parity.py:aligned_blob` already uses) it **decodes on both
targets as the same five-task program**, its section being the 256-byte table
plus the -inf fill laid out at a 376-halfword row stride (375 + 1). H14 keeps
the table at 256 bytes (2,258,048-byte section on H13, 2,256,256 on H14). The
encoder covers the runtime form the island binds (three runtime operands);
the const-fill form stays refused with a precise remaining-work note.

The H14 encoder gained the `apple-parity-select` family (three-input programs;
ANEC now allows three inputs, cond on channel 7 with element code 3).

## (c) Islands A/C: batched runtime-runtime matmul

The headline: **H14 does not tile per batch the way H13 does.** The same
geometries that cost H13 26 tasks per batch (208 tasks at b8) decode on H14 as
**2 tasks** (`[8,375,128]x[8,128,375]` and `x[8,128,749]`), 3 at b16 (the
compute task repeats for the upper half-batch, split between b8 and b16), and
5 for the probs x V geometry `[8,375,375]x[8,375,128]` (four compute tasks
whose kernel-DMA chunk words step 0x1..0x4). The rank-3 `[8,M,K]` and rank-4
`[1,8,M,K]` head forms emit **identical task streams and program
descriptors**; only the tensor descriptors differ. Batch size rides in one
register: Common `0x28` low halfword (`0x1000B | batch`) — b2/b4/b8 streams
differ in exactly that word. `transpose_y=true` costs one extra task.

The encoder gained the `apple-parity-batched-matmul` family: whole-program
replay per decoded `(rows, reduction, columns, batch, heads-form, ty)` point,
tensor descriptors carried verbatim (they record one batch element's bytes as
the total, the known batch trap). Off-point batches and geometries refuse with
`h14.outside-parity-envelope`.

## Refusals recorded (both targets unless noted)

| Case | Status |
|---|---|
| `garms_reduce_reduce_{max,mean}_1x2048x1x1_kd1` | `callback_status=1` — flat-C standalone reductions |
| `gasel_ninf_*` with `om.blob` sub-header | `Cannot retrieve file blob properties` (header-layout finding; decodes with the aligned header) |
| select with fp16 cond | refused (upstream H13 finding, unchanged) |
| `garms_chain_*` | covered by `apple-parity-rms-norm` (c128, c128_gamma, c2048, c2048_gamma) |
| `gasel_ninf_1x8x375x375` | covered by `apple-parity-select` const-fill (H14 only) |

## Remaining work

1. rms_norm chain support beyond {128, 2048} channels and the uniform
   fp16(0x1p-1) gamma — generic per-channel weight scaling and shapes with
   non-trivial spatial extent are decoded-only.
2. select const-fill shapes beyond `[1, 8, 375, 375]` (the lone decoded ninf
   point): other mask-fill shapes need their own
   `fillBlocks / fillBlockData / fillBlockStride / fillBits` measurements
   from the captured section.
3. Batched matmul between decoded points (batch 0x28 patch is understood for
   b<=8; the b16 half-batch split and the 4-task pv chunking are decoded but
   not generalized).
4. Semantic caveat: the two select fp16 descriptors are identical, so the
   a-vs-b channel order (5/6) is pinned to MIL declaration order, not proven
   against execution.
