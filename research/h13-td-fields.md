# H13 task-descriptor and constant-section fields

Two encoders are documented here: the `matmul` (matvec) programs, and the
`softmax` / `layer_norm` / `reduce_*` programs. Both emit Apple's decoded task
stream verbatim for a covered geometry and refuse anything else.

# Matvec

## Scope and provenance

This report documents the H13 `matmul` (`transpose_y=true`) programs in the
checked-in oracle campaign under `research/oracles/h13`, minted on
`MacStudio.local` (Apple M1 Ultra, macOS 26.6.2 arm64) with
`/tmp/h13-oracle/bin/ane-compile-hwx`, SHA-256
`b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe`. All 36
accepted matmuls per target use `transpose_y=true`; all 36 `transpose_y=false`
attempts were rejected by Apple's compiler.

Two additional research drivers back the field derivations here. Neither
retains Apple HWX bytes:

- `research/mint_matvec_probes.py` mints matmuls whose weight payload carries a
  known pattern (uniform `fp16 0.5` cannot distinguish one constant-section
  permutation from another). It covers `zero`, `mask`, `index`, per-element
  `onehot<logical>`, and `rowstain<n>` / `colstain<k>` payloads, and records
  only the same hashes and decoded-word tables `mint_oracles.py` records.
- `research/analyze_matvec_oracles.py` reports, for every decoded word, whether
  it is invariant across the 36 cases or varies with M, K, or N.

Apple's compiler was also asked for the Mach-O segment table of two freshly
compiled matvec objects, on the host, to resolve the surface addresses. Only
segment metadata was read; no object bytes left the host.

## Program shape

Every decoded matmul in the grid M ∈ {1, 2, 8, 64}, K ∈ {256, 512, 1024},
N ∈ {256, 512, 1024} is one program with two tasks, except
`matmul_m64_k256_n1024_ty1`, which has three:

| Field | Value |
|---|---|
| Task 0, "preparation" | 126 words (504 bytes), `header[7] = 0x200` |
| Task 1, "compute" | 157 words (628 bytes), `header[7] = 0` |
| Task stream | 1140 bytes of tasks in a 1152-byte constant offset (0x480) |
| `task_words_minus_one` | 125, so the program declares task 0's size |
| Task 1's size | task 0's `header[1][24:16] + 1` = 157 words |
| Constant section | exactly `K * N * 2` bytes |
| Surfaces | input channel 5, output channel 4, no separate kernel resource |

`matmul_m64_k256_n1024_ty1` instead emits 126 + 157 + 126 words, links task 2 at
`0x500`, and takes a 1792-byte (0x700) constant offset; task 2 is a second copy
task whose destination is the output surface.

## Constant section: Apple's weight packing

The section is a pure permutation of the `K * N` fp16 halfwords the compiler
reads for the `[N, K]` weight constant. In halfword units, with

- `g = min(16, N / 16)` interleaved weight rows per plane,
- `planes = N / g`,
- `p = n / g` the plane of output column `n`,
- `q = (p % 16) * (planes / 16) + p / 16` the destination plane,

the element `W[n][k]` lands at

```
destination = q * (K * g) + k * g + (n % g)
```

`plugins/H13/H13Program.cpp:packMatvecWeights` implements exactly this.

Evidence: 502 of 505 pattern probes reproduce Apple's whole-section SHA-256
bit-for-bit, including every `(K, N)` in the shipped grid and the wider set
(K, N) ∈ {16, 32, 64, 128, 256, 512, 1024}²  that stays inside the compiler's
accepted range. The three mismatches are `K ∈ {8, 16}`, where the section is
padded to a 64-byte row stride and its size is no longer `K * N * 2`; those
shapes are outside the shipped envelope. All 36 checked-in matmul oracles also
match, and `tests/test_h13_parity.py` re-derives that equality from the
compiler's own output for every case.

The permutation was read out, not guessed. A `zero` payload leaves only the
weight file's own leading bytes in the section, which located the first
destinations (`0 -> 0`, `2 -> N/8`); `onehot` payloads then pinned individual
destinations by brute-forcing the 2 KiB chunk hashes; and `rowstain` /
`colstain` payloads confirmed that each weight row stays inside one plane.

The 16-way outer grouping in `q` is not arbitrary: task 1 programs 16 kernel-DMA
descriptors whose offsets and sizes are `K * N * 2 / 16` bytes each
(`0x1f848`-`0x1f884` offsets, `0x1f888`-`0x1f8c4` sizes), so the permutation is
what makes each of the 16 DMA chunks hold every sixteenth row group.

Note on oracle weight payloads: `research/mint_oracles.py` writes its blob
entry metadata unaligned relative to the CoreML `BlobMetadata` layout, so the
declared data offset is 0 and both Apple's compiler and this repository's
`ANEBlobResolver` read the weight from file byte 0. The 128-byte blob header is
therefore part of the weight stream the oracle compiled, which is why 56 of the
uniform-`0.5` section's halfwords are not `0x3800`. Parity and numerical
correctness coincide: both compilers permute the same input bytes.

## Program descriptor and surface addresses

Apple's H13 matvec object carries a `__DATA/__bss` scratch allocation below
every surface, and lays the output surface out before the input:

```
scratch  = align(max(M * K * 2, K * 64), 0x4000)   __DATA/__bss at 0x30000000
output   = scratch                                 __FVMLIB/__data, align(M*N*2, 0x4000)
input    = output + align(M * N * 2, 0x4000)       __FVMLIB/__const, align(M*K*2, 0x4000)
text     = input  + align(M * K * 2, 0x4000)       __TEXT/__text, file offset 0x4000
constant = align(text + task stream, 0x40)         __TEXT/__const
```

This reproduces `resource_addresses`, `text_address`, and `constant_address` for
all 35 two-task cases. `__TEXT` keeps segment flags `0x0` even with the scratch
present. `matmul_m64_k256_n1024_ty1` uses a 0x20000 scratch that the two-task
formula does not predict; its template carries the decoded value.

The scratch is not a DMA destination in the two-task form: task 0's tile-DMA
destination word `0x17800` is `0x000000c0`, while every task that writes the
output surface sets bit 26 (`0x040000c1`), and task 0 configures the L2 result
registers instead. The ANEC container therefore keeps the same channels as
before (input 5, output 4) and declares no extra channel.

## Task words

Word counts below are over the 35 two-task cases.

### Task 0, preparation (126 words, 21 vary)

| Word | Value |
|---|---|
| `0x0000c`, `0x00010` | `M` |
| `0x1380c`, `0x13814`, `0x13818` | `M * K * 2` |
| `0x13810` | `K * 2` |
| `header[1]` | `0x009c0000` for M ≤ 8; `0x009c0001` for M = 64. Bits 24:16 are task 1's size minus one; the low bits are unresolved. |
| `0x00000`, `0x00014` | packed 16-bit pair, `0x00010100` at K = 256, `0x00010200` at K = 512, `0x00010400` at K = 1024, and `0x00080100` in the three-task case |
| `0x00034`, `0x04808`, `0x04810`, `0x04814`, `0x04818`, `0x04838`, `0x0483c`, `0x04840`, `0x17808`, `0x1780c`, `0x17810`, `0x17814` | vary with M, K, or N without a resolved formula; the templates carry the decoded values |

The remaining 105 words are invariant and are emitted verbatim, including
`header[2] = 0x00000408`, `header[4] = 0x00000068`, `header[6] = 0x1000d800`,
`header[7] = 0x00000200`, `header[8] = 0x05823025`, `0x0001c = 0x5000a021`,
`0x04804 = 0x00500172`, `0x04830 = 0x00500170`, and `0x0c800 = 0x00000082`.

### Task 1, compute (157 words, 49 vary)

| Word | Value |
|---|---|
| `0x0000c` | `K` |
| `0x00010` | `N` |
| `0x17808`, `0x17810`, `0x17814` | `M * N * 2` |
| `0x1780c` | `N * 2` |
| `0x1f848`-`0x1f884` | 16 kernel-DMA offsets, `i * K * N * 2 / 16` |
| `0x1f888`-`0x1f8c4` | 16 kernel-DMA sizes, `K * N * 2 / 16` |
| `header[1]` | `N / 64 - 1` |
| `0x00000`, `0x00014`, `0x00034`, `0x0480c`, `0x04810`, `0x04814`, `0x04834`, `0x1380c`, `0x13810`, `0x13814`, `0x13818` | vary with M, K, or N without a resolved formula |

Invariant task 1 words include `header[6] = 0x2000b800`,
`header[8] = 0x05024023`, `header[9] = 0x00000021`, `0x0001c = 0x5000b021`,
`0x00034 = 0x10244405` for M ≤ 8, `0x0c800 = 0x00000082`,
`0x0c804 = 0x00101c00`, `0x0c810 = 0x00003c00`, and
`0x17818 = 0x01302031`.

## Coverage and limits

- Covered as one program: every (M, K, N) with M ∈ {1, 2, 8, 64} and
  K, N ∈ {256, 512, 1024}, i.e. all 36 decoded matmul oracles.
- `transpose_y=false` is handled by transposing the constant exactly on the
  host and using the same encoder; a transposed single-row `x` reduces to the
  same contiguous surface and uses the same program.
- K > 1024, N > 1024, or any geometry off the grid keeps the existing
  source-qualified encoder with its reduction chunking and 512-column slices.
- Unresolved: the low bits of task 0's `header[1]`, the packed pair at
  `0x00000`/`0x00014`, the L2 configuration words listed above, the three-task
  scratch size, and which ANEC channel Apple's runtime would use if a matvec
  ever needed the scratch as a DMA surface.

# Softmax, layer_norm, and reductions

## Scope and provenance

This documents the `softmax`, `layer_norm`, `reduce_sum`, `reduce_max`, and
`reduce_mean` programs, minted on the same host and tool as the matvec section
above. The shipped campaign in `research/mint_oracles.py` contributes 10 cases
(two softmax shapes, two layer_norm shapes, six reductions), which cannot
separate a word that tracks C from one that tracks the reduced extent, because
every one of its normalization cases reduces the same axis.

`research/mint_norm_probes.py` adds the sweep. It attempted 226 cases and
Apple's compiler accepted 209:

| Sweep | Shapes | Purpose |
|---|---|---|
| Flat channel | `[1, C, 1, 1]`, C ∈ {64, 128, 256, 512, 1024, 2048, 4096} | The transformer residual width; softmax over C and over the last axis, layer_norm over C and over every non-batch axis, all three reductions over C |
| Sequence | `[1, C, 1, W]`, W ∈ {32, 64, 128, 512} | Softmax and reduction over the last axis with a nonunit W |
| Attention scores | `[1, H, S, S]`, H ∈ {1, 8, 12}, S ∈ {64, 128, 256} | Last-axis softmax on a score matrix |
| Spatial | `[1, C, S, S]`, (C, S) ∈ {(32, 4), (64, 8), (128, 16)} | Reduction and normalization over C, H, W, HW, and CHW |
| Rank reduction | `keep_dims = false` on the flat and spatial forms | Apple's dense output surface for a rank-reduced result |

The 17 rejections are all of one kind: **every affine `layer_norm` form is
rejected by this compiler**. `research/oracles/h13/*_affine.json` records them
with `callback_status=1`. Before concluding that, seven forms were tried:
inline rank-1 `gamma`/`beta` matching the reduced axis, inline rank-1 over a
unit last axis, inline rank-3 `[C, 1, 1]` with `axes = [1, 2, 3]`, `gamma`
without `beta`, a rank-2 `x` with `axes = [-1]`, a rank-4 `x` normalized over a
64-wide last axis, and `BLOBFILE`-backed `gamma`/`beta`. A `BLOBFILE`-backed
`mul` compiled in the same harness, so the blob path itself works. There is
therefore no oracle for an affine layer_norm, and both the encoder and
`tools/h13_reference.py` refuse one rather than emit a program Apple never
produced.

## Program shape

219 decoded records (105 normalization, 114 reduction) collapse to 186
templates keyed on `(operation, input CHW, reduced-axis mask, keep_dims)`, with
177 distinct task streams; `research/mint_norm_probes.py --emit-templates`
asserts that two records sharing a key agree on every decoded word, section
hash, and descriptor before it writes `plugins/H13/H13NormTemplates.inc`.

Task counts depend on the operation and on which axis reduces, not on the
extent:

| Operation | Reduced axes | Tasks | Constant section |
|---|---|---|---|
| `softmax` | C | 4 (W = 1, C = 1) or 5 | 256 / 1152 / 2176 bytes |
| `softmax` | H | 5 | 640 / 1152 / 2176 bytes |
| `softmax` | W | 5, 6 (H > 1), 8 (S = 256) | 128 / 1024 / 1536 / 2048 bytes |
| `layer_norm` | C | 5, 6 (H > 1) | 16384 zero bytes |
| `layer_norm` | W | 3, 5 (H > 1) | 16384 zero bytes |
| `layer_norm` | HW, CHW | 3 | 16384 zero bytes |
| `reduce_sum`, `reduce_mean` | C | 1 (H = W = 1) or 2 | 16384 zero bytes |
| `reduce_max` | C | 1 | 16384 zero bytes |
| all three reductions | H | 3 | 16384 zero bytes |
| all three reductions | W, HW, CHW | 1 | 16384 zero bytes |

Every task stream starts with a 126-word (504-byte) first task, so
`task_words_minus_one` is 125 in all 186 templates, and each later task's size
comes from the previous task's `header[1][24:16] + 1`, exactly as in the matvec
form.

## Constant section

`layer_norm` and all three reductions carry a 16384-byte all-zero section: the
decoded SHA-256 equals `sha256(bytes(16384))` for all 143 such records.

`softmax` carries fp16 lookup tables, and its section size varies with the
shape (128 to 2176 bytes). The content rule is exact for all 69 decoded softmax
records:

- the exponential table (`kExpKERNWords`, 128 bytes, the same block the unary
  `exp` encoder already uses) at offset 0;
- when the reduced axis is C or H, the reciprocal table (`kRecipKERNWords`,
  128 bytes) in the final 128 bytes of the section, i.e. at `size - 128`;
- zero everywhere else.

A last-axis softmax carries no reciprocal table. The generator derives which
of the three layouts applies by hashing each candidate against the recorded
section SHA-256, so `plugins/H13/H13NormTemplates.inc` never asserts a layout
that was not observed. The offsets were read out, not assumed: brute-forcing
every 2-byte-aligned position for both blocks returns `(0, size - 128)` for all
21 channel-axis and height-axis cases.

## Surfaces and scratch

Both surfaces follow the elementwise rule already in
`plugins/H13/H13Program.cpp:elementwiseTensor`, with the logical MIL shape
mapped to CHW by dropping leading unit dimensions above rank 3 and padding
below it:

```
row   = max(64, W * 2)
plane = row * H
bytes = plane * C
```

This reproduces `shape`, `strides`, and `total_bytes` for all 438 decoded input
and output surfaces, including the rank-reduced results: `keep_dims = false`
over `[1, 64, 8, 8]`'s spatial axes gives a logical `[1, 64]`, which Apple lays
out as `[1, 1, 1, 64]` with a 128-byte row stride, not as the channel-major
`[1, 64, 1, 1]` the `keep_dims = true` form uses. That difference is why
`keep_dims` is part of the template key.

The input surface is laid out first and the output second (unlike matvec).
Scratch is 0 for 179 templates and 16384 for 7 (channel-axis `reduce_sum` and
`reduce_mean` with two tasks, and some softmax shapes); the templates carry the
decoded value.

## Task words

A whole family mixes shapes that differ in C, H, and W at once, so words were
fitted inside 32 sweeps where exactly one of C, H, W moves. Across those
sweeps, of 12966 word slots: 11937 are invariant within their sweep, 562 have a
fitted formula, and 467 are unresolved.

Resolved words worth naming, all from the flat `[1, C, 1, 1]` sweep unless
stated:

| Word | Value |
|---|---|
| `0x0000c`, `0x00010` | `C` |
| `0x00028` | `C` (reductions and `layer_norm` over CHW) |
| `0x00000`, `0x00014` | `C + 0x10000`, or `C` in the high halfword with `1` in the low one, in the tasks that stream a plane per channel |
| `0x00024` | `C` in the high halfword with `0x4001` in the low one (`reduce_max`, `softmax` task 0) |
| `0x0480c`-`0x04840` | `C * 2`, `C * 16`, or an affine form such as `16 * C + 0x20`; the L2 base and stride block |
| `0x0880c` | `fp32(1 / reduced extent)`: exactly `1/C` for a channel reduction. This is the mean divisor, and it is the one word that carries the reduction's arithmetic rather than its geometry |
| `0x0002c` | `min(log2(reduced extent), 9)` in `layer_norm` (6, 7, 8, 9, 9, 9, 9 for C = 64..4096); a constant `84` in `reduce_mean` |
| `0x13814`, `0x13818` | `C * 64`, the source DMA byte count of the padded surface |
| `0x17810`, `0x17814` | `C * 64` on the output side |
| `0x1380c`, `0x13810`, `0x17808`, `0x1780c` | `CHW * 2` in the W sweeps, `rowIn` or `planeIn` elsewhere |
| `header[1]` | bits 24:16 are the next task's size minus one (`0x7d` for a 126-word successor, `0x9c` for a 157-word one); low bits vary and are unresolved |

The softmax kernel-DMA block is partly readable. The 16 offsets at
`0x1f848`-`0x1f884` are 0 when the task loads the exponential table and
`size - 128` when it loads the reciprocal table (`0x800` for a 2176-byte
section), which is the same offset the constant-section rule above gives. The
16 sizes at `0x1f888`-`0x1f8c4` are a mix of `0x80` and `0x40` whose split
tracks the number of active lanes (all `0x80` for `[1, 64, 1, 64]`, eight
`0x80` then eight `0x40` for `[1, 8, 64, 64]`, one `0x80` then fifteen `0x40`
for the flat shapes); the rule for the split is not resolved.

### Unresolved words

Listed per operation and per swept dimension. Parity does not depend on any of
them: the encoder emits Apple's decoded stream for a covered geometry and
refuses anything else, so an unresolved word is a limit on extrapolation, not
on correctness.

| Operation | Sweep | Unresolved |
|---|---|---|
| `layer_norm` | C | `0x0002c` (the clamp above 512 is observed, not derived), `header[1]` low bits |
| `reduce_max` | C | `0x04800`, `0x04810`, `0x04834`, `0x0c804`, `header[1]`, `header[4]` |
| `reduce_sum`, `reduce_mean` | C | `0x04810`, `0x04834`, `header[1]` |
| all three reductions | W | `0x00000`, `0x00014`, `0x00028`, `0x0002c`, `0x0480c`, `0x04810`, `0x04814`, `0x04818`, `0x04834`, `0x1380c` |
| `softmax` | C | `0x0000c`, `0x00010`, `0x00024`, `0x00030`, `0x00034`, `0x04808`-`0x04844`, the kernel-DMA block `0x1f80c`-`0x1f8c4`, `header[1]`, `header[8]` |
| `softmax` | W | the same set plus `0x13800`-`0x13838`, `0x17808`-`0x17818`, `header[4]`, `header[6]` |

The softmax lists are long because its L2 configuration and LUT DMA change
shape with the tiling, not because those sweeps disagree: each individual
template still reproduces Apple's words exactly.

## Coverage and limits

- Covered as one program: the 186 templates above, i.e. all 219 decoded
  normalization and reduction oracles. `tests/test_h13_parity.py` re-derives
  every task word, section hash, program descriptor, and tensor descriptor
  from the compiler's own output for each of them, in both `anec` and `hwx`.
- MIL rank is normalized before lookup, so the rank-2 and rank-3 forms a
  transformer emits reach the same program as the rank-4 surface they collapse
  onto: `softmax(axis = -1)` on `[1, 512]` compiles to the same bytes as
  `axis = 3` on `[1, 1, 1, 512]`.
- Refused, with `h13.norm-outside-envelope`: any shape off the sweep, an
  epsilon other than 1e-5, `gamma`/`beta`, a non-constant axes operand, and a
  reduction over the batch axis.
- Numerics are documented separately from parity. Apple's softmax evaluates
  `exp` and the reciprocal through the fp16 LUTs above, so device output differs
  from `tools/h13_reference.py`, which accumulates in float32 and rounds once.
  The reference is the numerical contract; the oracle bytes are the parity
  contract. They are not the same claim.
- `research/inspect_anec.py` reads a spatial or batched surface now: one
  formula, `row = max(64, W * 2)` with `plane = row * H`, covers the
  elementwise 64-byte lane, the padded spatial plane, and the matmul's dense
  rows. Its operation whitelist still excludes softmax and the reductions, so
  those packages remain unreadable by it; that is separate work.

# Runtime-runtime matmul, extended matvec, and broadcast

## Provenance

`research/mint_rrmm_probes.py` mints the probes behind this section and emits
`plugins/H13/H13EnvelopeTemplates.inc` from what decoded: 90 `rrmm_*` records
minted on `MacStudio.local` (Apple M1 Ultra, macOS 26.6.2 arm64) with Apple's
`/tmp/h13-oracle/bin/ane-compile-hwx`, alongside the 224 `env_*` matmul and
broadcast records from the envelope campaign. No HWX bytes are retained. The
generator verifies each record's surfaces, VM layout, scratch, binding order,
and constant-section hash against the encoder's own formulas before it writes
a template, so a template never encodes a case the encoder would build
differently.

## Program shape

| Form | Surfaces | Descriptor order | Address order | Constant section |
|---|---|---|---|---|
| Constant weight | `x`, result | `x`, result | result, `x` | `K * N * 2` packed weight |
| Both operands runtime | `y`, `x`, result | `y`, `x`, result | result, `y`, `x` | 16 KiB zero |
| Identical-shape binary | `x`, `y`, result | `x`, `y`, result | `x`, `y`, result | 16 KiB zero |
| Broadcast binary | `x`, `y`, result | `x`, `y`, result | `x`, result, `y` | 16 KiB zero |
| Scalar or per-channel constant | `x`, result | `x`, result | `x`, result | 16 KiB zero, or `4 * C` |

The address order is not the declaration order, and Apple varies it per
family: a matmul lays the result out below both operands, and a broadcast puts
the result between them. Both orders are derived, not assumed — the generator
reproduces every recorded `resource_addresses` list from the surface sizes and
the `__DATA/__bss` scratch below them, and fails the case if it cannot.

Every surface follows `row = max(64, W * 2)`, `plane = row * H`,
`element = plane * C`. A batched surface records `element` as its
tensor-descriptor total size, one batch element, while the object spaces
surfaces by `N * element`: `[8, 64, 8, 8]` records strides
`[32768, 512, 64, 2]` and total 32,768 bytes but occupies 0x40000. The runners
must size buffers from the manifest's `allocationBytes` for that reason, and
`HWXObjectBinding.allocationByteLength` carries it separately from
`storageByteLength` in the writer.

## Task structure

A runtime-runtime matmul emits a 126-word staging task per partition followed
by one 157-word compute task, and the compute task uses the extended header:
`header[9] & 3 == 3` adds one word between the fixed header and the first
register record, so the compute task is 158 words. `transpose_y=true` costs
exactly one task more than `transpose_y=false` at the same geometry (2 against
1 at M=16, 3 against 2 at M=128), the reverse of the constant-weight case.
Task counts over the minted M sweep at K = N = 64, `transpose_y=false`, are
1, 2, 3, 2, 4, 2, 4, 3, 4, 5 for M of 16, 32, 48, 64, 96, 128, 192, 256, 384,
512: the count is not monotonic in M, and no closed form reproduces it.

Rank does not change the stream. `rrmm_r2rr_m64_k64_n64_tx0_ty0` and
`rrmm_r3rr_m64_k64_n64_tx0_ty0_b1` decode to identical words, descriptors, and
section hashes, which is why the encoder normalizes rank before lookup.

## Constant sections

- A runtime second operand leaves the section 16 KiB of zeros: nothing is
  packed, because both operands arrive as surfaces.
- A constant weight is the `[columns, reduction]` fp16 matrix in Apple's
  permutation: `group` rows interleave at halfword granularity, each group is
  one contiguous reduction-major plane, and the planes are ordered by
  `(plane % 16) * (groups / 16) + plane / 16`. `group` is
  `min(16, N / 16)`, and above 128 x rows also `min(group, 32768 / K)`.
  Measured: M = 128 with K = 4096 packs 16 rows per group, M = 144, 160, 176,
  192, 224, 256 and 512 pack 8, and M = 192 with K = 8192 packs 4. Every
  decoded constant-weight section hash is reproduced by that rule and by no
  other rule in the corpus.
- A per-channel constant becomes `4 * C` bytes: `C` fp16 bias values then `C`
  fp16 scale values. `add` stores the operand as the bias and 1.0 as the
  scale; `mul` leaves the bias zero and stores the operand as the scale. Both
  layouts are proven against the recorded section hash at C = 64, 256, 768,
  1024 and 3072.

## Unresolved words

Parity does not depend on any of these: the encoder emits Apple's decoded
stream for a covered geometry and refuses anything else, so an unresolved word
limits extrapolation, not correctness.

- The geometry-scaled register words have no derived closed form. Between
  K = N = 2048 and K = N = 4096 at M = 128 (`transpose_y=false`), 58 of the
  284 stream words change; between `transpose_x=false` and `transpose_x=true`
  at one geometry, 24 of 410 change. The templates carry the measured words
  instead.
- Apple's task count for a runtime-runtime matmul (above) and the exact row
  count between 129 and 144 at which the packing group starts tracking
  `32768 / K` are both unresolved; every probed row count is a multiple of 16.
- The 16 KiB all-zero constant section every runtime broadcast and every
  runtime-runtime matmul carries is unexplained: no register in the decoded
  stream references it, and its size does not move with any swept dimension.
- `transpose_y=false` with a BLOBFILE constant weight stays refused. Three
  fresh probes at (M, K, N) of (1, 256, 256), (32, 2048, 2048) and
  (128, 2048, 2048) return `ANE internal validation error: Metadata data type
  does not match requested type.`, matching the first campaign's 36 refusals,
  so the host transpose to the accepted `transpose_y=true` program is a
  deliberate deviation rather than parity.
- The three envelope refusals at M = 128 with K = 8192 (N of 2048, 4096 and
  8192) are recorded as rejections with `callback_status=1` and no validation
  report; they remain unexplained, and the same K and N compile at every other
  probed M.
