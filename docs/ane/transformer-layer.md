# Transformer layers: chains, fusion, and task scheduling

This page records how a pre-norm transformer block maps onto ANE tasks on H13 (A14/M1) and H14 (A15/M2), as measured by the chain campaign in [fusion-rules.md](../../research/fusion-rules.md) and the envelope campaign in [oracle-envelope.md](../../research/oracle-envelope.md). Every claim names its research document and, where it rests on measured points, the `chain_*` oracle case that carries it. The oracle records keep decoded task words, descriptors, constant-section hashes and sizes, and compiler status; they do not keep Apple-generated HWX bytes. **Evidence: high for the recorded corpus.**

The block being scheduled is the standard pre-norm form, `x + attn(ln(x))` followed by `x + ffn(ln(x))`, with attention projections where stated. The campaign minted it at `d_model` in {256, 512, 768, 1024}, head counts in {4, 8, 12}, sequence lengths in {64, 128, 256, 512}, and a 4× feed-forward, as 127 `chain_*` records per target. See the coverage table in [fusion-rules.md](../../research/fusion-rules.md).

## One program, many tasks

Apple never splits a multi-operation model across programs. Every accepted chain object — including a 512-block stack at 10,240 tasks (`chain_stack512_d128_s64_proj0`) and an 8,192-unit stack at an 85,934,080-byte object (`chain_deep8192_d256_s64`) — carries exactly one program load command. Partitioning is by task inside the one program. **Evidence: high over the envelope and chain corpora.** See the partitioning section of [oracle-envelope.md](../../research/oracle-envelope.md) and section 7 of [fusion-rules.md](../../research/fusion-rules.md).

## Fusion is a post-operation field, not a rewrite

A matmul followed by `relu`, `gelu`, `silu`, a per-channel bias, a bias plus `relu`, or a multiply by a scalar costs exactly as many tasks as the matmul alone. The consumer operation disappears into one register word of the matmul's compute task, in the NE block: `0x0c804` on H13 and `0x00d04` on H14. Read against the lone matmul `chain_base_matmul_d256_s64`:

| Meaning | `0x0c804` / `0x00d04` | Evidence |
|---|---|---|
| Matmul, no post-operation | `0x00101c00` | `chain_base_matmul_d256_s64` |
| Binary elementwise, no post-operation | `0x00100000` | `chain_base_add_d256_s64` |
| Clamp / `relu` post-operation | bit `0x00010000` set (`0x00111c00`) | `chain_pair_mm_relu_d256_s64` |
| Lookup-table unary post-operation | bit `0x00020000` set (`0x00121c00`) | `chain_pair_mm_gelu_d256_s64`, `chain_pair_mm_silu_d256_s64` |
| Per-channel bias | low nibble `0x10` (`0x00101c10`) | `chain_pair_mm_bias_d256_s64` |
| Bias and clamp together | both (`0x00111c10`) | `chain_pair_mm_bias_relu_d256_s64` |

**Evidence: high; byte-exact diffs.** See section 2 of [fusion-rules.md](../../research/fusion-rules.md).

Two further words complete the mechanism:

- **Output scale.** `0x0c810` (H13) and `0x00d10` (H14) hold an fp16 multiplier. A matmul alone carries fp16 1.0 (`0x00003c00`); a multiply by fp16 0.125 writes `0x00003000` and changes nothing else (`chain_pair_mm_mul_d256_s64`). This is the word that carries attention's `1/sqrt(d_head)` scaling for free.
- **Kernel-DMA extension.** A fused lookup table or bias also extends the kernel-DMA lane descriptors — `0x1f848`–`0x1f8c4` on H13, `0x0195c`–`0x019d4` on H14: each of the 16 lane offsets and 16 lane sizes grows by the appended constant bytes (`0x80` per lane for a 2,048-byte table, `0x40` for a 1,024-byte bias). That is the whole of the "32 changed words" a fused `gelu` diff shows: one post-operation word plus 31 DMA words. **Evidence: high.** See [fusion-rules.md](../../research/fusion-rules.md) §2.

Nothing else fuses. Fusion reaches exactly one operation forward from a matmul and only into the post-operation field. A residual `add` against a runtime tensor costs two extra tasks and rewrites 108 words (`chain_pair_mm_add_d256_s64`); a second matmul, a `layer_norm` and a `softmax` each keep their own tasks in both directions (`chain_pair_mm_mm_d256_s64`, `chain_pair_ln_gelu_d256_s64`, `chain_pair_mm_softmax_d256_s64`). Feeding a matmul is never free: `gelu_mm`, `mul_mm` and `softmax_mm` all pay for the producer separately. **Evidence: high.** See the pair table in [fusion-rules.md](../../research/fusion-rules.md) §2.

The task stream does not distinguish one lookup-table unary from another: `chain_base_gelu_d256_s64` and `chain_base_silu_d256_s64` decode to identical words, and only the 128-byte constant-section table separates them. An encoder must therefore carry the table identity out of band. **Evidence: high.** See section 1 of [fusion-rules.md](../../research/fusion-rules.md); see also [numerics.md](numerics.md) for what the table means arithmetically.

## The attention path

Attention is `QK^T -> scale -> softmax -> PV` over three runtime inputs. The whole chain is one program at every tested size, declaring exactly 4 surfaces (three operands plus the result) and no intermediates: the `[S, S]` score matrix never gets a tensor descriptor. **Evidence: high.** See `chain_attn_*` in section 3 of [fusion-rules.md](../../research/fusion-rules.md).

**Runtime-runtime matmul structure.** With both operands runtime — the form constant-weight probes never covered — every tested score and context shape was accepted with a 16 KiB all-zero constant section. Score matmuls `[1,S,D] x [1,S,D]^T` cost 3 tasks at S=64 and S=128, 4 (H13) / 3 (H14) at S=256, and 6 / 3 at S=512; context matmuls `[1,S,S] x [1,S,D]` cost 2, 3 / 2 and 5 / 2. `transpose_y=false` costs one task fewer than `transpose_y=true` in the runtime-runtime form — the opposite ranking of the constant-weight form. **Evidence: high.** See the runtime-runtime table in [oracle-envelope.md](../../research/oracle-envelope.md) §2, cases `env_mm_r3rr_*`.

These tasks use the extended task header: one extra word between the fixed header and the first register record, present in every runtime-runtime matmul except the rank-2 M=16 cases, and in every attention chain — 48 of 2,677 decoded H13 tasks and 30 of 2,251 H14 tasks. In `chain_attn_s64_dh64` the two trailing compute tasks are 632 bytes, not 628: 157 words plus the extra word. **Evidence: high.** See section 7 of [oracle-envelope.md](../../research/oracle-envelope.md) and [task-descriptors.md](task-descriptors.md).

**Softmax program structure.** A last-axis softmax is its own multi-task program: 6 tasks at `[1,1,64,64]` and `[1,1,128,128]`, 8 at `[1,1,256,256]`, and 6 for batched heads `[1,8,S,S]` and `[1,12,S,S]`, built from 126-word staging tasks around compute tasks. Its constant section is the 128-byte exponential table at offset 0; a last-axis softmax carries no reciprocal table. In the attention chain the constant section is the softmax table only — 2,048 bytes on H13, 128 on H14, independent of sequence length and head count. **Evidence: high.** See the chain table in [oracle-envelope.md](../../research/oracle-envelope.md) §5, the normalization section of [h13-td-fields.md](../../research/h13-td-fields.md), and every `chain_attn_*` record.

**Scale folding.** The `1/sqrt(d_head)` multiply is free: dropping it (`chain_attn_s128_dh64_noscale`, `chain_attn_s256_dh64_noscale`) changes neither the task count nor the constant section, because the scalar lands in the output-scale word `0x0c810` / `0x00d10` as an fp16 multiplier. **Evidence: high.** See section 2 and the attention table of [fusion-rules.md](../../research/fusion-rules.md).

**Task-count law.** H14 is flat at 9 tasks for every sequence length and head count. H13 emits `9 + 2 * (S/128 - 1)` for S ≥ 128 — 9, 11, 15 at 128, 256, 512 — and `9 + 3 * (heads - 1)` when heads arrive as a rank-3 batch: 18, 30, 42 at 4, 8, 12 heads (`chain_attn_s128_dh64_h4/h8/h12`). The head dimension does not matter; 64 and 96 give identical streams (`chain_attn_s64_dh64` vs `chain_attn_s64_dh96`). **Evidence: high.** See the attention table in [fusion-rules.md](../../research/fusion-rules.md) §3.

Scratch: H13 allocates 16,384 bytes at S=64 growing to 524,288 at S=512; H14 allocates none. **Evidence: high.** `chain_attn_*` scratch columns.

## The feed-forward path

The feed-forward block `matmul -> activation -> matmul` is **two compute tasks** on both targets at every width, because the activation is a post-operation on the first matmul:

| Case | Tasks H13/H14 | H13 task sizes | Constant section |
|---|---:|---|---:|
| `chain_ffn_d256_s64` | 4 / 3 | 504×2, 628×2 | 1,050,624 |
| `chain_ffn_d256_s128` | 3 / 3 | 504×1, 628×2 | 1,050,624 |
| `chain_ffn_d512_s64` | 4 / 3 | 504×2, 628×2 | 4,196,352 |
| `chain_ffn_d1024_s64` | 4 / 3 | 504×2, 628×2 | 16,779,264 |
| `chain_ffn_d512_s256` | 6 / 6 | 504×1, 628×5 | 4,200,448 |
| `chain_ffn_d256_s64_biasfull` | 7 / 7 | 504×4, 628×3 | 1,212,544 |

`relu`, `gelu` and `silu` give the same task count and differ only in the constant section: a `relu` adds nothing, a lookup-table unary adds 2,048 bytes on both targets. A per-channel bias is also free in tasks and costs `ceil(2 * C / 1024) * 1024` bytes of constant section (`chain_ffn_d256_s64_biasbcast`); a bias shaped like the whole activation is not a per-channel bias and costs four extra tasks (`chain_ffn_d256_s64_biasfull`). Task counts do not move with `d_model`; they move only with the sequence length, and only on H13, which needs one extra 126-word task at `s=64` (4 tasks against H14's 3) and a 6-task form at `d_model=512, s=256`. **Evidence: high.** See the feed-forward table in [fusion-rules.md](../../research/fusion-rules.md) §3.

## Normalization feeding a projection

`layer_norm -> matmul` costs 5 tasks for every width, plus two per doubling of the sequence above 128: 5, 7, 9 at `s` of 128, 256, 512 (`chain_lnproj_d512_s256`, `chain_lnproj_d512_s512`). The projection contributes one 157-word compute task; the rest are the normalization's own 126-word tasks (`504×4, 628×1`). The constant section is exactly the weight bytes — the `layer_norm`'s 16 KiB all-zero section is absorbed. `layer_norm` never fuses in either direction. **Evidence: high.** See the `chain_lnproj` table in [fusion-rules.md](../../research/fusion-rules.md) §3.

## Residuals

A residual around a whole sub-block costs exactly one extra task over the sub-block itself, identical on both targets: 10 against attention's 9 (`chain_resattn_s64_d256`, `chain_resattn_s128_d256`), 5 against the feed-forward's 3 or 4 (`chain_resffn_d256_s64`), at every width measured. The `x + relu(x)` single-task collapse (`chain_res_relu_d256`, `chain_res_relu_d1024`) is the elementwise engine's own clamp post-operation, not a general add fusion; the `gelu` and `silu` residuals cost two tasks because they need their table (`chain_res_gelu_d256`, `chain_res_silu_d256`). **Evidence: high.** See the residual table in [fusion-rules.md](../../research/fusion-rules.md) §3 and the chain section of [oracle-envelope.md](../../research/oracle-envelope.md) §5.

## Whole blocks and stacks

| Case | Tasks H13/H14 | H13 task sizes | Constant H13/H14 | Scratch H13/H14 |
|---|---:|---|---:|---:|
| `chain_block_d256_s64_proj1` | 25 / 24 | 504×15, 628×7, 632×3 | 1,576,960 / 1,575,040 | 49,152 / 0 |
| `chain_block_d512_s64_proj1` | 27 / 24 | 504×15, 628×7, 632×5 | 6,295,552 / 6,293,632 | 81,920 / 0 |
| `chain_block_d768_s64_proj1` | 29 / 24 | 504×15, 628×7, 632×7 | 14,159,872 / 14,157,952 | 114,688 / 0 |
| `chain_block_d256_s64_proj0` | 20 / 20 | 504×15, 628×3, 632×2 | 1,052,672 / 1,050,752 | 32,768 / 0 |
| `chain_stack2_d256_s64_proj1` | 50 / 48 | 504×30, 628×14, 632×6 | 3,151,872 / 3,149,952 | 49,152 / 0 |
| `chain_stack8_d256_s64_proj1` | 200 / 192 | 504×120, 628×56, 632×24 | 12,601,344 / 12,599,424 | 49,152 / 0 |

A pre-norm block with four projections is 25 tasks on H13 at `d_model=256` plus two per additional 256 of width (27 at 512, 29 at 768); H14 is 24 for every width. Without projections both targets emit 20. Depth multiplies exactly — `tasks = depth × per-block`, verified with full task words at 2, 3, 4, 6 and 8 blocks and in summary records through 512 blocks — and there is no cross-block fusion and no common-subexpression elimination of the compute. On H13 the task-length sequence of a stack is exactly periodic with period one block. **Evidence: high.** See the block and stack tables in [fusion-rules.md](../../research/fusion-rules.md) §3.

The task order is MIL dataflow order in every case: a block's tasks appear as normalization, projections, scores, softmax, context, residual, normalization, feed-forward, residual. Sharing an input buys one task: two projections of one input (`chain_branch_d256_s64`, 4 tasks) cost one fewer than the same projections chained (`chain_serial_d256_s64`, 5), because the two compute tasks share a single staging task for the input they both read. **Evidence: high.** See section 6 of [fusion-rules.md](../../research/fusion-rules.md).

## Task sequence and staging tasks

H13 emits 126-word (504-byte) staging tasks and 157/158-word (628/632-byte) compute tasks. Written `S` for staging and `C` for compute, in emission order:

| Case | H13 sequence |
|---|---|
| `chain_base_matmul_d256_s64` | `S C` |
| `chain_ffn_d256_s64` | `S C C S` |
| `chain_ffn_d256_s128` | `S C C` |
| `chain_lnproj_d256_s64` | `S S S S C` |
| `chain_attn_s64_dh64` | `S S C S S C S S C` |
| `chain_serial_d256_s64` | `S C S C S` |
| `chain_block_d256_s64_proj0` | `S S S S S C S S C S S C S S S S S C C S` |
| `chain_block_d256_s64_proj1` | `S S S S C C C S S C S S S C C C C S S S S S C C S` |

The rule behind it: a compute task is preceded by a 126-word staging task that brings its operands into L2, consecutive compute tasks reading what is already resident share one staging task (`S C C` in the feed-forward; the task `chain_branch` saves), and at `s=64` a trailing 126-word task writes the result surface back. **Evidence: high for the recorded sequences; the predicate for the trailing task is an open question.** See section 6 of [fusion-rules.md](../../research/fusion-rules.md).

## Surfaces and scratch

No intermediate is ever declared. Every chain declares exactly its inputs and its result — 2 tensor descriptors for a single-input chain, 4 for attention's three operands plus output. Not the 4× feed-forward hidden state, not the `[S, S]` score matrix, not a block's post-residual activation gets a tensor descriptor or a resource address. **Evidence: high.** See section 4 of [fusion-rules.md](../../research/fusion-rules.md).

The intermediates live in the `__DATA/__bss` scratch below the declared surfaces, based at `0x30000000`. The measured behaviour:

- **H14 allocates no scratch at all** in any chain in the campaign.
- **H13's scratch tracks the largest live intermediate, and is reused.** A feed-forward at `d_model=256, s=64` holds a `[1,64,1024]` hidden state of 131,072 bytes and allocates exactly that; at `d_model=1024, s=64` both are 524,288.
- **Tiling shrinks the scratch below the intermediate.** The same feed-forward at `s=128` is a 3-task program with 65,536 bytes of scratch for a 262,144-byte hidden state — a quarter. The tiling factor is visible but its rule is not derived.
- **Depth does not add scratch.** One block and eight blocks at `d_model=256, s=64` both allocate 49,152 bytes; scratch scales with sequence length and width instead.

**Evidence: high.** See section 4 of [fusion-rules.md](../../research/fusion-rules.md) and [containers.md](containers.md) for the wrapper view.

## Constant-section composition

The constant section of a chain is the concatenation of what its operations need, with one dedup and one floor:

| Contribution | Bytes | Evidence |
|---|---|---|
| Each distinct BLOBFILE weight | `rows × columns × 2`, in Apple's packing | `chain_lnproj_*` at four widths |
| Lookup-table unary fused into a matmul | 2,048 on both targets | `chain_ffn_d256_s64` vs `chain_ffn_d256_s64_relu` |
| Lookup-table unary standing alone | 128 on both targets | `chain_base_gelu_d256_s64` |
| Fused per-channel bias | `ceil(2 * C / 1024) * 1024` per bias | `chain_ffn_*_biasbcast`: 3,072 for 1024+256, 5,120 for 2048+512 |
| Softmax | 2,048 H13, 128 H14 | every `chain_attn_*` |
| `layer_norm` or runtime binary, alone | 16,384 zero bytes | `chain_base_layer_norm`, `chain_base_add` |
| Whole projected block | weights + 4,096 H13, weights + 2,176 H14 | `chain_block_*` at three widths |

Two rules a scheduler must respect: the 16 KiB all-zero block is a **floor, not an addend** — `layer_norm -> gelu` (`chain_pair_ln_gelu_d256_s64`) is 16,256 bytes smaller than a lone `layer_norm`, because the table replaces the zero block; and identical constants are **stored once** — a projected block whose six weight matrices pointed at one shared blob offset declared 1,572,864 bytes and packed 1,183,744. Deduplication removes constants, not compute. **Evidence: high.** See sections 5 and the provenance note of [fusion-rules.md](../../research/fusion-rules.md). Weight packing itself is the permutation documented in the matvec section of [h13-td-fields.md](../../research/h13-td-fields.md); convolution packing, where relevant, is in its conv section.

## Task-count laws

Collected from the per-family tables of [fusion-rules.md](../../research/fusion-rules.md) §3:

| Chain family | H13 | H14 |
|---|---|---|
| Feed-forward (`matmul -> act -> matmul`) | 2 compute tasks; 4 total at `s=64`, 3 above, 6 at `d512, s256` | same, but 3 at `s=64` |
| `layer_norm -> projection` | 5, plus 2 per doubling of `s` above 128 (5, 7, 9) | same |
| Attention, single head | `9 + 2 * (S/128 - 1)` for S ≥ 128 | 9, flat |
| Attention, batched heads | `9 + 3 * (heads - 1)` | 9, flat |
| Residual around a sub-block | +1 task | +1 task |
| Projected block | 25 at `d256`, +2 per 256 of width | 24, flat |
| Stack | `depth × per-block` | `depth × per-block` |

## The geometry envelope and the ceilings

The campaign held `d_model` to {256, 512, 768, 1024}, heads to {4, 8, 12}, sequence to {64, 128, 256, 512} and the feed-forward to 4×, and found no chain-level ceiling anywhere inside it. The ceilings are not depth, task count, or weight bytes:

- **8,192 stacked weightless units** compile — 73,728 MIL operations, 740 seconds, an 85,934,080-byte object on H13 (`chain_deep8192_d256_s64`).
- **512 stacked blocks** with a feed-forward and an attention each compile to 10,240 tasks on both targets (`chain_stack512_d128_s64_proj0`).
- **64 fully projected blocks** compile with a 402,786,304-byte constant section (`chain_ceiling_weights64_d512`).
- The largest accepted single matmul chain is `M=256, K=8192, N=8192`: 385 tasks and 268,959,744 bytes of constants (`chain_ceiling_mlp_m256_k8192_n8192_ty1`).

**The single refusal is inherited from the per-operation envelope.** The same feed-forward built on `M=128, K=8192, N=8192` returns `callback_status=1` on both targets (`chain_ceiling_mlp_m128_k8192_n8192_ty1`) — the geometry the envelope campaign already refused as a lone matmul. Chain acceptance is the conjunction of per-operation acceptance; a scheduler needs no chain-level size check beyond the per-operation envelope. **Evidence: high.** See section 7 of [fusion-rules.md](../../research/fusion-rules.md) and §6 of [oracle-envelope.md](../../research/oracle-envelope.md).

One refusal-list correction matters here: a `transpose_y=false` BLOBFILE matmul that [h13-td-fields.md](../../research/h13-td-fields.md) records as refused standing alone is **accepted as the first matmul of a feed-forward chain** at three geometries (`chain_ceiling_mlp_m32_k2048_n2048_ty0` and two others). The geometry alone does not predict the verdict; the envelope check must be derived from probes using the same weight layout the scheduler will emit. **Evidence: high for the three accepted cases; open question which difference flips the verdict.** See the coverage section of [fusion-rules.md](../../research/fusion-rules.md).

What actually stopped the depth sweep was compile time, not acceptance: compile seconds scale roughly linearly in depth (130 s at 2,048, 281 s at 4,096, 740 s at 8,192), and depth 16,384 exceeded the repository's 900-second budget (`chain_deep16384_d256_s64`, recorded as `compiler timed out`, not refused). **Evidence: high.** See section 7 of [fusion-rules.md](../../research/fusion-rules.md).

## What this means for a scheduler

Section 8 of [fusion-rules.md](../../research/fusion-rules.md) derives the algorithm; the shape of it:

1. Concatenate per-operation task streams in MIL dataflow order and relink them (H13 `header[7]` chaining; H14 16-byte alignment).
2. Apply the fusion field instead of emitting the consumer: set the `0x0c804`/`0x00d04` bits, write the fp16 multiplier into `0x0c810`/`0x00d10`, append the table or bias to the constant section, and extend the 32 kernel-DMA words.
3. Place intermediates in `__DATA/__bss` scratch and declare none of them; reuse one scratch allocation for every block in a stack.
4. Insert the 126-word staging tasks, keyed to a measured template per covered chain shape.
5. Predict the task budget from the per-family laws above.
6. Refuse only what the per-operation envelope refuses.

## Open questions

- **Open question:** What is the tiling factor that makes a 3-task feed-forward allocate a quarter of its hidden state as scratch, while the 4-task form allocates all of it?
- **Open question:** What is the predicate for the trailing 126-word write-back task — present at `s=64`, absent at `s=128` and above in the feed-forward family?
- **Open question:** Why does H13 need `2 * (S/128 - 1)` extra attention tasks and 3 extra tasks per batched head where H14 needs none, and why does a projected block cost H13 two extra tasks per 256 of width?
- **Open question:** What composes the 4,096-byte (H13) and 2,176-byte (H14) constant-section overhead of a projected block? The softmax table accounts for 2,048 and 128 of it.
- **Open question:** Which lookup table a fused unary uses is not visible in the task words; how should a verifier confirm table identity beyond the constant-section hash?
- **Open question:** Does the fp16 output-scale word round scalars that fp16 cannot represent exactly (for example `1/sqrt(96)`)? The measured cases used exactly representable scalars (`chain_pair_mm_mul_d256_s64`, fp16 0.125).
- **Open question:** Why is `transpose_y=false` with a BLOBFILE weight refused alone but accepted inside a chain at the same geometry?
