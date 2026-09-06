# Task descriptors and register-block maps

This page records what the repository's Apple-oracle corpus establishes about task-descriptor encoding for H13 (A14/M1) and H14 (A15/M2). Each claim names the research document that derives it and, where the claim rests on one measured point, the oracle case that carries it. The oracle records keep decoded fields, hashes, and compiler status; they do not keep Apple-generated HWX bytes. **Evidence: high for the retention policy.** See the [differential report](../../research/oracle-diff.md), the [envelope report](../../research/oracle-envelope.md), and the repository [disclaimer](../../DISCLAIMER.md).
How chains of these tasks are scheduled, fused, and sized is covered in [transformer-layer.md](transformer-layer.md).

## Corpus behind this page

| Source | Attempts | Decoded | Refused | Where |
|---|---:|---:|---:|---|
| First campaign, one input changed at a time | 470 (235 per target) | 344 | 126 | [oracle-diff.md](../../research/oracle-diff.md) |
| Envelope campaign, `env_*` cases at the edge of the accepted forms | 548 (274 per target) | 542 | 6 | [oracle-envelope.md](../../research/oracle-envelope.md) |
| Chain campaign, `chain_*` cases at transformer sizes | 254 (127 per target) | 190 with full task words | 2 | [fusion-rules.md](../../research/fusion-rules.md) |
| Convolution probes, `conv_probe_*` and `conv_known_*` | 1,048 (524 per target) | 1,048 | 0 | conv section of [h13-td-fields.md](../../research/h13-td-fields.md) |

The pre-chain campaigns' committed records decode 4,928 tasks: 2,677 on H13 and 2,251 on H14, of which 1,790 and 1,736 come from the envelope campaign. **Evidence: high; recounted over `git ls-files research/oracles`.** The chain and convolution records extend the corpus but are counted in their own reports.

Both campaigns were minted on `MacStudio.local`, an M1 Ultra running macOS 26.6.2, with one Apple compiler build whose SHA-256 every record carries, alongside the hashes of the minting driver and the decoder. Each record states its own provenance, because the first campaign was deliberately not re-minted when the envelope campaign was added. **Evidence: high for the recorded provenance.** See [oracle-diff.md](../../research/oracle-diff.md) and [oracle-envelope.md](../../research/oracle-envelope.md).

Three narrower probe sets back specific field derivations: `research/mint_matvec_probes.py` and `research/mint_h14_matvec_probes.py` for constant-section packing (101 decoded H13 and all 125 H14 probe records), and `research/mint_norm_probes.py` for the normalization and reduction sweep. **Evidence: high for the checked-in probe records.** See [h13-td-fields.md](../../research/h13-td-fields.md) and the appendix of [h14-td-fields.md](../../research/h14-td-fields.md).

## Record encoding

A task is a fixed header followed by register-write records. Each record header carries a length and a target address inside a hardware register block; payload indices have meaning only within that block and generation. A value at NE word 1 must not be read with a Tile DMA word-1 definition.

| Property | H13 | H14 |
|---|---|---|
| Header words | 10 | 8 |
| Record form | Dense only: `count = (header >> 26) + 1`, byte address `header & 0x03ffffff` | Dense: `count = ((header >> 15) & 0x3f) + 1`, base is a **word** index in bits 14:0. Scatter: bit 31 set, 16-bit following-word mask in bits 30:15 |
| Record population in the first campaign | 2,032 dense records | 1,237 dense and 1,188 scatter records |

**Evidence: medium.** The encodings are reconstructed, not published. See [oracle-diff.md](../../research/oracle-diff.md) (task-descriptor encodings) and the task-stream delta in [h14-td-fields.md](../../research/h14-td-fields.md); the H13 form is also implemented by the pinned [coreml_to_ane_hwx parser](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump) and exercised by this repository's HWX inspector.

## Task sizing, linking, and alignment

H13 declares the first task's size in the program descriptor at `+0x818` plus one, and every later task's size in the preceding task's `header[1]` bits 24:16 plus one. `header[7]` holds the next task's section-relative byte offset and is zero in the final task, so an H13 task stream is a linked list. Zero padding separates linked sections: `matmul_m1_k256_n512_ty1` declares two tasks, a 126-word first task, a next pointer of `0x200`, eight zero bytes, then a 157-word second task. `matmul_m64_k256_n1024_ty1` links a third 126-word copy task at `0x500`. **Evidence: high for the decoded records; medium for the field names.** See [oracle-diff.md](../../research/oracle-diff.md) and the matvec section of [h13-td-fields.md](../../research/h13-td-fields.md).

H14 sizes each task independently from `header[0]`, which packs `task_words << 16 | task_id` and yields the exact word count in bits 26:16. There is no next-task pointer. Tasks are 16-byte aligned, and a zero-size 16-byte frame is present in the text section, so a reader walks the stream by header size and alignment and skips zero-size frames. `binary_add_1x64x1x1` is one 61-word task; `matmul_m1_k256_n512_ty1` is 38 words then 85. **Evidence: high for the decoded records.** See the task-stream delta in [h14-td-fields.md](../../research/h14-td-fields.md).

One published prediction fails here: a nine-word H14 header with a DTID field at word 8 is refuted for every one of the 265 decoded H14 tasks in the first campaign. Word 8 is the first record header. **Evidence: high for the refutation over that corpus.** See the map-predictions table in [h14-td-fields.md](../../research/h14-td-fields.md).

## The extended task header

The last fixed header word — H13 `header[9]`, H14 `header[7]` — carries a flag in its two low bits. When both bits are set, one extra word sits between the fixed header and the first register record:

```text
extended = (last_header_word & 0x3) == 0x3
```

Across the corpus that word takes four values on H13 (`0x0`, `0x21`, `0x23`, `0x26`) and six on H14 (`0x1`, `0x10001`, `0x30001`, `0x40001`, `0x50001`, `0x50003`). Only `0x23` and `0x50003` carry the extra word, so bit 1 alone is not the predicate: H13 `0x26` and H14 `0x30001` set bit 1 and carry no extra word. The declared task size already includes the extra word, so task splitting is unaffected — a 157-word compute task becomes 158 — and only the register stream shifts. **Evidence: high over the corpus; open question for the word's meaning.** See section 7 of [oracle-envelope.md](../../research/oracle-envelope.md) and [oracle-diff.md](../../research/oracle-diff.md).

The population is small and entirely inside two families. Re-counted over the committed records, 48 of 2,677 decoded H13 tasks and 30 of 2,251 decoded H14 tasks are extended, in 26 cases per target: every runtime-runtime matmul except the four rank-2 M=16 cases, plus the four attention chains. `env_mm_r2rr_m128_k2048_n2048_tx1_ty1` and `env_chain_attention_s128_d64` are extended; `env_mm_r2rr_m16_k2048_n2048_tx0_ty1` is not, so the boundary sits between M=16 and M=128 for rank 2, while every rank-3 runtime-runtime case from M=64 up is extended. The extra word is `0x0` in 43 H13 and 26 H14 tasks and `0x7` in 3 H13 and 4 H14 tasks; **two H13 tasks hold `0x8`**, tasks 9 and 10 of `env_chain_attention_s256_d64`, which the envelope report's "`0x0` or `0x7` in every observed case" does not cover. **Evidence: high; re-derived from the records rather than quoted.**

Both failure modes are worth knowing for anyone writing a parser. On H13 the misread raised an unaligned-address error, which a campaign driver can easily misfile as an Apple rejection. On H14 the misread did not raise at all: the extra word parsed as a plausible record header and produced silently wrong register words. **Evidence: high.** See section 7 of [oracle-envelope.md](../../research/oracle-envelope.md).

## Block map

| Block | H13 base / words | H14 base / words |
|---|---:|---:|
| Common | `0x00000` / 16 | `0x0000` / 19 |
| L2 | `0x04800` / 18 | `0x0500` / 25 |
| PE (planar engine) | `0x08800` / 4 | `0x0900` / 5 |
| NE (neural engine) | `0x0c800` / 5 | `0x0d00` / 5 |
| Tile DMA source | `0x13800` / 28 | `0x1100` / 53 |
| Tile DMA destination | `0x17800` / 7 | `0x1500` / 10 |
| Kernel DMA | `0x1f800` / 62 | `0x1900` / 70 |

**Evidence: medium.** The functional order is shared and confirmed by corresponding one-parameter changes on both targets; the H14 names come from the pinned [H14 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md). See [oracle-diff.md](../../research/oracle-diff.md).

Two corrections to that external map are established by the corpus. First, the `+0x3c00` remap for non-Common blocks is only the map's old-to-modern presentation transform, not a stream encoding rule: every H14 record in all 172 decoded H14 cases uses the old bases `0x0500`–`0x1900`, and none uses `0x4100`–`0x5500`. Second, H14 destination DMA extends to `0x1524`, which Apple writes in `binary_add_1x64x1x1`, so a decoder must use the exclusive end `0x1528` rather than stopping at nine words. No record in any decoded H14 task falls outside the declared blocks. **Evidence: high over the decoded corpus.** See the map-predictions and block tables in [h14-td-fields.md](../../research/h14-td-fields.md) and [oracle-diff.md](../../research/oracle-diff.md).

### Words with a resolved value or formula

| Word (H13 address) | Resolved meaning | Evidence |
|---|---|---|
| Common `0x00000`, `0x00014` | Packed 16-bit W/H: `1 → 0x00010001`, `8 → 0x00080008`, `224 → 0x00e000e0` on both targets | [oracle-diff.md](../../research/oracle-diff.md), `binary_add_1x64x1x1`, `binary_add_1x64x8x8`, `binary_add_1x3x224x224` |
| Common `0x0000c`, `0x00010` | Channel count C in elementwise and normalization tasks; K and N respectively in the matvec compute task | [oracle-diff.md](../../research/oracle-diff.md), [h13-td-fields.md](../../research/h13-td-fields.md) |
| PE `0x08800` first word | Operation selector: `0x00080000` add, `0x00080004` mul, `0x00080008` max, `0x0008000c` min, `0x000c0000` sub, on both targets | [oracle-diff.md](../../research/oracle-diff.md) |
| PE `0x0880c` | `fp32(1 / reduced extent)`, the mean divisor — the one word carrying a reduction's arithmetic rather than its geometry | [h13-td-fields.md](../../research/h13-td-fields.md), flat `[1,C,1,1]` sweep |
| Common `0x0002c` | `min(log2(reduced extent), 9)` in `layer_norm`; a constant `84` in `reduce_mean` | [h13-td-fields.md](../../research/h13-td-fields.md) |
| Tile DMA `0x13814`, `0x13818`, `0x17810`, `0x17814` | `C * 64`, the byte count of the 64-byte-padded surface on the source and destination sides | [h13-td-fields.md](../../research/h13-td-fields.md) |
| Kernel DMA `0x1f848`–`0x1f884` and `0x1f888`–`0x1f8c4` | In matvec, 16 coefficient offsets `i * K * N * 2 / 16` and 16 sizes `K * N * 2 / 16`; in softmax, offset 0 for the exponential table and `size - 128` for the reciprocal table | [h13-td-fields.md](../../research/h13-td-fields.md) |
| `header[1]` bits 24:16 (H13) | The next task's size minus one (`0x7d` for a 126-word successor, `0x9c` for a 157-word one) | [h13-td-fields.md](../../research/h13-td-fields.md) |
| NE `0x0c800+4` / `0x0d00+4` (`0x0c804` / `0x00d04`) | The post-operation field: bit `0x00010000` clamp, bit `0x00020000` lookup-table unary, low nibble `0x10` per-channel bias; `0x00101c00` bare matmul, `0x00100000` bare binary elementwise | [fusion-rules.md](../../research/fusion-rules.md) §2, `chain_pair_mm_relu_d256_s64`, `chain_pair_mm_gelu_d256_s64`, `chain_pair_mm_bias_d256_s64` |
| NE `0x0c810` / `0x00d10` | The output scale, an fp16 multiplier: `0x00003c00` (1.0) on a bare matmul, `0x00003000` (0.125) when the MIL multiplies by fp16 1/8 — the free word for attention's `1/sqrt(d_head)` | [fusion-rules.md](../../research/fusion-rules.md) §2, `chain_pair_mm_mul_d256_s64` |

The remaining variation is recorded, not named. For H14 the emitter-readiness count is Common 12 of 19 words resolved, L2 10 of 25, PE 3 of 5, NE 3 of 5, Tile DMA source 47 of 53, Tile DMA destination 7 of 10, and Kernel DMA 16 of 70, where "resolved" means invariant, never written, or covered by a verified one-parameter formula in at least one sampled family. For H13's normalization sweeps, 11,937 of 12,966 word slots are invariant within their sweep, 562 have a fitted formula, and 467 are unresolved. An unresolved word limits extrapolation, not parity: a template emits Apple's decoded stream for a covered geometry and refuses anything else. **Evidence: high as counts over the corpus.** See [h14-td-fields.md](../../research/h14-td-fields.md) and [h13-td-fields.md](../../research/h13-td-fields.md).

An absent write is not a zero write. The decoded records distinguish the two, and so must any comparison between compilers. **Evidence: high.** See [h14-td-fields.md](../../research/h14-td-fields.md) and the native-versus-Apple tables in [oracle-diff.md](../../research/oracle-diff.md).

## The matvec two-task form

Apple's constant-weight `matmul` with `transpose_y=true` is one program with two tasks over the whole decoded grid M ∈ {1, 2, 8, 64}, K, N ∈ {256, 512, 1024}:

| Field | Value |
|---|---|
| Task 0, preparation | 126 words, `header[7] = 0x200` |
| Task 1, compute | 157 words, `header[7] = 0` |
| Constant section | exactly `K * N * 2` bytes |
| Surfaces | input channel 5, output channel 4, no separate kernel resource |

`matmul_m64_k256_n1024_ty1` is the single exception, with a third 126-word copy task whose destination is the output surface. Apple lays the output surface out **below** the input and reserves a `__DATA/__bss` scratch under both; the closed form for those addresses reproduces every recorded resource address, text address, and constant address for all 35 two-task cases. Task 0 does not use the scratch as a DMA destination: its tile-DMA destination word is `0x000000c0`, while every task that writes the output surface sets bit 26 (`0x040000c1`). **Evidence: high over the decoded grid.** See the matvec section of [h13-td-fields.md](../../research/h13-td-fields.md).

### Weight permutation

The constant section is a pure permutation of the `K * N` fp16 halfwords of the `[N, K]` weight. With `g = min(16, N / 16)` interleaved rows per plane, `planes = N / g`, `p = n / g`, and `q = (p % 16) * (planes / 16) + p / 16`:

```text
destination_halfword = q * (K * g) + k * g + (n % g)
```

502 of 505 known-pattern probes reproduce Apple's whole-section SHA-256 bit for bit, over the shipped grid and the wider set (K, N) ∈ {16, 32, 64, 128, 256, 512, 1024}². The three mismatches are K ∈ {8, 16}, where the section gains a 64-byte row stride and its size is no longer `K * N * 2`; those shapes are outside the accepted envelope. The permutation was read out with zero, index, one-hot, and row/column-stain payloads, not guessed, because a uniform `fp16 0.5` weight cannot distinguish one permutation from another. The 16-way outer grouping in `q` is what makes each of the 16 kernel-DMA chunks hold every sixteenth row group. **Evidence: high.** See the matvec section of [h13-td-fields.md](../../research/h13-td-fields.md).

H14 uses the same permutation unchanged: all 125 known-weight `h14mv_*` probes decoded, `--verify` rebuilds every recorded section hash and nonzero count, and every uniform-weight H14 section has the same SHA-256 as its H13 twin. Two H14-specific details: the section length is `max(1024, K * N * 2)` with `stride = section_halfwords / planes`, and the harness compiler packs the weight file from byte 0, so the 128-byte blob header is part of the packed stream and the last 64 payload halfwords are dropped. The `h14mv_zero_*` probes isolate that header effect. **Evidence: high for the probe corpus.** See the appendix of [h14-td-fields.md](../../research/h14-td-fields.md).

## Softmax, layer_norm, and reduction programs

219 decoded H13 records (105 normalization, 114 reduction) collapse to 186 templates keyed on `(operation, input CHW, reduced-axis mask, keep_dims)`, with 177 distinct task streams. Task count depends on the operation and on which axis reduces, not on the extent:

| Operation | Reduced axis | Tasks | Constant section |
|---|---|---|---|
| `softmax` | C | 4 (W = 1, C = 1) or 5 | 256 / 1152 / 2176 bytes |
| `softmax` | H | 5 | 640 / 1152 / 2176 bytes |
| `softmax` | W (last axis) | 5, 6 (H > 1), 8 (S = 256) | 128 / 1024 / 1536 / 2048 bytes |
| `layer_norm` | C | 5, 6 (H > 1) | 16,384 zero bytes |
| `layer_norm` | W, HW, CHW | 3, 5 (H > 1) | 16,384 zero bytes |
| `reduce_sum`, `reduce_mean` | C | 1 (H = W = 1) or 2 | 16,384 zero bytes |
| `reduce_max` | C | 1 | 16,384 zero bytes |
| all three reductions | H | 3 | 16,384 zero bytes |
| all three reductions | W, HW, CHW | 1 | 16,384 zero bytes |

Every stream starts with a 126-word first task, so the program declares `task_words_minus_one = 125` in all 186 templates. Surfaces follow the elementwise rule `row = max(64, W * 2)`, `plane = row * H`, `bytes = plane * C`, which reproduces the shape, strides, and total bytes of all 438 decoded input and output surfaces — including rank-reduced results, where `keep_dims = false` over the spatial axes of `[1, 64, 8, 8]` gives a dense `[1, 1, 1, 64]` surface with a 128-byte row stride rather than the channel-major layout `keep_dims = true` uses. That difference is why `keep_dims` is part of the template key. **Evidence: high over the decoded sweep.** See the normalization section of [h13-td-fields.md](../../research/h13-td-fields.md).

### Constant sections and lookup tables

`layer_norm` and all three reductions carry a 16,384-byte all-zero section: the decoded SHA-256 equals `sha256(bytes(16384))` for all 143 such records. `softmax` carries fp16 lookup tables under an exact rule for all 69 decoded records — the 128-byte exponential table at offset 0, the 128-byte reciprocal table in the final 128 bytes when the reduced axis is C or H, and zeros elsewhere. A last-axis softmax carries no reciprocal table. The offsets were found by brute-forcing every 2-byte-aligned position for both blocks, which returns `(0, size - 128)` for all 21 channel-axis and height-axis cases. **Evidence: high over the decoded sweep.** See the normalization section of [h13-td-fields.md](../../research/h13-td-fields.md).

The seven nonlinear unary tables (`sigmoid`, `tanh`, `gelu`, `silu`, `exp`, `sqrt`, `rsqrt`) are target-independent 128-byte blocks. H14 stores exactly that block; H13 stores it followed by zeros to 2 KiB. The tables do not vary with shape or rank: `[1,64,8,8]`, `[1,128,16,16]`, `[1,256,32,32]`, `[1,768,16,16]`, and `[1,3072,1,1]` all produce the same `gelu` table, and `gelu` with `mode="TANH_APPROXIMATION"` is byte-identical to `mode="EXACT"`, while `SIGMOID_APPROXIMATION` is a different table. On H13 a multi-head softmax section scales at 128 bytes per head while H14 keeps 128 bytes regardless, the same padding split as the unary tables. **Evidence: high for the recorded hashes.** See [oracle-diff.md](../../research/oracle-diff.md) and sections 4 and 5 of [oracle-envelope.md](../../research/oracle-envelope.md).

## Apple emits one program per object

Across the 542 accepted envelope objects — 271 per target, holding 1,790 H13 tasks and 1,736 H14 tasks — every object carries exactly one program load command. The minting driver parses multi-program objects: it collects every program descriptor, gives each its own text region, splits each region separately, and records `program_count` with a per-program task section. None appeared, including at 129 tasks and a 134,217,728-byte constant section. Apple's partitioning unit is the task, not the object, and for a rank-2 constant-weight matmul at M = 256 or 512 the count follows `tasks = 1 + K * N / 2^19` across all 18 measured points. **Evidence: high over the accepted corpus.** See the partitioning section of [oracle-envelope.md](../../research/oracle-envelope.md).

The two targets accept and refuse exactly the same 274 envelope cases and differ only in how much they split: 41 of 271 decoded pairs have different task counts, and H14 uses fewer tasks in every one. A backend must therefore not assume equal task counts across generations; `matmul_m64_k256_n1024_ty1` is three H13 tasks and two H14 tasks. **Evidence: high.** See [oracle-envelope.md](../../research/oracle-envelope.md) and the task-stream delta in [h14-td-fields.md](../../research/h14-td-fields.md).

## Accepted geometry

| Family | What was accepted | Evidence |
|---|---|---|
| Elementwise | Identical runtime shapes cost one task; **any** shape difference costs two, whether per-channel, spatial, scalar-tensor, or batch broadcast. Channel counts 64, 96, 200, 300, 512, 768, 1024, 3072, 8192, 16384 — odd counts behave like powers of two — with batches 1, 2, 8 and spatial 1×1, 8×8, 16×16 | [oracle-envelope.md](../../research/oracle-envelope.md) §1, `env_bcast_add_1x768x16x16_runtime_1x1x16x16`, `env_bcast_add_1x200x1x1_runtime_1x200x1x1` |
| Constant-weight matmul | `[M,K] x const[N,K]`, `transpose_y=true`, over M ∈ {1, 16, 32, 128, 256, 512} and K, N ∈ {2048, 4096, 8192}, minus the refusal below. Largest accepted single program: M=512, K=N=8192, 129 tasks and a 134 MiB constant section | [oracle-envelope.md](../../research/oracle-envelope.md) §2, `env_mm_r2rb_m512_k8192_n8192_tx0_ty1` |
| Runtime-runtime matmul | All four `transpose_x`/`transpose_y` combinations, rank 2 and rank 3, including attention score and context shapes to S=512, always with a 16 KiB all-zero constant section. `transpose_y=false` costs one task **fewer** than `transpose_y=true` | [oracle-envelope.md](../../research/oracle-envelope.md) §2, `env_mm_r2rr_m128_k2048_n2048_tx1_ty1`, `env_mm_r3rr_m512_k64_n512_tx0_ty1_b1` |
| Convolution | Kernel 1 and 3, stride 1 and 2, groups 1, 4, 64, with and without a BLOBFILE bias, `pad_type="same"` — always one task in one program | [oracle-envelope.md](../../research/oracle-envelope.md) §3, `env_conv_k1_c256_n256_s8_st1_g1_bias1` |
| Chains | `x + relu(x)` folds to one task; `gelu`/`silu` residuals cost two because they need their table; `matmul → gelu → matmul` is two compute tasks at every transformer width with the activation as a post-operation field; attention stays one program at 9 to 15 tasks; whole pre-norm blocks and stacks to 512 blocks stay one program | [oracle-envelope.md](../../research/oracle-envelope.md) §5, `env_chain_residual_relu_1x768x1x1`, `env_chain_attention_s256_d64`; task-count laws and fusion words in [fusion-rules.md](../../research/fusion-rules.md), `chain_ffn_d256_s64`, `chain_attn_s512_dh64`, `chain_stack8_d256_s64_proj1` |

Convolution constant sections were first measured, not explained; the convolution probes have since recovered the packing. Dense (`groups = 1`) sections are sixteen plane groups of `lanes` interleaved output channels over the reduction, the group padded to 64 bytes; grouped sections are 64 planes of `Cout / 64` channels, each padded to 64 bytes; depthwise sections are 16 lanes of `Cout / 16` channels' taps back to back; a bias adds one row of `lanes` halfwords ahead of a plane's weight rows, which is why a bias is a fixed 1,024 or 2,048 bytes rather than `outputs * 2`; and stride 2 switches to a zero-skipping form whose per-row mask bytes make the section size depend on which weights are zero. **Evidence: high; 284 decoded convolutions per target reproduce byte-for-byte through the parity tests.** See the conv section of [h13-td-fields.md](../../research/h13-td-fields.md) and the conv appendix of [h14-td-fields.md](../../research/h14-td-fields.md); the envelope measurement is section 3 of [oracle-envelope.md](../../research/oracle-envelope.md).

## Refusals

| Refused form | Scope | Evidence |
|---|---|---|
| Inline fp16 tensor constants as a binary operand | All 12 cases per target | [oracle-diff.md](../../research/oracle-diff.md), `binary_add_c64_constant_inline` |
| Affine `layer_norm` (`gamma`, `beta`) | All 17 attempted forms, in seven variants including inline rank-1, rank-3, gamma-only, rank-2 input, and BLOBFILE-backed parameters; a BLOBFILE `mul` compiled in the same harness, so the blob path itself works | [h13-td-fields.md](../../research/h13-td-fields.md), `norm_layer_norm_ax1_1x2048x1x1_affine` |
| `transpose_y=false` with a BLOBFILE weight | Reports `ANE internal validation error: Metadata data type does not match requested type.` The transpose itself is fine: with both operands runtime, all four combinations are accepted. The refusal does not extend into chains: the same form is accepted as the first matmul of a feed-forward at three geometries | [oracle-diff.md](../../research/oracle-diff.md), [oracle-envelope.md](../../research/oracle-envelope.md), [fusion-rules.md](../../research/fusion-rules.md), `chain_ceiling_mlp_m128_k2048_n2048_ty0` |
| `[128,8192] x const[N,8192]`, `transpose_y=true` | The envelope campaign's only refusal, three cases per target, `callback_status=1` with no MIL validation report. Not a size ceiling: the same K and N are accepted at M = 1, 16, 32, 256, 512, and M=512 K=N=8192 moves four times the data and compiles | [oracle-envelope.md](../../research/oracle-envelope.md) §6, `env_mm_r2rb_m128_k8192_n2048_tx0_ty1` |
| BLOBFILE `real_div` | Both cases per target | [oracle-diff.md](../../research/oracle-diff.md) |

Two conclusions from the narrow first campaign were **wrong** and the envelope campaign corrected them: `transpose_y=false` is not refused as such, and convolution bias is not refused. Both original refusals belonged to the storage of the constant operand, not to the attribute under test. A refusal record is evidence of the attempted MIL and the compiler's response, not a decoded oracle, and a refusal observed at one sampled point is not a general rule. **Evidence: high as a method statement.** See [oracle-diff.md](../../research/oracle-diff.md) and [oracle-envelope.md](../../research/oracle-envelope.md).

## Shape and stride fields

Logical shape is the tensor shape seen by the operation. Physical shape includes row padding, planes, batches, and allocation size required by transfer hardware. Each row is padded to 64 bytes: an elementwise descriptor records strides `[N-stride, C*64 or H*64, 64, 2]` and a total of `C * H * 64`, which is 12,800 bytes for `[1,200,1,1]` and 1,048,576 bytes for `[1,16384,1,1]`. This is why odd channel counts cost nothing. **Evidence: medium for the decoded field meanings.** See section 1 of [oracle-envelope.md](../../research/oracle-envelope.md) and the local [H13 field ledger](../../research/h13-hwx-fields.md).

One trap for a backend: with a batch above 1 the descriptor's total size still covers a single batch element and equals the batch stride. `[8,64,8,8]` records strides `[32768,512,64,2]` and total 32,768 bytes, not 262,144. **Evidence: high for the decoded descriptors.** See section 1 of [oracle-envelope.md](../../research/oracle-envelope.md).

Do not infer logical dimensions by dividing allocation size by element size. Padding and tiling break that equivalence. **Evidence: high as a consequence of the distinct logical and physical fields.**

## What byte equality proves

Byte equality with an oracle is a statement about one input program, target, and compiler version. It is not device execution, and it does not mean every field is understood. The method and its boundary are stated in [parity-method.md](parity-method.md).

## Open questions

- **Open question:** What does the extended header word (`0x0` or `0x7`) select, and why do only `0x23` on H13 and `0x50003` on H14 carry it?
- **Open question:** Which H13 task-header words encode task identity, generation, dependency, and scheduling state? The low bits of `header[1]` remain unresolved in every family.
- **Open question:** What do the 54 unresolved Kernel DMA words on H14 and the 467 unresolved normalization word slots on H13 mean?
- **Open question:** What partitions the weights of the 57 grid points per target whose convolutions emit 2 to 32 tasks, and what closes the two-byte gaps in the stride-2 body-count model? Stride-2 grouped and depthwise sections (68 grid points per target) have no byte-level probe. See the coverage-and-limits table in the conv section of [h13-td-fields.md](../../research/h13-td-fields.md).
- **Open question:** Why does H13 split chains into more tasks than H14 — two extra attention tasks per 128 of sequence, three per batched head, and two per 256 of width in a projected block? See section 9 of [fusion-rules.md](../../research/fusion-rules.md).
- **Open question:** Why is M=128 with K=8192 refused when both larger and smaller row counts at the same K and N compile?
- **Open question:** Which block fields are firmware-facing contracts and which are direct hardware register images?
- **Open question:** Which descriptor fields may vary without changing observable computation?
