# H14 task-descriptor fields and H13→H14 emitter delta

This file is generated from the checked-in JSON. Edit `research/h14-oracle-analysis.py`, not this file.

Regenerate and compare it with:

```sh
python3 research/h14-oracle-analysis.py > /tmp/h14-td-fields.md
cmp research/h14-td-fields.md /tmp/h14-td-fields.md
```

## Scope and evidence

The analysis ran against repository commit `72a03870db43e03c91e362bcdbd6e7b438fef3db`. All 470 oracle records name that source commit; 172 H13 and 172 H14 records decoded, and 63 per target record Apple rejection. The external names come from freedomtan's `h14_register_map.md` at `ce54664e787976b646c450ceabed1731b506a4cd`. The tables cover all 265 decoded H14 tasks. An absent write is distinct from a zero write.

| Family | Decoded H14 cases |
|---|---|
| binary_constant | 22 |
| binary_runtime | 70 |
| chain | 2 |
| convolution | 7 |
| matmul | 36 |
| normalization | 4 |
| reduction | 6 |
| unary | 25 |

## Block coverage and resolution

A word is **evidence-resolved** when it is invariant, never written, or has a verified one-parameter formula for at least one named family below. A word is **unresolved** when two or more values occur and no sampled family explains them. This is an emitter-readiness count, not a claim that an invariant or partly resolved word has a complete hardware meaning.

| Block | H14 old base | Words | Resolved | Unresolved |
|---|---|---|---|---|
| Common | `0x0000` | 19 | 12 | 7 |
| L2 | `0x0500` | 25 | 10 | 15 |
| PE | `0x0900` | 5 | 3 | 2 |
| NE | `0x0d00` | 5 | 3 | 2 |
| TileDMA source | `0x1100` | 53 | 47 | 6 |
| TileDMA destination | `0x1500` | 10 | 7 | 3 |
| KernelDMA | `0x1900` | 70 | 16 | 54 |

## Map predictions tested against the oracles

| Prediction | Result | Oracle evidence |
|---|---|---|
| Nine-word header / DTID at h8 | Refuted for emitted H14 tasks. Each decoded task has eight header words; the word at index 8 is the first dense/scatter record header, not DTID. | `binary_add_1x64x1x1`, `matmul_m1_k256_n512_ty1`; all 265 decoded H14 tasks parse this way |
| Packed 15-bit W/H | Confirmed at Common `0x0000` and `0x0014`: `1→0x00010001`, `8→0x00080008`, `224→0x00e000e0`. | `binary_add_1x64x1x1`, `binary_add_1x64x8x8`, `binary_add_1x3x224x224` |
| Non-Common `+0x3c00` remap | Refuted as an H14 stream encoding rule. Every H14 record uses old bases `0x0500`..`0x1900`; no record uses `0x4100`..`0x5500`. The delta is only the map's old-to-modern presentation transform. | all 172 decoded H14 cases |
| TileDMA DataSetId bits 8:15 | Location is consistent but not experimentally confirmed: source-1/source-2/destination config words carry `0x0e` in bits 8:15 throughout this campaign; no MIL parameter controls the ID. | `binary_add_1x64x1x1`, `matmul_m1_k256_n512_ty1`, `softmax_1x64x8x8` |
| KernelDMA DataSetId bits 8:15 | Not confirmed: all written `CoeffDMAConfig0..15` words have zero in bits 8:15. Their bits 16:23 vary with task position/allocation instead. | `conv_k1_c1024_n1024_s1_bias0`, `matmul_m1_k256_n512_ty1`, `binary_pow_1x1024x1x1` |
| SparseKernelBlockSize | The detailed map's `0x19f8` prediction is refuted: that word is never written. The prose prediction at `0x1a00` is supported: the same pow case writes `0x00000000` in t1 and `0x00000080` in t2. The campaign does not establish the value formula. | `binary_pow_1x1024x1x1#t1`, `binary_pow_1x1024x1x1#t2` |
| Block extents | Common 19, L2 25, PE 5, NE 5, TileDMA source 53, TileDMA destination 10, and KernelDMA 70 match decoded writes. This makes `0x11d4` and `0x1528` outside the mapped stream blocks despite the map listing them; `0x1524` is written. | `binary_add_1x64x1x1` |

## Task-stream delta

| Property | H13 | H14 | Emitter delta |
|---|---|---|---|
| Header | 10 words | 8 words | Drop H13 next-pointer/linked-size conventions; emit H14 `TID | task_words<<16` in h0 and seven control words. |
| Task size | First: program `task_words_minus_one+1`; later: prior h1 bits 16:24 plus one | h0 bits 16:26 give exact words | Compute each H14 task independently. `binary_add_1x64x1x1` is 61 words; `matmul_m1_k256_n512_ty1` is 38 then 85. |
| Link/alignment | h7 is next section-relative byte offset; final h7=0 | No link; tasks are 16-byte aligned and zero-size 16-byte prefixes/padding are skipped | Replace linked traversal with aligned sequential tasks. |
| Register records | 2032 dense records; header `((count-1)<<26)|byte_address` | 1237 dense + 1188 scatter; base is a word index in bits 0:14 | Dense: count-minus-one bits 15:20. Scatter: bit31 plus a 16-bit following-word mask in bits 15:30. |
| Block addresses | Common `0x00000`; others `0x04800`..`0x1f800` | Common `0x0000`; others `0x0500`..`0x1900` | Change every non-Common base; do not apply the modern `+0x3c00` display remap. |
| Task decomposition | 1/2/3/5/6-task objects: 98/59/13/1/1 | 1/2/3/5/6-task objects: 98/60/12/1/1 | Do not assume identical task count: `matmul_m64_k256_n1024_ty1` is H13 3 tasks and H14 2. |

## Program and tensor descriptor delta

| Field | H13 | H14 | Evidence/result |
|---|---|---|---|
| Program command kind/size | kind 1, `0x880` bytes | kind 4, `0x890` bytes | All 172 pairs |
| Text address | resource-dependent | resource-dependent | Same in 135/172 pairs; H14 is lower in 37 because resources moved |
| Text→const offset | 128-byte alignment of linked-task extent | `align_up(text_words*4, 64)` | All 172 pairs; add C64 is `0x200` vs `0x140` |
| Task metadata | code 538, count, first size-minus-one | count and `text_words`; no H13 code/first-size fields | `binary_add_1x64x1x1`, `matmul_m1_k256_n512_ty1` |
| Resources | slots 0..2 carry allocation addresses as needed; slots 3..4 are zero | slots 0..3 are zero; slot 4 is `0x30000000` in every case | All 172 pairs |
| H14 32-word trailer | absent | text address, kind=4, text words, task count, constants/sentinels, function name `main`, plus unresolved words 18/20/28 | All H14 cases; compare `binary_add_1x64x1x1` and `matmul_m1_k256_n512_ty1` |
| Tensor descriptors | binding, element code, shape, strides, total bytes | byte-for-byte same decoded fields | All 172 paired cases; element code is 5 for campaign fp16 tensors |

## Complete H14 block tables

The map column reports freedomtan's name, even where observed behavior challenges that name. The H13 relation compares the same block-relative slot for the same MIL case and task index. `same`, `changed`, and one-sided counts are write comparisons, not semantic equivalence.

### Common (`0x0000`, 19 words)

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x0000` | `0x0000` | InDim (packed W+H) — W bits 0:14; H bits 16:30; pad bits 15,31 | `0x00010001`, `0x00010002`, `0x00010008`, `0x00010040`, `0x00010100`, `0x00010200`, `0x00010400`, `0x00020040`, `0x00020200`, `0x00080008`, `0x00100010`, `0x00400008`, `0x00e000e0`, `0x02000001` (265/265 tasks) | one-task elementwise/unary t0: `(H<<16)|W`; matvec t0: `(1<<16)|K`, t1: `(1<<16)|M` | `0x00000` | same 263; changed 2; H13-only 1 | `binary_add_1x1024x1x1#t0`, `binary_add_c64_constant_blob#t0`, `layer_norm_1x512x1x1#t0` |
| `0x0004` | `0x0004` | InDepth — input depth | `0x00000008` (2/265 tasks) | invariant/omitted | `0x00004` | same 2; H13-only 264 | `layer_norm_1x64x8x8#t0` |
| `0x0008` | `0x0008` | ChannelCfg — input/source-2/output formats | not written (0/265 tasks) | invariant/omitted | `0x00008` | H13-only 266 | all 172 decoded H14 cases |
| `0x000c` | `0x000c` | InChannels — input channels | `0x00000001`, `0x00000002`, `0x00000003`, `0x00000006`, `0x00000008`, `0x00000040`, `0x00000080`, `0x00000100`, `0x00000200`, `0x00000400`, `0x00000800`, `0x00001000`, `0x00002000` (265/265 tasks) | one-task elementwise/unary t0: `C`; matvec t0: `M`, t1: `K` | `0x0000c` | same 264; changed 1; H13-only 1 | `binary_add_c512_constant_blob#t0`, `binary_add_1x128x16x16#t0`, `binary_pow_1x4096x1x1#t2` |
| `0x0010` | `0x0010` | OutChannels — output channels | `0x00000001`, `0x00000002`, `0x00000003`, `0x00000008`, `0x00000040`, `0x00000080`, `0x00000100`, `0x00000200`, `0x00000400`, `0x00000800`, `0x00001000` (265/265 tasks) | one-task elementwise/unary t0: `C`; matvec t0: `M`, t1: `N` | `0x00010` | same 264; changed 1; H13-only 1 | `binary_add_c512_constant_blob#t0`, `binary_add_1x128x16x16#t0`, `binary_add_1x4096x1x1#t0` |
| `0x0014` | `0x0014` | OutDim (packed W+H) — W bits 0:14; H bits 16:30; pad bits 15,31 | `0x00010001`, `0x00010002`, `0x00010008`, `0x00010040`, `0x00010100`, `0x00010200`, `0x00010400`, `0x00020040`, `0x00020200`, `0x00080008`, `0x00100010`, `0x00400008`, `0x00e000e0`, `0x02000001` (265/265 tasks) | one-task elementwise/unary t0: `(H<<16)|W`; matvec t0: `(1<<16)|K`, t1: `(1<<16)|M` | `0x00014` | same 263; changed 2; H13-only 1 | `binary_add_1x1024x1x1#t0`, `binary_add_c64_constant_blob#t0`, `layer_norm_1x512x1x1#t0` |
| `0x0018` | `0x0018` | OutDepth | `0x00000008` (2/265 tasks) | invariant/omitted | `0x00018` | same 2; H13-only 264 | `layer_norm_1x64x8x8#t0` |
| `0x001c` | `0x001c` | OCGSize | `0x00000000`, `0x00000002`, `0x00000004`, `0x00000005` (154/265 tasks) | unresolved | `0x0001c` | changed 154; H13-only 112 | `binary_add_c512_constant_blob#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x0020` | `0x0020` | ConvCfg | `0x5042a0c3` (1/265 tasks) | invariant/omitted | `0x00020` | changed 1; H13-only 265 | `conv_k3_c64_n64_s8_bias0#t0` |
| `0x0024` | `0x0024` | ConvCfg3d | not written (0/265 tasks) | invariant/omitted | `0x00024` | H13-only 266 | all 172 decoded H14 cases |
| `0x0028` | `0x0028` | GroupConvCfg | `0x00010008`, `0x00014001`, `0x0002c001`, `0x00044001`, `0x00084001`, `0x00404001` (120/265 tasks) | unresolved | `0x00028` | changed 120; H13-only 146 | `reduce_mean_axes_1#t1`, `binary_real_div_1x64x1x1#t0`, `reduce_max_axes_1#t0` |
| `0x002c` | `0x002c` | TileCfg | `0x00000001`, `0x00000002`, `0x00000008`, `0x00000010`, `0x00000020`, `0x00000040`, `0x000000e0`, `0x00000200` (265/265 tasks) | square spatial sweep: `S` where `H=W=S` | `0x0002c` | changed 265; H13-only 1 | `binary_add_1x1024x1x1#t0`, `layer_norm_1x512x1x1#t0`, `layer_norm_1x512x1x1#t1` |
| `0x0030` | `0x0030` | TileOverlap | `0x00000004`, `0x00000006`, `0x00000008`, `0x00000034`, `0x00000044`, `0x00000054` (105/265 tasks) | unresolved | `0x00030` | changed 105; H13-only 161 | `binary_add_1x1024x1x1#t0`, `binary_add_1x64x8x8#t0`, `layer_norm_1x512x1x1#t0` |
| `0x0034` | `0x0034` | NECfg | `0x00000001`, `0x00000002`, `0x00000007`, `0x00000010`, `0x00000080`, `0x00000200`, `0x00000400`, `0x00000410` (29/265 tasks) | unresolved | `0x00034` | changed 29; H13-only 237 | `softmax_1x512x1x1#t4`, `layer_norm_1x512x1x1#t2`, `softmax_1x512x1x1#t1` |
| `0x0038` | `0x0038` | Cfg | `0x00000021`, `0x00000031`, `0x00000033`, `0x00000041`, `0x00000051`, `0x00000061`, `0x00010001`, `0x00040001`, `0x00120001`, `0x00200001`, `0x00200005`, `0x00211101`, `0x00230001`, `0x00231101`, `0x00233305`, `0x00240001`, `0x00241101`, `0x00244401`, `0x00244405`, `0x10200001`, `0x10200005`, `0x10230301`, `0x30211101`, `0x30220201`, `0x30222201`, `0x30242201` (257/265 tasks) | unresolved | `0x00038` | changed 257; H13-only 9 | `binary_add_c512_constant_scalar#t0`, `binary_pow_1x128x16x16#t2`, `reduce_mean_axes_1#t0` |
| `0x003c` | `0x003c` | TaskInfo | `0x00000000`, `0x00100000` (265/265 tasks) | unresolved | `0x0003c` | same 105; changed 160; H13-only 1 | `binary_add_1x1024x1x1#t0`, `binary_add_c512_constant_blob#t0` |
| `0x0040` | `0x0040` | DPE | `0x00000001`, `0x00000002` (12/265 tasks) | unresolved | — | new H14 slot | `unary_rsqrt_c512#t0`, `binary_pow_1x1024x1x1#t2` |
| `0x0044` | `0x0044` | Spare0 | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x0048` | `0x0048` | Spare1 | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |

### L2 (`0x0500`, 25 words)

Addresses/strides use 16-byte units in the freedomtan map.

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x0500` | `0x4100` | Control | `0x00000004` (4/265 tasks) | invariant/omitted | `0x04800` | same 4; H13-only 262 | `reduce_max_axes_1#t0` |
| `0x0504` | `0x4104` | Src1Cfg | `0x00000108`, `0x00000120`, `0x00000142`, `0x00000172`, `0x00400100`, `0x00400142`, `0x00500171`, `0x00500172`, `0x01500172` (265/265 tasks) | unresolved | `0x04804` | same 228; changed 37; H13-only 1 | `softmax_1x512x1x1#t4`, `layer_norm_1x512x1x1#t1`, `binary_add_1x1024x1x1#t0` |
| `0x0508` | `0x4108` | Src2Cfg | `0x00000040`, `0x00000080`, `0x00000100`, `0x00000200`, `0x00000400`, `0x00000800`, `0x00000840`, `0x00001000`, `0x00002000`, `0x00002040`, `0x00002100`, `0x00002400`, `0x00004000`, `0x00004100`, `0x00008000`, `0x00009000`, `0x00010000`, `0x00010200`, `0x00012000`, `0x00020000`, `0x00020300`, `0x00024000`, `0x00049800`, `0x00093000` (113/265 tasks) | unresolved | `0x04808` | same 112; changed 1; H13-only 153 | `layer_norm_1x512x1x1#t0`, `binary_pow_1x512x1x1#t0`, `binary_pow_1x3x224x224#t0` |
| `0x050c` | `0x410c` | Src1Base | `0x00000010`, `0x00000020`, `0x00000080`, `0x00000090`, `0x00000100`, `0x000001c0` (262/265 tasks) | unresolved | `0x0480c` | same 260; changed 2; H13-only 4 | `binary_add_1x1024x1x1#t0`, `matmul_m64_k1024_n1024_ty1#t1`, `binary_add_1x3x224x224#t0` |
| `0x0510` | `0x4110` | Src1ChannelStride | `0x00000020`, `0x00000080`, `0x00000100`, `0x00000200`, `0x00000300`, `0x00000400`, `0x00000420`, `0x00000430`, `0x00000450`, `0x00000540`, `0x00000800`, `0x00000820`, `0x00000a80`, `0x00001000`, `0x00001020`, `0x00001030`, `0x00002000`, `0x00002020`, `0x00002030`, `0x00004000`, `0x00004020`, `0x00004030`, `0x00008000`, `0x00008020`, `0x00010000`, `0x00010020` (212/265 tasks) | unresolved | `0x04810` | same 198; changed 14; H13-only 54 | `layer_norm_1x512x1x1#t0`, `binary_pow_1x256x1x1#t0`, `binary_add_1x4096x1x1#t0` |
| `0x0514` | `0x4114` | Src1RowStride | `0x00000010`, `0x00000080`, `0x00000100`, `0x00000200`, `0x00000300`, `0x00000400`, `0x00000540`, `0x00000800`, `0x00001000`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000` (206/265 tasks) | unresolved | `0x04814` | same 205; changed 1; H13-only 60 | `layer_norm_1x512x1x1#t0`, `binary_add_1x3x224x224#t0`, `binary_add_1x4096x1x1#t0` |
| `0x0518` | `0x4118` | Src1DepthStride | `0x00000010`, `0x00000080`, `0x00000100`, `0x00000200`, `0x00000300`, `0x00000400`, `0x00000420`, `0x00000540`, `0x00000800`, `0x00001000`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000` (210/265 tasks) | unresolved | `0x04818` | same 209; changed 1; H13-only 56 | `layer_norm_1x512x1x1#t0`, `binary_add_1x3x224x224#t0`, `binary_add_1x4096x1x1#t0` |
| `0x051c` | `0x411c` | Src1GroupStride | `0x00000040`, `0x00000440`, `0x00000840`, `0x00001040`, `0x00002040`, `0x00002100`, `0x00004040`, `0x00004200`, `0x00008040`, `0x00010040`, `0x00010200`, `0x00049840` (55/265 tasks) | unresolved | `0x0481c` | same 55; H13-only 211 | `layer_norm_1x512x1x1#t1`, `binary_add_1x1024x1x1#t0`, `binary_add_1x3x224x224#t0` |
| `0x0520` | `0x4120` | Src2Base | `0x00000010`, `0x00000020`, `0x000001c0` (58/265 tasks) | unresolved | `0x04820` | same 58; H13-only 208 | `binary_add_1x1024x1x1#t0`, `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0` |
| `0x0524` | `0x4124` | Src2ChannelStride | `0x00000020`, `0x00000080`, `0x00000400`, `0x00000420`, `0x00000540`, `0x00000820`, `0x00001020`, `0x00002020`, `0x00004020`, `0x00008020`, `0x00010020` (68/265 tasks) | unresolved | `0x04824` | same 68; H13-only 198 | `layer_norm_1x512x1x1#t1`, `binary_add_1x128x1x1#t0`, `binary_add_1x4096x1x1#t0` |
| `0x0528` | `0x4128` | Src2RowStride | `0x00000010`, `0x00000400`, `0x00000540`, `0x00000800`, `0x00001000`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000` (52/265 tasks) | unresolved | `0x04828` | same 52; H13-only 214 | `layer_norm_1x64x8x8#t1`, `binary_add_1x128x16x16#t0`, `binary_add_1x4096x1x1#t0` |
| `0x052c` | `0x412c` | Src2DepthStride | `0x00000400`, `0x00000540`, `0x00000800`, `0x00001000`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000` (51/265 tasks) | unresolved | `0x0482c` | same 51; H13-only 215 | `binary_add_1x64x1x1#t0`, `binary_add_1x512x1x1#t0`, `binary_add_1x4096x1x1#t0` |
| `0x0530` | `0x4130` | Src2GroupStride | `0x00000140`, `0x00000141`, `0x0000014a`, `0x00500130`, `0x00500170`, `0x0050017a` (258/265 tasks) | unresolved | `0x04830` | same 254; changed 4; H13-only 8 | `binary_pow_1x1024x1x1#t1`, `layer_norm_1x512x1x1#t0`, `binary_add_1x1024x1x1#t0` |
| `0x0534` | `0x4134` | ResultCfg | `0x00000010`, `0x00000020`, `0x000001c0`, `0x00000400`, `0x00000420`, `0x000004a0`, `0x00000520`, `0x00000800`, `0x00000820`, `0x00000860`, `0x00000920`, `0x00001000`, `0x00001020`, `0x00001060`, `0x00001220`, `0x00002000`, `0x00002020`, `0x00002040`, `0x00002060`, `0x00002100`, `0x00002140`, `0x00002180`, `0x00002420`, `0x00002860`, `0x00004000`, `0x00004020`, `0x00004030`, `0x00004040`, `0x00004060`, `0x00004100`, `0x00004200`, `0x00004820`, `0x00008000`, `0x00008020`, `0x00008060`, `0x00009000`, `0x00009020`, `0x00010000`, `0x00010020`, `0x00010060`, `0x00012000`, `0x00012020`, `0x00020000`, `0x00020060`, `0x00020300`, `0x00020400`, `0x00024000`, `0x00093000`, `0x00093040` (189/265 tasks) | unresolved | `0x04834` | same 182; changed 7; H13-only 77 | `binary_pow_1x1024x1x1#t1`, `binary_pow_1x512x1x1#t2`, `binary_add_1x3x224x224#t0` |
| `0x0538` | `0x4138` | ResultBase | `0x00000010`, `0x00000020`, `0x00000040`, `0x00000080`, `0x00000090`, `0x000001c0`, `0x00000380` (71/265 tasks) | unresolved | `0x04838` | same 70; changed 1; H13-only 195 | `binary_real_div_1x64x8x8#t0`, `reduce_mean_axes_1#t0`, `binary_pow_1x3x224x224#t0` |
| `0x053c` | `0x413c` | ResultChannelStride | `0x00000080`, `0x00000420`, `0x00000540`, `0x00000820`, `0x00000a80`, `0x00001020`, `0x00002030` (24/265 tasks) | unresolved | `0x0483c` | same 24; H13-only 242 | `binary_add_c64_constant_blob#t0`, `binary_pow_1x64x8x8#t0`, `binary_pow_1x128x16x16#t0` |
| `0x0540` | `0x4140` | ResultRowStride | `0x00000010` (2/265 tasks) | invariant/omitted | `0x04840` | same 2; H13-only 264 | `reduce_mean_axes_1#t1` |
| `0x0544` | `0x4144` | ResultDepthStride | `0x00000400` (2/265 tasks) | invariant/omitted | `0x04844` | same 2; H13-only 264 | `softmax_1x64x8x8#t1` |
| `0x0548` | `0x4148` | ResultGroupStride | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x054c` | `0x414c` | SrcAndResultWrapCfg | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x0550` | `0x4150` | Src1WrapStart | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x0554` | `0x4154` | Src2WrapStart | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x0558` | `0x4158` | L2Reserved | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x055c` | `0x415c` | ResultWrapIndex | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x0560` | `0x4160` | ResultWrapStartOffset | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |

### PE (`0x0900`, 5 words)

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x0900` | `0x4500` | PEConfig — pool mode, operation, nonlinear mode | `0x00000002`, `0x00000040`, `0x00001000`, `0x00080000`, `0x00080004`, `0x00080008`, `0x0008000c`, `0x00080020`, `0x00092010`, `0x000c0000` (93/265 tasks) | binary op: add `0x80000`, mul `0x80004`, max `0x80008`, min `0x8000c`, sub `0xc0000` | `0x08800` | same 93; H13-only 173 | `reduce_max_axes_2_3#t0`, `binary_maximum_1x1024x1x1#t0`, `binary_sub_1x1024x1x1#t0` |
| `0x0904` | `0x4504` | BiasScale | `0x38000000`, `0x3c003800`, `0x3c00b800`, `0x3dc50000` (8/265 tasks) | unresolved | `0x08804` | same 8; H13-only 258 | `binary_mul_c512_constant_scalar#t0`, `binary_sub_c512_constant_scalar#t0`, `softmax_1x512x1x1#t1` |
| `0x0908` | `0x4508` | PreScale | `0x3c0000a8` (2/265 tasks) | invariant/omitted | `0x08808` | same 2; H13-only 264 | `layer_norm_1x512x1x1#t1` |
| `0x090c` | `0x450c` | FinalScale | `0x39800000`, `0x3b000000`, `0x3c800000` (6/265 tasks) | unresolved | `0x0880c` | same 6; H13-only 260 | `layer_norm_1x64x8x8#t0`, `layer_norm_1x512x1x1#t0`, `reduce_mean_axes_1#t1` |
| `0x0910` | `0x4510` | Quant | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |

### NE (`0x0d00`, 5 words)

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x0d00` | `0x4900` | KernelCfg | `0x00010082` (52/265 tasks) | invariant/omitted | `0x0c800` | changed 52; H13-only 214 | `binary_maximum_c512_constant_scalar#t0` |
| `0x0d04` | `0x4904` | MacCfg | `0x00101c00`, `0x00101c0b`, `0x00101c0c`, `0x00111c00`, `0x00111c0c`, `0x00121c08`, `0x00121c0a`, `0x00121c0c` (160/265 tasks) | unresolved | `0x0c804` | same 160; H13-only 106 | `conv_k1_c1024_n1024_s1_bias0#t0`, `unary_relu_c512#t0`, `binary_maximum_c512_constant_scalar#t0` |
| `0x0d08` | `0x4908` | MatrixVectorBias | not written (0/265 tasks) | invariant/omitted | `0x0c808` | H13-only 266 | all 172 decoded H14 cases |
| `0x0d0c` | `0x490c` | AccBias | `0x00000011` (2/265 tasks) | invariant/omitted | `0x0c80c` | same 2; H13-only 264 | `unary_rsqrt_c512#t0` |
| `0x0d10` | `0x4910` | PostScale | `0x00003c00`, `0x00003dc5` (160/265 tasks) | unresolved | `0x0c810` | same 160; H13-only 106 | `binary_add_c512_constant_blob#t0`, `unary_exp_c512#t0` |

### TileDMA source (`0x1100`, 53 words)

Base/stride fields use 64-byte units; format and pixel-offset fields are packed.

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x1100` | `0x4d00` | Src1DMAConfig — enable/cache/dependency; DataSetId predicted at bits 8:15 | `0x00000e21`, `0x00000ee1`, `0x00010e21`, `0x00020ec1` (206/265 tasks) | unresolved | `0x13800` | changed 206; H13-only 60 | `binary_add_1x1024x1x1#t0`, `binary_add_c512_constant_blob#t1`, `layer_norm_1x512x1x1#t2` |
| `0x1104` | `0x4d04` | Src2DMAConfig — enable/cache/dependency; DataSetId predicted at bits 8:15 | `0x00000e20` (51/265 tasks) | invariant/omitted | `0x13804` | changed 51; H13-only 215 | `binary_add_1x1024x1x1#t0` |
| `0x1108` | `0x4d08` | Src1DMAConfigExt | `0x000000ce`, `0x000000ee` (206/265 tasks) | unresolved | `0x13808` | changed 206; H13-only 60 | `binary_add_1x1024x1x1#t0`, `layer_norm_1x512x1x1#t0` |
| `0x110c` | `0x4d0c` | Src2DMAConfigExt | `0x000000ce` (51/265 tasks) | invariant/omitted | `0x1380c` | changed 51; H13-only 215 | `binary_add_1x1024x1x1#t0` |
| `0x1110` | `0x4d10` | Src1BaseAddrLo | `0x00000000` (206/265 tasks) | invariant/omitted | `0x13810` | changed 206; H13-only 60 | `binary_add_1x1024x1x1#t0` |
| `0x1114` | `0x4d14` | Src1BaseAddrHi | `0x00000000` (206/265 tasks) | invariant/omitted | `0x13814` | changed 206; H13-only 60 | `binary_add_1x1024x1x1#t0` |
| `0x1118` | `0x4d18` | Src1RowStride | `0x00000040`, `0x00000080`, `0x000001c0`, `0x00000200`, `0x00000400` (47/265 tasks) | unresolved | `0x13818` | changed 47; H13-only 219 | `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0`, `binary_add_c512_constant_blob#t0` |
| `0x111c` | `0x4d1c` | Src1PlaneStride | `0x00000040`, `0x00000200`, `0x00000400`, `0x00000800`, `0x00018800` (183/265 tasks) | unresolved | `0x1381c` | changed 183; H13-only 83 | `binary_add_1x1024x1x1#t0`, `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0` |
| `0x1120` | `0x4d20` | Src1DepthStride | `0x00000040` (1/265 tasks) | invariant/omitted | `0x13820` | changed 1; H13-only 265 | `layer_norm_1x64x8x8#t0` |
| `0x1124` | `0x4d24` | Src1GroupStride | not written (0/265 tasks) | invariant/omitted | `0x13824` | H13-only 266 | all 172 decoded H14 cases |
| `0x1128` | `0x4d28` | Src2BaseAddrLo | `0x00000000` (51/265 tasks) | invariant/omitted | `0x13828` | changed 51; H13-only 215 | `binary_add_1x1024x1x1#t0` |
| `0x112c` | `0x4d2c` | Src2BaseAddrHi | `0x00000000` (51/265 tasks) | invariant/omitted | `0x1382c` | changed 51; H13-only 215 | `binary_add_1x1024x1x1#t0` |
| `0x1130` | `0x4d30` | Src2RowStride | `0x00000040`, `0x000001c0` (15/265 tasks) | unresolved | `0x13830` | changed 15; H13-only 251 | `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0` |
| `0x1134` | `0x4d34` | Src2PlaneStride | `0x00000040`, `0x00000200`, `0x00000400`, `0x00018800` (51/265 tasks) | unresolved | `0x13834` | changed 51; H13-only 215 | `binary_add_1x1024x1x1#t0`, `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0` |
| `0x1138` | `0x4d38` | Src2DepthStride | not written (0/265 tasks) | invariant/omitted | `0x13838` | H13-only 266 | all 172 decoded H14 cases |
| `0x113c` | `0x4d3c` | Src2GroupStride | not written (0/265 tasks) | invariant/omitted | `0x1383c` | H13-only 266 | all 172 decoded H14 cases |
| `0x1140` | `0x4d40` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x13840` | H13-only 266 | all 172 decoded H14 cases |
| `0x1144` | `0x4d44` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x13844` | H13-only 266 | all 172 decoded H14 cases |
| `0x1148` | `0x4d48` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x13848` | H13-only 266 | all 172 decoded H14 cases |
| `0x114c` | `0x4d4c` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x1384c` | H13-only 266 | all 172 decoded H14 cases |
| `0x1150` | `0x4d50` | Src1Fmt | `0x01002031` (206/265 tasks) | invariant/omitted | `0x13850` | changed 206; H13-only 60 | `binary_add_1x1024x1x1#t0` |
| `0x1154` | `0x4d54` | Src2Fmt | `0x01002031` (51/265 tasks) | invariant/omitted | `0x13854` | changed 51; H13-only 215 | `binary_add_1x1024x1x1#t0` |
| `0x1158` | `0x4d58` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x13858` | H13-only 266 | all 172 decoded H14 cases |
| `0x115c` | `0x4d5c` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x1385c` | H13-only 266 | all 172 decoded H14 cases |
| `0x1160` | `0x4d60` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x13860` | H13-only 266 | all 172 decoded H14 cases |
| `0x1164` | `0x4d64` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | `0x13864` | H13-only 266 | all 172 decoded H14 cases |
| `0x1168` | `0x4d68` | PixelOffset[0] | not written (0/265 tasks) | invariant/omitted | `0x13868` | H13-only 266 | all 172 decoded H14 cases |
| `0x116c` | `0x4d6c` | PixelOffset[1] | not written (0/265 tasks) | invariant/omitted | `0x1386c` | H13-only 266 | all 172 decoded H14 cases |
| `0x1170` | `0x4d70` | PixelOffset[2] | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1174` | `0x4d74` | PixelOffset[3] | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1178` | `0x4d78` | TileDmaSrcReserved | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x117c` | `0x4d7c` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1180` | `0x4d80` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1184` | `0x4d84` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1188` | `0x4d88` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x118c` | `0x4d8c` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1190` | `0x4d90` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1194` | `0x4d94` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1198` | `0x4d98` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x119c` | `0x4d9c` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11a0` | `0x4da0` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11a4` | `0x4da4` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11a8` | `0x4da8` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11ac` | `0x4dac` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11b0` | `0x4db0` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11b4` | `0x4db4` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11b8` | `0x4db8` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11bc` | `0x4dbc` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11c0` | `0x4dc0` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11c4` | `0x4dc4` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11c8` | `0x4dc8` | — | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11cc` | `0x4dcc` | Spare0 | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x11d0` | `0x4dd0` | Spare1 | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |

### TileDMA destination (`0x1500`, 10 words)

Base/stride fields use 64-byte units; DstFmt is packed.

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x1500` | `0x5100` | DstDMAConfig — enable/cache/L2 mode; DataSetId predicted at bits 8:15 | `0x00010e31`, `0x01000e31`, `0x01010e31`, `0x01020e31`, `0x01040e31`, `0x01050e31` (172/265 tasks) | unresolved | `0x17800` | changed 172; H13-only 94 | `reduce_mean_axes_1#t1`, `binary_pow_1x1024x1x1#t2`, `softmax_1x64x8x8#t5` |
| `0x1504` | `0x5104` | DstReserved | not written (0/265 tasks) | invariant/omitted | `0x17804` | H13-only 266 | all 172 decoded H14 cases |
| `0x1508` | `0x5108` | DstBaseAddrLo | `0x00000000` (172/265 tasks) | invariant/omitted | `0x17808` | same 1; changed 171; H13-only 94 | `binary_add_1x1024x1x1#t0` |
| `0x150c` | `0x510c` | DstBaseAddrHi | `0x00000000` (172/265 tasks) | invariant/omitted | `0x1780c` | same 1; changed 171; H13-only 94 | `binary_add_1x1024x1x1#t0` |
| `0x1510` | `0x5110` | DstRowStride | `0x00000040`, `0x000001c0` (26/265 tasks) | unresolved | `0x17810` | changed 26; H13-only 240 | `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0` |
| `0x1514` | `0x5114` | DstPlaneStride | `0x00000040`, `0x00000200`, `0x00000400`, `0x00000800`, `0x00018800` (160/265 tasks) | unresolved | `0x17814` | changed 160; H13-only 106 | `binary_add_1x1024x1x1#t0`, `binary_add_1x128x16x16#t0`, `binary_add_1x3x224x224#t0` |
| `0x1518` | `0x5118` | DstDepthStride | not written (0/265 tasks) | invariant/omitted | `0x17818` | H13-only 266 | all 172 decoded H14 cases |
| `0x151c` | `0x511c` | DstGroupStride | `0x00000040` (2/265 tasks) | invariant/omitted | — | new H14 slot | `reduce_mean_axes_1#t1` |
| `0x1520` | `0x5120` | DstFmt | `0x01302031` (172/265 tasks) | invariant/omitted | — | new H14 slot | `binary_add_1x1024x1x1#t0` |
| `0x1524` | `0x5124` | Spare0 | `0x0000dead` (66/265 tasks) | invariant/omitted | — | new H14 slot | `binary_add_1x1024x1x1#t0` |

### KernelDMA (`0x1900`, 70 words)

Config words contain enable/cache/DataSetId/UserTag; base and size fields use 64-byte units.

| H14 old | Modern | Map name/meaning | Observed values | Formula/status | H13 address | Paired value relation | Oracle evidence |
|---|---|---|---|---|---|---|---|
| `0x1900` | `0x5500` | MasterConfig | `0x00000080`, `0x00010240`, `0x000102c0`, `0x00020240`, `0x000202c0`, `0x00030240`, `0x000302c0`, `0x00040240` (194/265 tasks) | unresolved | `0x1f800` | changed 194; H13-only 72 | `binary_add_1x1024x1x1#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1904` | `0x5504` | AlignedCoeffSizePerCh | not written (0/265 tasks) | invariant/omitted | `0x1f804` | H13-only 266 | all 172 decoded H14 cases |
| `0x1908` | `0x5508` | Prefetch | `0x00000000` (44/265 tasks) | invariant/omitted | `0x1f808` | changed 44; H13-only 222 | `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x190c` | `0x550c` | Reserved0 | `0x00000000` (44/265 tasks) | invariant/omitted | `0x1f80c` | changed 44; H13-only 222 | `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1910` | `0x5510` | Reserved1 | not written (0/265 tasks) | invariant/omitted | `0x1f810` | H13-only 266 | all 172 decoded H14 cases |
| `0x1914` | `0x5514` | Reserved2 | not written (0/265 tasks) | invariant/omitted | `0x1f814` | H13-only 266 | all 172 decoded H14 cases |
| `0x1918` | `0x5518` | CoeffDMAConfig0 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f818` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x191c` | `0x551c` | CoeffDMAConfig1 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f81c` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1920` | `0x5520` | CoeffDMAConfig2 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f820` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1924` | `0x5524` | CoeffDMAConfig3 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f824` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1928` | `0x5528` | CoeffDMAConfig4 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f828` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x192c` | `0x552c` | CoeffDMAConfig5 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f82c` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1930` | `0x5530` | CoeffDMAConfig6 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f830` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1934` | `0x5534` | CoeffDMAConfig7 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f834` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1938` | `0x5538` | CoeffDMAConfig8 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f838` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x193c` | `0x553c` | CoeffDMAConfig9 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f83c` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1940` | `0x5540` | CoeffDMAConfig10 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f840` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1944` | `0x5544` | CoeffDMAConfig11 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f844` | changed 66; H13-only 200 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1948` | `0x5548` | CoeffDMAConfig12 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f848` | changed 66; H13-only 30 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x194c` | `0x554c` | CoeffDMAConfig13 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f84c` | changed 66; H13-only 30 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1950` | `0x5550` | CoeffDMAConfig14 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f850` | changed 66; H13-only 30 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1954` | `0x5554` | CoeffDMAConfig15 | `0x00000021`, `0x00010020`, `0x00010021`, `0x00020020`, `0x00030020` (66/265 tasks) | unresolved | `0x1f854` | changed 66; H13-only 30 | `conv_k1_c1024_n1024_s1_bias0#t0`, `matmul_m1_k1024_n1024_ty1#t1`, `softmax_1x64x8x8#t3` |
| `0x1958` | `0x5558` | CoeffBaseAddr0 | not written (0/265 tasks) | invariant/omitted | `0x1f858` | H13-only 96 | all 172 decoded H14 cases |
| `0x195c` | `0x555c` | CoeffBaseAddr1 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f85c` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1960` | `0x5560` | CoeffBaseAddr2 | `0x00000400`, `0x00002400`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000`, `0x00040000` (44/265 tasks) | unresolved | `0x1f860` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1964` | `0x5564` | CoeffBaseAddr3 | `0x00000600`, `0x00003600`, `0x00006000`, `0x0000c000`, `0x00018000`, `0x00030000`, `0x00060000` (44/265 tasks) | unresolved | `0x1f864` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1968` | `0x5568` | CoeffBaseAddr4 | `0x00000800`, `0x00004800`, `0x00008000`, `0x00010000`, `0x00020000`, `0x00040000`, `0x00080000` (44/265 tasks) | unresolved | `0x1f868` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x196c` | `0x556c` | CoeffBaseAddr5 | `0x00000a00`, `0x00005a00`, `0x0000a000`, `0x00014000`, `0x00028000`, `0x00050000`, `0x000a0000` (44/265 tasks) | unresolved | `0x1f86c` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1970` | `0x5570` | CoeffBaseAddr6 | `0x00000c00`, `0x00006c00`, `0x0000c000`, `0x00018000`, `0x00030000`, `0x00060000`, `0x000c0000` (44/265 tasks) | unresolved | `0x1f870` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1974` | `0x5574` | CoeffBaseAddr7 | `0x00000e00`, `0x00007e00`, `0x0000e000`, `0x0001c000`, `0x00038000`, `0x00070000`, `0x000e0000` (44/265 tasks) | unresolved | `0x1f874` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1978` | `0x5578` | CoeffBaseAddr8 | `0x00001000`, `0x00009000`, `0x00010000`, `0x00020000`, `0x00040000`, `0x00080000`, `0x00100000` (44/265 tasks) | unresolved | `0x1f878` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x197c` | `0x557c` | CoeffBaseAddr9 | `0x00001200`, `0x0000a200`, `0x00012000`, `0x00024000`, `0x00048000`, `0x00090000`, `0x00120000` (44/265 tasks) | unresolved | `0x1f87c` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1980` | `0x5580` | CoeffBaseAddr10 | `0x00001400`, `0x0000b400`, `0x00014000`, `0x00028000`, `0x00050000`, `0x000a0000`, `0x00140000` (44/265 tasks) | unresolved | `0x1f880` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1984` | `0x5584` | CoeffBaseAddr11 | `0x00001600`, `0x0000c600`, `0x00016000`, `0x0002c000`, `0x00058000`, `0x000b0000`, `0x00160000` (44/265 tasks) | unresolved | `0x1f884` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1988` | `0x5588` | CoeffBaseAddr12 | `0x00001800`, `0x0000d800`, `0x00018000`, `0x00030000`, `0x00060000`, `0x000c0000`, `0x00180000` (44/265 tasks) | unresolved | `0x1f888` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x198c` | `0x558c` | CoeffBaseAddr13 | `0x00001a00`, `0x0000ea00`, `0x0001a000`, `0x00034000`, `0x00068000`, `0x000d0000`, `0x001a0000` (44/265 tasks) | unresolved | `0x1f88c` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1990` | `0x5590` | CoeffBaseAddr14 | `0x00001c00`, `0x0000fc00`, `0x0001c000`, `0x00038000`, `0x00070000`, `0x000e0000`, `0x001c0000` (44/265 tasks) | unresolved | `0x1f890` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1994` | `0x5594` | CoeffBaseAddr15 | `0x00001e00`, `0x00010e00`, `0x0001e000`, `0x0003c000`, `0x00078000`, `0x000f0000`, `0x001e0000` (44/265 tasks) | unresolved | `0x1f894` | changed 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x1998` | `0x5598` | CoeffBfrSize0 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f898` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x199c` | `0x559c` | CoeffBfrSize1 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f89c` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19a0` | `0x55a0` | CoeffBfrSize2 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8a0` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19a4` | `0x55a4` | CoeffBfrSize3 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8a4` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19a8` | `0x55a8` | CoeffBfrSize4 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8a8` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19ac` | `0x55ac` | CoeffBfrSize5 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8ac` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19b0` | `0x55b0` | CoeffBfrSize6 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8b0` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19b4` | `0x55b4` | CoeffBfrSize7 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8b4` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19b8` | `0x55b8` | CoeffBfrSize8 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8b8` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19bc` | `0x55bc` | CoeffBfrSize9 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8bc` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19c0` | `0x55c0` | CoeffBfrSize10 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8c0` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19c4` | `0x55c4` | CoeffBfrSize11 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8c4` | same 44; H13-only 52 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19c8` | `0x55c8` | CoeffBfrSize12 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8c8` | changed 44; H13-only 222 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19cc` | `0x55cc` | CoeffBfrSize13 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8cc` | changed 44; H13-only 222 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19d0` | `0x55d0` | CoeffBfrSize14 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8d0` | changed 44; H13-only 222 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19d4` | `0x55d4` | CoeffBfrSize15 | `0x00000200`, `0x00001200`, `0x00002000`, `0x00004000`, `0x00008000`, `0x00010000`, `0x00020000` (44/265 tasks) | unresolved | `0x1f8d4` | changed 44; H13-only 222 | `conv_k1_c64_n64_s8_bias0_relu#t0`, `conv_k1_c256_n512_s1_bias0#t0`, `conv_k1_c1024_n1024_s1_bias0#t0` |
| `0x19d8` | `0x55d8` | BiasDMAConfig | `0x00010020`, `0x00020020`, `0x00030020` (22/265 tasks) | unresolved | `0x1f8d8` | changed 22; H13-only 244 | `binary_pow_1x1024x1x1#t1`, `binary_pow_1x1024x1x1#t2`, `softmax_1x64x8x8#t3` |
| `0x19dc` | `0x55dc` | BiasBaseAddr | `0x00010020`, `0x00020020`, `0x00030020` (22/265 tasks) | unresolved | `0x1f8dc` | changed 22; H13-only 244 | `binary_pow_1x1024x1x1#t1`, `binary_pow_1x1024x1x1#t2`, `softmax_1x64x8x8#t3` |
| `0x19e0` | `0x55e0` | BiasReserved0 | `0x00010020`, `0x00020020`, `0x00030020` (22/265 tasks) | unresolved | `0x1f8e0` | changed 22; H13-only 244 | `binary_pow_1x1024x1x1#t1`, `binary_pow_1x1024x1x1#t2`, `softmax_1x64x8x8#t3` |
| `0x19e4` | `0x55e4` | BiasReserved1 | `0x00000021`, `0x00010021`, `0x00020021`, `0x00030021` (52/265 tasks) | unresolved | `0x1f8e4` | changed 52; H13-only 214 | `binary_maximum_c512_constant_scalar#t0`, `binary_pow_1x1024x1x1#t2`, `softmax_1x64x8x8#t3` |
| `0x19e8` | `0x55e8` | PostScaleDMAConfig | not written (0/265 tasks) | invariant/omitted | `0x1f8e8` | H13-only 266 | all 172 decoded H14 cases |
| `0x19ec` | `0x55ec` | PostScaleBaseAddr | not written (0/265 tasks) | invariant/omitted | `0x1f8ec` | H13-only 266 | all 172 decoded H14 cases |
| `0x19f0` | `0x55f0` | PostScaleReserved0 | not written (0/265 tasks) | invariant/omitted | `0x1f8f0` | H13-only 266 | all 172 decoded H14 cases |
| `0x19f4` | `0x55f4` | PostScaleReserved1 | `0x00000000`, `0x00000080` (52/265 tasks) | unresolved | `0x1f8f4` | changed 52; H13-only 214 | `binary_maximum_c512_constant_scalar#t0`, `binary_pow_1x1024x1x1#t2` |
| `0x19f8` | `0x55f8` | SparseBlockSizeCfg — BlockSize bits 0:7 in the detailed map | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x19fc` | `0x55fc` | Reserved | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1a00` | `0x5600` | Reserved — reserved in the detailed map; SparseKernelBlockSize in map prose | `0x00000000`, `0x00000080` (52/265 tasks) | unresolved | — | new H14 slot | `binary_maximum_c512_constant_scalar#t0`, `binary_pow_1x1024x1x1#t2` |
| `0x1a04` | `0x5604` | Reserved | `0x00000000` (52/265 tasks) | invariant/omitted | — | new H14 slot | `binary_maximum_c512_constant_scalar#t0` |
| `0x1a08` | `0x5608` | Spare0 | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1a0c` | `0x560c` | Spare1 | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1a10` | `0x5610` | Reserved | not written (0/265 tasks) | invariant/omitted | — | new H14 slot | all 172 decoded H14 cases |
| `0x1a14` | `0x5614` | Reserved | `0x00000080` (52/265 tasks) | invariant/omitted | — | new H14 slot | `binary_maximum_c512_constant_scalar#t0` |

## Records outside the declared blocks

| Address |
|---|
| None across all 265 decoded H14 tasks |

## Ranked H14 backend implementation plan

“Template-encodable” means an emitter can reproduce a checked-in point exactly. It does not mean interpolation is safe. The unresolved lists contain every word that varies within the named evidence family without a formula above.

| Rank | Family | Evidence envelope now | Exact blockers to general encoding | Oracle evidence |
|---|---|---|---|---|
| 1 | Elementwise add/mul/max/min/sub | Template-encodable at the ten sampled fp16 shapes: 1×1024×1×1, 1×128×16×16, 1×128×1×1, 1×2048×1×1, 1×256×1×1, 1×3×224×224, 1×4096×1×1, 1×512×1×1, 1×64×1×1, 1×64×8×8 | Resolve geometry-varying t0:`0x0030`, t0:`0x050c`, t0:`0x0510`, t0:`0x0514`, t0:`0x0518`, t0:`0x051c`, t0:`0x0520`, t0:`0x0524`, t0:`0x0528`, t0:`0x052c`, t0:`0x0534`, t0:`0x1118`, t0:`0x111c`, t0:`0x1130`, t0:`0x1134`, t0:`0x1510`, t0:`0x1514`; header t0:h1. Operation word `0x0900` and Common geometry are resolved. | `binary_add_1x64x1x1`, `binary_mul_1x128x16x16`, `binary_sub_1x3x224x224` |
| 2 | Unary | Template-encodable only at sampled channels. abs: C=64,128,256,512,1024,2048,4096; exp: C=64,512; gelu: C=64,512; leaky_relu: C=64,512; relu: C=64,512; rsqrt: C=64,512; sigmoid: C=64,512; silu: C=64,512; sqrt: C=64,512; tanh: C=64,512 | Resolve operation-dependent t0:`0x0038`, t0:`0x003c`, t0:`0x0510`, t0:`0x0514`, t0:`0x0518`, t0:`0x0534`, t0:`0x0d04`, t0:`0x0d10`, t0:`0x1900`; header t0:h1, t0:h4, t0:h7. Each sampled object is one task, but the campaign lacks independent H/W sweeps. | `unary_abs_c64`, `unary_abs_c4096`, `unary_exp_c512`, `unary_sqrt_c512` |
| 3 | Matvec (`transpose_y=true` matmul) | Template-encodable grid M={1,2,8,64}, K={256,512,1024}, N={256,512,1024}; all 36 combinations decoded as two H14 tasks | Resolve t0:`0x001c`, t0:`0x0028`, t0:`0x0038`, t0:`0x0508`, t0:`0x0510`, t0:`0x0514`, t0:`0x0518`, t0:`0x0538`, t0:`0x111c`, t1:`0x0038`, t1:`0x050c`, t1:`0x0534`, t1:`0x1514`, t1:`0x195c`, t1:`0x1960`, t1:`0x1964`, t1:`0x1968`, t1:`0x196c`, t1:`0x1970`, t1:`0x1974`, t1:`0x1978`, t1:`0x197c`, t1:`0x1980`, t1:`0x1984`, t1:`0x1988`, t1:`0x198c`, t1:`0x1990`, t1:`0x1994`, t1:`0x1998`, t1:`0x199c`, t1:`0x19a0`, t1:`0x19a4`, t1:`0x19a8`, t1:`0x19ac`, t1:`0x19b0`, t1:`0x19b4`, t1:`0x19b8`, t1:`0x19bc`, t1:`0x19c0`, t1:`0x19c4`, t1:`0x19c8`, t1:`0x19cc`, t1:`0x19d0`, t1:`0x19d4`; header t0:h1, t0:h6, t1:h1. The 16 coefficient base/size words dominate the gap. `transpose_y=false` has no accepted oracle. | `matmul_m1_k256_n256_ty1`, `matmul_m64_k1024_n1024_ty1`; `matmul_m1_k256_n256_ty0` is rejected |

Implement the shared H14 container and task-record writer first, then copy exact templates for smoke probes, then derive the listed changing words one family at a time. Do not start with convolution, reductions, normalization, real_div, or pow: their extra task graphs add unresolved allocation and SparseKernelBlockSize behavior beyond the three ranked families.

No extra oracle was minted for this report. The checked-in campaign already supplies at least three spatial points, seven channel points, five operation points, and the full 4×3×3 matvec grid needed for the formulas and blockers stated here.
