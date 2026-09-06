# H13/H14 Apple-oracle differential report

## Scope and provenance

This report records source-native compiler research that began at repository commit `d140e0026bc8641b96c9f5580e923c00637c4fd1`. The checked-in campaign was regenerated against repository base commit `9483fe699b9416a2bd0537606b36f5b5014638b9` on `MacStudio.local`, Apple M1 Ultra, macOS 26.6.2 arm64. The oracle executable was `/tmp/h13-oracle/bin/ane-compile-hwx`, SHA-256 `b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe`.

The campaign retained MIL, weight descriptions, decoded register words, constant-section hashes and compact fp16 field tables, compiler status, and the SHA-256 of each Apple HWX. It did not retain Apple-generated HWX bytes. The records are under `research/oracles/h13` and `research/oracles/h14`.

The exact campaign source hashes are `cda43658a57026468a90dc088909559e9fbb0c04597a691119da93338984b648` for `research/mint_oracles.py` and `fc85d0664a4e567e617189d53d22252f46b6cbca2e4246976ee9ae4deaa8aed5` for `research/h13_td.py`. Each JSON record carries both hashes.

A second campaign, the envelope sweep, was added later at repository commit `e1cb580` on the same host and with the same oracle executable. Its 548 records are named `env_*` under the same two directories and carry their own source hashes: `a6e8f70d438973e60913815be66dbd6de19d7200df9053a9efc2a615823d8074` for `research/mint_oracles.py` and `125da0249ca3c3ba0ca1f9f0cabfd1db54a965abb75f9f4cdcc2598e255c8fbc` for `research/h13_td.py`. The 470 first-campaign records were deliberately not regenerated, so the corpus carries two provenance sets and every record states its own. `research/oracle-envelope.md` reports the envelope sweep in full; the coverage table below summarizes it and the corrections it forces on this report.

The external register-name evidence is freedomtan's H14 register map at commit [`ce54664e787976b646c450ceabed1731b506a4cd`](https://github.com/freedomtan/coreml_to_ane_hwx/commit/ce54664e787976b646c450ceabed1731b506a4cd), specifically [`hwx_dump/h14_register_map.md`](https://raw.githubusercontent.com/freedomtan/coreml_to_ane_hwx/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md). All other claims below come from the checked-in oracle records or the source-native comparison described here.

## Method

`research/mint_oracles.py` changes one input at a time where practical. `campaign()` covers binary operations and geometry, unary operations, reductions, normalization, convolution, matmul, linear, constants, and short fused chains; `envelope_campaign()` adds the `env_*` cases that probe the outer edge of the accepted forms. It invokes Apple's compiler for both targets, parses the Mach-O load commands, splits every task descriptor of every program the object declares, decodes the register stream, writes one JSON record per attempt, and deletes the temporary HWX. A task whose register stream cannot be split is stored with its header words and a `decode_error` rather than failing the case, so a parser gap is never recorded as an Apple rejection.

The complete command was:

`python3 research/mint_oracles.py --host macstudio --targets h13 h14 --force`

It produced `SUMMARY cases=470 decoded=344 rejected=126`.

The envelope sweep was minted separately with `--case 'env_*' --force`, and produced `SUMMARY cases=548 decoded=542 rejected=6`.

## Campaign coverage

| Target | Attempts | Decoded | Apple rejection | Task-count distribution among decoded objects |
|---|---:|---:|---:|---|
| H13 | 235 | 172 | 63 | 1: 98; 2: 59; 3: 13; 5: 1; 6: 1 |
| H14 | 235 | 172 | 63 | 1: 98; 2: 60; 3: 12; 5: 1; 6: 1 |

The family results were identical for H13 and H14:

| Family | Attempts per target | Decoded per target |
|---|---:|---:|
| Binary, two runtime inputs | 70 | 70 |
| Binary, constant operand | 36 | 22 |
| Unary | 29 | 25 |
| Reduction | 6 | 6 |
| Softmax/layer normalization | 4 | 4 |
| Convolution | 13 | 7 |
| Matmul | 72 | 36 |
| Linear | 2 | 0 |
| Short chains | 3 | 2 |

The envelope sweep, 274 further cases per target:

| Family | Attempts per target | Decoded per target |
|---|---:|---:|
| Broadcast elementwise | 93 | 93 |
| Matmul, runtime and constant operands | 117 | 114 |
| Convolution with bias, groups, stride 2 | 22 | 22 |
| gelu and silu on spatial shapes | 13 | 13 |
| Fused chains | 29 | 29 |

Within this first campaign, all 36 accepted matmuls per target use `transpose_y=true` and all 36 `transpose_y=false` attempts were rejected. Inline tensor constants were rejected in all 12 binary-operation cases per target. Scalar constants were accepted in all 12 cases. BLOBFILE constants were accepted in 10 of 12; both `real_div` BLOBFILE cases were rejected. The oracle returned `callback_status=1` for 61 failures per target and `callback_status=22` for two. A rejection JSON is evidence of the attempted MIL and compiler response, not a decoded oracle.

The envelope sweep corrects two conclusions that this narrow sampling invited. `transpose_y=false` is not refused by Apple: with both matmul operands as runtime inputs it is accepted in every attempted geometry, and it costs one task fewer than `transpose_y=true`. What is refused is a `transpose_y=false` BLOBFILE weight, which reports `ANE internal validation error: Metadata data type does not match requested type.` Likewise convolution bias is not refused: all 6 bias rejections here used an inline fp16 tensor constant, the same storage refused for binary operands, and all 12 envelope convolutions with a BLOBFILE bias decoded. The envelope sweep's own 6 rejections are a single geometry, `[128,8192] x const[N,8192]` with `transpose_y=true`, three per target, all `callback_status=1`.

## Task descriptor encodings

| Property | H13 | H14 |
|---|---|---|
| Task header | 10 words | 8 words |
| Extended header | One extra word precedes the first register record when `header[9] & 0x3 == 0x3` | Same rule on `header[7]` |
| Task sizing | First size is program word at `+0x818` plus one; later size is the preceding task's `header[1][24:16]` plus one | `header[0][26:16]` words |
| Task linking | `header[7]` is the next section-relative byte offset; final value is zero | Tasks are 16-byte aligned; zero-size 16-byte prefix/alignment records are skipped |
| Register record | `count=(header>>26)+1`; byte address is `header&0x03ffffff` | Dense: `count=((header>>15)&0x3f)+1`. Scatter: bit 31 set, 16-bit mask in bits 30:15. Base word index is bits 14:0. |

The multi-task rule is observed directly in `matmul_m1_k256_n512_ty1.json`: one program declaring two tasks, first task 126 words, next pointer `0x200`, 8 bytes of zero alignment, and second task 157 words. Across the first campaign, the linked parser decoded H13 objects containing one, two, three, five, and six tasks; the envelope sweep reaches 129 tasks, still in one program.

The extended-header flag was found by the envelope sweep. Across the 5,549 decoded tasks in the checked-in corpus, H13 `header[9]` is `0x0`, `0x21`, `0x23` or `0x26` and H14 `header[7]` is `0x1`, `0x10001`, `0x30001`, `0x40001`, `0x50001` or `0x50003`; only `0x23` and `0x50003` carry the extra word, so bit 1 alone is not the predicate. The extra word held `0x0`, `0x7`, or `0x8`, and the task's declared size already includes it, so only the register stream shifts. `research/oracle-envelope.md` records how a silent mis-parse was caught.

## Register block map

The H14 names come from the cited register map. H13 uses the same functional block order at different bases, confirmed by corresponding one-parameter changes.

| Block | H13 base / words | H14 base / words |
|---|---:|---:|
| Common | `0x00000` / 16 | `0x0000` / 19 |
| L2 | `0x04800` / 18 | `0x0500` / 25 |
| PE | `0x08800` / 4 | `0x0900` / 5 |
| NE | `0x0c800` / 5 | `0x0d00` / 5 |
| Tile DMA source | `0x13800` / 28 | `0x1100` / 53 |
| Tile DMA destination | `0x17800` / 7 | `0x1500` / 10 |
| Kernel DMA source | `0x1f800` / 62 | `0x1900` / 70 |

H14 destination DMA covers ten words through `0x1524`; Apple writes that address in the baseline add object. Therefore the decoder uses the exclusive range end `0x1528`, rather than stopping at nine words.

## One-parameter field correlations

These are correlations, not invented semantic names:

| Changed input | Observed word behavior |
|---|---|
| Spatial H/W: 1, 8, 224 | Common word `+0x00` is respectively `0x00010001`, `0x00080008`, `0x00e000e0` on both targets. This confirms packed 16-bit W/H. |
| Channels: 64 through 4096 | Common words `+0x0c` and `+0x10` equal C on both targets. H13 L2 and DMA size/stride words scale with C; corresponding H14 block words change in the same sweep. |
| Operation: add/mul/max/min/sub | First PE word is `0x00080000`, `0x00080004`, `0x00080008`, `0x0008000c`, `0x000c0000` on both targets. |
| Operation: real_div/pow | Apple emits two tasks for real_div and three for pow at C64, rather than one PE-only binary task. |
| Runtime input to scalar constant | Input count falls from two to one while the scalar form remains one task and retains a 16 KiB constant section. |
| Runtime input to BLOBFILE constant | Apple emits two tasks and a 256-byte constant section for add C64. |
| Target H13 to H14 | Tensor descriptors retain the same shapes, element code 5, strides, and total byte sizes; program and task encodings and all non-common register bases change. |

For `[1,64,1,1]` fp16 add, each tensor descriptor is shape `[1,64,1,1]`, strides `[4096,64,64,2]`, and total size 4096. The campaign only uses valid fp16 tensor fixtures. It does not establish how another element type changes descriptor or register words.

Header words, allocation selectors, many L2 words, and several sparse H14 fields vary consistently but remain unresolved. The JSON preserves them without assigning unsupported names.

## Constant sections and reusable LUTs

Every accepted record includes the constant section's size, SHA-256, nonzero-byte count, first-128-byte SHA-256, and tail nonzero count. Sections up to 64 KiB also include 2 KiB chunk hashes. Sections no larger than 256 bytes include every nonzero fp16 word as an index/value table. This preserves exact field evidence without checking in Apple-generated bytes.

The seven nonlinear unary tables are target-independent 128-byte prefixes. H14 stores exactly the prefix. H13 stores the same prefix followed by zeros to 2 KiB. C64 and C512 produce identical hashes.

| Operation | Shared 128-byte SHA-256 | H13 padded 2 KiB SHA-256 |
|---|---|---|
| sigmoid | `73f5680aa5f7b3833479e0ecd5a9dd0e3ec221e3aad5170e9db1e40e7a0c7469` | `08e8e724d22e1c05bb530b2081b3a3546957483748d74f41a38f3e25318b76f6` |
| tanh | `f4a38468b2a29430c8ffa4bb04cef84b8347394bbe2d6eb1081093ed4eaa57c6` | `9db44445cb09ea51fea932641deb5e647faa4d5d8213cab5c50539a3909d4cbf` |
| gelu | `34540958c4c1928918d1f50b00a88a56a3b8291f13d0b010fe4646d1d7f89838` | `a78bcf09c60241a2397c8ce8f41c061c429a62278f6c295760a65896a948bc6b` |
| silu | `0d48f9b9a9a791fcef312a765cd621e43acb2b56306b9bf491ee0957c5842897` | `96f9d957c685199015329e8c2156cb65cfba3e5e3f68d2e55fb7cbca710af94d` |
| exp | `b7b6085a1edc7def0f0bb2fc1fe345f1ba9d9a1a47e55b4516273149323b54d2` | `c079709c1ed8870f250a87e7e18901b1f5fc52f0edee020ac9171e221bc30f1a` |
| sqrt | `07dff4d87f5df889ef0981865b54a4253e00c12eb444430215be401fe2d69607` | `8bc99873c811a8c9c484f114618b0ab6cc7b9982373105af4b561f36d42fae4b` |
| rsqrt | `182158a2cbc0c3da91e3447d7afae8825d45a1e6396d611b4a48fb2ed8f0a6d3` | `d451a1e49a6781f5a58e59cd299701d79bc5a4bac10b564f7a7a7295da7f16ef` |

The checked-in `kSigmoidKERNWords`, `kTanhKERNWords`, `kGeluKERNWords`, `kSiluKERNWords`, `kExpKERNWords`, `kSqrtKERNWords`, and `kRsqrtKERNWords` arrays in `plugins/H16G/Encoding/H16GLUTEncoderData.inc` hash exactly to those shared prefixes. A source-native H13 encoder can therefore reuse each 32-word array and append 1,920 zero bytes.

The remaining scalar forms have compact generation rules. All indices below count little-endian fp16 words; unspecified words are zero.

- abs, relu, and scalar add, multiply, and subtract use a 16 KiB all-zero section, SHA-256 `4fe7b59af6de3b665b67788cc2f99892ab827efae3a467342b3bb4e3bc8e5bfe`.
- leaky_relu uses 2 KiB on H13 and 128 bytes on H14: `[0]=-inf`, `[1]=+inf`, `[2]=-inf`, `[3]=+inf`, `[37]=alpha`, `[39]=1`, `[41]=2^-18`, and `[42]=2^-24`. For alpha 0.125, the H13 SHA-256 is `137349d1334abe41447c43bb17c9c33d8a3513ffa2b9084c3f0b622a99c6d0f9`.
- scalar maximum uses 2 KiB on H13 and 128 bytes on H14: `[0]=v`, `[1]=+inf`, `[2]=v`, `[3]=+inf`, `[4..36]=v`, then the same metadata at 37, 39, 41, and 42. For v=0.5, the H13 SHA-256 is `df1ae3605b08b3031177204191133b76a5447af3b090eae33b7147d0a8b92f03`.
- scalar minimum uses 2 KiB on H13 and 128 bytes on H14: `[0]=-inf`, `[1]=v`, `[2]=-inf`, `[3]=v`, then the same metadata. For v=0.5, the H13 SHA-256 is `ce09200c6aa706965dbc08f7f74234c9145563914b6adb9df629857f4ce1844d`.
- scalar real_div uses 256 bytes on both targets. The first 128 bytes are zero and `[64..127]=1/v`. For v=0.5, the SHA-256 is `244990b7ace7886514f167a4b37bc8f0ca2536bc10dfec1c211453bbecbc9907`.

## Apple H13 versus current native H13

The native comparison used `build/mil-hwxc --target H13 --format hwx` for four implemented binary operations and two implemented matvec geometries, then decoded the result with the same parser. The tables compare emitted words exactly. `not written` means the stream has no record for that address; it is different from an explicit zero write.

Apple emits one 126-word task for each binary case while native emits one 157-word task. Apple emits tasks `[126,157]` for each matvec case while native emits one 157-word task. For matvec, the table compares Apple's task 1 with native task 0; native has no counterpart for Apple's 126-word preparation task 0.

### Four supported binary operations: exact shared table

Apple HWX file SHA-256 values vary between compiler runs: the same MIL compiled twice yields different file bytes but identical decoded task words. The tables below were re-verified word by word against the regenerated corpus, so the hashes here identify the checked-in records while the word tables are the stable evidence.

| Case | Apple HWX SHA-256 | Native HWX SHA-256 |
|---|---|---|
| `binary_add_1x64x1x1` | `e430ea1a624f35555e7ba7f68be1f8c1becfbe247c9073a0634c22753caf4cbf` | `675d57d3ca84ad7f56390ceaf3717d22293eba796166e5e23fb07b509d7652f1` |
| `binary_mul_1x64x1x1` | `c5dffa1aa1a5bf19209d9a02deabbac27099c72c91db7148aed7363a4d4040ca` | `1756565a4f2c8bbc0bd535b8cc10cbb5e213b161eee1a9f84ea2ebbda57cc36c` |
| `binary_maximum_1x64x1x1` | `fa8b87d2eb3e9beb6d1b4f5f4c7a47574e1173745efa0da8a152fd158ea349e8` | `b6ee7ae0d1a490068fd9644e20c985780e5e0acf7d7b41aff0ca90ecd42c09b1` |
| `binary_minimum_1x64x1x1` | `a5fb3388109d598ca1007725a8f119355fb10fdac612759aaed5ef4f7f88a834` | `a4fd35f6034b3f45b15f4931a277c32bc5eb6be94e0fefc2c6837c5c6b4b94f0` |

Apple task 0 is 126 words for each case; native task 0 is 157 words. The difference locations and Apple values are identical across all four operations. Native add, maximum, and minimum are also identical at every difference location. Multiply differs only at NE word `0x0c804`, represented by its own native column. Thus this one table is the exact per-word table for all four cases without repeating 66 rows four times.

| Kind | Word | Apple, all four | Native add/max/min | Native mul |
|---|---:|---:|---:|---:|
| header | `header[0]` | `0x02000000` | `0x02400000` | `0x02400000` |
| header | `header[2]` | `0x0000042a` | `0x00000422` | `0x00000422` |
| header | `header[8]` | `0x000259a4` | `0x00024966` | `0x00024966` |
| register | `0x0c800` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x0c804` | `0x00100000` | `0x00000000` | `0x00000030` |
| register | `0x13818` | `0x00001000` | `0x00000000` | `0x00000000` |
| register | `0x1382c` | `0x00001000` | `0x00000000` | `0x00000000` |
| register | `0x13840` | `0x00000100` | `0x00000000` | `0x00000000` |
| register | `0x17814` | `0x00001000` | `0x00000000` | `0x00000000` |
| register | `0x17818` | `0x01302031` | `0x01002031` | `0x01002031` |
| register | `0x1f808` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f80c` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f810` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f814` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f818` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f81c` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f820` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f824` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f828` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f82c` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f830` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f834` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f838` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f83c` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f840` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f844` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f848` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f84c` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f850` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f854` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f858` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f85c` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f860` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f864` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f868` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f86c` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f870` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f874` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f878` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f87c` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f880` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f884` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f888` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f88c` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f890` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f894` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f898` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f89c` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8a0` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8a4` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8a8` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8ac` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8b0` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8b4` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8b8` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8bc` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8c0` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8c4` | `not written` | `0x00000000` | `0x00000000` |
| register | `0x1f8c8` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f8cc` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f8d0` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f8d4` | `0x00000080` | `0x00000000` | `0x00000000` |
| register | `0x1f8e8` | `0x00000040` | `0x00000000` | `0x00000000` |
| register | `0x1f8ec` | `0x00000040` | `0x00000000` | `0x00000000` |
| register | `0x1f8f0` | `0x00000040` | `0x00000000` | `0x00000000` |
| register | `0x1f8f4` | `0x00000040` | `0x00000000` | `0x00000000` |

### `matmul_m1_k256_n512_ty1`

Apple HWX SHA-256: `426adefddc00a7ab14c713e3b4d7cafae2c4ccebbf8d4b7c204de929c4521b48`. Native HWX SHA-256: `9602d87e88adef371ce519b1af47b3be4ace093f8b3649dbbb52545175b5eb6c`. Apple task compared: 1; task sizes: Apple 157 words, native 157 words.

| Kind | Word | Apple | Native |
|---|---:|---:|---:|
| header | `header[0]` | `0x03000001` | `0x02400000` |
| header | `header[1]` | `0x00000007` | `0x00000000` |
| header | `header[6]` | `0x2000b800` | `0x30009800` |
| header | `header[8]` | `0x05024023` | `0x05024025` |
| register | `0x0000c` | `0x00000100` | `0x00000200` |
| register | `0x0001c` | `0x5000b021` | `0x5000b421` |
| register | `0x00034` | `0x10244405` | `0x00244405` |
| register | `0x04804` | `0x00000160` | `0x00500172` |
| register | `0x04810` | `0x00001000` | `0x00002030` |
| register | `0x04814` | `0x00001000` | `0x00002000` |
| register | `0x04818` | `0x00000000` | `0x00002000` |
| register | `0x04830` | `0x0000014a` | `0x00500172` |
| register | `0x04834` | `0x00001000` | `0x00002030` |
| register | `0x04838` | `0x00000000` | `0x00000010` |
| register | `0x0483c` | `0x00000000` | `0x00002020` |
| register | `0x04840` | `0x00000000` | `0x00002000` |
| register | `0x04844` | `0x00000000` | `0x00002000` |
| register | `0x13800` | `0x00048880` | `0x00033881` |
| register | `0x13814` | `0x00004000` | `0x00008000` |
| register | `0x13818` | `0x00004000` | `0x00000000` |
| register | `0x17800` | `0x040000c1` | `0x000000c1` |
| register | `0x17808` | `0x00000400` | `0x00000040` |
| register | `0x1780c` | `0x00000400` | `0x00000040` |
| register | `0x17810` | `0x00000400` | `0x00008000` |
| register | `0x17814` | `0x00000400` | `0x00000000` |
| register | `0x1f84c` | `0x00004000` | `0x00008000` |
| register | `0x1f850` | `0x00008000` | `0x00010000` |
| register | `0x1f854` | `0x0000c000` | `0x00018000` |
| register | `0x1f858` | `0x00010000` | `0x00020000` |
| register | `0x1f85c` | `0x00014000` | `0x00028000` |
| register | `0x1f860` | `0x00018000` | `0x00030000` |
| register | `0x1f864` | `0x0001c000` | `0x00038000` |
| register | `0x1f868` | `0x00020000` | `0x00040000` |
| register | `0x1f86c` | `0x00024000` | `0x00048000` |
| register | `0x1f870` | `0x00028000` | `0x00050000` |
| register | `0x1f874` | `0x0002c000` | `0x00058000` |
| register | `0x1f878` | `0x00030000` | `0x00060000` |
| register | `0x1f87c` | `0x00034000` | `0x00068000` |
| register | `0x1f880` | `0x00038000` | `0x00070000` |
| register | `0x1f884` | `0x0003c000` | `0x00078000` |
| register | `0x1f888` | `0x00004000` | `0x00008000` |
| register | `0x1f88c` | `0x00004000` | `0x00008000` |
| register | `0x1f890` | `0x00004000` | `0x00008000` |
| register | `0x1f894` | `0x00004000` | `0x00008000` |
| register | `0x1f898` | `0x00004000` | `0x00008000` |
| register | `0x1f89c` | `0x00004000` | `0x00008000` |
| register | `0x1f8a0` | `0x00004000` | `0x00008000` |
| register | `0x1f8a4` | `0x00004000` | `0x00008000` |
| register | `0x1f8a8` | `0x00004000` | `0x00008000` |
| register | `0x1f8ac` | `0x00004000` | `0x00008000` |
| register | `0x1f8b0` | `0x00004000` | `0x00008000` |
| register | `0x1f8b4` | `0x00004000` | `0x00008000` |
| register | `0x1f8b8` | `0x00004000` | `0x00008000` |
| register | `0x1f8bc` | `0x00004000` | `0x00008000` |
| register | `0x1f8c0` | `0x00004000` | `0x00008000` |
| register | `0x1f8c4` | `0x00004000` | `0x00008000` |
| register | `0x1f8e8` | `0x00000040` | `0x00000000` |
| register | `0x1f8ec` | `0x00000040` | `0x00000000` |
| register | `0x1f8f0` | `0x00000040` | `0x00000000` |
| register | `0x1f8f4` | `0x00000040` | `0x00000000` |

### `matmul_m1_k512_n512_ty1`

Apple HWX SHA-256: `0fd127c28797e927a56ddcd336a3d43adc607440609b1d19e1d705d6a43af487`. Native HWX SHA-256: `933572b8781f3674e271c825ac9b1309171db0d57d64ac1e5a2453d710016f6d`. Apple task compared: 1; task sizes: Apple 157 words, native 157 words.

| Kind | Word | Apple | Native |
|---|---:|---:|---:|
| header | `header[0]` | `0x03000001` | `0x02400000` |
| header | `header[1]` | `0x0000000f` | `0x00000000` |
| header | `header[6]` | `0x2000b800` | `0x30009800` |
| header | `header[8]` | `0x05024023` | `0x05024025` |
| register | `0x0001c` | `0x5000b021` | `0x5000b421` |
| register | `0x00034` | `0x10244405` | `0x00244405` |
| register | `0x04804` | `0x00000160` | `0x00500172` |
| register | `0x04810` | `0x00002000` | `0x00002030` |
| register | `0x04818` | `0x00000000` | `0x00002000` |
| register | `0x04830` | `0x0000014a` | `0x00500172` |
| register | `0x04834` | `0x00002000` | `0x00002030` |
| register | `0x04838` | `0x00000000` | `0x00000010` |
| register | `0x0483c` | `0x00000000` | `0x00002020` |
| register | `0x04840` | `0x00000000` | `0x00002000` |
| register | `0x04844` | `0x00000000` | `0x00002000` |
| register | `0x13800` | `0x00048880` | `0x00033881` |
| register | `0x13818` | `0x00008000` | `0x00000000` |
| register | `0x17800` | `0x040000c1` | `0x000000c1` |
| register | `0x17808` | `0x00000400` | `0x00000040` |
| register | `0x1780c` | `0x00000400` | `0x00000040` |
| register | `0x17810` | `0x00000400` | `0x00008000` |
| register | `0x17814` | `0x00000400` | `0x00000000` |
| register | `0x1f8e8` | `0x00000040` | `0x00000000` |
| register | `0x1f8ec` | `0x00000040` | `0x00000000` |
| register | `0x1f8f0` | `0x00000040` | `0x00000000` |
| register | `0x1f8f4` | `0x00000040` | `0x00000000` |

## What the comparison establishes

For the four binary operations, all words outside the shared table match after register-stream decoding. The large difference set is mostly stream shape: Apple writes sparse kernel DMA ranges, while native writes a dense 62-word zero block. The two matvec tables show the exact remaining values after pairing Apple's computational task with native's only task; the missing Apple preparation task remains a separate structural difference.
