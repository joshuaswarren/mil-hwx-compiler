# H13/H14 Apple-oracle differential report

## Scope and provenance

This report records source-native compiler research at repository commit `d140e0026bc8641b96c9f5580e923c00637c4fd1`. The oracle ran on `MacStudio.local`, Apple M1 Ultra, macOS 26.6.2 arm64. The oracle executable was `/tmp/h13-oracle/bin/ane-compile-hwx`, SHA-256 `b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe`.

The campaign retained MIL, weight descriptions, decoded words, compiler status, and the SHA-256 of each Apple HWX. It did not retain Apple-generated HWX bytes. The records are under `research/oracles/h13` and `research/oracles/h14`.

The external register-name evidence is freedomtan's H14 register map at commit [`ce54664e787976b646c450ceabed1731b506a4cd`](https://github.com/freedomtan/coreml_to_ane_hwx/commit/ce54664e787976b646c450ceabed1731b506a4cd), specifically [`hwx_dump/h14_register_map.md`](https://raw.githubusercontent.com/freedomtan/coreml_to_ane_hwx/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md). All other claims below come from the checked-in oracle records or the source-native comparison described here.

## Method

`research/mint_oracles.py` changes one input at a time where practical. It covers binary operations and geometry, unary operations, reductions, normalization, convolution, matmul, linear, constants, and short fused chains. It invokes Apple's compiler for both targets, parses the Mach-O load commands, splits every task descriptor, decodes the register stream, writes one JSON record per attempt, and deletes the temporary HWX.

The complete command was:

`python3 research/mint_oracles.py --host macstudio --targets h13 h14 --force`

It produced `SUMMARY cases=470 decoded=344 rejected=126`.

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

All 36 accepted matmuls per target use `transpose_y=true`; all 36 `transpose_y=false` attempts were rejected. Inline tensor constants were rejected in all 12 binary-operation cases per target. Scalar constants were accepted in all 12 cases. BLOBFILE constants were accepted in 10 of 12; both `real_div` BLOBFILE cases were rejected. The oracle returned `callback_status=1` for 61 failures per target and `callback_status=22` for two. A rejection JSON is evidence of the attempted MIL and compiler response, not a decoded oracle.

## Task descriptor encodings

| Property | H13 | H14 |
|---|---|---|
| Task header | 10 words | 8 words |
| Task sizing | First size is program word at `+0x818` plus one; later size is the preceding task's `header[1][24:16]` plus one | `header[0][26:16]` words |
| Task linking | `header[7]` is the next section-relative byte offset; final value is zero | Tasks are 16-byte aligned; zero-size 16-byte prefix/alignment records are skipped |
| Register record | `count=(header>>26)+1`; byte address is `header&0x03ffffff` | Dense: `count=((header>>15)&0x3f)+1`. Scatter: bit 31 set, 16-bit mask in bits 30:15. Base word index is bits 14:0. |

The multi-task rule is observed directly in `matmul_m1_k256_n512_ty1.json`: program count 2, first task 126 words, next pointer `0x200`, 8 bytes of zero alignment, and second task 157 words. Across the campaign, the linked parser decoded H13 objects containing one, two, three, five, and six tasks.

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

## Apple H13 versus current native H13

The native comparison used `build/mil-hwxc --target H13 --format hwx` for four implemented binary operations and two implemented matvec geometries, then decoded the result with the same parser. The tables compare emitted words exactly. `not written` means the stream has no record for that address; it is different from an explicit zero write.

Apple emits one 126-word task for each binary case while native emits one 157-word task. Apple emits tasks `[126,157]` for each matvec case while native emits one 157-word task. For matvec, the table compares Apple's task 1 with native task 0; native has no counterpart for Apple's 126-word preparation task 0.

### Four supported binary operations: exact shared table

| Case | Apple HWX SHA-256 | Native HWX SHA-256 |
|---|---|---|
| `binary_add_1x64x1x1` | `a184d54174dec72cd0b26224e2a49cd2e474f6728ec5ce7ba93e4a85b6ac40b4` | `675d57d3ca84ad7f56390ceaf3717d22293eba796166e5e23fb07b509d7652f1` |
| `binary_mul_1x64x1x1` | `854fcd320569bf820010fb1527364ccb23d7398be1f52637453b72ae7dbe3b6b` | `1756565a4f2c8bbc0bd535b8cc10cbb5e213b161eee1a9f84ea2ebbda57cc36c` |
| `binary_maximum_1x64x1x1` | `4994ab4094b53ce013513d79181a19037f9613f53cedb0b7324619ac43fa2f5b` | `b6ee7ae0d1a490068fd9644e20c985780e5e0acf7d7b41aff0ca90ecd42c09b1` |
| `binary_minimum_1x64x1x1` | `6936257cafc1871e7f2192733841f830f8d55fc37946c1afa4acd7d337cd66a1` | `a4fd35f6034b3f45b15f4931a277c32bc5eb6be94e0fefc2c6837c5c6b4b94f0` |

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

Apple HWX SHA-256: `3bd4d7ed6b909e42e5902ec697f9df85d48f1b6825eb33550b7375f047ce7fd1`. Native HWX SHA-256: `9602d87e88adef371ce519b1af47b3be4ace093f8b3649dbbb52545175b5eb6c`. Apple task compared: 1; task sizes: Apple 157 words, native 157 words.

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

Apple HWX SHA-256: `aea0dc19cc7505fde54253e556d388b976d121f9996b63d6ea47a2b05424e8ab`. Native HWX SHA-256: `933572b8781f3674e271c825ac9b1309171db0d57d64ac1e5a2453d710016f6d`. Apple task compared: 1; task sizes: Apple 157 words, native 157 words.

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
