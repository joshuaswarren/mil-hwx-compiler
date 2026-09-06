# Apple single-program envelope for H13 and H14

## What this measures

`research/oracle-diff.md` records the first campaign, which changed one input at a
time around small shapes. This report answers a different question: how far do
Apple's accepted forms actually reach, and what does Apple emit at the edge? It
sweeps the geometry and the operations a transformer needs — per-channel and
spatial broadcasts, batches, odd and very large channel counts, matmul with both
operands runtime, convolution with bias, groups and stride 2, gelu and silu on
spatial shapes, and five fused chains — then records exactly what Apple's
compiler returned for every attempt, including the refusals.

The headline results:

- Apple accepted 542 of 548 attempts. All six refusals are the same form: a
  rank-2 constant-weight matmul with M=128 and K=8192.
- **Apple never emitted more than one HWX program.** Every one of the 542
  accepted objects carries exactly one program load command. Apple partitions
  work by emitting more tasks inside that single program, up to 129 tasks for
  one matmul, not by splitting the object.
- `transpose_y=false` is accepted after all. The first campaign concluded that
  all 36 `transpose_y=false` matmuls were rejected; this sweep shows the
  refusal belongs to the BLOBFILE constant operand, not to the transpose. With
  both operands runtime, `transpose_x` and `transpose_y` are accepted in all
  four combinations.
- Convolution bias is accepted. The first campaign's 6 bias refusals per target
  used an inline fp16 tensor constant, the same storage that was refused for
  binary operands. With a BLOBFILE bias, all 12 bias convolutions here decoded.
- `gelu` with `mode="TANH_APPROXIMATION"` produces byte-identical constants to
  `mode="EXACT"`; only `SIGMOID_APPROXIMATION` produces a different table.

## Provenance

All 548 records were minted in one run on `MacStudio.local`, Apple M1 Ultra,
macOS 26.6.2 arm64, at repository commit `e1cb580`, with Apple's compiler
`/tmp/h13-oracle/bin/ane-compile-hwx`, SHA-256
`b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe`. The campaign
source is `research/mint_oracles.py`, SHA-256
`a6e8f70d438973e60913815be66dbd6de19d7200df9053a9efc2a615823d8074`; the decoder is
`research/h13_td.py`, SHA-256
`125da0249ca3c3ba0ca1f9f0cabfd1db54a965abb75f9f4cdcc2598e255c8fbc`. Every record
carries all four hashes, so this provenance is checkable per file rather than
only asserted here.

The command was:

```sh
python3 research/mint_oracles.py --host macstudio --targets h13 h14 --case 'env_*' --force
```

It printed `SUMMARY cases=548 decoded=542 rejected=6` in 8 minutes 47 seconds.
The records are `research/oracles/h13/env_*.json` and
`research/oracles/h14/env_*.json`, 274 cases per target. No Apple HWX bytes are
retained: each record keeps the MIL, the weight description, the decoded task
words, the program and tensor descriptors, constant-section hashes and sizes,
and the compiler's status string.

The 470 records from the first campaign were not regenerated. They keep the
driver and decoder hashes from commit `9483fe6`, so the corpus carries two
provenance sets on purpose; each file states its own.

## Coverage and outcome

| Target | Family | Attempts | Decoded | Apple refusals | Task counts among decoded | Programs per object |
|---|---|---:|---:|---:|---|---|
| H13 | Broadcast elementwise | 93 | 93 | 0 | 1: 45; 2: 48 | 1 for all 93 |
| H13 | Matmul | 117 | 114 | 3 | 1: 4; 2: 37; 3: 36; 4: 4; 5: 2; 6: 2; 8: 1; 9: 3; 17: 6; 32: 1; 33: 9; 65: 6; 129: 3 | 1 for all 114 |
| H13 | Convolution | 22 | 22 | 0 | 1: 22 | 1 for all 22 |
| H13 | Activation | 13 | 13 | 0 | 1: 13 | 1 for all 13 |
| H13 | Chain | 29 | 29 | 0 | 1: 2; 2: 4; 3: 3; 4: 2; 5: 4; 6: 9; 8: 1; 9: 3; 11: 1 | 1 for all 29 |
| H14 | Broadcast elementwise | 93 | 93 | 0 | 1: 45; 2: 48 | 1 for all 93 |
| H14 | Matmul | 117 | 114 | 3 | 1: 6; 2: 68; 3: 11; 8: 1; 9: 3; 17: 6; 32: 1; 33: 9; 65: 6; 129: 3 | 1 for all 114 |
| H14 | Convolution | 22 | 22 | 0 | 1: 22 | 1 for all 22 |
| H14 | Activation | 13 | 13 | 0 | 1: 13 | 1 for all 13 |
| H14 | Chain | 29 | 29 | 0 | 1: 2; 2: 4; 3: 5; 5: 5; 6: 8; 8: 1; 9: 4 | 1 for all 29 |

The two targets accept and refuse exactly the same 274 cases. They differ in how
much work they split: 41 of the 271 decoded pairs have different task counts, and
in every one of those H14 uses fewer tasks than H13.

## Apple's partitioning

Across the 542 accepted objects — 271 per target, holding 1,790 H13 tasks and
1,736 H14 tasks — the parser found exactly one program load command in every one.
`research/mint_oracles.py` now parses multi-program objects — it collects every
program descriptor, gives each its own `__TEXT/__text` region, splits that region
separately, and records `program_count` plus a per-program `task_section` — so a
second program would have been recorded rather than mis-parsed. None appeared,
including at 134 MiB of weights and 129 tasks.

Apple's actual partitioning unit is the task. For a rank-2 constant-weight matmul
with M at 256 or 512, the count follows a closed form over the tested grid:

    tasks = 1 + K * N / 2^19

which reproduces all 18 measured values (9, 17, 33, 17, 33, 65, 33, 65, 129 for
K and N over 2048, 4096, 8192, identical at M=256 and M=512).

## 1. Elementwise and broadcast

Every attempted broadcast form was accepted on both targets, and the task count
follows one rule. The identical-shape row covers `add`, `mul` and `sub`; the
other rows cover `add` and `mul`.

| Operand form | Example | Tasks | Constant section | Cases |
|---|---|---:|---:|---:|
| Identical shapes, runtime | `[1,3072,1,1] add [1,3072,1,1]` | 1 | 16 KiB, all zero | 37 |
| Per-channel, runtime | `[1,768,16,16] add [1,768,1,1]` | 2 | 16 KiB, all zero | 12 |
| Spatial, runtime | `[1,96,8,8] mul [1,1,8,8]` | 2 | 16 KiB, all zero | 12 |
| Scalar tensor, runtime | `[1,64,16,16] add [1,1,1,1]` | 2 | 16 KiB, all zero | 12 |
| Batch broadcast, runtime | `[8,64,8,8] add [1,64,1,1]` | 2 | 16 KiB, all zero | 4 |
| Scalar fp16 constant | `[1,768,8,8] mul fp16(0.5)` | 1 | 16 KiB, all zero | 8 |
| Per-channel BLOBFILE constant | `[1,768,16,16] mul const [1,768,1,1]` | 2 | 4 bytes per channel | 8 |

The rule is that identical shapes cost one task and any shape difference costs
two: Apple emits a 126-word materialization task before the 126-word compute
task on H13. This holds for per-channel, spatial, scalar-tensor and batch
broadcasts alike, and does not depend on the channel count.

Accepted geometry, all decoded on both targets:

- Channel counts 64, 96, 200, 300, 512, 768, 1024, 3072, 8192, 16384. Odd counts
  are not special: 96, 200 and 300 behave exactly like powers of two.
- Batch N of 1, 2 and 8, with `[N,C,1,1]` and `[N,C,8,8]`.
- Spatial 1x1, 8x8 and 16x16.

The tensor descriptors show why odd channel counts are free. Each row is padded
to 64 bytes: strides are `[N-stride, C*64 or H*64, 64, 2]`, and the recorded
total is `C * H * 64`. For `[1,200,1,1]` that is 12,800 bytes, and for
`[1,16384,1,1]` it is 1,048,576 bytes. One detail worth flagging for the native
backend: with N greater than 1 the descriptor's total size still covers a single
batch element and equals the batch stride — `[8,64,8,8]` records strides
`[32768,512,64,2]` and total 32,768 bytes, not 262,144.

## 2. Matmul

This is the family that matters for attention and feed-forward blocks, so it was
swept the hardest: 117 cases per target over rank-2 and rank-3 operands, both
transposes, and each operand as either a runtime input or a BLOBFILE constant.

### Constant-weight matmul, `[M,K] x const[N,K]`, `transpose_y=true`

Task counts, H13 / H14. The constant section is always exactly `K*N*2` bytes.

| M | K=2048 N=2048 | 2048/4096 | 2048/8192 | 4096/2048 | 4096/4096 | 4096/8192 | 8192/2048 | 8192/4096 | 8192/8192 |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| 16 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 |
| 32 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 | 3/2 |
| 128 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | refused | refused | refused |
| 256 | 9/9 | 17/17 | 33/33 | 17/17 | 33/33 | 65/65 | 33/33 | 65/65 | 129/129 |
| 512 | 9/9 | 17/17 | 33/33 | 17/17 | 33/33 | 65/65 | 33/33 | 65/65 | 129/129 |

The largest accepted single program is `M=512, K=8192, N=8192`: 129 tasks and a
134,217,728-byte constant section, in one program, on both targets.

M=128 is the discontinuity. It uses fewer tasks than M=32 for K of 2048 and 4096,
and it is the only geometry Apple refuses.

### Rank-3 `[1,M,K]`, constant weight

Rank 3 with a leading batch of 1 is accepted over the whole grid and produces the
same task counts as rank 2 at M=1, 32 and 256 — 2/2, 3/2 and 9/9 through 129/129
respectively. Rank 3 does not change the constant section.

### `transpose_x=true`, constant weight

Accepted for `[K,M] x const[N,K]`: M=1 gives 1 task, M=32 gives 4 on H13 and 1 on
H14, M=256 gives 8 tasks at K=N=2048 and 32 at K=N=4096. Note this is `K*N/2^19`
without the `+1` of the `transpose_x=false` form.

### Constant `x`, runtime `y`

`matmul(x = const[M,K], y = runtime[N,K])` is accepted at M=1 and M=32 for
K=N=2048 and K=N=4096: 2 tasks on both targets, constant section `M*K*2` bytes.

### Both operands runtime — the attention shape

This is the form the first campaign never tested, and it is the one attention
needs. All 26 cases were accepted on both targets, with a 16 KiB all-zero
constant section in every one.

| Form | Geometry | Tasks H13 / H14 |
|---|---|---|
| `[1,S,D] x [1,S,D]^T` (scores) | S=64 D=64 | 3/3 |
| | S=128 D=64, S=128 D=128 | 3/3 |
| | S=256 D=64, S=256 D=128 | 4/3 |
| | S=512 D=64, S=512 D=128 | 6/3 |
| `[1,S,S] x [1,S,D]` (context) | S=64, S=128 | 2/2 |
| | S=256 | 3/2 |
| | S=512 | 5/2 |
| Rank-2, `ty=false` | M=16, K=N=2048 and 4096 | 1/1 |
| Rank-2, `ty=true` | M=16, K=N=2048 and 4096 | 2/2 |
| Rank-2, `ty=false` | M=128, K=N=2048 and 4096 | 2/2 |
| Rank-2, `ty=true` | M=128, K=N=2048 and 4096 | 3/3 |
| Rank-2, `tx=true`, `ty=false` | M=128, K=N=2048 | 2/2 |
| Rank-2, `tx=true`, `ty=true` | M=128, K=N=2048 | 3/3 |

In every runtime-runtime pair that shares a geometry, `transpose_y=false` costs
one task fewer than `transpose_y=true` — the opposite of the constant-weight
case, where `transpose_y=false` is refused outright.

## 3. Convolution

All 22 cases decoded on both targets, always as a single task in a single
program. Kernel 1 and 3, stride 1 and 2, groups 1, 4 and 64 (depthwise), with and
without a BLOBFILE bias, `pad_type="same"`.

| Case | Weight bytes | Constant section | Constant minus weights |
|---|---:|---:|---:|
| `k1_c64_n64_s16_st1_g1_bias0` | 8,192 | 8,192 | 0 |
| `k1_c64_n64_s16_st1_g1_bias1` | 8,192 | 9,216 | 1,024 |
| `k1_c64_n64_s16_st1_g4_bias0` | 2,048 | 4,096 | 2,048 |
| `k1_c64_n64_s16_st2_g1_bias0` | 8,192 | 11,136 | 2,944 |
| `k1_c64_n64_s16_st2_g4_bias0` | 2,048 | 7,936 | 5,888 |
| `k1_c256_n256_s8_st1_g1_bias1` | 131,072 | 132,096 | 1,024 |
| `k1_c768_n768_s1_st1_g1_bias1` | 1,179,648 | 1,181,696 | 2,048 |
| `k3_c64_n64_s16_st1_g1_bias0` | 73,728 | 73,728 | 0 |
| `k3_c64_n64_s16_st1_g1_bias1` | 73,728 | 74,752 | 1,024 |
| `k3_c64_n64_s16_st1_g4_bias0` | 18,432 | 20,480 | 2,048 |
| `k3_c64_n64_s16_st1_g64_bias0` | 1,152 | 2,048 | 896 |
| `k3_c64_n64_s16_st2_g1_bias0` | 73,728 | 82,816 | 9,088 |
| `k3_c64_n64_s16_st2_g4_bias0` | 18,432 | 28,544 | 10,112 |
| `k3_c128_n128_s16_st2_g1_bias1` | 294,912 | 328,576 | 33,664 |

`research/mint_conv_probes.py` closed this campaign's convolution gaps with a
524-case grid and known-weight probes; `research/h13-td-fields.md` records the
packing those probes recovered, which reproduces 284 decoded convolutions per
target byte-for-byte. The added bytes this campaign left unexplained are now
named: a bias is one extra row per plane rather than `Cout * 2` bytes, grouping
pads every plane to 64 bytes, and stride 2 switches to a zero-skipping form
whose per-row mask bytes and body count make the section size depend on which
weights are zero.

Three shapes of the constant section are visible. A stride-1, groups-1
convolution stores exactly the weights. A bias adds a fixed 1,024 bytes for 64 or
256 output channels and 2,048 for 768, not `outputs*2`. Grouping and stride 2 both
add material beyond the weights, and the `bias0` and `bias1` sections are the same
size whenever grouping or stride 2 is in play, so that extra material absorbs the
bias. The added bytes are not explained by this campaign; the measured sizes are
recorded rather than named.

## 4. gelu and silu on spatial shapes

All 13 cases decoded on both targets as one task. The constant section is the
128-byte lookup table, stored bare on H14 and zero-padded to 2 KiB on H13, and it
does not vary with the shape: `[1,64,8,8]`, `[1,128,16,16]`, `[1,256,32,32]`,
`[1,768,16,16]` and `[1,3072,1,1]` all produce the same table.

| Operation | Shared 128-byte SHA-256 prefix |
|---|---|
| `gelu`, `mode="EXACT"` | `34540958c4c1928918d1f50b00a88a56a3b8291f13d0b010fe4646d1d7f89838` |
| `gelu`, `mode="TANH_APPROXIMATION"` | `34540958c4c1928918d1f50b00a88a56a3b8291f13d0b010fe4646d1d7f89838` |
| `gelu`, `mode="SIGMOID_APPROXIMATION"` | `2d42a6d5c643896e96c2c722f50b95e3c4edc2bcbaae90733eef82d0d0d7f6fb` |
| `silu` | `0d48f9b9a9a791fcef312a765cd621e43acb2b56306b9bf491ee0957c5842897` |

`EXACT` and `TANH_APPROXIMATION` are the same table, so Apple treats the tanh
approximation as the exact form. The `EXACT` and `silu` hashes match the ones
already recorded in `research/oracle-diff.md` from `[1,64,1,1]` and `[1,512,1,1]`,
which confirms the table is independent of both shape and rank.

## 5. Chains: what Apple fuses

| Chain | Geometry | Tasks H13 / H14 | Constant section H13 / H14 |
|---|---|---|---|
| `x + relu(x)` | `[1,768,1,1]`, `[1,64,8,8]` | 1/1 | 16,384 / 16,384 |
| `x + gelu(x)` | `[1,768,1,1]` | 2/2 | 1,024 / 128 |
| `x + gelu(x)` | `[1,64,8,8]` | 2/2 | 2,048 / 128 |
| `x + silu(x)` | `[1,768,1,1]` | 2/2 | 1,024 / 128 |
| `x + silu(x)` | `[1,64,8,8]` | 2/2 | 2,048 / 128 |
| `softmax` axis -1 | `[1,1,64,64]`, `[1,1,128,128]` | 6/6 | 128 / 128 |
| `softmax` axis -1 | `[1,1,256,256]` | 8/8 | 128 / 128 |
| `softmax` axis -1 | `[1,8,S,S]`, S=64,128,256 | 6/6 | 1,024 / 128 |
| `softmax` axis -1 | `[1,12,S,S]`, S=64,128,256 | 6/6 | 1,536 / 128 |
| `matmul -> gelu -> matmul` | M=1, K=2048, H=4096 | 3/3 | 33,556,480 / same |
| `matmul -> gelu -> matmul` | M=32, K=2048, H=4096 | 4/3 | 33,556,480 / same |
| `matmul -> gelu -> matmul` | M=128, K=2048, H=4096 | 3/3 | 33,556,480 / same |
| `matmul -> gelu -> matmul` | M=32, K=4096, H=4096 | 4/3 | 67,110,912 / same |
| `layer_norm -> matmul` | M=1, K and N over 2048, 4096 | 5/5 | `K*N*2` |
| `layer_norm -> matmul` | M=32, rank 3 | 6/5 | 8,388,608 / same |
| `matmul -> softmax -> matmul` | S=64 D=64; S=128 D=64 and D=128 | 9/9 | 2,048 / 128 |
| `matmul -> softmax -> matmul` | S=256, D=64 | 11/9 | 2,048 / 128 |

The fusion rules this exposes, refined by the chain campaign in
`research/fusion-rules.md` (127 `chain_*` records per target, minted on the
same host and tool):

- `x + relu(x)` collapses to a single task with an all-zero 16 KiB constant
  section. The chain campaign shows why: a clamp is a post-operation field in
  the compute task's NE block — `0x0c804` on H13, `0x00d04` on H14 — and
  setting bit `0x00010000` there costs no task at all. `gelu` and `silu` set
  bit `0x00020000` instead and need their lookup table, which is why those
  residuals cost two tasks.
- A residual `add` against a runtime tensor does **not** fuse. A matmul
  followed by such an add is 4 tasks against the lone matmul's 2, and rewrites
  108 words. The `x + relu(x)` collapse is the elementwise engine's clamp, not
  a general add fusion.
- The feed-forward block `matmul -> gelu -> matmul` costs exactly one task more
  than a lone matmul at the same M: 3 tasks at M=1 and M=128, and 4 on H13 at
  M=32 where the lone matmul takes 3. The activation between two matmuls is
  nearly free, and both weight matrices plus the 2,048-byte gelu table live in
  one constant section (`2*H*K*2 + 2048` bytes). The chain campaign measures
  the same thing at transformer sizes: two compute tasks for the whole block at
  every `d_model` from 256 to 1024, with a per-channel bias and the activation
  both folded into the post-operation field.
- Attention as one program — `matmul -> softmax -> matmul` over three runtime
  inputs — is accepted at every tested size and stays a single program with 9 to
  11 tasks. Adding the `1/sqrt(d_head)` scaling costs nothing: it lands in the
  output-scale word `0x0c810` / `0x00d10` as an fp16 multiplier.
- On H13 the softmax constant section scales with the head count, 128 bytes per
  head, while H14 stores 128 bytes regardless. This is the same padding split
  seen for the unary tables.
- Two cautions for anyone reading the constant-section column above. Apple
  stores one copy of a constant that repeats, so a chain whose weights share a
  blob offset packs less than it declares — a projected block declaring
  1,572,864 bytes packed 1,183,744. And the 16 KiB all-zero block is a floor,
  not an addend: `layer_norm -> gelu` carries 128 bytes, 16,256 fewer than the
  lone `layer_norm`.
- Depth is not the limit. Stacked pre-norm blocks stay one program at 512
  blocks and 10,240 tasks, and 8,192 weightless block units compile to an
  85,934,080-byte object; the only chain refusal found is inherited from the
  per-operation envelope (`M=128, K=8192` below), and depth 16,384 was not
  refused but exceeded this repository's 900-second compile budget.
- One correction the chain campaign forces on the refusal list below: a
  `transpose_y=false` BLOBFILE matmul at `(32, 2048, 2048)` and
  `(128, 2048, 2048)` — recorded refused as a lone matmul in
  `research/h13-td-fields.md` — is accepted as the first matmul of a
  feed-forward chain on both targets. The geometry alone does not predict the
  refusal.

## 6. Every refusal

Six of 548 attempts were refused, three per target, and they are the same three
geometries on both:

| Case | Target | Compiler status |
|---|---|---|
| `env_mm_r2rb_m128_k8192_n2048_tx0_ty1` | H13, H14 | `callback_status=1` |
| `env_mm_r2rb_m128_k8192_n4096_tx0_ty1` | H13, H14 | `callback_status=1` |
| `env_mm_r2rb_m128_k8192_n8192_tx0_ty1` | H13, H14 | `callback_status=1` |

All three are `[128,8192] x const[N,8192]` with `transpose_y=true`. The same K and
N are accepted at M=1, 16, 32, 256 and 512, and M=128 is accepted at K=2048 and
K=4096, so the refusal is specific to the pairing of M=128 with K=8192 and is not
a size ceiling: `M=512, K=8192, N=8192` moves four times the data and compiles.
The compiler emitted no MIL validation report for these, only the callback
status, unlike the first campaign's `transpose_y=false` refusals which reported
`ANE internal validation error: Metadata data type does not match requested
type.`

A rejection record stores the attempted MIL and the compiler's response. It is
evidence of the refusal, not a decoded oracle.

## 7. The extended task header

This sweep found a task-header form the decoder did not know about, and it is
recorded here because it changes how a task stream must be read.

The last fixed header word — H13 `header[9]`, H14 `header[7]` — carries a flag in
its two low bits. When both are set, one extra word sits between the fixed header
and the first register record. Across the 4,928 decoded tasks in the committed
corpus (2,677 H13, 2,251 H14) the
word takes four values on H13 (`0x0`, `0x21`, `0x23`, `0x26`) and six on H14
(`0x1`, `0x10001`, `0x30001`, `0x40001`, `0x50001`, `0x50003`); only `0x23` and
`0x50003` carry the extra word. Bit 1 alone is not the predicate: H13 `0x26` and
H14 `0x30001` have bit 1 set and no extra word.

The extra word held `0x0` (43 H13 / 26 H14 tasks), `0x7` (3 / 4), or `0x8` (tasks
9 and 10 of `env_chain_attention_s256_d64` on H13), and the task's declared size
already includes it, so task splitting is unaffected — a 157-word compute task
becomes 158 words. Only the register stream shifts.

Two failure modes made this worth chasing. On H13 the misread raised
`H13 record[N] has unaligned address 0x7`, which `run_case` had been recording as
a rejection even though Apple had accepted the program. On H14 the misread did
not raise at all: the extra word parsed as a plausible record header
(`0x00000000` at address `0x0`, `0x00000007` at `0x1c`) and produced silently
wrong register words. `research/h13_td.py` handles the flag as of commit
`c8224a3`, and `research/mint_oracles.py` now decodes each task defensively — a
task whose register stream cannot be split is stored with its header words and a
`decode_error`, and the case still counts as accepted, with a top-level
`task_decode_errors` count. In the final corpus that count is zero for all 542
accepted cases.

The forms that use the extended header are the attention chains and every
runtime-runtime matmul except rank-2 M=16 (rank-2 from M=128, rank-3 from M=64):
48 of 2,677 H13 tasks and 30 of 2,251 H14 tasks, 26 cases per target.

## Reproducing

```sh
# the whole envelope sweep, both targets
python3 research/mint_oracles.py --host macstudio --targets h13 h14 --case 'env_*' --force

# one family
python3 research/mint_oracles.py --host macstudio --targets h13 --case 'env_mm_*'

# list without compiling
python3 research/mint_oracles.py --local --list --case 'env_*'
```

`envelope_campaign()` in `research/mint_oracles.py` builds these cases; every name
it produces starts with `env_`. Records already present are skipped unless
`--force` is given, so a family can be extended without re-minting the rest.
