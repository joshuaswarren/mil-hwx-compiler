# Apple's chain partitioning and fusion rules for H13 and H14

## What this measures

`research/oracle-envelope.md` established the two facts a native scheduler
starts from: Apple never emits more than one HWX program, and it partitions
work by emitting more tasks inside that program. It measured five chain shapes
at a handful of sizes, which is enough to see that `x + relu(x)` collapses to
one task while `x + gelu(x)` costs two, and not enough to schedule a
transformer. It never ran a projection, a bias, a scaled score matrix, a
pre-norm block, or two stacked blocks, and it never said what any individual
task in a chain actually was.

This campaign mints the chains a scheduler has to place, at the sizes a
transformer uses — `d_model` in {256, 512, 768, 1024}, heads in {4, 8, 12},
sequence in {64, 128, 256, 512}, feed-forward 4x — and answers four questions
per chain: how many tasks Apple emits and what each task is, which
intermediates get a declared surface, which adjacent operations fuse into one
task and which register words carry the fusion, and where the ceiling is.

The headline results:

- **Fusion is a post-operation field, not a rewrite.** A matmul followed by
  `relu`, `gelu`, `silu`, a per-channel bias, a bias and a `relu` together, or
  a multiply by a scalar all cost exactly as many tasks as the matmul alone.
  One register word changes: `0x0c804` on H13, `0x00d04` on H14, in the NE
  block. A scalar multiply is even cheaper — it lands in the output-scale word
  `0x0c810` / `0x00d10` as an fp16 multiplier and changes nothing else.
- **Nothing else fuses.** A second matmul, a `layer_norm`, a `softmax`, or a
  residual `add` against a runtime tensor each cost their own tasks. Fusion
  reaches exactly one operation forward from a matmul, and only into the
  post-operation field.
- **Task order is MIL dataflow order, and stacks are exactly linear.** A
  projected block at `d_model=256` is 25 tasks on H13 and 24 on H14; two
  blocks are 50 and 48; eight blocks are 200 and 192. On H13 the task-length
  sequence of a stack repeats with a period of exactly one block. There is no
  cross-block fusion and no common-subexpression elimination of the compute.
- **No intermediate is ever declared.** Every chain declares only its inputs
  and its result; every intermediate lives in the `__DATA/__bss` scratch below
  the surfaces, whose size Apple reuses across blocks — a 1-block and an
  8-block stack at the same shape both allocate 49,152 bytes.
- **The ceiling is not depth, task count, or weight bytes.** Apple accepted
  8,192 stacked weightless units, 512 stacked blocks with a feed-forward and
  an attention each (10,240 tasks), 64 stacked fully projected blocks, and a
  402,786,304-byte constant section. The one refusal in this campaign is
  a chain built on a matmul geometry the envelope campaign already found
  refused on its own: `M=128, K=8192, N=8192` returns `callback_status=1` on
  both targets, while the same chain at `M=256` compiles to 385 tasks and a
  268,959,744-byte constant section. Chain acceptance is the conjunction of
  per-operation acceptance.

## Provenance

Every record was minted on `MacStudio.local`, Apple M1 Ultra, macOS 26.6.2
arm64, with Apple's compiler `/tmp/h13-oracle/bin/ane-compile-hwx`, SHA-256
`b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe`. The
campaign source is `research/mint_chain_probes.py`; the record writer and HWX
parser are `research/mint_oracles.py`, SHA-256
`a6e8f70d438973e60913815be66dbd6de19d7200df9053a9efc2a615823d8074`, and the
task decoder is `research/h13_td.py`, SHA-256
`125da0249ca3c3ba0ca1f9f0cabfd1db54a965abb75f9f4cdcc2598e255c8fbc`. Every
record carries its own tool, driver and decoder hashes, its own source commit,
and the compiler's status string, so the provenance is checkable per file.

    python3 research/mint_chain_probes.py --host macstudio --targets h13 h14
    python3 research/mint_chain_probes.py --report --targets h13 h14

The records are `research/oracles/{h13,h14}/chain_*.json`, 127 cases per
target. No Apple HWX bytes are retained: a record keeps the MIL, the weight
description, the decoded task words, the program and tensor descriptors,
constant-section hashes and sizes, and the compiler's status. Chains above
eight blocks are recorded as `chain_ceiling` summaries — Apple's verdict,
compile time, HWX size, task count and constant size, without the task words —
because the decoded words of a 36,864-task object run to 227 MB of JSON and
say nothing the shallower stacks have not already said.

### One correctness note about the weights

Apple stores one copy of a constant that repeats. The first pass of this
campaign pointed every BLOBFILE constant at blob offset 64, as the earlier
campaigns do, and a projected block whose six weight matrices declare
1,572,864 bytes packed a 1,183,744-byte constant section — the three
identical `[256,256]` projections stored once. Every case here therefore gives
each constant its own metadata record and its own payload region with its own
fp16 value (bit patterns `0x3400 + index`), and the same block now packs
1,576,960 bytes. Deduplication removes constants, not compute: the task count
was 25 both before and after.

## Coverage and outcome

| Family | Cases per target | Decoded per target | Refused per target | What it isolates |
|---|---:|---:|---:|---|
| `chain_base` | 9 | 9 | 0 | One operation at the chain geometry, so a pair can be diffed against its own parts |
| `chain_pair` | 15 | 15 | 0 | One operation plus exactly one neighbour |
| `chain_ffn` | 18 | 18 | 0 | `matmul -> activation -> matmul`, with and without bias |
| `chain_norm` | 10 | 10 | 0 | `layer_norm -> projection` |
| `chain_attention` | 13 | 13 | 0 | `QK^T -> scale -> softmax -> PV`, single and batched heads |
| `chain_residual` | 12 | 12 | 0 | `x + f(x)` around an activation, a feed-forward, and an attention |
| `chain_block` | 8 | 8 | 0 | Whole pre-norm blocks, with and without projections |
| `chain_stack` | 8 | 8 | 0 | Two to eight stacked blocks, with task words |
| `chain_order` | 2 | 2 | 0 | Two projections of one input, parallel against chained |
| `chain_ceiling` | 32 | 30 accepted | 1 refused, 1 timed out | Depth, weight bytes and per-operation envelope, as summaries |

95 chains per target are recorded with full task words, of which 86 hold more
than one operation. The two targets accept and refuse exactly the same cases.

Two results from the ceiling family are worth reading before the tables, because
both correct something written elsewhere:

- `chain_deep16384_d256_s64` is recorded as **not accepted for a reason that is
  this campaign's, not Apple's**: the compiler ran past
  `COMPILE_TIMEOUT_SECONDS` (900 s) on both targets and was killed, after
  depth 8,192 had compiled in 740 s. It is a timeout, not a refusal, and the
  record's `compiler.status` says `compiler timed out` so the two cannot be
  confused.
- `transpose_y=false` against a BLOBFILE constant weight is **accepted inside a
  chain** at three geometries, two of which `research/h13-td-fields.md` records
  as refused when the same matmul stands alone with
  `ANE internal validation error: Metadata data type does not match requested
  type.`:

| Probe | (M, K, N) | As a lone matmul (`h13-td-fields.md`) | As the first matmul of `matmul -> gelu -> matmul` |
|---|---|---|---|
| `chain_ceiling_mlp_m64_k256_n256_ty0` | 64, 256, 256 | refused at M=1 | accepted, 3 / 3 tasks |
| `chain_ceiling_mlp_m32_k2048_n2048_ty0` | 32, 2048, 2048 | refused | accepted, 4 / 3 tasks |
| `chain_ceiling_mlp_m128_k2048_n2048_ty0` | 128, 2048, 2048 | refused | accepted, 3 / 3 tasks |

  The geometry is therefore not what Apple rejects. What differs besides the
  chain context is the weight layout — these constants are two distinct regions
  of one shared `weights.bin` — so this campaign cannot say which of the two
  flips the verdict. It is a single-operation question and belongs in a
  re-mint of the `rrmm` probes, not here; until then, a scheduler should not
  treat `transpose_y=false` as refused outright.

## 1. How a task is labelled

A chain task is labelled by matching its register words against the
single-operation corpus, so a label is a measurement rather than a guess. The
procedure is derived in three steps, all in `--report`:

1. **Find the words that carry operation identity.** Group every decoded
   single-operation task by (input shape, task count, task index). Inside such
   a group the shape and the task's role are fixed, so the operation is the
   only difference; any word that moves is an operation word. 115 words
   qualify on H13.
2. **Build a signature per operation and role.** For each (operation, task
   count, task index) keep the operation words that every observation agrees
   on — a word that tracks the geometry falls out on its own. Just under 400
   signatures survive on H13 (397 against the corpus as of this run; the count
   moves as other campaigns add single-operation records, the derived words do
   not). A signature below 64 words is discarded, because the matmul staging
   role alone contributes hundreds of groups and its thinnest agreement is 28
   words and would match anything.
3. **Score, and name every winner.** A chain task takes the label of the
   best-scoring signature, and if several operations tie they are all named.

Two honest limits follow from the method and matter for the scheduler:

- **The task stream does not distinguish one lookup-table unary from
  another.** `chain_base_gelu_d256_s64` and `chain_base_silu_d256_s64` decode
  to identical words and both label `gelu|silu`; only the 128-byte table in
  the constant section separates them. The same holds for `exp`, `sigmoid`,
  `tanh` and `sqrt` in the corpus.
- **A fused or geometry-shifted task scores below 1.0.** The report prints the
  score and the words that missed, which is exactly the fusion evidence in
  section 2. Labels below 1.0 are marked with `~score` in the report's task
  sequences and should be read as "closest single operation", not as identity.

## 2. What fuses into one task

Every row below is a byte-exact diff between a two-operation chain and the
same first operation minted alone at the same geometry, `[1, 64, 256]` with a
`[256, 256]` BLOBFILE weight. "Changed words" counts every header and register
word that differs across the tasks the two programs share.

| Pair | Chain | Tasks fused H13/H14 | Lone first + lone second | Const fused - lone first | Post-op word `0x0c804`/`0x00d04` (lone -> fused) | Changed words H13/H14 | One task? |
|---|---|---:|---:|---:|---|---:|---|
| `mm_relu` | matmul then relu | 2 / 2 | 2 + 1 | +0 | `0x00101c00` -> `0x00111c00` | 1 / 1 | yes |
| `mm_gelu` | matmul then gelu | 2 / 2 | 2 + 1 | +2048 | `0x00101c00` -> `0x00121c00` | 32 / 32 | yes |
| `mm_silu` | matmul then silu | 2 / 2 | 2 + 1 | +2048 | `0x00101c00` -> `0x00121c00` | 32 / 32 | yes |
| `mm_bias` | matmul then per-channel bias | 2 / 2 | 2 + 2 | +1024 | `0x00101c00` -> `0x00101c10` | 32 / 32 | yes |
| `mm_bias_relu` | matmul, bias, then relu | 2 / 2 | 2 + 1 | +1024 | `0x00101c00` -> `0x00111c10` | 32 / 32 | yes |
| `mm_mul` | matmul then multiply by fp16 1/8 | 2 / 2 | 2 + 1 | +0 | unchanged | 1 / 1 | yes |
| `mm_add` | matmul then add a runtime tensor | 4 / 4 | 2 + 1 | +0 | `0x00101c00` -> `0x00101c0c` | 108 / 91 | no (+2 over first) |
| `mm_mm` | matmul then matmul | 3 / 3 | 2 + 2 | +131072 | unchanged | 18 / 14 | no (+1 over first) |
| `gelu_mm` | gelu then matmul | 3 / 3 | 1 + 2 | +131072 | unchanged | 19 / 13 | no (+2 over first) |
| `mul_mm` | multiply by a scalar then matmul | 3 / 3 | 1 + 2 | +114688 | `0x00100000` -> `0x00101c0c` | 40 / 35 | no (+2 over first) |
| `ln_gelu` | layer_norm then gelu | 4 / 4 | 3 + 1 | -16256 | unchanged | 21 / 18 | no (+1 over first) |
| `ln_ln` | layer_norm then layer_norm | 6 / 6 | 3 + 3 | +0 | unchanged | 17 / 18 | no (+3 over first) |
| `add_ln` | matmul, add, then layer_norm | 6 / 6 | 2 + 3 | +0 | unchanged | 40 / 28 | no (+4 over first) |
| `mm_softmax` | matmul then softmax | 7 / 7 | 2 + 6 | +128 | unchanged | 21 / 21 | no (+5 over first) |
| `softmax_mm` | softmax then matmul | 8 / 8 | 6 + 2 | +131072 | unchanged | 17 / 13 | no (+2 over first) |

### The words that carry the fusion

`0x0c804` (H13) and `0x00d04` (H14) hold the same values, so one rule covers
both targets. Read against the lone operations at this geometry:

| Meaning | `0x0c804` / `0x00d04` | Evidence |
|---|---|---|
| matmul, no post-operation | `0x00101c00` | `chain_base_matmul_d256_s64` |
| binary elementwise, no post-operation | `0x00100000` | `chain_base_add_d256_s64`, `chain_base_mul_scalar_d256_s64` |
| clamp / `relu` post-operation | bit `0x00010000` set | `mm_relu` `0x00111c00`; lone `relu` `0x00111c0c` |
| lookup-table unary post-operation | bit `0x00020000` set | `mm_gelu`, `mm_silu` `0x00121c00`; lone `gelu` and `silu` `0x00121c0c` |
| per-channel bias | low nibble `0x10` | `mm_bias` `0x00101c10` |
| bias and clamp together | both | `mm_bias_relu` `0x00111c10` |

The low byte differs between a fused post-operation (`0x00`) and the same
operation standing alone (`0x0c`), which is why a fused task scores 0.99
rather than 1.0 against the lone-unary signature: the post-operation field
selects the function while the standalone task also configures its own input
plumbing.

The output scale is a separate word, `0x0c810` on H13 and `0x00d10` on H14.
A matmul alone carries fp16 1.0 (`0x00003c00`); `mm_mul` carries fp16 0.125
(`0x00003000`), the exact scalar the MIL multiplied by, and no other word in
the program changes. That is the one word an attention scheduler needs for the
`1/sqrt(d_head)` scaling, and section 4 confirms the scaling is free in a real
attention chain too.

A fused lookup table or bias also extends the kernel-DMA descriptor block —
`0x1f848`-`0x1f8c4` on H13, `0x0195c`-`0x019d4` on H14. Each of the 16 lane
offsets grows by the table or bias size (`0x80` per lane for the 2,048-byte
gelu table, `0x40` for the 1,024-byte bias) and each of the 16 lane sizes
grows by the same amount, which is the whole of the "32 changed words" in
those rows: 1 post-operation word plus 31 DMA words.

### What the pair table means for a scheduler

- Fusion reaches **one operation forward from a matmul**, and only into the
  post-operation field: clamp, a lookup-table unary, a per-channel bias, an
  output scale, or a bias-plus-clamp pair.
- A residual `add` against a runtime tensor is **not** a post-operation: it
  costs two extra tasks and rewrites 108 words. The `x + relu(x)` collapse
  the envelope campaign found is the elementwise engine's own clamp, not a
  general add fusion — `chain_res_relu_d256` is 1 task, but
  `chain_pair_mm_add_d256_s64` is 4.
- `layer_norm` and `softmax` never fuse in either direction.
- Feeding a matmul is never free: `gelu_mm`, `mul_mm` and `softmax_mm` all
  pay for the producer separately.
- A `layer_norm` whose 16 KiB all-zero constant section meets an operation
  with a real table keeps only the table (`ln_gelu`, -16,256 bytes). The
  16 KiB block is a floor, not an addend.

## 3. Task-count laws per chain family

### Feed-forward, `matmul -> activation -> matmul`

| Case | Tasks H13/H14 | H13 task sizes | Weight bytes | Const H13/H14 | Scratch H13/H14 |
|---|---:|---:|---:|---:|---:|
| `chain_ffn_d256_s64` | 4 / 3 | 504x2,628x2 | 1048576 | 1050624 / 1050624 | 131072 / 0 |
| `chain_ffn_d256_s128` | 3 / 3 | 504x1,628x2 | 1048576 | 1050624 / 1050624 | 65536 / 0 |
| `chain_ffn_d512_s64` | 4 / 3 | 504x2,628x2 | 4194304 | 4196352 / 4196352 | 262144 / 0 |
| `chain_ffn_d512_s128` | 3 / 3 | 504x1,628x2 | 4194304 | 4196352 / 4196352 | 131072 / 0 |
| `chain_ffn_d768_s64` | 4 / 3 | 504x2,628x2 | 9437184 | 9439232 / 9439232 | 393216 / 0 |
| `chain_ffn_d768_s128` | 3 / 3 | 504x1,628x2 | 9437184 | 9439232 / 9439232 | 196608 / 0 |
| `chain_ffn_d1024_s64` | 4 / 3 | 504x2,628x2 | 16777216 | 16779264 / 16779264 | 524288 / 0 |
| `chain_ffn_d1024_s128` | 3 / 3 | 504x1,628x2 | 16777216 | 16779264 / 16779264 | 262144 / 0 |
| `chain_ffn_d256_s256` | 3 / 3 | 504x1,628x2 | 1048576 | 1050624 / 1050624 | 131072 / 0 |
| `chain_ffn_d256_s512` | 3 / 3 | 504x1,628x2 | 1048576 | 1050624 / 1050624 | 262144 / 0 |
| `chain_ffn_d512_s256` | 6 / 6 | 504x1,628x5 | 4194304 | 4200448 / 4200448 | 262144 / 0 |
| `chain_ffn_d256_s64_relu` | 4 / 3 | 504x2,628x2 | 1048576 | 1048576 / 1048576 | 131072 / 0 |
| `chain_ffn_d256_s64_silu` | 4 / 3 | 504x2,628x2 | 1048576 | 1050624 / 1050624 | 131072 / 0 |
| `chain_ffn_d256_s64_biasbcast` | 4 / 3 | 504x2,628x2 | 1051136 | 1053696 / 1053696 | 131072 / 0 |
| `chain_ffn_d512_s128_biasbcast` | 3 / 3 | 504x1,628x2 | 4199424 | 4201472 / 4201472 | 131072 / 0 |
| `chain_ffn_d768_s128_biasbcast` | 3 / 3 | 504x1,628x2 | 9444864 | 9447424 / 9447424 | 196608 / 0 |
| `chain_ffn_d256_s64_biasfull` | 7 / 7 | 504x4,628x3 | 1212416 | 1212544 / 1212544 | 131072 / 0 |
| `chain_ffn_d512_s128_biasfull` | 7 / 7 | 504x4,628x3 | 4849664 | 4849792 / 4849792 | 524288 / 0 |

The whole block is **two compute tasks** on both targets: the activation is a
post-operation on the first matmul, which is why `relu`, `gelu` and `silu`
give the same task count and differ only in the constant section (a `relu`
adds nothing, a lookup-table unary adds 2,048 bytes on both targets). A
per-channel bias is also free in tasks and costs `ceil(2 * C / 1024) * 1024`
bytes of constant section; a bias shaped like the whole activation is not a
per-channel bias and costs 4 extra tasks. Task counts do not move with
`d_model` at all, only with the sequence length and only on H13, which needs
one extra 126-word task at `s=64`.

### `layer_norm` feeding a projection

| Case | Tasks H13/H14 | H13 task sizes | Weight bytes | Const H13/H14 | Task bytes H13/H14 |
|---|---:|---:|---:|---:|---:|
| `chain_lnproj_d256_s64` | 5 / 5 | 504x4,628x1 | 131072 | 131072 / 131072 | 2676 / 876 |
| `chain_lnproj_d256_s128` | 5 / 5 | 504x4,628x1 | 131072 | 131072 / 131072 | 2676 / 876 |
| `chain_lnproj_d512_s64` | 5 / 5 | 504x4,628x1 | 524288 | 524288 / 524288 | 2676 / 876 |
| `chain_lnproj_d512_s128` | 5 / 5 | 504x4,628x1 | 524288 | 524288 / 524288 | 2676 / 876 |
| `chain_lnproj_d768_s64` | 5 / 5 | 504x4,628x1 | 1179648 | 1179648 / 1179648 | 2676 / 876 |
| `chain_lnproj_d768_s128` | 5 / 5 | 504x4,628x1 | 1179648 | 1179648 / 1179648 | 2676 / 876 |
| `chain_lnproj_d1024_s64` | 5 / 5 | 504x4,628x1 | 2097152 | 2097152 / 2097152 | 2676 / 876 |
| `chain_lnproj_d1024_s128` | 5 / 5 | 504x4,628x1 | 2097152 | 2097152 / 2097152 | 2676 / 876 |
| `chain_lnproj_d512_s256` | 7 / 7 | 504x6,628x1 | 524288 | 524288 / 524288 | 3700 / 1324 |
| `chain_lnproj_d512_s512` | 9 / 9 | 504x8,628x1 | 524288 | 524288 / 524288 | 4724 / 1644 |

Five tasks for every width, and two more per doubling of the sequence above
128: 5, 7, 9 at `s` of 128, 256, 512. The projection contributes one 157-word
compute task; the rest are the normalization's own 126-word tasks. The
constant section is exactly the weight bytes — the `layer_norm`'s 16 KiB
all-zero section is absorbed.

### Attention, `QK^T -> scale -> softmax -> PV`

| Case | Tasks H13/H14 | H13 task sizes | Const H13/H14 | Scratch H13/H14 | Surfaces |
|---|---:|---:|---:|---:|---:|
| `chain_attn_s64_dh64` | 9 / 9 | 504x6,628x1,632x2 | 2048 / 128 | 16384 / 0 | 4 |
| `chain_attn_s64_dh96` | 9 / 9 | 504x6,628x1,632x2 | 2048 / 128 | 16384 / 0 | 4 |
| `chain_attn_s128_dh64` | 9 / 9 | 504x6,628x1,632x2 | 2048 / 128 | 32768 / 0 | 4 |
| `chain_attn_s128_dh96` | 9 / 9 | 504x6,628x1,632x2 | 2048 / 128 | 32768 / 0 | 4 |
| `chain_attn_s256_dh64` | 11 / 9 | 504x6,628x1,632x4 | 2048 / 128 | 131072 / 0 | 4 |
| `chain_attn_s256_dh96` | 11 / 9 | 504x6,628x1,632x4 | 2048 / 128 | 131072 / 0 | 4 |
| `chain_attn_s512_dh64` | 15 / 9 | 504x6,628x1,632x8 | 2048 / 128 | 524288 / 0 | 4 |
| `chain_attn_s512_dh96` | 15 / 9 | 504x6,628x1,632x8 | 2048 / 128 | 524288 / 0 | 4 |
| `chain_attn_s128_dh64_h4` | 18 / 9 | 504x9,628x1,632x8 | 2048 / 128 | 131072 / 0 | 4 |
| `chain_attn_s128_dh64_h8` | 30 / 9 | 504x13,628x1,632x16 | 2048 / 128 | 262144 / 0 | 4 |
| `chain_attn_s128_dh64_h12` | 42 / 9 | 504x17,628x1,632x24 | 2048 / 128 | 393216 / 0 | 4 |
| `chain_attn_s128_dh64_noscale` | 9 / 9 | 504x6,628x1,632x2 | 2048 / 128 | 32768 / 0 | 4 |
| `chain_attn_s256_dh64_noscale` | 11 / 9 | 504x6,628x1,632x4 | 2048 / 128 | 131072 / 0 | 4 |

- **The scale is free.** Dropping the `mul` by `1/8` changes neither the task
  count nor the constant section at either size, which is the whole-chain
  confirmation of the `0x0c810` finding in section 2.
- **The head dimension does not matter**; 64 and 96 give identical streams.
- **H14 is flat at 9 tasks** for every sequence length and every head count.
  H13 emits `9 + 2 * (S/128 - 1)` for S ≥ 128 — 9, 11, 15 at 128, 256, 512 —
  and `9 + 3 * (heads - 1)` when the heads arrive as a rank-3 batch: 18, 30,
  42 at 4, 8 and 12 heads.
- The constant section is the softmax table only: 2,048 bytes on H13,
  128 on H14, independent of size and head count.

### Residuals

| Case | Tasks H13/H14 | H13 task sizes | Const H13/H14 | Scratch H13/H14 |
|---|---:|---:|---:|---:|
| `chain_res_relu_d256` | 1 / 1 | 504x1 | 16384 / 16384 | 0 / 0 |
| `chain_res_relu_d1024` | 1 / 1 | 504x1 | 16384 / 16384 | 0 / 0 |
| `chain_res_gelu_d256` | 2 / 2 | 504x1,628x1 | 1024 / 128 | 0 / 0 |
| `chain_res_gelu_d1024` | 2 / 2 | 504x1,628x1 | 1024 / 128 | 0 / 0 |
| `chain_res_silu_d256` | 2 / 2 | 504x1,628x1 | 1024 / 128 | 0 / 0 |
| `chain_res_silu_d1024` | 2 / 2 | 504x1,628x1 | 1024 / 128 | 0 / 0 |
| `chain_resffn_d256_s64` | 5 / 5 | 504x3,628x2 | 1050624 / 1050624 | 0 / 0 |
| `chain_resffn_d256_s128` | 5 / 5 | 504x3,628x2 | 1050624 / 1050624 | 0 / 0 |
| `chain_resffn_d512_s64` | 5 / 5 | 504x3,628x2 | 4196352 / 4196352 | 0 / 0 |
| `chain_resffn_d512_s128` | 5 / 5 | 504x3,628x2 | 4196352 / 4196352 | 0 / 0 |
| `chain_resattn_s64_d256` | 10 / 10 | 504x7,628x1,632x2 | 2048 / 128 | 32768 / 0 |
| `chain_resattn_s128_d256` | 10 / 10 | 504x7,628x1,632x2 | 2048 / 128 | 65536 / 0 |

A residual around a whole sub-block costs exactly one extra task over the
sub-block itself: 10 against attention's 9, and 5 against the feed-forward's
3 or 4 — the same count at every width, and identical on both targets. The
`x + relu(x)` single task and the two-task `gelu`/`silu` residuals reproduce
the envelope campaign at these widths.

### Whole blocks and stacks

| Case | Tasks H13/H14 | H13 task sizes | Weight bytes | Const H13/H14 | Scratch H13/H14 | Task bytes H13/H14 |
|---|---:|---:|---:|---:|---:|---:|
| `chain_block_d256_s64_proj1` | 25 / 24 | 504x15,628x7,632x3 | 1572864 | 1576960 / 1575040 | 49152 / 0 | 15352 / 4872 |
| `chain_block_d256_s128_proj1` | 25 / 24 | 504x15,628x7,632x3 | 1572864 | 1576960 / 1575040 | 98304 / 0 | 15352 / 4872 |
| `chain_block_d512_s64_proj1` | 27 / 24 | 504x15,628x7,632x5 | 6291456 | 6295552 / 6293632 | 81920 / 0 | 16888 / 4872 |
| `chain_block_d512_s128_proj1` | 27 / 24 | 504x15,628x7,632x5 | 6291456 | 6295552 / 6293632 | 163840 / 0 | 16888 / 4872 |
| `chain_block_d768_s64_proj1` | 29 / 24 | 504x15,628x7,632x7 | 14155776 | 14159872 / 14157952 | 114688 / 0 | 18424 / 4872 |
| `chain_block_d768_s128_proj1` | 29 / 24 | 504x15,628x7,632x7 | 14155776 | 14159872 / 14157952 | 229376 / 0 | 18424 / 4872 |
| `chain_block_d256_s64_proj0` | 20 / 20 | 504x15,628x3,632x2 | 1048576 | 1052672 / 1050752 | 32768 / 0 | 11512 / 3528 |
| `chain_block_d512_s128_proj0` | 20 / 20 | 504x15,628x3,632x2 | 4194304 | 4198400 / 4196480 | 131072 / 0 | 11512 / 3528 |
| `chain_stack2_d256_s64_proj1` | 50 / 48 | 504x30,628x14,632x6 | 3145728 | 3151872 / 3149952 | 49152 / 0 | 30712 / 9672 |
| `chain_stack2_d256_s128_proj1` | 50 / 48 | 504x30,628x14,632x6 | 3145728 | 3151872 / 3149952 | 98304 / 0 | 30712 / 9672 |
| `chain_stack2_d512_s64_proj1` | 54 / 48 | 504x30,628x14,632x10 | 12582912 | 12589056 / 12587136 | 81920 / 0 | 34040 / 9672 |
| `chain_stack2_d512_s128_proj1` | 54 / 48 | 504x30,628x14,632x10 | 12582912 | 12589056 / 12587136 | 163840 / 0 | 34040 / 9672 |
| `chain_stack3_d256_s64_proj1` | 75 / 72 | 504x45,628x21,632x9 | 4718592 | 4726784 / 4724864 | 49152 / 0 | 46072 / 14472 |
| `chain_stack4_d256_s64_proj1` | 100 / 96 | 504x60,628x28,632x12 | 6291456 | 6301696 / 6299776 | 49152 / 0 | 61688 / 19272 |
| `chain_stack6_d256_s64_proj1` | 150 / 144 | 504x90,628x42,632x18 | 9437184 | 9451520 / 9449600 | 49152 / 0 | 92920 / 28872 |
| `chain_stack8_d256_s64_proj1` | 200 / 192 | 504x120,628x56,632x24 | 12582912 | 12601344 / 12599424 | 49152 / 0 | 123896 / 38472 |

A pre-norm block with four projections is 25 tasks on H13 at `d_model=256` and
two more per additional 256 of width (27 at 512, 29 at 768); H14 is 24 for
every width. Without the projections both targets emit 20. Depth multiplies
exactly: `tasks = depth * per-block`, verified at 2, 3, 4, 6 and 8 blocks with
full task words, and at 12 through 512 blocks in the summary records — 10,240
tasks for `chain_stack512_d128_s64_proj0` on both targets. The task-section
bytes scale the same way, and the scratch does not scale at all.

### Order, and what sharing an input buys

| Case | Tasks H13/H14 | H13 task sizes | Weight bytes | Const H13/H14 | Scratch H13/H14 |
|---|---:|---:|---:|---:|---:|
| `chain_base_matmul_d256_s64` | 2 / 2 | 504x1,628x1 | 131072 | 131072 / 131072 | 32768 / 0 |
| `chain_pair_mm_mm_d256_s64` | 3 / 3 | 504x1,628x2 | 262144 | 262144 / 262144 | 32768 / 0 |
| `chain_branch_d256_s64` | 4 / 4 | 504x2,628x2 | 262144 | 262144 / 262144 | 32768 / 0 |
| `chain_serial_d256_s64` | 5 / 5 | 504x3,628x2 | 262144 | 262144 / 262144 | 32768 / 0 |

`chain_branch` projects one input twice and adds the results; `chain_serial`
chains the two projections and adds. Same operations, same weights, same
constant section; the parallel form costs one task fewer, because the two
compute tasks share a single staging task for the input they both read. Task
order in both follows MIL dataflow order.

## 4. Surfaces, scratch, and where the intermediates live

Every chain in this campaign declares exactly its inputs and its result and
nothing else: 2 tensor descriptors for a single-input chain, 4 for attention's
three operands plus its output. No intermediate — not the 4x feed-forward
hidden state, not the `[S,S]` score matrix, not a block's post-residual
activation — ever gets a tensor descriptor or a resource address.

The intermediates live in the `__DATA/__bss` scratch that sits below the
declared surfaces, at `0x30000000`. Its size is the first resource address
minus that base, and the measured behaviour is:

- **H14 allocates no scratch at all** in any chain in this campaign. Every
  H14 case reports 0.
- **H13's scratch tracks the largest live intermediate, and is reused.** A
  feed-forward at `d_model=256, s=64` holds a `[1,64,1024]` hidden state,
  131,072 bytes, and allocates exactly 131,072. At `d_model=1024, s=64` the
  hidden state is 524,288 bytes and the scratch is 524,288.
- **When Apple tiles, the scratch shrinks below the intermediate.** The same
  feed-forward at `s=128` is a 3-task program with 65,536 bytes of scratch for
  a 262,144-byte hidden state, a quarter of it; at `s=256` and `s=512` it is
  131,072 and 262,144. The tiling factor is visible but its rule is not
  derived here.
- **Depth does not add scratch.** One block and eight blocks at
  `d_model=256, s=64` both allocate 49,152 bytes: Apple reuses the same
  scratch for every block's intermediates. Scratch scales with the sequence
  length (49,152 at `s=64`, 98,304 at `s=128`) and with the width (81,920 at
  512, 114,688 at 768).
- A `layer_norm`-headed chain, a residual feed-forward and every elementwise
  chain allocate no scratch on either target.

## 5. Constant-section layout in a chain

The constant section of a chain is the concatenation of what its operations
need, with one dedup and one floor:

| Contribution | Bytes | Evidence |
|---|---|---|
| Each distinct BLOBFILE weight | `rows * columns * 2`, in Apple's packing | `chain_lnproj_*`: section equals the weight bytes exactly, at four widths |
| A lookup-table unary fused into a matmul | 2,048 on both targets | `mm_gelu`, `mm_silu`, `chain_ffn_*` (`+2048` over the `relu` variant) |
| A lookup-table unary standing alone | 128 on both targets | `chain_base_gelu_d256_s64`, `chain_base_silu_d256_s64` |
| A fused per-channel bias | `ceil(2 * C / 1024) * 1024`, over every bias in the chain | `mm_bias` 1,024 for C=256; `chain_ffn_*_biasbcast` 3,072 for 1024+256, 5,120 for 2048+512, 8,192 for 3072+768 — exact on both targets |
| A standalone per-channel add of a `[1,1,C]` constant | `2 * C`, unpadded | `chain_base_bias_d256_s64`: 512 bytes for C=256 |
| A softmax | 2,048 on H13, 128 on H14 | every `chain_attn_*` |
| A `layer_norm` or a runtime binary, alone | 16,384 zero bytes | `chain_base_layer_norm`, `chain_base_add`, `chain_res_relu` |
| A whole projected block | weights + 4,096 on H13, weights + 2,176 on H14 | `chain_block_*`, at three widths |

Two rules a scheduler must respect:

- **The 16 KiB all-zero block is a floor, not an addend.** `ln_gelu` is
  16,256 bytes *smaller* than a lone `layer_norm`, because the table replaces
  the zero block rather than following it.
- **Identical constants are stored once.** Section sizes here are only
  additive because every constant in this campaign is distinct by
  construction; the deduplicating measurement is in the provenance note above.

## 6. Task order and staging tasks

H13 emits two kinds of task in these chains: 126-word (504-byte) tasks and
157- or 158-word (628- or 632-byte) compute tasks. Written as `S` for a
126-word task and `C` for a compute task, in emission order:

| Case | Task sequence (H13) |
|---|---|
| `chain_base_matmul_d256_s64` | `S C` |
| `chain_ffn_d256_s64` | `S C C S` |
| `chain_ffn_d256_s128` | `S C C` |
| `chain_lnproj_d256_s64` | `S S S S C` |
| `chain_attn_s64_dh64` | `S S C S S C S S C` |
| `chain_branch_d256_s64` | `S C C S` |
| `chain_serial_d256_s64` | `S C S C S` |
| `chain_block_d256_s64_proj0` | `S S S S S C S S C S S C S S S S S C C S` |
| `chain_block_d256_s64_proj1` | `S S S S C C C S S C S S S C C C C S S S S S C C S` |

The pattern behind it: a compute task is preceded by a 126-word staging task
that brings its operands into L2, consecutive compute tasks that read what is
already resident share one staging task (`S C C` in the feed-forward, and the
task the parallel branch saves), and at `s=64` a trailing 126-word task
writes the result surface back. The labelled sequences in `--report` follow
MIL dataflow order in every case: a block's tasks appear as normalization,
projections, scores, softmax, context, residual, normalization, feed-forward,
residual, in that order.

On H13 the task-length sequence of a stack is exactly periodic with period one
block — 25 lengths repeating 2, 4 and 8 times in `chain_stack2`, `chain_stack4`
and `chain_stack8`. H14's is not periodic, because its variable-length tasks
differ in the first and last block where real surfaces replace scratch.

## 7. The ceiling

Depth, task count and weight bytes were all pushed until either the compiler
refused or this repository's decoder gave up. Apple's verdict is recorded
separately from the decode result, because the two are not the same event.

| Probe | Depth | Weight bytes | Verdict | Tasks H13/H14 | HWX bytes H13 | Constant section H13/H14 | Compile seconds H13/H14 |
|---|---:|---:|---|---:|---:|---:|---|
| `chain_stack32_d256_s64_proj1` | 32 | 50,331,648 | accepted | 800 / 768 | 51,216,384 | 50,399,232 / 50,397,312 | 3.1 / 3.0 |
| `chain_stack64_d256_s64_proj1` | 64 | 100,663,296 | accepted | 1,600 / 1,536 | 102,416,384 | 100,796,416 / 100,794,496 | 5.9 / 5.7 |
| `chain_stack512_d128_s64_proj0` | 512 | 134,217,728 | accepted | 10,240 / 10,240 | 142,917,632 | 135,268,352 / 135,266,432 | 33.6 / 34.8 |
| `chain_ceiling_weights64_d512` | 64 | 402,653,184 | accepted | 1,728 / 1,536 | 404,504,576 | 402,786,304 / 402,784,384 | 7.3 / 7.7 |
| `chain_deep2048_d256_s64` | 2,048 | 0 | accepted | 36,864 / 36,864 | 21,495,808 | 2,176 / 256 | 129.9 / 118.9 |
| `chain_deep4096_d256_s64` | 4,096 | 0 | accepted | undecoded | 42,975,232 | undecoded | 281.1 / 271.1 |
| `chain_deep8192_d256_s64` | 8,192 | 0 | accepted | undecoded | 85,934,080 | undecoded | 740.5 / 694.6 |
| `chain_ceiling_mlp_m256_k8192_n8192_ty1` | 1 | 268,435,456 | accepted | 385 / 385 | 269,877,248 | 268,959,744 / 268,959,744 | 0.6 / 0.7 |
| `chain_ceiling_mlp_m128_k8192_n8192_ty1` | 1 | 268,435,456 | **refused** | — | — | — | 0.0 / 0.0 |
| `chain_ceiling_mlp_m128_k2048_n2048_ty0` | 1 | 16,777,216 | accepted | 3 / 3 | 16,809,984 | 16,779,264 / 16,779,264 | 0.2 / 0.2 |
| `chain_deep16384_d256_s64` | 16,384 | 0 | timed out here at 900 s | — | — | — | >900 / >900 |

- **The largest accepted chain in this campaign is 8,192 stacked block units**
  — 73,728 MIL operations — which compiles in 740 seconds into an
  85,934,080-byte object on H13. Every decoded object in the campaign, up to
  36,864 tasks and a 402,786,304-byte constant section, carries exactly one
  program load command; the two undecoded objects and the timed-out one are
  not counted in that claim.
- **The only refusal is inherited from the per-operation envelope.** A
  feed-forward built on `M=128, K=8192, N=8192` returns `callback_status=1` on
  both targets, with no validation report — the same status the envelope
  campaign recorded for that geometry as a lone matmul. The same chain at
  `M=256` is accepted, so the refusal is the matmul's, not the chain's. A
  scheduler therefore needs no chain-level size check beyond the
  per-operation envelope it already has.
- **Two of these "accepted" objects do not decode here.** At depth 4,096 and
  8,192 the compiler exits 0 and writes the object, and
  `research/mint_oracles.py`'s task splitter then fails with
  `H13 task[0] extends beyond __TEXT/__text` on H13 and
  `program[0] declares 53638 tasks but decoded 0` on H14. That is this
  repository's limit, not Apple's, and it is why the `chain_ceiling` family
  records `accepted` and `parse_error` as separate fields. Fixing the splitter
  for multi-megabyte task sections is outside this campaign.
- **What actually stops this search is compile time, not acceptance.** Compile
  seconds scale roughly linearly in depth — 130 s at 2,048, 281 s at 4,096,
  740 s at 8,192 — so depth 16,384 exceeded the 900-second budget in
  `mint_oracles.py` on both targets and is recorded as `compiler timed out`.
  Apple's own bound on depth is therefore still unmeasured, and it is above
  8,192 blocks.
- **The refusal is per-operation, but "refused as a lone matmul" is not
  sufficient to predict it.** A `transpose_y=false` BLOBFILE matmul that
  `research/h13-td-fields.md` records as refused at `(32, 2048, 2048)` and
  `(128, 2048, 2048)` compiles inside a feed-forward chain (three tasks,
  16,779,264 bytes of constants, both targets). The coverage section above
  records this in full; the practical consequence for a scheduler is that the
  envelope check must be re-derived from single-operation probes that use the
  same weight layout the scheduler will emit.

## 8. A native scheduling algorithm

### What the single-operation templates already cover

`plugins/H13/H13NormTemplates.inc`, `plugins/H13/H13EnvelopeTemplates.inc` and
their H14 counterparts already carry, per covered geometry: the exact task
stream for one operation, the surface layout (`row = max(64, W * 2)`,
`plane = row * H`, `element = plane * C`), the descriptor and address ordering
per family, the constant packing for a weight matrix, the lookup-table offsets,
the per-channel bias layout, and the scratch size. For a chain, those
are the per-operation bodies — every measurement in this document says Apple
emits, for each operation in a chain, the same task stream it emits for that
operation alone, up to the fusion field and the surface addresses.

### What a chain encoder must add

1. **Concatenate in dataflow order.** Emit the operations in MIL topological
   order — measured in every case in section 6 — and relink the stream: on
   H13 set each task's `header[7]` to the next task's offset and
   `header[1][24:16]` to the next task's word count minus one; on H14 pack
   tasks to a 16-byte alignment. The program descriptor's task count and
   `task_words_minus_one` follow from the first task.
2. **Apply the fusion field instead of emitting the consumer.** When a matmul
   is followed by exactly one of `relu`, a lookup-table unary, a per-channel
   bias add, a multiply by a scalar, or a bias-and-`relu` pair, drop the
   consumer's tasks and instead, in the matmul's compute task:
   - set bit `0x00010000` of `0x0c804` / `0x00d04` for a clamp,
   - set bit `0x00020000` for a lookup-table unary,
   - set the low nibble to `0x10` for a per-channel bias,
   - write the fp16 multiplier into `0x0c810` / `0x00d10` for a scalar
     multiply (fp16 1.0 when there is none),
   - append the 2,048-byte table, or the bias padded to a 1,024-byte
     multiple, to the constant section, and extend all 32 kernel-DMA words
     (`0x1f848`-`0x1f8c4` on H13, `0x0195c`-`0x019d4` on H14) by the
     appended size.
   Fuse nothing else: a residual add against a runtime tensor, a second
   matmul, a `layer_norm` and a `softmax` each keep their own tasks.
3. **Place intermediates in scratch, and declare none of them.** Give tensor
   descriptors and resource addresses only to the chain's inputs and result.
   Size the `__DATA/__bss` scratch from the largest live intermediate for the
   untiled forms, and reuse one scratch allocation for every block in a stack.
   H14 needs none. Where a template exists for the same shape, take the
   measured scratch from the record rather than the formula — section 4 shows
   the tiled forms allocate a fraction of the intermediate and that fraction
   is not derived here.
4. **Insert the staging tasks.** Emit a 126-word staging task before a compute
   task whose operands are not already resident, share one staging task
   between consecutive compute tasks that read the same operands, and emit the
   trailing 126-word write-back task in the forms that have one (`s=64` in the
   feed-forward family). This is the one part of the encoding where the
   observed pattern is a rule of thumb rather than a derived predicate; the
   safe implementation keys the staging layout to a measured template per
   covered chain shape.
5. **Predict the task budget from the per-family laws.** Section 3 gives them:
   two compute tasks per feed-forward, `5 + 2 * (S/128)` for a normalized
   projection above `s=128`, `9 + 2 * (S/128 - 1)` for H13 attention and a
   flat 9 on H14, one extra task for a residual around a sub-block,
   `depth * per-block` for a stack, and `9 + 3 * (heads - 1)` for
   batch-dimension heads on H13.
6. **Refuse only what the per-operation envelope refuses.** Section 7 shows a
   chain inherits its operations' refusals and adds none of its own up to
   8,192 blocks, 10,240 tasks and 402 MB of weights. Validate each operation
   against its envelope and emit the chain.

## 9. Unresolved

- The tiling factor that makes a 3-task feed-forward allocate a quarter of its
  hidden state as scratch, while the 4-task form allocates all of it.
- The predicate for the trailing 126-word task: present at `s=64`, absent at
  `s=128` and above in the feed-forward family.
- Why H13 needs `2 * (S/128 - 1)` extra attention tasks and 3 extra tasks per
  batched head where H14 needs none, and why a projected block costs H13 two
  extra tasks per 256 of width.
- The 4,096-byte (H13) and 2,176-byte (H14) constant-section overhead of a
  projected block: the softmax table accounts for 2,048 and 128 of it.
- Which lookup table a fused unary uses is not visible in the task words at
  all; only the constant section separates `gelu` from `silu`. An encoder must
  therefore carry the table identity out of band.
- The decoder cannot read a task section above roughly 21 MB, so the depth
  4,096 and 8,192 objects are recorded as accepted-but-undecoded.
