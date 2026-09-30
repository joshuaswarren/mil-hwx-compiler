# H14 model coverage: Parakeet islands + Qwen staged decode

Date: 2026-09-30. Actor: H14Batch2. Compiler worktree:
`/home/joshuawarren/src/mil-hwx-h14-integ-wt`, branch `agent/h14-integ`,
head `02ea9ba` (merge of `agent/h14-mint-model-gaps` @ `6d42002` +
`agent/h14-shape-ext` @ `ff13ce4`). Compiler binary:
`./build/mil-hwxc --target H14 --format anec`. GNUstep runtime at
`$HOME/.local/mil-hwx-gnustep/lib`. Parity receipt:
`make test-h14-parity` = **830 cases PASS** at head `02ea9ba`
(234 elementwise, 57 matvec, 14 known-weight matvec probes over 9
(K, N) grid points, 109 softmax/layer_norm, 114 reduction over 190
norm templates, 284 convolution, 14 island select/batched-matmul
(ANEC), 4 rms_norm chain (ANEC), 1642 artifacts).

Harness: `/var/tmp/qwen-h14-matrix/run_qwen_matrix.py` (38 staged Qwen
programs) and `/var/tmp/parakeet-h14-integ/run_parakeet_islands.py`
(25 coverage cases). Notebook archive:
`~/.local/share/apple-silicon-lab/artifacts/H14Integ/` (scripts,
results) and `~/.local/share/apple-silicon-lab/artifacts/H14Batch2/`
(this batch's mints). Pre-registered entry:
`~/.local/share/apple-silicon-lab/entries/H14Batch2/2026-09-30T0321Z-macstudio-h14-batch2-mint.md`.

## Definitions

A staged program counts as **compiled** only when the WHOLE program —
every op the staged MIL defines, fused as one graph — emits exactly
one ANEC package. When the H14 compiler lowers a program's ops as
separate per-op ANECs the row is **partial**; per-op programs do not
count as compiled. Compile/refused outcomes establish package
emit/accept, not on-chip behavior (parity is not device execution).

A program class is a distinct staged-graph section, derived from the
manifest lane/ctx/state port tables:

| Class | n_prog | srcs (lane + state_in + ctx) | dsts |
|---|---:|---|---|
| A-first (group_start) | 2 | `x` + state0 | `q,k,v,beta,gt,z` |
| A-next (carry residual + gate) | 11 | `x,o,z` + state0 | `x,q,k,v,beta,gt,z` |
| B-state (DeltaNet recurrence) | 18 | `q,k,v,beta,gt` + state0 | `o` |
| C-readout (group_end chunk 0) | 1 | `x,o,z` | `h` |
| D-attn (gated attention) | 5 | `x,o,z` + 5 ctx + 3 state | `x,q,k,v,beta,gt,z` |
| E-final (group_end chunk 1) | 1 | `x,o,z` + 5 ctx + 2 state | `h` |
| **Total** | **38** | | |

Program indexes per class (derived from manifest lane/state/ctx
counts): A-first = [0, 21]; A-next = [2, 4, 8, 10, 14, 16, 23, 27, 29,
33, 35]; B-state = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22, 24, 26, 28,
30, 32, 34, 36]; D-attn = [6, 12, 18, 25, 31]; C-readout = [20];
E-final = [37].

## Op families per class

The H14-mappable core (the ops the H14 dispatcher carries:
elementwise, unary, norm, matmul/linear, select, rms_norm chain;
reshape/slice/concat/softplus have no H14 encoder and refuse by
name). Each program's first-op refusal identifies its blocker.

### A-first / A-next (13 programs)

Required op sequence (host block, no carry residual for A-first;
`x·x` first op then full rms_norm chain for A-next):

1. `mul(x, x)` at `[1, 2048]` rank-2 (sum-of-squares lane)
2. `reduce_sum` at `[1, 2048]` axes=[1] (rank-2 sibling of `[1, 2048, 1, 1]`)
3. `mul` at `[1, 1]` with eps `[1, 1]`
4. `add` at `[1, 1]` to get sum+eps
5. `sqrt` at `[1, 1]` for `mean_eps`
6. `mul` at `[1, 1]` with `1/sqrt(mean_eps)` inverse
7. `mul(x, inv_rms)` at `[1, 2048]` for normalised
8. `mul(normalised, gamma)` at `[1, 2048]` with constant gamma
9. `qkv linear` at `[1, 2048] × [6144, 2048]`
10. `sigmoid(gate)` + `mul(z_gate)` at `[1, 6144, 1, 1]`
11. `silu(conv)` + `mul(ffn_gate)` at `[1, 6144, 1, 1]` (SwiGLU)
12. `ffn_down linear` at `[1, 6144] × [2048, 6144]`
13. `rms_norm_gated(o · z)` at `[1, 2048]` (head × head_dim = 128)

Refusal point: step 2 — `reduce_sum` at `[1, 2048]` axes=[1]. The norm
envelope decodes reductions at `[1, C, 1, 1]` for C ∈ 64..4096 but
not the rank-2 `[1, 2048]` spelling. Apple's own compiler rejected
this (`callback_status=1` per the H14Shape ext campaign).

**Family fix**: rewrite to rank-3 `[1, 16, 128]` axes=[1] (the
per-head form Apple accepts) by changing `x` from rank-2 to rank-3
`[1, 16, 128]` before the reduce; reshape back after. Same numerical
result, fp16 envelope already accommodates the tolerance.

### B-state (18 programs)

Required op sequence (DeltaNet recurrence):

1. `state·gt` broadcast `mul` at `[1, 16, 128, 128] × [16, 1, 1]` (state decay)
2. `matmul(k, state)` at `[1, 16, 1, 128] × [1, 16, 128, 128]` rank-4
   b16 → `[1, 16, 1, 128]` (rows=1)
3. `sub(v, kv)` at `[1, 16, 1, 128]`
4. `mul(delta, beta)` at `[1, 16, 1, 128] × [1, 16, 1, 1]` (broadcast
   per-head)
5. `matmul(k^T, delta)` at `[1, 16, 128, 1] × [1, 16, 1, 128]` rank-4
   b16 → `[1, 16, 128, 128]` (rows=128 outer product, ty=true)
6. `add(state', outer)` at `[1, 16, 128, 128]`
7. `matmul(q, state')` at `[1, 16, 1, 128] × [1, 16, 128, 128]` rank-4
   b16 → `[1, 16, 1, 128]` (rows=1)
8. `reshape` at `[1, 16, 128]` for output `o`

Refusal point: step 1 (broadcast mul not in decoded channel set) and
steps 2/5/7 (rank-4 b16 batched matmul has no row=1 / row=128 template;
Parakeet covers rows=375 only).

**Family fixes**:
- (F1) Broadcast `[1, 16, 128, 128] × [16, 1, 1]` mul — Apple
  decoder's `(C,1,1)` form covers C ∈ {1, 64, 96, 128, 200, 256, 300,
  512, 768, 1024, 2048, 3072, 4096, 8192, 16384}; 16 is in the set
  but the broadcast layout `[1, 16, 128, 128] × [16, 1, 1]` (head-major
  not channel-major) is not. Requires a per-head channel-broadcast
  Apple decode.
- (F3) Rank-4 b16 batched matmul: rows=1 (step 2), rows=128 outer
  product with `ty=true` (step 5), rows=1 (step 7). Parakeet's
  `kH14BatchedMatmulTasks` covers `rows=375` only.

### C-readout (1 program)

Required op sequence (chunk-0 final readout):

1. `sigmoid(gate)` at `[16, 128]` (or `[1, 16, 128, 1]`)
2. `mul(o, gate)` at `[16, 128] × [16, 128]`
3. `add(o, h_residual)` at `[16, 128] + [16, 128]` for `h`

Refusal: chain encoder supports exactly one producer + relu, so
`sigmoid → mul → add` does not fuse into one program; it lowers as
3 per-op ANECs at the merged build.

**Family fix**: chain template needs a 3-op producer (sigmoid → mul
→ add) — Apple-side decode of the chained form, separate from the
2-op sigmoid → mul chain at `(2048, 1, 1)` that's already in the
decoder.

### D-attn (5 programs)

Required op sequence (gated-attention with KV cache):

1. All of A-next (rms_norm + qkv + gates + conv + l2-norm)
2. `RoPE` (mul+add pair on q, k at `[1, 16, 1, 128]`)
3. `matmul(Q, K^T)` at `[1, 16, 1, 50] × [1, 16, 50, 128]` rank-4
   b16 → `[1, 16, 1, 50]`
4. `mul(scores, 1/sqrt(128))` at `[1, 16, 1, 50]`
5. `add(scores, mask)` at `[1, 16, 1, 50]` (mask `[1, 1, 1, 50]`
   broadcast)
6. `softmax` at `[16, 50]` axis=-1 (or `[1, 16, 1, 50]`)
7. `matmul(probs, V)` at `[1, 16, 1, 50] × [1, 16, 50, 128]` rank-4
   b16 → `[1, 16, 1, 128]`
8. `sigmoid(gate)` + `mul(o, gate)`
9. `o_proj linear` at `[1, 2048] × [2048, 2048]`

Refusal point: step 1's reduce_sum (same as class A); plus steps
3/7 (no batched matmul decoder at rows=1 b16); plus step 6
upstream chain (bare softmax at `[16, 50]` already compiles — only
the upstream chain does not).

**Family fixes**: F1+F2+F3 plus F4 (softmax attention pipeline at
`[16, 50]`).

### E-final (1 program)

Required op sequence: A-next base + C-readout, with full attention
pipeline baked (no RoPE / KV cache — this is chunk 1 final).

Refusal point: same as A-next (reduce_sum at rank-2).

**Family fix**: F2 alone unblocks E-final.

## Smallest family set completing whole programs (per class)

A program is "completed by family set F" iff every op in its MIL
matches a decoded shape/operand under F. The smallest F that
completes ≥ 1 program per class:

| Class | Smallest family set to complete ≥ 1 program | n_prog completed |
|---|---|---:|
| A-first / A-next | F2 (reduce_sum rank-3 [1,16,128] axes=[1] + Qwen residual elementwise at 2048/4096/6144) | 13 |
| B-state | F1 (broadcast [1,16,128,128] × [16,1,1] mul) + F3 (b16 rank-4 batched matmul rows=1, 128) | 18 |
| C-readout | F5 (3-op chain sigmoid → mul → add at (16, 128)) | 1 |
| D-attn | F2 + F3 + F4 (softmax attention pipeline at [16,50]) | 5 |
| E-final | F2 (the class carries through A-next + C-readout base) | 1 |
| **Whole staged Qwen matrix** | F1 ∪ F2 ∪ F3 ∪ F4 ∪ F5 | **38** |

Each family corresponds to one Apple-mint batch:

- **F1**: 1 oracle — `binary_mul [1,16,128,128] × [16,1,1]`. Same
  shape needed for `add`. 2 oracles total.
- **F2**: 3 oracles — `reduce_sum [1,16,128] axes=[1]` rank-3 (Apple
  accepts), `[1,16,128,1] axes=[2]` rank-4 twin (already in
  gxreduce_*), and the `(1,2048,1,1) axes=[2,3]` rank-4 twin
  (Apple rejects). Family fix is the rank-3 rewrite.
- **F3**: 3 oracles — `gabmm_r4 [1,16,1,128] × [1,16,128,128] ty=false`
  (rows=1), `[1,16,128,1] × [1,16,1,128] ty=true` (rows=128 outer
  product), `[1,16,1,128] × [1,16,128,128] ty=false` (rows=1 again
  for the third matmul).
- **F4**: 1 oracle — softmax attention pipeline at `[16,50]`
  (`mul(x, scale) → add(x, mask) → softmax`). The bare softmax at
  `[16,50]` already compiles (control-softmax-pure); only the
  upstream chain needs the new decode.
- **F5**: 1 oracle — 3-op chain sigmoid → mul → add at `(16, 128)`.

Total: 10 new Apple-decoded oracles to land.

## Per-program matrix (38 staged programs)

| Prog | Class | Refusal point | Smallest fix family |
|---:|---|---|---|
| 0 | A-first | reduce_sum at [1,2048] axes=[1] | F2 |
| 1 | B-state | broadcast mul [1,16,128,128]×[16,1,1] | F1+F3 |
| 2–5 | A-next / B-state | as classes | F2 / F1+F3 |
| 6 | D-attn | reduce_sum at [1,2048] | F2+F3+F4 |
| 7–11 | B-state / A-next | as classes | F1+F3 / F2 |
| 12 | D-attn | as 6 | F2+F3+F4 |
| 13–17 | B-state / A-next | as classes | F1+F3 / F2 |
| 18 | D-attn | as 6 | F2+F3+F4 |
| 19 | B-state | as 1 | F1+F3 |
| 20 | C-readout | chain encoder: 3-op sigmoid→mul→add | F5 |
| 21 | A-first | reduce_sum at [1,2048] | F2 |
| 22–36 | A-next / B-state | as classes | F2 / F1+F3 |
| 37 | E-final | reduce_sum at [1,2048] | F2 |

**Whole-program compiled count before this batch: 0 of 38.** Per-op
partial: 1 (program 20 only).

## Limits flagged per assignment

- BO_INIT max 16 MiB: any compiled H14 package whose `constantBytes`
  exceeds 16 MiB is flagged. (Verified: control-matvec-1x2048x5120 at
  20 MiB flagged.)
- 250 distinct programs per boot: per-model-chain cap.
- 16-byte-frame descriptor walk: flagged when emitted
  `firstTaskBytes` is not 16-byte aligned or exceeds the 7-bit
  encode (`> 0x1f8`).

## Apple-decoded envelope (H14 minted-oracle point sets)

Decoded shape/operand sets extracted from the H14 plugin inc files;
each case below is a "compiles inside the decoded envelope" set.
Decoded shapes outside these tables emit
`h14.outside-parity-envelope`, `h14.norm-outside-envelope`, or
`h14.unsupported-program`.

- Matvec (constant weight, runtime x): rows ∈ {1, 2, 8, 64}.
  (reduction, columns) point set: {256, 512, 1024}², plus
  (1536, 1536), (1536, 2048), (1536, 3072), (2048, 1536),
  (2048, 3072), (2048, 5120), (3072, 1536), (3072, 2048),
  (3072, 3072). Refusal text `{256, 512, 1024}` is stale.
- Batched matmul (runtime × runtime): rank-3 `[B, M, K]` or rank-4
  `[1, B, M, K]`, batch ≥ 2; decoded batches 8 and 16 (b16 3 tasks,
  b8 2 tasks; pv 5 tasks). Parakeet `[8, 375, 128] × [8, 128, 375]`
  and `[8, 375, 128] × [8, 128, 749]` and `[8, 375, 375] × [8, 375,
  128]` all decoded; rank-3 and rank-4 emit identical task streams.
- Select (runtime, bool cond): NCHW `(1, C, H, W)` for C ∈ decoded
  elementwise channel set; rank-3 `[1, C, W]` lowers to the same
  form. Channel order in the emitted ANEC: cond on channel 7, a on 5,
  b on 6, result on 4.
- Elementwise binary runtime: identical shapes at `(C, 1, 1)` for
  C ∈ {1, 64, 96, 128, 200, 256, 300, 512, 768, 1024, 2048, 3072,
  4096, 8192, 16384}; spatial `(64/96/768, 8, 8)`, `(64/96/128/768,
  16, 16)`, `(3, 224, 224)`. Broadcast forms: spatial operands
  `(1, 1, 1)`, `(1, H, W)`, `(C, 1, 1)` at spatial shapes; for `mul`
  and `real_div` only, the broadcast `(2048, 1, 1) × (1, 1, 1)` is
  decoded. **No broadcast at head-major `[1, 16, 128, 128] ×
  [16, 1, 1]`** — F1 family needed.
- Elementwise binary scalar: identical-shape `(C, 1, 1)` C ∈ {64,
  512}, `(64, 8, 8)`, `(64, 16, 16)`, `(768, 8/16, 8/16)`; scalar
  must be the inline fp16 literal `0.5` (`0x3800`).
- Elementwise unary: op-set × shape set is sparse.
  `sigmoid/silu/sqrt/rsqrt/tanh/abs/relu/leaky_relu/exp/gelu` each
  decode on a per-shape subset (sigmoid and silu are limited to
  `(64, 1, 1)` and `(512, 1, 1)`, plus spatial for silu/gelu; `sqrt`
  is the only unary at `(1, 1, 1)`). [1, 2048, 1, 1] sigmoid, silu,
  and sqrt now decode after the shape-ext merge.
- Norm (softmax / layer_norm / reduce_sum/mean/max):
  channel-flattened `(C, 1, 1)` for C ∈ {64, 128, 256, 512, 1024,
  2048, 4096} plus (8192, 1, 1) for `add/sub/mul` only, plus spatial
  forms (32, 4, 4), (64, 8, 8), (128, 16, 16). Softmax additionally
  at `(8, 1, 64)`, `(8, 64, 64)`, `(8, 128, 128)`, `(12, 64, 64)`,
  `(1, 64, 64)`, `(1, 256, 256)`, `(16, 50)` (control-softmax-pure
  at 02ea9ba).
- Apple-rejected reduce points: `reduce_sum / mean / max` at
  `(1, 16, 128, 1)` `axes=[2,3]`/`[3]`, `(1, 16, 1, 128)`
  `axes=[2,3]`/`[3]`, `(1, 2048, 1, 1)` axes where `reduce_sum`
  refuses. **Family fix**: rewrite to rank-3 form `[1, 16, 128]`
  axes=[1] (Apple accepts).

## Parakeet island programs

`run_parakeet_islands.py` against the merged build
(`/var/tmp/parakeet-h14-integ/results.json`):

| Program | Geometry | H14 status | Apple H13 TDs | H14 emitted TDs |
|---|---|---|---:|---:|
| `island-attn-a-kt/p0` rank-4 | `[1,8,375,128] × [1,8,128,749]` | compiled | 208 | 2 |
| `island-attn-a-kt/p0` rank-3 | `[8,375,128] × [8,128,749]` | compiled (same task stream) | 208 | 2 |
| `island-attn-a-kt/p1` | `[1,8,375,128] × [1,8,128,375]` | compiled | 208 | 2 |
| `island-select-8head` | select at `[1,8,375,375]` | compiled | 5 | 5 |
| `island-pv` | `[1,8,375,375] × [1,8,375,128]` | compiled | 208 | 5 |
| `island-oproj` | matvec rows=375 | refused (envelope) | 208 | n/a |
| `island-ffn-chain` | linear × 2 + silu | refused | 28 | n/a |
| `parakeet-encoder-whole` | 1,230-op encoder | refused (MIL lexer 0x2d) | 13,701 | n/a |

Per-program refusal strings (Parakeet):

- `island-oproj`: `h14.outside-parity-envelope`: rows=375 ∉ {1,2,8,64}.
- `island-ffn-chain`: same envelope at first `linear(x=...)`.
- `parakeet-whole-encoder`: `mil.lex.unexpected-character`: byte
  `0x2d` at `model.mil:229:84` (H14 MIL-frontend gap on the H13
  Apple-captured dialect).

## Qwen staged-decode program classes (manifest)

Manifest: `~/.local/share/apple-silicon-lab/artifacts/QwenChain/manifest.json`,
38 programs, `max_len 50`. Class inventory per the lane/state/ctx
port tables (see definitions above).

The op mix per class, reconstructed from `qwen-fused-mil/`
(`ane-m1rt-fused-wt/.local/qwen-fused-mil/`), `qwen-recurrent-mil.py`,
and the contract GGUF
(`~/.local/share/apple-silicon-lab/artifacts/QwenChain/Qwen3.8-2B-Q4_K_M.gguf`):

- **Class A (host block, 14 programs)**:
  rms_norm(x) [1,2048] → qkv linear (`blk.N.attn_qkv` [6144, 2048])
  → causal_conv1d (`ssm_conv1d` [4, 6144] over qkv) → sigmoid/softplus
  gates (`blk.N.ssm_beta.weight` [2048, 16] → 16 heads; decay via
  exp/maximum/real_div chain) → l2_q / l2_k per-head normalize
  (`ssm_norm` [128]) → z projection (`attn_gate` [2048, 2048]).
- **Class B (state block, 18 programs)**:
  decay·state → matmul(k, state_decay) → sub v-kv → ·beta →
  transpose → matmul(outer) → add(state, outer) → matmul(q,
  state') → o.
- **Class C (mid, 11 programs)**: residual_add (z gate) → rms_norm(post)
  → silu_mul_6144 (ffn_gate ⊗ ffn_up; weights [6144, 2048] each) →
  ffn_down linear [2048, 6144] (`blk.N.ffn_down`) → next layer's host
  block (qkv/conv/gates/l2).
- **Class D (mid + attention, 5 programs)**: class C composition plus
  RoPE (rope_ha/hb on q, k), batched matmul `Q × K^T` and `probs × V`
  ([16, 1, 50] × [16, 1, 128] etc.), softmax over 50 positions, KV
  cache `[2, 50, 256]` (2 kv-heads × 128 head-dim), ctx tables
  `[1, 50, 1]` (oh/inv/mask) and `[1, 256]` (cos/sin).
- **Classes E/F (chunk/mid final, 2 programs)**:
  readout tail = rms_norm(z·o over head-dim [128]) → ssm_out
  (`blk.N.ssm_out` [2048, 2048]) → residual_add → final_norm (chunk
  1 only) → h.

Compiled-package ANEC facts (Parakeet islands, from the merged build):

| Program | `constantBytes` | `firstTaskBytes` | pkg bytes | io channels | flags |
|---|---:|---:|---:|---|---|
| `attn-a-kt/p0` (rank-4) | 16384 | 184 | 21184 | in ch5/6, out ch4 | `firstTaskBytes % 16 = 8` |
| `attn-a-kt/p0` (rank-3) | 16384 | 184 | 21184 | in ch5/6, out ch4 | same |
| `attn-a-kt/p1` | 16384 | 184 | 21184 | in ch5/6, out ch4 | same |
| `select-8head` | 256 | 244 | 5696 | in ch5/6/7, out ch4 | `firstTaskBytes % 16 = 4` |
| `pv` | 16384 | 192 | 22528 | in ch5/6, out ch4 | none |

---

# H14 Batch 3 follow-up (date 2026-09-30, actor H14Batch3)

## F3 (b16 rank-4 batched matmul rows=1 / rows=128) — LANDED

Two Apple-decoded oracles (`env_mm_r3rr_m1_k128_n128_tx0_ty0_b16` and
`env_mm_r3rr_m128_k1_n128_tx1_ty1_b16`) routed into
`research/oracles/h14/gabmm_r4_*.json`. `batchedMatmulPlan` dispatch
fixed to honor `transpose_x=true` (logical rows/reduction post-transpose).
Generator emits both `headsForm=false` and `headsForm=true` from each r3
oracle (Parakeet r3/r4 streams are identical).

Parity: 832 -> 834 PASS (16 island templates).

Matrix unchanged: B-state still refuses because the per-head broadcast
at `{16,1,128} x {16,1,1}` is not in the F1 broadcast table. F3 alone
is correctly compiled but gated on additional F1 broadcast shapes.

## F2 / F4 / F5 — DECODED-FINDINGS

See `~/.local/share/apple-silicon-lab/entries/H14Batch3/2026-09-30T0001Z-h14-batch3-f2-f4-f5-decoded-findings.md`.

- F2: decoder fix landed in `fb9e715` (H14Batch2). Dispatch still
  refuses A/A-next/D/E classes — root cause is the harness's separate
  `op1_axes` const op spelling; needs inline-axes rewrite in harness.
- F5: 3-op chain oracle decoded (49152 bytes, 3 tasks). Chain encoder
  only handles producer+relu. Needs OracleChainTemplate table extension.
- F4: Apple rejected `mul+add` upstream at both rank-4 and rank-2
  softmax attention. No decoded oracle exists; family cannot be finished.

## Next mint batch priority — historical (superseded by H14Batch7 below)

The following list records the plan after Batch3, not the current blockers.

1. F1b: per-head broadcast at `{16,1,128} x {16,1,1}` and
   `{16,1,1} x {16,1,128}`. Apple mint. Once landed, F1b+F3 unlocks 18 B-state.
2. F2: inline the axes literal in the staged MIL.
3. F5: wire the decoded 3-op chain template.
4. F4: mint a chainable upstream variant on Apple.
## H14Batch7 — transpose and Qwen linear probes (2026-09-30)

The latest local parity run passes **838 cases**: 238 elementwise, 58 matvec, 14 known-weight matvec probes over 9 `(K,N)` grid points, 109 softmax/layer_norm, 114 reductions over 190 norm templates, 284 convolution, 16 island, 4 rms_norm chain, and 1 sigmoid/mul/add chain. The 38-program Batch5 Qwen matrix now has **1 ONE-ANEC** (`C-readout`, program 20) and 37 refusals: 2 A-first, 11 A-next, 18 B-state, 5 D-attn, and 1 E-final. No partial programs remain.

The frontend now folds `transpose_x` and `transpose_y` from scalar bool `const` producers and defaults either omitted flag to `false`. The first matmul in every B-state program now passes this flag gate. The next refusal in all 18 programs (`qwen-prog-01,03,05,07,09,11,13,15,17,19,22,24,26,28,30,32,34,36`) is source `model.mil:12:5`: `matmul(transpose_x=true, transpose_y=false, x=[1,16,1,128], y=[1,16,1,128]) -> [1,16,128,128]`. The frontend reports `H14 matmul does not support transpose_x = true` at `plugins/H14/ANEH14Compiler.mm:701`.

The other 19 refusals all first stop at source `model.mil:20:5`, a `linear` with input `[1,2048,1,1]`, weight `[N,2048]`, and output `[1,N]`. A-first programs 00 and 21 and A-next programs 02,04,08,10,14,16,23,27,29,33,35 use `N=6144`; D-attn programs 06,12,18,25,31 and E-final program 37 use `N=4096`. `matvecGeometry` rejects the rank mismatch (`input rank 4`, `output rank 2`) at `plugins/H14/ANEH14Compiler.mm:659`, with the refusal emitted at lines 708-713.

The required standalone matvec point `(M=1,K=2048,N=4096)` is now Apple-decoded and wired alongside the existing `(1,2048,6144)` point; both shape/orientation entries are in `H14MatvecTemplates.inc`. Apple also accepted one-task 1x1 convolution forms at `x=[1,2048,1,1]`, `w=[N,2048,1,1]` for both widths; their oracle records and rows are in `research/oracles/h14/` and `H14ConvTemplates.inc`. The exact `linear -> conv1x1` semantic mapping and fp16 accumulation caveat are in `research/h14-rewrites.md`. Standalone matmul K=2048,N=4096 emits two tasks; reshape+matmul from the rank-4 chain surface emits one task, so they are distinct oracle surfaces and are not conflated.

The matrix's only ONE-ANEC artifact is `/var/tmp/h14-qwen-anec/program-0.anec`; the manifest includes program/class/ops, SHA-256, tasks, IO shapes/channels/bytes, and kernel bytes. The 37 current refusals remain; the rewrite proposal does not claim that staged Qwen MIL was changed or that a device executed the convolution.
## H14Batch8 — B-state outer product and Qwen conv1x1 matrix (2026-09-30)

Parity passes **840 cases**: 239 elementwise, 58 matvec, 14 known-weight matvec probes over 9 (K,N) points, 109 softmax/layer_norm, 114 reductions over 190 norm templates, 284 convolution, 17 island select/batched-matmul, 4 rms_norm chains, and 1 sigmoid/mul/add chain.

The 38-program matrix has an original column and an explicit rw-conv1x1 column. In original, 18 B-state programs compile into 7 ANEC programs each, and C-readout program 20 compiles as one ANEC; 19 A/D/E programs refuse at the linear on source line 20. No staged Qwen MIL was rewritten. rw-conv1x1 changes only the final projection in A-first/A-next/D-attn/E-final and declares its adapted output shape [1,N,1,1]; all 19 compile into two ANEC programs each (7-task rms_norm chain + 1-task conv), none into one ANEC. The change in fp16 accumulation order still requires STAGED-QWEN-REF token-equality validation before promotion.

The 18 B-state ids 01,03,05,07,09,11,13,15,17,19,22,24,26,28,30,32,34,36 now compile as complete seven-program outputs. Apple accepted the exact staged outer matmul tx=true, ty=false at [1,16,1,128] x [1,16,1,128] -> [1,16,128,128] as a two-task program. The rank-3 neighbor, the tx=true,ty=true spelling with transpose(y), and the tx=false,ty=false spelling with transpose(x) also decoded; all four task-descriptor streams match exactly. The staged same-shape add add([1,16,128,128], [1,16,128,128]) was the next local refusal and is now Apple-decoded and wired. The later q/k-by-state matmuls use the already-decoded {M=1,K=128,N=128,batch=16,ty=false} oracle.

The remaining first refusals are all 19 original A/D/E projections at model.mil:20:5, with exact diagnostic error [h14.unsupported-program]: H14 matmul requires a runtime fp16 x, a constant rank-2 weight matching its transpose flag, and a matching output shape. N=6144 ids: 00,02,04,08,10,14,16,21,23,27,29,33,35; N=4096 ids: 06,12,18,25,31,37. The conv1x1 rewrite removes this refusal in its labeled column, but is not yet a single ANEC and is not token-validated.

Matrix receipt: /var/tmp/qwen-h14-matrix-batch8/results.json. Apple mint inputs, emitted HWX, decoded descriptors, transcripts, and SHA256SUMS are under the private ~/.local/share/apple-silicon-lab/artifacts/H14Batch8/ notebook. No Apple device execution is claimed.
