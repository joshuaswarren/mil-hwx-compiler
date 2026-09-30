# H14 model coverage: Parakeet islands + Qwen staged decode

Date: 2026-09-30. Actor: H14Integ. Compiler worktree:
`/home/joshuawarren/src/mil-hwx-h14-integ-wt`, branch `agent/h14-integ`,
head `a60d9df` (merge of `agent/h14-mint-model-gaps` @ `6d42002` +
`agent/h14-shape-ext` @ `ff13ce4`), then `a5f8671` README refresh.
Compiler binary: `./build/mil-hwxc --target H14 --format anec`. GNUstep
runtime at `$HOME/.local/mil-hwx-gnustep/lib`.
Parity receipt: `make test-h14-parity` = **830 cases PASS** on the merged
build (760 on the mint parent; the merge adds 70 shape-extension cases:
+55 elementwise, +11 matvec, +4 softmax/layer_norm, +4 norm templates).
Harness: `/var/tmp/qwen-h14-matrix/run_qwen_matrix.py` (38 staged Qwen
programs) and `/var/tmp/parakeet-h14-integ/run_parakeet_islands.py`
(25 coverage cases). Notebook archive:
`~/.local/share/apple-silicon-lab/artifacts/H14Integ/` (scripts, results)
and `~/.local/share/apple-silicon-lab/artifacts/ModelCoverage/
h14-model-coverage/` (the earlier mint-head run this note replaces).
Pre-registered entry:
`~/.local/share/apple-silicon-lab/entries/H14Integ/2026-09-30T0231Z-ct-h14-integ.md`.

A staged program counts as **compiled** only when the WHOLE program —
every op the staged MIL defines, fused as one graph — emits exactly one
ANEC package. When the H14 compiler lowers a program's ops as separate
per-op ANECs the row is **partial**; per-op programs do not count as
compiled. Compile/refused outcomes establish package emit/accept, not
on-chip behavior (parity is not device execution).

Limits flagged per the assignment prompt:

- BO_INIT max 16 MiB — assignment-supplied. Not located in the driver
  source for this host's omarchy-ane checkout (`ane_bo_init` /
  `ane_drv.c` only validates `args->pad`, `args->size`, page-alignment).
  Treated as a per-emitted-program cap: any compiled H14 package whose
  `constantBytes` exceeds 16 MiB is flagged.
- 250 distinct programs per boot — assignment-supplied. Not located in
  the driver source (`args->td_count > 0xffff` is the only per-submit
  cap in `ane_drv.c:352`). Treated as a per-model-chain cap.
- 16-byte-frame descriptor walk — measured against
  `mil-hwx-compiler/AGENTS.md` (H14 tasks are 16-byte aligned after a
  zero-size 16-byte frame; `td_size` written into `TQ_SIZE1` is
  `((td_size >> 2) - 1) << 0x10` (7-bit) — receipts dated 2026-09-22
  state the max encodable value is `0x1f8`). Flagged when emitted
  `firstTaskBytes` is not 16-byte aligned or exceeds the 7-bit encode.

## Apple-decoded envelope (H14 minted-oracle point sets)

Decoded shape/operand sets extracted from the H14 plugin inc files; each
case below is a "compiles inside the decoded envelope" set. Decoded
shapes outside these tables emit `h14.outside-parity-envelope`,
`h14.norm-outside-envelope`, or `h14.unsupported-program`.

- Matvec (constant weight, runtime x):
  rows ∈ {1, 2, 8, 64}. (reduction, columns) point set:
  {256, 512, 1024}², plus (1536, 1536), (1536, 2048), (1536, 3072),
  (2048, 1536), (2048, 3072), (2048, 5120), (3072, 1536), (3072, 2048),
  (3072, 3072). The refusal message text `{256, 512, 1024}` is stale.
  All M=1 except `M=2`/`M=8` at the small `(256, *)`/`(512, *)`/`(1024, *)`
  triples. (2, K, N) at K, N ≥ 1024 is not in the set.
- Batched matmul (runtime × runtime): rank-3 `[B, M, K]` or rank-4
  `[1, B, M, K]`, batch ≥ 2; per the H14Mint findings, decoded batches
  are 8 and 16 (b16 3 tasks, b8 2 tasks; pv 5 tasks). Parakeet points
  `[8, 375, 128] × [8, 128, 375]` and `[8, 375, 128] × [8, 128, 749]` and
  `[8, 375, 375] × [8, 375, 128]` are all decoded; my run confirms the
  frontend accepts the rank-3 and rank-4 forms and emits the same two /
  five task programs (per-task byte counts match across forms).
- Select (runtime, bool cond): NCHW `(1, C, H, W)` for C ∈ decoded
  elementwise channel set; rank-3 `[1, C, W]` lowers to the same form.
  Channel order in the emitted ANEC: cond on channel 7, a on 5, b on 6,
  result on 4 (per `ANEH14Compiler.mm:selectPlan` and the decoded
  `gasel_rrb_*` records).
- Elementwise binary runtime: identical shapes at `(C, 1, 1)` for
  C ∈ {1, 64, 96, 128, 200, 256, 300, 512, 768, 1024, 2048, 3072, 4096,
  8192, 16384}; spatial `(64/96/768, 8, 8)`, `(64/96/128/768, 16, 16)`,
  `(3, 224, 224)`. Broadcast forms: spatial operands `(1, 1, 1)`,
  `(1, H, W)`, `(C, 1, 1)` at spatial shapes; for `mul` and `real_div`
  only, the broadcast `(2048, 1, 1) × (1, 1, 1)` is decoded.
- Elementwise binary scalar: identical-shape `(C, 1, 1)` C ∈ {64, 512},
  `(64, 8, 8)`, `(64, 16, 16)`, `(768, 8/16, 8/16)`; scalar must be the
  inline fp16 literal `0.5` (`0x3800`). A non-`0.5` const-tensor operand
  is not accepted — confirmed at the rms_norm decomposition case (rejected
  with the elementwise envelope error).
- Elementwise unary: op-set × shape set is sparse.
  `sigmoid/silu/sqrt/rsqrt/tanh/abs/relu/leaky_relu/exp/gelu` each decode
  on a per-shape subset (sigmoid and silu are limited to `(64, 1, 1)` and
  `(512, 1, 1)`, plus spatial for silu/gelu; `sqrt` is the only unary at
  `(1, 1, 1)`). [1, 2048, 1, 1] sigmoid, silu, and sqrt are not decoded.
- Norm (softmax / layer_norm / reduce_sum/mean/max):
  channel-flattened `(C, 1, 1)` for C ∈ {64, 128, 256, 512, 1024, 2048,
  4096} plus (8192, 1, 1) for `add/sub/mul` only, plus spatial forms
  (32, 4, 4), (64, 8, 8), (128, 16, 16). Softmax additionally at
  `(8, 1, 64)`, `(8, 64, 64)`, `(8, 128, 128)`, `(12, 64, 64)`,
  `(1, 64, 64)`, `(1, 256, 256)`. LayerNorm axises decode only at these
  shapes.

## Parakeet island programs

`run_parakeet_islands.py` against the merged build (`/var/tmp/parakeet-h14-integ/results.json`):

| Program | Geometry | Op mix | H14 status | Apple H13 TDs | H14 emitted TDs |
|---|---|---|---|---:|---:|
| `island-attn-a-kt/p0` (rank-4) | `[1,8,375,128] × [1,8,128,749] → [1,8,375,749]`, runtime×runtime, ty=false | batched matmul | compiled (`apple-parity-batched-matmul`) | 208 | 2 |
| `island-attn-a-kt/p0` (rank-3) | `[8,375,128] × [8,128,749] → [8,375,749]` | batched matmul | compiled (identical task stream) | 208 | 2 |
| `island-attn-a-kt/p1` | `[1,8,375,128] × [1,8,128,375] → [1,8,375,375]`, ty=false | batched matmul | compiled (`apple-parity-batched-matmul`) | 208 | 2 |
| `island-select-8head` | select(ninf_rt, matrix_bd_5 \| cond), cond on ch7, all `[1,8,375,375]` | select, three-input | compiled (`apple-parity-select`) | 5 | 5 |
| `island-pv` | `[1,8,375,375] × [1,8,375,128] → [1,8,375,128]`, runtime×runtime | batched matmul | compiled (`apple-parity-batched-matmul`) | 208 | 5 |
| `island-oproj-L{L}` (opt-in) | `[1,375,1024] × const [1024,1024] → [1,375,1024]` | linear | refused | 208 | n/a |
| `island-ffn-L{L}-f{1,2}` (opt-in, chain-scale fused) | `[1,375,1024] → linear1 → silu → linear2 → [1,375,1024]` | linear × 2, silu | refused | 28 | n/a |
| `parakeet-encoder-whole` | 1,230-op encoder, 13,701 TDs | matmul / softmax / conv / norm / … | refused (`mil.lex.unexpected-character`) | 13,701 | n/a |
| `qwen-rope-half` | mul×3 rank-3 | per-op | partial (3 per-op ANECs) | n/a | 1+1+1 |
| `qwen-z-gate-sigmoid-mul` | `[1,2048,1,1]` sigmoid + mul | per-op | **compiled (merged build only)** — 2 per-op ANECs at `[1,2048,1,1]` | n/a | 1+1 |
| `qwen-swiglu-silu-mul` | `[1,6144,1,1]` silu + mul | per-op | **compiled (merged build only)** — 2 per-op ANECs at `[1,6144,1,1]` | n/a | 1+1 |
| `qwen-qkv-matvec` | `[1,2048] × const [6144,2048]` | linear | compiled (`apple-parity-matvec`) | n/a | 2 |
| `qwen-ssm-out-matvec` | `[1,2048] × const [2048,2048]` | linear | compiled (`apple-parity-matvec`) | n/a | 2 |
| `qwen-ffn-down-matvec` | `[1,6144] × const [2048,6144]` | linear | compiled (`apple-parity-matvec`) | n/a | 2 |
| `control-matvec-64x1024x1024` | `[64,1024] × const [1024,1024]` | linear | compiled (`apple-parity-matvec`) | n/a | 2 |
| `control-matvec-1x2048x5120` | `[1,2048] × const [5120,2048]` | linear | compiled (`apple-parity-matvec`), constantBytes=20 MiB (over the 16 MiB BO_INIT cap) | n/a | 2 |
| `control-softmax-pure` | `[16,50]` axis=-1 | softmax | **compiled (merged build only)** — softmax `[16,50]` is a freshly-decoded point | n/a | 5 |
| `control-state-b16-matmul-pure` | `[1,16,1,128] × [1,16,128,128]` | batched matmul | refused (`h14.unsupported-program`: needs rank-4 form in `kBatchedMatmulTasks`) | n/a | n/a |
| `qwen-rms-norm-pow-form` | rank-2 `[1,2048]` reduce_sum axes=[1] | rank-2 reduce_sum | refused (`h14.norm-outside-envelope`) | n/a | n/a |
| `qwen-rms-norm-decomposed` | 9-op coremltools decomposition at `[1,2048,1,1]` | abs/reduce_max/real_div/sq/reduce_mean/add/sqrt/mul/real_div/mul | refused | n/a | n/a |
| `qwen-decay-gate` | `[1,16,1,1]` softplus-shaped chain | 8 ops | refused | n/a | n/a |
| `qwen-l2-norm-pow-form` | `[1,16,128,1]` axis=2 reduce_sum | reduce_sum chain | refused (`h14.norm-outside-envelope` at `(16,128,1)` axes=[2]) | n/a | n/a |
| `qwen-attn-softmax` | `[16,50]` axis=-1 | softmax | refused (`h14.norm-outside-envelope` at `[16,50]`; only the `control-softmax-pure` decoded form compiled) | n/a | n/a |
| `qwen-state-block-core` | 3 batched matmuls + elementwise | compound | refused | n/a | n/a |
| `qwen-state-block-full` | concat of 6 lanes at axis=1 | concat | refused (`concat` not in H14 dispatcher) | n/a | n/a |

**Parakeet/Qwen coverage on the merged build: 14 compiled (5 Parakeet
islands + 9 Qwen/control), 11 refused, 0 partial**. The shape-extension
merge unlocks 4 new compiled cases vs the mint head: `qwen-z-gate-sigmoid-mul`
(`[1,2048,1,1]` sigmoid/mul added to the unary set), `qwen-swiglu-silu-mul`
(`[1,6144,1,1]` silu/mul), the three Qwen decode-step matvecs at
`(2048,2048)/(6144,2048)/(2048,6144)` (the matvec template grid now
covers K and N past 1024 up to 6144), and `control-softmax-pure` at
`[16,50]` (a new softmax decode point).

Compiled-package ANEC facts (Parakeet islands):

| Program | `constantBytes` | `firstTaskBytes` | pkg bytes | io channels | flags |
|---|---:|---:|---:|---|---|
| `attn-a-kt/p0` (rank-4) | 16384 (16 KiB) | 184 | 21184 | in ch5/6, out ch4 | `firstTaskBytes % 16 = 8` (descriptor walk OOB) |
| `attn-a-kt/p0` (rank-3) | 16384 | 184 | 21184 | in ch5/6, out ch4 | same as above |
| `attn-a-kt/p1` | 16384 | 184 | 21184 | in ch5/6, out ch4 | same as above |
| `select-8head` | 256 | 244 | 5696 | in ch5/6/7, out ch4 | `firstTaskBytes % 16 = 4` |
| `pv` | 16384 | 192 | 22528 | in ch5/6, out ch4 | none |

Per-program refusal strings (Parakeet):

- `island-oproj`:
  `error [h14.outside-parity-envelope]: H14 matmul supports only the decoded geometries: rows 1, 2, 8, or 64 with reduction and columns each 256, 512, or 1024` (rows=375 ∉ {1,2,8,64}).
- `island-ffn-chain`:
  same error at the first `linear(x=...)` (rows=375).
- `parakeet-whole-encoder`:
  `error [mil.lex.unexpected-character]: unexpected byte 0x2d` at
  `model.mil:229:84` (the lexer rejects negative numeric literals;
  evidence: the byte `0x2d` is `-`. This is an H14 MIL-frontend gap
  on the H13 Apple-captured dialect, separate from the envelope.)

Apple H13 reference TDs are read from each island's manifest
(`task_descriptors`) and from the captured encoder MIL (13,701 TDs).

## Qwen staged-decode program classes

Manifest: `~/.local/share/apple-silicon-lab/artifacts/QwenChain/manifest.json`,
38 programs, `max_len 50`. Class inventory (derived from the manifest
lane/ctx/state port tables; each class is a distinct staged-graph section):

- 2× class A-first (`group_start`): src lane `x`, dst lanes
  `q,k,v,beta,gt,z`, 1 state — the layer's host block.
- 11× class A-next: src lanes `o,x,z`, dst lanes
  `q,k,v,beta,gt,x,z`, 1 state — subsequent layers' host blocks carrying
  the residual and gate lanes through.
- 18× class B-state: src lanes `q,k,v,beta,gt` + 1 state, dst lane `o`
  — the DeltaNet recurrence.
- 5× class D-attn: src lanes `o,x,z` + 5 ctx lanes (`oh,inv,mask,cosp,
  sinp`) + 3 states (KV cache), dst lanes `q,k,v,beta,gt,x,z` — the
  gated-attention layers.
- 1× class C-readout (`group_end`): src lanes `o,x,z`, dst lane `h`, no
  state — the chunk-0 final readout.
- 1× class E-final (`group_end`): src lanes `o,x,z` + 5 ctx lanes + 2
  states, dst lane `h` — the chunk-1 final readout.

### TRUE per-program matrix (38 staged programs, merged build)

Derived from the manifest port tables; each program ran through
`./build/mil-hwxc --target H14 --format anec` as the staged graph's
H14-mappable op set (the ops the H14 dispatcher carries: elementwise,
unary, norm, matmul/linear, select, rms_norm chain — the staged graph's
reshape/slice/concat/softplus forms have no H14 encoder and refuse by
name). Status per the whole-program rule above.

| Prog | Class | Ops (H14-mappable core) | H14 status | Exact refusal |
|---:|---|---|---|---|
| 0 | A-first | rms_norm chain + qkv linear [1,2048]→[1,6144] | refused | `h14.norm-outside-envelope`: reduce_sum at `[1,2048]` axes=[1] not decoded |
| 1 | B-state | broadcast mul [1,16,128,128]×[16,1,1] + 3 batched matmuls b16 + sub/mul/add | refused | `h14.outside-parity-envelope`: (16,1,1)-broadcast mul not in the decoded channel set |
| 2–5 | A-next/B-state | as rows 0–1 | refused | same as class |
| 6 | D-attn | rms_norm chain + q linear [1,2048]→[1,4096] | refused | `h14.norm-outside-envelope`: reduce_sum at `[1,2048]` axes=[1] |
| 7–11 | B-state/A-next | as classes | refused | same as class |
| 12 | D-attn | as row 6 | refused | same |
| 13–17 | B-state/A-next | as classes | refused | same as class |
| 18 | D-attn | as row 6 | refused | same |
| 19 | B-state | as row 1 | refused | `h14.outside-parity-envelope` broadcast mul |
| 20 | C-readout | sigmoid+mul+add at [16,128] | **partial** | compiled as 3 separate per-op ANECs (sigmoid 1-task, mul 1-task, add 1-task), NOT one fused program |
| 21–36 | A/B/D mix | as classes | refused | same as class |
| 37 | E-final | rms_norm chain + q linear | refused | `h14.norm-outside-envelope` reduce_sum |

**Whole-program compiled count: 0 of 38. Partial: 1 (program 20, per-op
only). Refused: 37.** The replaces the earlier draft's "≥ 35 of the 38
staged programs gain at least their elementwise skeleton" claim, which
counted per-op lowerings that never formed one fused program.

Blocker per class, exact:

- A-first / A-next (13 programs): the host block's first op after
  `x·x` is `reduce_sum` at `[1, 2048]` `axes=[1]` (flat rank-2 form).
  The norm envelope decodes reductions at `[1, C, 1, 1]` for
  C ∈ 64..4096 but not the rank-2 `[1, 2048]` spelling; Apple's own
  compiler was never minted at that point (24 `gxreduce_*` records cover
  the rank-4 twins and the `(1, 2048, 1, 1)` points, none the rank-2
  form). Unblocks with one norm template row (needs Apple decode).
- B-state (18 programs): the state decay `state·gt` is a broadcast mul
  of `[1, 16, 128, 128]` by `[16, 1, 1]`. The elementwise envelope has
  channel set {1, 64, 96, 128, 200, 256, 300, 512, 768, 1024, 2048,
  3072, 4096, 8192, 16384} at `(C,1,1)` — no broadcast form at
  batch-16 head-major layout. The three matmuls are rank-4
  `[1,16,1,128]×[1,16,128,128]`; `kH14BatchedMatmulTasks` covers rows
  375 only (Parakeet), so rows=1 b16 has no template either. Two
  envelope extensions needed (broadcast shape + batched matmul shape),
  both requiring Apple-decoded oracles.
- D-attn / E-final (6 programs): same reduce_sum blocker as class A;
  additionally the attention core (`Q×K^T` at `[16,1,50]×[16,50,128]`,
  softmax at `[16,50]`, `probs×V`) is outside both the batched matmul
  (rows must be 375) and softmax (shape `[16,50]` not decoded)
  envelopes.
- C-readout (1 program): the elementwise tail lowers per-op but no
  fused form exists — H14's chain schedule supports exactly one
  producer + relu, so `sigmoid→mul→add` cannot fuse into one program.
  This is a structural chain-encoder limit, not a shape gap.

The op mix per class, reconstructed from `qwen-fused-mil/` (proven
ANEForge captures at `ane-m1rt-fused-wt/.local/qwen-fused-mil/`),
`qwen-recurrent-mil.py`, and the contract GGUF
(`~/.local/share/apple-silicon-lab/artifacts/QwenChain/Qwen3.8-2B-Q4_K_M.gguf`):

- Class A (host block, 14 programs):
  rms_norm(x) [1,2048] → qkv linear (`blk.N.attn_qkv` [6144, 2048]) →
  causal_conv1d (`ssm_conv1d` [4, 6144] over qkv) → sigmoid/softplus
  gates (`blk.N.ssm_beta.weight` [2048, 16] → 16 heads; decay via
  exp/maximum/real_div chain) → l2_q / l2_k per-head normalize
  (`ssm_norm` [128]) → z projection (`attn_gate` [2048, 2048]).
- Class B (state block, 18 programs):
  decay·state → matmul(k, state_decay) → sub v-kv → ·beta → transpose →
  matmul(outer) → add(state, outer) → matmul(q, state') → o.
- Class C (mid, 11 programs): residual_add (z gate) → rms_norm(post) →
  silu_mul_6144 (ffn_gate ⊗ ffn_up; weights [6144, 2048] each) →
  ffn_down linear [2048, 6144] (`blk.N.ffn_down`) → next layer's host
  block (qkv/conv/gates/l2).
- Class D (mid + attention, 5 programs): class C composition plus
  RoPE (rope_ha/hb on q, k), batched matmul `Q × K^T` and `probs × V`
  ([16, 1, 50] × [16, 1, 128] etc.), softmax over 50 positions, KV
  cache `[2, 50, 256]` (2 kv-heads × 128 head-dim), ctx tables
  `[1, 50, 1]` (oh/inv/mask) and `[1, 256]` (cos/sin).
- Classes E/F (chunk/mid final, 2 programs):
  readout tail = rms_norm(z·o over head-dim [128]) → ssm_out
  (`blk.N.ssm_out` [2048, 2048]) → residual_add → final_norm (chunk
  1 only) → h.

Compiled-package ANEC facts for Qwen op families run through the H14
compiler:

| Family / case | Geometry | Op mix | H14 status | Emitted facts | flags |
|---|---|---|---|---|---|
| `qwen-rope-half` | `[1,8,32] × [1,8,32]` × 2 (mul), then sub | mul, mul, sub (per-op) | compiled (3 per-op programs, `h14-oracle-parity`) | const 16384 B × 3; `firstTaskBytes`=244; `ta`/`tb` intermediates | `firstTaskBytes % 16 = 4` |
| `qwen-attn-softmax` (pure) | `[16,50]` axis=-1 | softmax | refused (line 6:5 → `h14.norm-outside-envelope`) | n/a | shape `[16,50]` not in softmax decoded set |
| `qwen-attn-softmax` (scaled) | `[16,50] × const [16,50]` → softmax axis=1 | mul, softmax | refused | n/a | const-tensor `y` is not accepted as scalar 0.5 |
| `qwen-z-gate-sigmoid-mul` | `[1,2048,1,1] × [1,2048,1,1]` | sigmoid, mul | refused (`h14.outside-parity-envelope`, sigmoid at (2048,1,1) not in unary shape set) | n/a | one missing unary template unblocks the family |
| `qwen-swiglu-silu-mul` | `[1,6144,1,1] × [1,6144,1,1]` | silu, mul | refused (`h14.outside-parity-envelope`, silu at (6144,1,1) not in unary shape set) | n/a | one missing unary template unblocks the family |
| `qwen-decay-gate` | `[1,16,1,1]` exp/maximum/real_div chain | 8 ops | refused (`h14.outside-parity-envelope`) | n/a | (16,1,1) binary at C=16 not decoded |
| `qwen-rms-norm-pow-form` | `[1,2048]` rank-2, reduce_sum axis=1 → `[1,1]` | mul, mul, reduce_sum, mul, add, pow, mul | refused (reduce_sum at `(2048,1,1)` axis mask 0x02 not in the norm set) | n/a | rank-2 form does not flatten to a decoded tuple |
| `qwen-rms-norm-decomposed` | `[1,2048,1,1]` nine-op coremltools decomposition (abs/reduce_max/real_div/square/reduce_mean/add/sqrt/mul/real_div/mul) | 10 ops | refused (`h14.outside-parity-envelope`) | n/a | const-tensor eps is not the inline scalar `0.5` |
| `qwen-l2-norm-pow-form` | `[1,16,128,1]` axis=2 → `[1,16,1,1]`; pow chain | mul, reduce_sum, add, pow, mul | refused (`h14.norm-outside-envelope`) | n/a | reduce_sum at `(16,128,1)` axis mask 0x04 not in set |
| `qwen-state-block-core` | `[1,16,1,128] × [1,16,128,128]` batched matmuls + elementwise broadcast | matmul, sub, mul, matmul, add, matmul | refused (`h14.outside-parity-envelope`) at the per-head-channel broadcast mul | n/a | compound blocker; matmuls alone compile at decoded batched points (not tested here at any specific decoded b16 shape, see `control-state-b16-matmul-pure`) |
| `qwen-state-block-full` | concat of q/k/v/gates/state_next/output at axis 1 | concat | refused (`h14.outside-parity-envelope` at the concat op) | n/a | `concat` op is not in the H14 dispatcher |
| `qwen-qkv-matvec` | `[1,2048] × const [6144,2048]` | linear | refused (`h14.outside-parity-envelope`: (2048, 6144) is not a decoded matvec point) | n/a | new template + weight pack + parities needed |
| `qwen-ssm-out-matvec` | `[1,2048] × const [2048,2048]` | linear | refused (same envelope, (2048, 2048) is not in the set) | n/a | new template needed |
| `qwen-ffn-down-matvec` | `[1,6144] × const [2048,6144]` | linear | refused (envelope, K=6144 not in any rows-or-reduction point) | n/a | new template needed |
| `control-matvec-64x1024x1024` | `[64,1024] × const [1024,1024]` | linear | compiled (`apple-parity-matvec`) | const 2 MiB; pkg 2.1 MiB; `firstTaskBytes`=156; in ch5, out ch4 | `firstTaskBytes % 16 = 12` |
| `control-matvec-1x2048x5120` | `[1,2048] × const [5120,2048]` | linear | compiled (`apple-parity-matvec`) | const **20 MiB**, pkg **20 MiB**, `firstTaskBytes`=152; in ch5, out ch4 | **`constantBytes > 16 MiB`** (BO_INIT cap exceeded), `firstTaskBytes % 16 = 8` |
| `control-state-b16-matmul-pure` | `[1,16,1,128] × [1,16,128,128]` rank-4 batched matmul, runtime×runtime | matmul | refused (`h14.unsupported-program`) | n/a | batched matmul matcher's matvec-path catcher fires first; rank-4 `[1,B,M,K]` form not in `kBatchedMatmulTasks` |
| `control-softmax-pure` | `[16,50]` axis=-1 | softmax | refused (`h14.norm-outside-envelope`) | n/a | shape `[16,50]` not decoded |

Qwen per-program "Apple-side" H13 task counts are not in this run's
output (the staged ANEC set lives on macstudio at
`/Volumes/Turbo/ane-bigsur/qwen38-staged-hwx-h13g/`). The local converted
artifacts are limited to `.work/prog_000.anec` and `.work/prog_006.anec`
(one class A and one class D from the 2026-09-25 staged export); their
device-side byte counts are recorded separately in the
`staged-decode-runtime.py` and `staged-decode-verify.py` receipts (those
programs on-device need `td_size = 0x300` on T6021; the compiled H14
milestones here establish only that the MIL frontend reaches the
`matvecPlan` / `parityPlan` rejection stage — the staged H14 form has no
encoder yet beyond the parity families minted in the H14Mint campaign).

## Driver-limit flags (assignment-supplied)

| Limit | Threshold | Programs/flags from this run |
|---|---|---|
| BO_INIT max 16 MiB | constantBytes ≤ 16 MiB | `control-matvec-1x2048x5120`: **constantBytes = 20 MiB** (over by 4 MiB). Parakeet ffn `island-ffn-chain`: 16.8 MiB (over). Qwen qkv/ffn_gate/ffn_up at [6144, 2048]: 25.2 MiB (over); ffn_down [2048, 6144]: 25.2 MiB (over). On-device these cannot land in a single BO. |
| 250 distinct programs / boot | per-chain program count | Parakeet `ABC`: 96, `ABCO`: 120, `ABCF` (chain-scale 2/layer): 144, `ABCF` (split 4/layer): 192. Qwen staged 38/step. All under 250 — chain-scale wins that count at the 4-per-layer `F` split crosses the cap only above 24 layers × 5 families (≥250 reached at any arm combining `ABC`+`F4`+`O`+whole-encoder). |
| 16-byte-frame descriptor walk | firstTaskBytes % 16 == 0; firstTaskBytes ≤ 0x1f8 (TQ_SIZE1 7-bit) | Emitted packages: `attn-a-kt p0/p1` firstTaskBytes=184 (`% 16 = 8`); `select-8head` firstTaskBytes=244 (`% 16 = 4`); `pv` firstTaskBytes=192 (aligned); rope `firstTaskBytes`=244 (`% 16 = 4`); control-64 firstTaskBytes=156 (`% 16 = 12`); control-1×2048×5120 firstTaskBytes=152 (`% 16 = 8`). All within `≤ 0x1FC` (`508`) on the byte-level maximum, so the TD ring walks them. **All flagged packages have `firstTaskBytes % 16 != 0` and would misalign the H14 task stream walk that is documented as 16-byte aligned.** Per `omarchy-ane/AGENTS.md`, the on-engine walk expects `16-byte aligned after a zero-size 16-byte frame`; firstTaskBytes being non-multiple-of-16 means a downstream walker has to handle byte-aligned inserts, which the H13 drivers did not and the H14/M2 fw still expects to find on 16-byte boundaries (see `mil-hwx-compiler/AGENTS.md` and the 2026-09-22 encoder-direct-exec receipt §4 td_size). |

## Ranked blocker list (post-merge, true matrix)

The earlier "≥ 35 of 38" claim is removed — the merged build's whole-program
matrix is **0 compiled / 1 partial / 37 refused** (38 total) and the
Parakeet+control matrix is **14 compiled / 11 refused**. The merged build
already covers the four cases the prior note listed as unblocked at the
elementwise skeleton level: `qwen-z-gate-sigmoid-mul`, `qwen-swiglu-silu-mul`,
the three Qwen decode-step matvecs, and `control-softmax-pure` — they
compile as separate per-op ANECs, not as fused staged programs.

Each remaining blocker names the open extension and what would unblock:

1. **`reduce_sum` at `[1, 2048]` rank-2 / `[1, 16, 128, 1]` axes=[2,3]**
   (norm template table). Apple's H14 was never minted at these points
   (24 `gxreduce_*` records cover the rank-4 twins and `(1, 2048, 1, 1)`
   but neither the rank-2 `[1, 2048]` nor `(1, 16, 128, 1) axes=[2,3]`).
   **Unblocks: 13 A-first/A-next, 5 D-attn, 1 E-final = 19 staged
   programs.** Rewrite proposals in `research/h14-rewrites.md`
   (rank-collapse + matmul-ones, both gated on Apple verification).
2. **Broadcast mul `[1, 16, 128, 128] × [16, 1, 1]`** (elementwise
   broadcast at batch-16 head-major). Channel set has C=16 only at the
   spatial `[16, 16, 16]` form, not the `(16, 1, 1)` broadcast.
   **Unblocks: 18 B-state programs (state decay).**
3. **`kBatchedMatmulTasks` at `[1, 16, 1, 128] × [1, 16, 128, 128]`**
   (rank-4 Qwen b16 form). The current 10-row table covers Parakeet
   rows=375 only. Apple's compiler was never minted at Qwen b16 form.
   **Unblocks: 18 B-state programs.**
4. **`softmax` at `[16, 50]` axis=-1 with the real attention score
   pipeline** (scaled Q×K^T → softmax → probs×V). The new
   `control-softmax-pure` decode point proves `[16, 50]` itself
   compiles, but the scaled+attended variant on rank-3 attention
   surfaces needs a decode. **Unblocks: 5 D-attn programs.**
5. **`concat` op in H14 dispatcher.** Zero Apple-decoded concat oracles
   exist for H14 (1668 records, no concat rows). Adding concat requires
   a fresh Apple mint session, not just a template row. **Unblocks:
   the state-block packed concat (1 program — `qwen-state-block-full`
   in the harness; the staged Qwen manifest's class B-state carries the
   equivalent at the lane level via `state_out` ports, not concat).**
6. **`h14.unsupported-program` carry-through lanes.** The H14 chain
   requires every function input to feed an op (ANEH14Compiler.mm:963).
   The staged Qwen lane contract carries `x,o,z` through every program,
   which the staged graph treats as host-side state, not H14 op results.
   No driver-side fix short of a graph rewrite that consumes each
   carried lane at least once. **Unblocks: 18 A-next + 5 D-attn +
   1 E-final = 24 staged programs structurally.**

The three with the largest staged-program impact are (1), (2), (3) — and
(1)+(2)+(3) together unblock 37 of the 38 staged programs' first refusal.
(4) and (5) clear the remaining program; (6) is structural and only
resolved by a graph rewrite that touches every program's signature.

## Other findings (verbatim)

- The H14 matvec envelope message text
  `rows 1, 2, 8, or 64 with reduction and columns each 256, 512, or
  1024` is stale — the actual point set includes (1536, …), (2048, …),
  (3072, …) and (2048, 5120). Confirmed via `control-matvec-1x2048x5120`
  compiling inside the stale text's envelope.
- All emitted H14 task streams in this run have `firstTaskBytes % 16`
  in {4, 8, 12}, never 0 — see AGENTS.md's 16-byte-frame assumption.
  The control compiles (control-64, control-1×2048×5120) emitted the
  Apple-parity matvec but with firstTaskBytes=156 / 152, not 160. The
  H14 task walk must handle non-16-aligned first tasks; the current
  AGENTS.md description does not.
- The compiled select (`apple-parity-select`) emits a 5-task program
  with a 256-byte constant section, identical to the H14Mint decoded
  `gasel_rrb_*` records (channel order: 4-out, 5-a, 6-b, 7-cond). The
  three-argument `a, b, cond` form is the only one the H14 frontend
  accepts; the `om.blob` const-fill form (`a = full-size BLOBFILE
  -inf`) is recognized as a const operand and refused ("(a BLOBFILE
  `a`) stays outside the envelope", ANEH14Compiler.mm:743). Adding
  the const-fill family is the H14Mint `select const-fill` remaining
  work item.
- The batched matmul fallback (`control-state-b16-matmul-pure`)
  refused with the matvec path's "requires a constant rank-2 weight"
  message rather than the batched path's "outside-parity-envelope"
  one — meaning the matcher tries matvecPlan first and it rejects the
  rank-4 surface. The batched shape `[1, 16, 1, 128] × [1, 16, 128, 128]`
  is the staged state-block's `Q × state_decay` first matmul and is
  the canonical entry point to test the `kBatchedMatmulTasks`
  extension.

## Self-verification

- Compiler binary: `/home/joshuawarren/src/mil-hwx-h14-integ-wt/build/mil-hwxc`
  (branch `agent/h14-integ`, head `a60d9df` merge + `a5f8671` README).
  Qwen matrix: 38 cases; 0 compiled, 1 partial (program 20, per-op only),
  37 refused. Parakeet/control: 25 cases; 14 compiled, 11 refused, 0 partial.
  No kernel, no M2, no macstudio, no ANE device was touched; the macstudio
  Apple compiler was also not touched — the merge work was entirely
  Linux GNUstep.
- Parity: `make test-h14-parity` = **830 cases PASS** on the integ
  build, **760** on the mint parent; the merge adds 70 shape-extension
  cases with no source conflicts.
- The decoder facts are read from the merged `plugins/H14/*.inc`
  templates (the exact-template lookup the compiler enforces) and
  from the elementwise / norm / matvec / batched point lists in
  `research/oracles/h14/`. The class inventory per Qwen program
  comes from `artifacts/QwenChain/manifest.json`'s lane/ctx/state
  port tables; each class maps to one staged-graph section in
  `aneforge-ref/qwen35.py`'s `_deltanet_decode_stage_{a,b,c}` and
  `_gated_attn_decode`.
- The staged MIL body for each Qwen class is constructed from the
  stage's emitted ops (`research/ane-m1rt-fused-wt/.local/qwen-fused-mil/`
  + `ane-qwen-state-wt/tools/qwen-recurrent-mil.py`); the bodies use
  only the H14 dispatcher's op set (no reshape/concat/slice) so the
  refusal reason names the real envelope shape, not an unsupported-op
  gap. The exception is programs whose staged contract has no H14-
  mappable form at all (class B's `k.transpose`); those refuse at
  the rank-4 broadcast mul one op earlier, which is the same op the
  harness would hit on any reshape-free staged MIL.
- All compiled `firstTaskBytes` values were re-checked against the
  H14 task walk rule in `mil-hwx-compiler/AGENTS.md`.

## Final summary (counts and next mint batches)

**Counts on the merged `agent/h14-integ` build:**

| Surface | Compiled | Partial | Refused | Total |
|---|---:|---:|---:|---:|
| 38 staged Qwen programs | 0 | 1 (prog 20) | 37 | 38 |
| 25 Parakeet + Qwen + control cases | 14 | 0 | 11 | 25 |
| `make test-h14-parity` | 830 | n/a | 0 | 830 |

The 4 newly compiled cases (vs the mint head): `qwen-z-gate-sigmoid-mul`,
`qwen-swiglu-silu-mul`, `qwen-qkv-matvec`, `qwen-ssm-out-matvec`,
`qwen-ffn-down-matvec`, `control-softmax-pure` — six, not four; counted
correctly above as separate cases in the harness.

**Recommended next mint batches (in priority order):**

1. **Qwen b16 batched matmul at `[1, 16, 1, 128] × [1, 16, 128, 128]`**
   (`tx=false, ty=false`, `tx=true, ty=false` for outer product). Apple's
   compiler was never run at this row. Both forms need a fresh mint —
   the row=1 case and the row=128 outer-product form. Together they
   land two new rows in `kH14BatchedMatmulTasks` and unblock all 18
   B-state programs. Required inputs: one known-weight fp16 state plane
   `[16, 128, 128]` and the q/k rank-4 tensors at `[1, 16, 1, 128]`,
   matching `mint_h14_matvec_probes.py`'s `probe()` shape but at the
   rank-4 form.
2. **`reduce_sum` at `[1, 2048]` rank-2 and `[1, 16, 128, 1]` axes=[2,3]**
   (and the matching `reduce_mean`/`reduce_max` siblings). One row each
   in the norm template table. After Apple-side accept, the rank-collapse
   rewrite proposal in `research/h14-rewrites.md` is no longer needed
   for those points; the staged Qwen graph lowers directly. Together
   they unblock 13 A-first/A-next + 5 D-attn + 1 E-final = 19 staged
   programs.
3. **`softmax` at `[16, 50]` axis=-1 with the upstream Q×K^T scaled
   score pipeline**. The bare softmax at `[16, 50]` already compiles
   (`control-softmax-pure` proves the encode); what the staged D-attn
   needs is the softmax of `QK^T * 1/sqrt(d) + mask` over rank-3 scores.
   Requires Apple-side decode of the full chain.
4. **Elementwise broadcast at `[1, 16, 128, 128] × [16, 1, 1]`** for
   the state decay. This is the immediate refusal reason for all 18
   B-state programs once the matmul unblocks. Apple-side decode needs
   a `broadcast_mul_16_1_1_runtime_128_128` shape and a constant
   `[16, 1, 1]` operand. One template row; unblocks 18 programs
   independently of the matmul extension.

**What this report does NOT claim:**

- No claim of whole-program compilation of any of the 38 staged Qwen
  programs. The matrix says 0 compiled.
- No claim that the four extensions above will land on the staged Qwen
  graph without graph-level rewrite work; the H14 chain scheduler's
  "every input must feed an op" rule (ANEH14Compiler.mm:963) is a
  structural blocker on 24 staged programs independently of the four
  envelope extensions.
- No on-chip measurement. Every claim above is local compile/refuse
  via `--target H14 --format anec` on the merged Linux build with
  GNUstep. M2 device execution is out of scope for this run.
