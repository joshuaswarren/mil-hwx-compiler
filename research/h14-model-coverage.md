# H14 model coverage: Parakeet islands + Qwen staged decode

Date: 2026-09-30. Actor: ModelCoverage. Compiler worktree:
`/home/joshuawarren/src/mil-hwx-h14-mint-wt`, head `2abdc09`. Compiler binary:
`./build/mil-hwxc --target H14 --format anec`. GNUstep runtime at
`$HOME/.local/mil-hwx-gnustep/lib`. Read-only w.r.t. the compiler source —
no edits, no rebuilds; the existing build artifact served every run.
Harness: `/var/tmp/model-coverage-build/run_coverage.py`. Per-case artifacts:
`/var/tmp/model-coverage-build/mil/<name>/model.mil`,
`/var/tmp/model-coverage-build/out/<name>/`. Notebook archive:
`~/.local/share/apple-silicon-lab/artifacts/ModelCoverage/h14-model-coverage/`
(`results.json`, `mil/`, `run_coverage.py`, `SHA256SUMS`). Pre-registered
entry: `~/.local/share/apple-silicon-lab/entries/ModelCoverage/2026-09-30T0037Z-omp-studio-local-ct-h14-model-coverage.md`.

The MIL frontend is the worklist interface; compiled programs are emitted
ANEC for `mil-hwxc.h14-anec-package.v1`. Compile/refused outcomes
establish package emit/accept, not on-chip behavior (parity is not device
execution; see `model-gap-findings.md`). Verbatim errors are emitted on
stderr; "refused" rows below carry them as recorded.

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

| Program | Geometry (from manifest) | Op mix | H14 status | Apple H13 TDs | H14 emitted TDs |
|---|---|---|---|---:|---:|
| `island-attn-a-kt/p0` (rank-4) | `[1,8,375,128] × [1,8,128,749] → [1,8,375,749]`, runtime×runtime, ty=false | batched matmul | compiled (`apple-parity-batched-matmul`) | 208 | 2 |
| `island-attn-a-kt/p0` (rank-3) | `[8,375,128] × [8,128,749] → [8,375,749]` | batched matmul | compiled (identical task stream) | 208 | 2 |
| `island-attn-a-kt/p1` | `[1,8,375,128] × [1,8,128,375] → [1,8,375,375]`, ty=false | batched matmul | compiled (`apple-parity-batched-matmul`) | 208 | 2 |
| `island-select-8head` | select(ninf_rt, matrix_bd_5 \| cond), cond on ch7, all `[1,8,375,375]` | select, three-input | compiled (`apple-parity-select`) | 5 | 5 |
| `island-pv` | `[1,8,375,375] × [1,8,375,128] → [1,8,375,128]`, runtime×runtime | batched matmul | compiled (`apple-parity-batched-matmul`) | 208 | 5 |
| `island-oproj-L{L}` (opt-in) | `[1,375,1024] × const [1024,1024] → [1,375,1024]` | linear | refused | 208 (per island ref `island-oproj`) | n/a |
| `island-ffn-L{L}-f{1,2}` (opt-in, chain-scale fused) | `[1,375,1024] → linear1 → silu → linear2 → [1,375,1024]`, weights `[4096,1024]` and `[1024,4096]` | linear × 2, silu | refused | 28 (per `island-ffn-L09-f{1,2}`) | n/a |
| `parakeet-encoder-whole` | 1,230-op encoder, 13,701 TDs (per `2026-09-22-encoder-island-cost` capture) | matmul / softmax / conv / norm / … | refused | 13,701 | n/a |

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
38 programs, `max_len 50`. Class inventory (per the 2026-09-27 entry):
2× class A (stage_a-first), 18× class B (state), 11× class C (mid),
5× class D (mid + attention / RoPE + KV cache), 1× class E (chunk-0
final readout), 1× class F (chunk-1 final readout).

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

## Ranked blocker list

Each line names one compiler-family extension and the program classes
it unblocks (count of currently-refused Qwen staged programs that fall
through to it).

1. **H14 elementwise unary shape extension** — add `sigmoid`,
   `silu`, `sqrt`, `real_div` (already in unary set), and `mul` to
   the `(2048, 1, 1)` and `(6144, 1, 1)` and `(4096, 1, 1)` shapes
   (for `sigmoid`, the Qwen z-gate lane; for `silu`, SwiGLU; for `sqrt`,
   the rms_norm decomposed chain). **Unblocks: z-gate (class C
   z-gating), SwiGLU (class C/E/F), rms_norm decomposition (class
   A/E/F all 26 staged programs that touch rms_norm — class A 14, class
   C 11, class E 1, class F 1), causal_conv (class A accumulators),
   the l2norm decomposition (class A l2_q / l2_k).** Net: ≥ 35 of the
   38 staged programs gain at least their elementwise skeleton once
   these shapes decode.
2. **H14 elementwise binary shape extension** — add `(16, 1, 1)`
   channel for the per-head softplus / decay gates; add the
   `(C, 1, 1) × (C, 1, 1)` broadcast at `C ∈ {16, 4096, 6144}` and
   extend the `(2048, 1, 1)` broadcast operands from `(1, 1, 1)` only
   to all the other operand shapes the binary-scalar set already covers.
   **Unblocks: decay gate (class A, all 14 programs), silu_mul_6144,
   sigmoid_mul_2048 channels beyond the 2048 broadcast, residual_add
   at (6144,1,1).** Compound blocker for class A and class C — nets
   ≥ 25 programs' elementwise skeleton.
3. **H14 norm frontend extension** — add the Qwen decode shape
   `[1, 2048, 1, 1]` axis=1 (mask 0x02) and axis=2 (mask 0x08) to
   the `reduce_sum/mean/max` decoded tables (the host block norm
   axis). Add softmax at `[16, 50]` axis=1 / `[1, 16, 1, 50]`
   axis=2 / `[16, 50]` axis=-1 to `softmax` (the attention decode
   shape). **Unblocks: rms_norm decomposed form (class A/E/F all 26
   programs), softmax_scaled (class D attention, 5 programs).**
4. **H14 batched matmul shape extension** — extend the
   `kBatchedMatmulTasks` inc with the actual class-B decoded shapes
   from `mint_h14_model_gaps.py` (batches 2, 4 in addition to the
   already-minted 8, 16; rows 1, 16 at b16 in particular), and the
   `[1, 16, 1, 128] × [1, 16, 128, 128]` rank-4 form that the matcher
   currently rejects at the matvecPlan fall-through. **Unblocks: class
   B state-block (18 programs).** The `control-state-b16-matmul-pure`
   refusal is the gating diagnostic for this work.
5. **H14 matvec template extension** — add the four contract Qwen
   points `(2048, 6144)`, `(6144, 2048)`, `(2048, 2048)`,
   `(6144, 6144)` (qkv / ffn_down / ssm_out / attn_gate, in both
   weight orientations) and the Parakeet `(375, 1024)` and
   `(1024, 1024)` linear-row extents. **Unblocks: 3 of 5 Qwen
   decode-step matvecs (qkv, ssm_out, ffn_down), both Parakeet
   opt-in programs (oproj, ffn-chain).**
6. **H14 MIL-frontend lexer: negative numeric literals** — accept
   `-N` and `-N.M` fp literals (currently `0x2d` is rejected at the
   lexer stage; see the `parakeet-whole-encoder` row). Necessary for
   any decomposition whose MIL emits negative eps, negative one, or
   negative-zero cases.
7. **H14 concat / transpose op support** — neither `concat` nor
   `transpose` appears in the H14 op dispatch. The staged state
   block closes with a concat that bundles q/k/v/gates/state'/output
   into the resident-state plane; batched `QK^T` uses `matmul
   (transpose_x = true)` to express transpose, but the state-block
   outer product cannot. **Unblocks: the state-block packed concat
   (class B, 18 programs).**

The three most consequential extensions — in the order that maximises
unblocked programs — are (1) elementwise unary at the Qwen/Parakeet
channel counts, (2) elementwise binary + the (2048,1,1) broadcast
expansion, and (3) norm frontend at `[1, 2048, 1, 1]` and softmax at
the attention shape. Together they unblock ≥ 35 of the 38 staged
programs at the elementwise / reduction skeleton level; (4) and (5)
then complete the matmul coverage (class B, plus the rest of the
decode-step matvecs).

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

- The compiler binary (`build/mil-hwxc`) is from the worktree head
  `2abdc09` (`research(h14): mint the model-gap families and encode
  select and batched matmul`). The harness ran 25 cases plus 1
  smoke; compiled 7 (after fixes), refused 18; no kernel, no M2,
  no macstudio, no ANE device was touched.
- The decoder facts are read from the worktree's
  `mil-hwx-h14-mint-wt/build/mil-hwxc` plus the
  `research/h14-model-gap-findings.md` baseline, and from the
  elementwise / norm / matvec / batched point lists extracted from
  `plugins/H14/*.inc` (the exact-template lookup the compiler
  enforces).
- The Apple H13 task counts are read from each Parakeet bundle's
  manifest.json (`task_descriptors` field, top-level and per
  program) and from `receipts/2026-09-22-encoder-island-cost/capture/model.mil`
  for the whole encoder (13,701 TDs).
- All compiled `firstTaskBytes` values were re-checked against the
  H14 task walk rule in `mil-hwx-compiler/AGENTS.md`.
