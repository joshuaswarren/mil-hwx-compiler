# H13 batched round-2 implementation receipt

## Source

- Compiler base: `eb7263d` (`feature/h13-slice-mid-range`, merged with
  `fix/h13-parity-reconcile` 703b3af)
- Branch: `feature/h13-boolean-and-round2`
- Oracles: `research/oracles-batched-round2` (614e2eb) merged — receipt
  `receipts/2026-09-13-batched-oracle-mint-round2/RECEIPT.md`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback)

## Delivered this commit (5229cc4): the batched wave is complete

### Packed-const packing, derived and proven

The round-2 `_idx` captures retain input weights and Apple's constant
sections, so the packing is settled from raw bytes, not inference:

- **Layout**: a 64-halfword (128-byte) kernel header, then the weight
  matrix — **transposed for ty=1 forms** — cut into packCols-wide chunks
  placed at the padded row stride with a **-64 phase**: row 0 starts at
  halfword 64 carrying 685 of 749 (packCols - 64), every later row starts
  at its row boundary carrying the previous row's displaced tail; the
  final 64 source halves stay unwritten.
- **Headers**: 8 distinct across the corpus, clustered by packed shape
  and batch (not by flags or rank); emitted as embedded byte tables.
- **Verification**: the generator rebuilds every retained pair and aborts
  on any byte mismatch — 14/14 exact, including both ty=1 transposed
  forms and both r4heads rank-4 twins.

### Envelope: 26 generator-verified entries

Round 1's 16 plus: V-projection (375,128,375) tx0/ty0 and tx0/ty1, both
storage kinds; the 209-task fold-flag forms (scores/attn-output × tx1/ty1
× runtime/blob) as an optional prefix task plus 26 tasks per batch; and
the packed `_idx` twins now carrying headers. Rank-3/rank-4 equivalence
re-verified per entry.

### Compiler

- `packBatchedWeights` implements the derived layout; the packed-const
  gate is **lifted** — const-KV scores and attn-output paths lower.
- `batchedMatmulParse` accepts explicit transpose flags (tx/ty forms
  parse with flag-aware rows/reduction/columns extraction; a
  boolean-sense bug in the first explicit-flag guard was caught by the
  ty=1 capture and fixed).
- The prefix task emits before the per-batch groups; taskCount carries
  the +1; firstTaskBytes is the prefix size.
- Inspector: the batched channel plan is canonical now — source channel
  5, destination channel 4, and channel 1 on the packed form's source
  selector when constantBytes > 0 (the round-2 selector survey: const
  (1,0,4) on the first tasks, runtime (5,0,4)).

### Suite

Every capture in `research/oracles/h13/batched/` verifies: task streams
byte-equal to the reconstruction (after the standard link-marker
rewrite), **constant sections byte-equal to Apple's retained
`const.bin`**, deterministic in both formats, inspector-validated.
Round-1 uniform captures synthesize their recorded uniform payloads.
The stale gate-era rejection tests were replaced: the V-projection ty=1
form now compiles (covered by its capture), the both-flags form rejects
at the shape gate, and a wrong-size constant blob rejects at the
resolver. `make test-h13` green; parity reconcile (846 cases) intact.

## Surveyed for the next commits on this branch (captures in hand)

- **less**: bool result only (fp16 result is Apple-rejected,
  `callback_status=1` at every shape). Decoded: 3 tasks (504/504/628),
  const 3072–5120, shapes (375/750/1500,1,1) and 1x64x1x1, plus the
  4-task bool-then-cast-fp16 form. Bindings: x,y input channel 1, bool
  result channel 2, NCHW [1,C,1,1].
- **floor**: 4 tasks (628/628/628/504), const 384, shapes [1] (Apple
  rewrites to [1,1,1,1]), 1x64x1x1, 1x512x1x1, blob-idx twin retained.
- **select**: bool cond only (fp16 0/1 cond Apple-rejected). Decoded:
  5 tasks (628/504/628/504/504), const 2048 (runtime-a, 1x64x1x1,
  [1,8,375,375]) and the -inf blob form (const 2258048, raw bytes
  retained). Three fp16 inputs + bool cond, output channel 2.
- **floor_div**: native decoded (5–6 tasks; scalar fp16(2.0) divisor —
  the inline tensor [1](2.0) form is Apple-rejected). Composition
  (real_div+floor) also decoded with equal constant hashes and task
  counts but different HWX — implement the native form.
- **tile**: materialized, 1 task, encoder form 768→288000 output bytes,
  const-x keeps the unexpanded payload in `__const`; raw bytes retained.
- **pad**: Apple rejects every form including the identity view — stays
  rejected with the mint receipt as evidence.

The bool-typed operands (less result, select cond) require a dtype
boundary at the frontend per the stream-3 decision doc; the compiler
encoders follow the captures' bool forms.

## Commands and observed results

```text
python3 research/generate_batched_matmul_tables.py
  # verified 26 entries, 8 headers (packing verified against raw bytes)
make -j8 build/mil-hwxc                    # 0 errors
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS
  h13 composite cli: PASS
  h13 batched cli: PASS                    # every capture, const bytes compared
  h13 registry cli: PASS
python3 tests/test_h13_parity.py build/mil-hwxc
  H13 oracle parity: PASS (846 cases …)    # reconcile intact
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change,
or Mac execution was performed by this agent. No files under
`ane-linux-experiments` were edited or executed.
