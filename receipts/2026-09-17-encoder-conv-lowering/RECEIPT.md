# Encoder conv leftover round 2: distinct-payload remint, depthwise and
# pointwise forms lower (2026-09-17)

Continuation of `receipts/2026-09-17-encoder-conv-lowering/RECEIPT.md` (same
worktree, `agent/slice-lastdim-lowering`). Host-only: the only remote contact
was the oracle compile on `macstudio` over BatchMode SSH, same binary as
round 1 (`/tmp/h13-oracle/bin/ane-compile-hwx`, sha256 `3d13fc85c2a6baa0…`,
52,336 B — byte-identical to the receipted round-1 tool). No jwm1/jw16, no
ANE execution, no pin change.

## What changed since round 1

`receipts/2026-09-17-encoder-conv-lowering/mint_encoder_conv_idx.py` re-minted
12 cases on the round-1 tool with the **per-element distinct payload**
(`uint16(index+1)` wrapping, `pattern: uint16_le_index_plus_one_wrapping`,
one running payload across both blob regions) — the permutation evidence the
uniform captures could not carry. Main's guard was applied quantitatively
before any derivation: every capture's section must contain at least half the
expected distinct values; **all 12 passed** (e.g. the k1x9 depthwise section
holds 10,241 distinct halfwords = 9,216 taps + 1,024 biases + wrap, the dense
n2048 section 65,536 per window × 32 windows).

## Covered vs refused (the nine forms, after round 2)

| # | encoder form (spelled) | n | verdict |
|---|---|---:|---|
| F3 | r3 out-proj k1x1 valid g1 nobias `[1,1024,375]` | 24 | **LOWERED** (round 1): W-major respell, 2-task program |
| F2 | r3 depthwise k9 g1024 **bias1** custom `[4,4]` | 24 | **LOWERED** (round 2): custom→same normalization (pad [4,4] is exactly the same pad of a unit-stride odd 9-tap kernel), W-major k1x9 respell `[1,1024,1,375]`, one linked 2-task slotted program |
| F7 | r4 pointwise k1x1 valid bias1 `[1,256,750,32]` | 1 | **LOWERED** (round 2): `pack_dense` with the bias row reproduces the distinct-payload section byte-exactly — round 1's 131,072 B uniform section was Apple constant-dedup on colliding values, not a missing bias row |
| F9 | r4 pointwise k1x1 valid bias1 `[1,256,375,16]` | 1 | **LOWERED** (round 2), same formula |
| F1 | r3 in-proj k1x1 valid g1 nobias `[1,1024,375]→[1,2048,375]` | 24 | **refused — payload-width aliasing bound**: the capture shows 16-lane planes with 256-row segments (rowblock order fully decoded, bias-free) but the column-supergroup order aliases mod 64: a 16-bit payload cannot distinguish colgroups 64 apart (`e → e + 64·1024` preserves every stored halfword), so no capture of this geometry can qualify the packer for columns ≥ 128. Row 1's `k9x1`-class section (20480 B) also reproduced under `pack_depthwise` + bias — the H-major row landed as a decoded row with no encoder consumer. |
| F4 | r4 padconv k1x1 g8 pad `[0,0,1,0]` custom | 24 | **refused — no pad rewrite**: the W-padded `[1,8,375,750]` surface section now reproduces (`pack_dense` grouped) and the row landed, but the encoder's 749-wide input needs an explicit pad column at W=0, which no decoded surface interpretation expresses; needs a pad-op emission or planner rewrite |
| F5 | r4 stem k3 s2 g1 bias1 custom=`same` | 1 | **refused — strided multi-tap transform**: the distinct-payload section carries 1,139 halfwords outside the weight/bias value ranges (negation/rescale-class transforms inside Apple's stride-2 macro), not a permutation; not decoded |
| F6 | r4 stem-dw k3 s2 g256 bias1 custom=`same` | 1 | refused — same strided transform |
| F8 | r4 stem-dw k3 s2 g256 `[1,256,750,32]→[1,256,375,16]` | 1 | refused — same strided transform (this surface pair now captured) |

**Covered: 50 of 101 instances** (F3 24, F2 24, F7 1, F9 1). Refused: 51,
every refusal carrying a named, evidence-backed gap.

## Table and code deltas

- `research/oracles/h13/`: +6 round-2 records (k1x9-bias1 W-major, k9x1-bias1
  H-major, pw375, pw750, pad-surface g8, tiny-g8) on top of round 1's three.
- `research/mint_conv_probes.py`: `case_weights` parses the
  `uint16_le_index_plus_one_wrapping` pattern; `pack_depthwise_slots` — the
  decoded 64-lane × 22-halfword slot layout (`b b t0 . t2 t1 t4 t3 t6 t5 t8
  t7 t1 t0 t3 t2 t5 t4 t7 t6 . t8` per slot, slot s ↔ channel
  `16·(s mod 64) + s div 64`) — dispatched for W-major depthwise-with-bias;
  `MULTI_TASK_CASES` gains the k1x9 record.
- `plugins/H13`: `packConvDepthwiseSlots` (C++ mirror of the slot formula);
  `packConvWeights` dispatches W-major depthwise-with-bias to it;
  `convParityPlan` normalizes the rank-3 custom spelling to `valid` (zero
  pads) or `same` (symmetric `(k-1)/2`, unit stride, odd kernel) — the only
  custom form any oracle qualifies; the conv reject message updates.
- `tests`: encoding asserts extended (new rows supported, rect-flip and
  strided forms refused); conv envelope CLI gains the F2 positive case (2
  task descriptors) and drops it from the refusal set.

## Verification (this host + the macstudio oracle compile)

- F2 word-exactness: the rank-3 custom-padded depthwise+bias spell compiled
  against the capture's exact blob; `parity.assert_tasks` +
  `parity.assert_constants` pass — byte-identical to
  `encoder_conv_idx_c1024_n1024_k1x9_s1_g1024_bias1_same_wmaj`.
- `make test-h13` PASS (full battery incl. `mint_conv_probes --check`).
- `make test-h13-parity`: the round-2 records run as parity cases (anec +
  hwx, byte-compared); suite total 882 → **888 cases / 1776 artifacts**,
  293 convolution.

## Bounds discovered (for the record)

1. **16-bit payload aliasing**: a uint16 payload cannot qualify dense
   colgroup order beyond 64 columns (`e → e + 64·1024` is value-invariant).
   Dense geometries above 64 output columns need either the consecutive
   order to be the true one (as at n=256, where pack_dense matched
   exactly and no other order could have) or device measurement, as for the
   linear plane permutation.
2. **Strided multi-tap sections are transforms, not permutations**: the
   k3-stride-2 sections embed computed halfwords — a micro-architecture
   decode, not a packing read-off.
3. Apple constant-dedup: the uniform captures' sections can be SMALLER than
   the true layout when payload values collide; section-shape arguments on
   uniform captures are unsound (round 1's F7 "missing bias" was this).

## What remains for a round 3

- F1 (24): decide the colgroup order by device measurement (jw16-class
  gate), or find a payload scheme that breaks the 64-column aliasing.
- F4 (24): a pad-op emission (or planner rewrite) feeding the qualified
  padconv row.
- F5/F6/F8 (3): decode the stride-2 transform semantics.
