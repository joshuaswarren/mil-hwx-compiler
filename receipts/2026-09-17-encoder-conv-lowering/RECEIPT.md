# Encoder conv leftover: covered lowerings land, the rest refuse by name (2026-09-17)

Branch `agent/slice-lastdim-lowering` (own worktree off f5cd370). Host-only:
no ANE execution, no jwm1/jw16 contact, no pin change, no performance-parity
claim.

## What the leftover is

The Parakeet encoder MIL
(`mlx-omarchy/receipts/2026-09-14-encoder-leftover/model-fp16-allpeels.mil`)
carries **101 convs in 9 distinct forms** (re-census this session, lines
resolved per op). This was the last whole family still refusing H13 with
`h13.conv-outside-envelope` after transpose (bfa4aed) and slice (f5cd370).

## The decisive oracle fact

The 12 checked-in conv captures
(`receipts/2026-09-16-compiler-leftover/oracles-round1/encoder_conv_*.json`,
all rank-4 same/valid respells) were minted with the `weight_blob` payload:
**one uniform fp16 halfword per constant region** (`0x3400` + ordinal). The
section histograms prove every weight element compiled to the same bits
(e.g. the 2 MB k1x1 section is 1,048,576 copies of `0x3400`), so **no
constant-section permutation is readable from any of them**. What each
record still qualifies: Apple's exact task stream (linked task arrays,
kernel/stride/DMA geometry) and the section's shape (size, zero structure,
bias placement) against the packing formulas the distinct-payload probe
corpus already qualified (`conv_probe_*` / `conv_known_*`, bit-plane
payloads).

## Covered vs refused (the nine forms)

| # | encoder form (spelled) | n | respelled geometry | oracle record | verdict |
|---|---|---:|---|---|---|
| F3 | r3 `[1,1024,375]→[1,1024,375]` k1 valid g1 nobias (out-proj) | 24 | W-major `[1,1024,1,375]` | `encoder_conv_c1024_n1024_k1x1_s1_g1_bias0_valid` (2 tasks) | **LOWERED** |
| F1 | r3 `[1,1024,375]→[1,2048,375]` k1 valid g1 nobias (in-proj) | 24 | W-major `[1,1024,1,375]→[1,2048,1,375]` | none survives — the checked-in n2048 capture is **H-major** `[1,1024,375,1]` (6 tasks), whose input surface the rank-3 tensor does not bind as | **refused** (`h13.conv-outside-envelope`, named) |
| F2 | r3 `[1,1024,375]` k9 g1024 **bias** custom `[4,4]` (depthwise) | 24 | W-major k1x9 | only `k9x1` bias0 decoded; the `k1x9` capture's section (40,960 B = 9,216 weights + 2,048 zero bytes) has unmodeled structure; bias1 never minted | **refused** |
| F4 | r4 `[1,8,375,749]→[1,8,375,750]` k1x1 s1 g8 pad `[0,0,1,0]` custom (rel-pos padconv) | 24 | pad-then-valid respell `[1,8,375,750]` | no record at that surface (the round-2 capture was overwritten by the round-3 `[1,8,8,8]` control under the same case name) | **refused** |
| F5 | r4 `[1,1,3000,128]→[1,256,1500,64]` k3 s2 g1 bias1 custom=`same` (stem) | 1 | same | `encoder_conv_c1_n256_k3x3_s2_g1_bias1_same` decoded, but its 6,144 B section does not reproduce (`pack_strided` is k1-only; multi-tap stride-2 layout unmodeled) | **refused** |
| F6 | r4 `[1,256,1500,64]→[1,256,750,32]` k3 s2 g256 bias1 custom=`same` (stem dw) | 1 | same | `encoder_conv_c256_n256_k3x3_s2_g256_bias1_same` decoded, section 7,168 B unmodeled | **refused** |
| F7 | r4 `[1,256,750,32]` k1x1 s1 g1 bias1 valid (pointwise) | 1 | — | `encoder_conv_c256_n256_k1x1_s1_g1_bias1_valid` decoded, but its section is 65,536 halves = **weights only**; the 256-half bias is absent and its placement is unresolved | **refused** |
| F8 | r4 `[1,256,750,32]→[1,256,375,16]` k3 s2 g256 bias1 custom (stem dw) | 1 | same | no record at this surface pair (round-2 case deduped by name before submission) | **refused** |
| F9 | r4 `[1,256,375,16]` k1x1 s1 g1 bias1 valid | 1 | — | no record (same dedupe) | **refused** |

**Covered: 24 of 101 instances (F3).** Refused: 77 instances across 8 forms,
every refusal carrying its named gap.

## What landed

- `research/oracles/h13/`: +3 records — F3 (above), F1's H-major capture
  `encoder_conv_c1024_n2048_k1x1_s1_g1_bias0_valid` (6-task row, rank-4
  geometry — no encoder consumer but a decoded row), and the rect key
  `encoder_conv_c1024_n1024_k9x1_s1_g1024_bias0_same` (1-task 9x1 row — the
  "rect keys not in table" item; the encoder's bias-bearing depthwise still
  refuses on the bias flag).
- `research/mint_conv_probes.py`:
  - `case_weights` parses the `weight_blob` payload ("fp16 bits 0x3400 +
    index, one value per constant" = per-constant uniform regions, no inline
    blob header). The description's sha256 records what the mint generated;
    the constant-section comparison in `covered()` adjudicates what actually
    compiled — fail-closed either way.
  - `weight_taps` + rect-aware `conv_constants`/`template_key`/`conv_row`:
    keys are now `(kh, kw, stride, groups, bias, in, out)`.
  - `covered(record, allow_multi_task=False)` + `MULTI_TASK_CASES`: only the
    two named encoder captures may carry linked task arrays; every other
    multi-task conv capture (the k3 stride-2 corpus) stays outside the
    tables. `stream_words` rebuilds linked streams with their alignment
    padding (matvec-emitter semantics).
- `plugins/H13`: `ConvShape`/`OracleConvTemplate` gain `kernelWidth`;
  `convParityPlan` accepts rectangular kernels (per-axis extent checks
  against the axis's own kernel extent) and the encoder's rank-3 spell
  (rank-2 `[p0,p1]` pad, rank-1 stride/dilation vectors — including their
  bare-scalar `tensor<int32,[1]>(1)` spelling — respelled onto the W-major
  surface the rank-3 tensor binds as). `packConvWeights` and the lowering
  site use `kh·kw` taps. The conv reject message names the encoder gaps.
- `plugins/H14`: same `kernelWidth` plumbing so the regenerated
  `H14ConvTemplates.inc` row format parses (H14 gate stays square-only).
- `tests/test_h13_encoding.cpp`: conv envelope asserts moved to the
  `(kh, kw, …)` contract; the three new rows assert supported, the rect
  flip (`1x9` on the 9x1 surface) and the bias-bearing depthwise assert
  refused.
- `tests/test_h13_conv_envelope_cli.py`: F3's rank-3 spell compiles as one
  `apple-parity-conv` program with 2 task descriptors; F6/F8 refusals added
  to the existing nine-form refusal set.
- `tests/test_h13_parity.py`: `conv_covered` passes the same multi-task
  opt-in, so the three new records join the parity set (a first run with the
  gate unpatched silently selected only the 1-task record — the suite's
  family assert is self-derived and would not have caught it).
- `Makefile`: `test-h13` now runs `mint_conv_probes.py --check` so the
  committed conv tables cannot drift from the oracle set.

## Verification (this host, no device)

- F3 word-exactness: the rank-3 encoder spell compiled with the oracle's
  exact payload; `parity.assert_tasks` + `parity.assert_constants` pass —
  task stream and constant section byte-identical to
  `encoder_conv_c1024_n1024_k1x1_s1_g1_bias0_valid`.
- `make test-h13` PASS (full battery, this worktree).
- `make test-h13-parity`: the three new records run as parity cases (anec +
  hwx, byte-compared against the decoded streams); suite total 879 → **882
  cases / 1764 artifacts**.

## What the next mint round needs (all host work on the oracle Mac)

1. Re-mint F1 W-major (`(1,1024,1,375)→(1,2048,1,375)` k1x1 valid), F2 bias1
   W-major k1x9, F4's `[1,8,375,750]` padconv surface, F7's bias1 pointwise,
   F8/F9's `[1,256,375,16]` pair — with **per-element distinct payloads**
   (the `uint16_le_index_plus_one_wrapping` pattern) so the packings
   qualify.
2. Derive the multi-tap stride-2 section layout (F5/F6) and F7's bias
   placement from those captures; the k1x9 depthwise section's 2 KB zero
   structure likewise.

## Commit pins

| item | value |
| --- | --- |
| base | mil-hwx-compiler `f5cd370` (`agent/slice-lastdim-lowering`) |
| oracle records | `receipts/2026-09-16-compiler-leftover/oracles-round1` (unchanged) |
| compiler source | this worktree at the commit carrying this receipt |
