# F4 padconv round: the rank-4 custom-pad respell decodes, the W-padded
# rel-pos conv lowers end to end (2026-09-17)

Continuation of `receipts/2026-09-17-encoder-conv-lowering/RECEIPT.md`
(round 2) on the same base commit `073c7b8`. Host-only: the only remote
contact was the oracle compile on `macstudio` over BatchMode SSH, same
receipted binary as rounds 1-2 (`/tmp/h13-oracle/bin/ane-compile-hwx`, sha256
`3d13fc85c2a6baa0…`, verified before every mint). No jwm1/jw16, no ANE
execution.

## What broke open

Round 2 recorded F4 as "refused — no pad rewrite: … needs a pad-op emission
or planner rewrite", and `convParityPlan` carried the comment "Apple's own
tool refuses the custom spelling". That refusal was a wrong generalization:
round 1/2 evidence only ever covered **rank-3** custom spells and the `pad`
op. The encoder's actual F4 form is a **rank-4** conv — `k1x1 s1 g8 bias0`,
input `[1,8,375,749]`, output `[1,8,375,750]`, `pad_type="custom"`,
`pad=[0,0,1,0]` — and that spelling had never been minted.

`mint_encoder_padconv.py` minted it. **All three cases decoded**: the F4
geometry as a **two-task program, 1536-byte section**, a tiny control
(`[1,8,8,7]→[1,8,8,8]`, byte-identical section to the F4 capture), and a
symmetric-custom k3x3 control (1 task, 1024 bytes — kept as evidence, not
curated). Apple's task stream reads the **749-wide input directly** (surface
binding 1, shape `[1,8,375,749]`, row stride 1536) and writes the 750-wide
output — the zero column at W=0 is produced inside Apple's own stream. No
pad-op emission and no planner rewrite is needed; the padconv is its own
whole-program capture.

## The section, decoded by same-format payload runs

`covered()` could not reproduce the 1536-byte section, and it carries too few
distinct values to look like a weight permutation (the round-2 distinctness
gate would fail it). Three payload runs with the round-2 multi-record blob
format (payloads `1..8`, `2,4,…,16`, `0x3801×8`) — the first byte-identical
to the landed capture as the control — decode it exactly:

- three 512-byte chunks of eight 64-byte lanes;
- chunks one and two are structural and payload-independent: even lanes hold
  `0x3c00` (fp16 1.0), odd lanes hold `0x0001` then `0x0100`;
- chunk three holds one **raw weight halfword per lane in channel order** —
  proven identity-ordered by the strictly ascending payloads (any
  non-identity lane permutation would break monotonicity).

Methodology note kept for the record: a first probe batch used the
`om.blob` single-record blob layout, which declares its payload at file
offset 128 where the corpus layout declares `0x18080`. Those compiles read
the wrong payload bytes and produced sections that differ from the corpus
format in exactly the weight words — they initially looked like proof that
the section ignores the weights. They are invalid compiles, retained in
`oracles-padconv/` with the other mint-lab records; the curated corpus rows
all use the corpus blob layout.

## Landing (this commit)

- `research/oracles/h13/`: +2 curated records with their const sections —
  `encoder_conv_pad_c8_n8_k1x1_s1_g8_bias0_p0010_f4` and `..._tiny`
  (`uint16_le_index_plus_one_wrapping` payload, rank 4, `pad_type` custom,
  `pad` `[0,0,1,0]`).
- `research/mint_conv_probes.py`: `pack_padconv_columns` (the three-chunk
  formula, capture-pinned to the 8-channel grouped k1x1); `conv_constants`
  dispatches it for the no-bias k1x1 custom-pad depthwise form;
  `MULTI_TASK_CASES` gains both rows.
- `plugins/H13/H13ConvTemplates.inc`: regenerated
  (`--emit-templates --targets h13`); two new rows, e.g.
  `{1,1,1,8,false,{8,375,749},{8,375,750}, kH13ConvText76, 628, 2, 1408,
  1536, 4521984}`. H14 output checked unchanged.
- `plugins/H13/H13Program.cpp`: `packConvPadconvColumns` (C++ mirror of the
  formula) and a dispatch in `packConvWeights`'s depthwise branch — a
  width-changing k1x1 no-bias grouped depthwise is exactly the custom-pad
  form; the valid/same rows keep `packConvDepthwise` because their widths
  agree.
- `plugins/H13/ANEH13Compiler.mm`: `convParityPlan` now splits the custom
  spelling by rank — rank-3 keeps the `same`/`valid` normalization (F2's
  path, unchanged), rank 4 accepts the declared pads when they reproduce the
  declared output per axis (`floor((D + pad_lo + pad_hi − k)/s) + 1`); the
  template key still pins the exact surfaces, so any custom form without a
  covering row refuses as before. Stale comments and the conv reject
  message updated; F1/F5/F6/F8 refusals stand.
- `tests/test_h13_encoding.cpp`: the F4 geometry flips from the refusal set
  to `supportsConvParity` (tiny too); an asymmetric pair with no capture
  (`{8,100,49}→{8,100,50}`) pins the refusal.
- `tests/test_h13_conv_envelope_cli.py`: `enc-padconv` moves to the positive
  set (apple-parity-conv, 2 task descriptors).

## Verification (this host + the macstudio oracle compiles)

- Section reproduction: `conv_constants` reproduces both curated records
  byte-exactly (1536 B, sha256 match), and the control payload run
  reproduces the landed F4 section byte-for-byte before any formula was
  written down.
- `make test-h13`: **PASS** — encoding asserts (flipped F4), conv envelope
  CLI (enc-padconv lowers with 2 task descriptors), and
  `mint_conv_probes --check` ("H13 conv templates: up to date", "H14 conv
  templates: up to date").
- `make test-h13-parity`: **PASS 890 cases / 1780 artifacts (was 888/1776),
  295 convolution** — both padconv records compile through the real encoder
  and byte-compare against the Apple captures on anec and hwx.
- `make test-hwx-inspection`: PASS.

## Pre-existing failures at the base commit (not this round's)

`make test-h13-reference` (`inspect_anec.binding_interval`: "slice has
incorrect fields"), `make test-h13-simulation` (`inspect_anec.py:134`
`NameError: destination is not defined`), and `make test-h14-parity`
(`H14NormTemplates.inc` stale) fail identically on the pristine `073c7b8`
checkout (`worktrees/mil-hwx-compiler/slice-lastdim`, clean status). They
are in-flight tooling breakage from another lane; none of the three files is
touched here.

## State of the encoder conv leftovers after this round

| # | form | n | verdict |
|---|---|---:|---|
| F3 | r3 out-proj k1x1 valid g1 nobias | 24 | lowered (round 1) |
| F2 | r3 depthwise k9 g1024 bias1 custom `[4,4]` | 24 | lowered (round 2) |
| **F4** | **r4 padconv k1x1 g8 pad `[0,0,1,0]` custom** | **24** | **lowered (this round)** |
| F1 | r3 in-proj k1x1 valid g1 nobias → n2048 | 24 | refused (payload-width aliasing; colgroup-order decode in flight on `agent/f1-u32-colgroup-order`) |
| F5/F6/F8 | stem k3 s2 forms | 3 | refused (stride-2 sections carry transformed halfwords) |

Covered: **74 of 101 instances.**
