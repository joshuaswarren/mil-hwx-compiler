# F1 in-projection round 3: the 64-colgroup alias broken with a 32-bit-pair
# payload; colgroup order decoded past column 128 and lowered (2026-09-17)

Continuation of `RECEIPT.md` (round 2) in this directory. Base commit
`073c7b8` (round-2 remint, parity 888), branch `agent/f1-u32-colgroups`,
worktree `~/.config/superpowers/worktrees/mil-hwx-compiler/f1-u32-colgroups`.
Host-only: the only remote contact was the oracle compile on `macstudio`
(BatchMode SSH, macOS 26.6.2), same receipted binary
`/tmp/h13-oracle/bin/ane-compile-hwx`, sha256 `3d13fc85c2a6baa0…` (52,336 B,
byte-identical to rounds 1-2). No jwm1/jw16, no ANE execution, no pin change.

## The alias, and the payload that breaks it

Round 2 refused F1 (in-proj k1x1 valid g1 nobias `[1,1024,375]→[1,2048,375]`,
24 encoder instances) because a uint16 payload cannot distinguish colgroups
64 apart: with `uint16(index+1)`, the map `e → e + 64·1024` preserves every
stored halfword, so chunk order (start ≡ 0 mod 64·1024) is invisible and
plane-group order is visible only mod 4.

`mint_encoder_conv_u32.py` re-mints the **same geometry with a byte-identical
MIL** (verified `mil == round-2 mil`) under a payload of 32-bit-index pairs:
halfwords `2k`/`2k+1` carry the little-endian uint32 `k+1` (pattern
`uint32le_pair_index_plus_one`, one running stream across every constant).
Under the alias shift the pair counter moves by 32768, changing the stored
low halfword at every even element. Distinctness gate (Main's bar, applied
before any derivation): 1,507,329 of 2,031,616 comparable positions (74.2%)
differ under the 65536-halfword shift; 100% of even positions. Gate ≥ 50%.

The first mint attempt failed MIL validation because the finisher wrote the
raw payload without the BLOBFILE container (`blob_bytes_index`'s header +
4096-slot record table + 64-byte-aligned payload area); fixed and verified
against the round-2 container before re-running.

## Capture and decode

`oracles-u32/h13/encoder_conv_idx_c1024_n2048_k1x1_s1_g1_bias0_valid_wmaj_u32.*`:
decoded, 4 tasks, constant section 4,194,304 B (sha256 `9ff369be1c136a2b…`),
same section size as round 2.

`decode_f1_colgroups.py` decodes the placement without assuming the colgroup
order. Payload parity is row parity (`e = c·1024 + r`), so

- even rows carry `low16(k+1) = (512·(c mod 128) + r/2 + 1) mod 2^16`
  → `c mod 128` and `r/2` per position;
- odd rows carry `high16(k+1) = c // 128` (with one exact carry exception:
  `c mod 128 == 127`, `r ≥ 1022` reads one block higher — 16 stripes, all
  found, exactly the predicted signature).

Capture-side gate: 65,536 even-row stripes all 16-distinct-spread, 65,520
odd-row stripes 16-uniform-small, 16 boundary-carry stripes, 0 violations.

**Decoded layout** (`colgroup-order-u32.json`): 16 regions × 8 slots; slot
`(g, j)` is a 16-lane plane over 1024 consecutive rows holding colgroup

    G(g, j) = 8·(4·(3 − g//4) + ((g//2) mod 2) + 2·(j mod 2))
              + 4·(g mod 2) + j//2

Section colgroup order (region-major): `[96, 112, 97, 113, …]` — 32-colgroup
blocks in order 3, 2, 1, 0, each split low/high-16 interleave. All 128
colgroups appear exactly once; columns ≥ 128 are fully qualified.

**Verification**: the decoded layout reconstructs the u32 capture
byte-exactly AND the round-2 uint16 capture byte-exactly (`u16
byte-exact: True`) — two independent payloads, one layout. The uint16 round-2
record is thereby retroactively qualified; the 64-colgroup alias is closed.

Note: AneConvLeftover's interim hint `a(t) = ((t&1)<<2)|(t>>1)` does **not**
match the decoded order; it is consistent with a partial read under the
uint16 alias. The byte-exact two-payload decode supersedes it.

## Lowering

- `research/mint_conv_probes.py`: `pack_dense_colgroups` (mirror of the
  decoded layout, gated to the qualified 1024→2048 bias-free geometry);
  `conv_constants` dense dispatch selects it; the F1 record joins
  `MULTI_TASK_CASES` (4 tasks); `lane_cap` now takes the max surface extent
  (`spatial`, `spatial_width`) — the rank-4 respell record has height 1 and
  needs the 16-lane cap, matching the C++ `laneCap(max(h, w))`. Older
  records without `spatial_width` default to the old behavior.
- `research/oracles/h13/`: the round-2 uint16 F1 record
  (`encoder_conv_idx_c1024_n2048_k1x1_s1_g1_bias0_valid_wmaj`) lands as a
  parity case. The u32 record stays in this receipt directory as decode
  evidence (its pattern is not a `case_weights` payload and needs no parity
  byte-compare).
- `plugins/H13/H13ConvTemplates.inc`: regenerated
  (`mint_conv_probes.py --emit-templates`); the F1 row is the W-major
  4-task entry; symbol renumbering churns the diff.
- `plugins/H13/H13Program.cpp`: `packConvDenseColgroups` + dispatch in
  `packConvWeights`'s groups-1 dense branch (`!bias && stride 1 && taps 1 &&
  reduction 1024 && outputs 2048`). A first version computed `base` in
  halfwords while the cursor advanced in bytes and left the tail of the
  section zeroed — caught by the parity byte-compare on the uniform
  H-major record and fixed (`* lanes * 2`); a standalone probe now shows
  0 zeros / 2,097,152 written halfwords.
- `tests/test_h13_conv_envelope_cli.py`: `enc-1d-pw` moves from the refusal
  set to a positive case (4 task descriptors); blob enlarged to 4 MiB for
  the in-projection weight.
- `tests/test_h13_encoding.cpp`: `supportsConvParity` gains the W-major
  in-projection row `{1024, 1, 375} → {2048, 1, 375}`.
- `tests/test_h13_layout_cli.py`: the `transpose-encoder-conv` feeder no
  longer refuses — with the row landed, the materialized transpose feeds a
  lowering 4-task parity conv (`transpose-encoder-conv` asserts
  `transpose/apple-parity-transpose/1` + `conv/apple-parity-conv/4`; the
  test now writes a valid blob subheader + 4 MiB payload).

## Verification

- `make test-h13`: **PASS** (encoding, anec, CLI battery, conv envelope CLI,
  `mint_conv_probes --check` templates up to date, H13 + H14).
- `make test-h13-parity`: **PASS — 889 cases / 1778 artifacts, 294
  convolution** (round 2: 888 / 1776 / 293). The F1 in-projection spell now
  byte-compares against Apple's capture in both anec and hwx formats.
- The 24 encoder F1 instances lower through the rank-3 → W-major respell the
  template row covers.

## Bounds

1. The colgroup order is qualified for the captured geometry only (k1x1 g1
   valid, 1024→2048, bias-free, 16-lane cap). Other dense geometries keep
   the consecutive `pack_dense` order their own captures qualified.
2. The uniform-payload round-1 H-major record (`encoder_conv_c1024_n2048…`)
   becomes trivially covered (one distinct halfword — every layout
   reproduces it); its task stream and section were already byte-compared
   before this change and remain so.
3. The u32-pair payload trick generalizes: any dense geometry whose colgroup
   order a uint16 capture cannot qualify can be reminted with
   `uint32le_pair_index_plus_one` and decoded by row-parity stripe analysis.

## Coordination

EncoderF4Pad (padconv lane, commit `80c7b4d` on `agent/f4-padconv-enc`) was
hubped before every shared-file edit; the hunks are disjoint by design
(their: padconv packers + convParityPlan pad acceptance + depthwise-branch
dispatch; mine: dense-branch dispatch + packer). `H13ConvTemplates.inc` is
regenerated on both branches — whoever integrates re-runs
`--emit-templates --targets h13` after combining the record sets.
