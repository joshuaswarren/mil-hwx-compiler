# Encoder leftover H13 oracle campaigns (2026-09-16)

Three rounds of Apple-oracle mints against the Parakeet encoder leftover op
families (`mlx-omarchy receipts/2026-09-14-encoder-leftover.md`), on the road
to lowering those families in this repository. Host-only: no ANE execution, no
jwm1/jw16 contact, no compiler plugin code modified, no pin change.

## Provenance (every record carries its own copy)

| item | value |
| --- | --- |
| base | mil-hwx-compiler `b61de468` (`feature/strict-bind-surface-naming`) |
| branch | `encoder-leftover-oracles` (this commit; assets only) |
| tool | `/tmp/h13-oracle/bin/ane-compile-hwx` 52336 B, SHA-256 `3d13fc85c2a6baa0b7628f0848269b16fb99f742180f0085f608c371bb9da6e0` (same binary the batched round-2 campaign used) |
| host | `MacStudio.local`, macOS 26.6.2 arm64, build 25G83 |
| decoder/driver | `research/h13_td.py` `125da024…`, `research/mint_oracles.py` `a6e8f70d…` (hashed per record) |
| target | h13 only (qualification target) |
| campaigns | `mint_encoder_leftover.py` (round 1, 28 cases), `mint_encoder_leftover2.py` (round 2, 27 cases), `mint_encoder_leftover3.py` (round 3 controls, 9 cases) |

Round totals: 64 cases, 49 decoded, 15 refused (`callback_status=1`), after
fixing a campaign-side MIL literal bug (below).

## What Apple's verdicts are

**Decoded — lowerings are now derivable from checked-in bytes:**

| family | cases | headline results |
| --- | ---: | --- |
| rank-3 `linear` | 12 | all five encoder geometries (`m375 k∈{1024,4096} n∈{128,640,1024,4096}`) compile as single programs of 2–17 tasks; const = weight bytes + a 2-block per-vector bias region when the bias is non-uniform (uniform biases are constant-folded into scalar register `0xc80c`, which is why round 1's uniform payload missed it) |
| B=8 head-projection bmm | 2 | `[8,375,1024]×[8,1024,128]` `ty=1` compiles as the same 209-task structure as the island-A captures |
| rank-3 unaries | 3 | `silu [1,375,4096]`, `silu [1,1024,375]`, `sigmoid [1,1024,375]`: 1 task, 128-byte section = the existing `kSiluKERNWords`/`kSigmoidKERNWords` tables verbatim |
| normalizations | 3 | `softmax [1,8,375,375]` = **5 tasks** (vs 8 at C=1), `softmax [1,1,375,375]` = 8 tasks, `layer_norm [1,375,1024]` no-params = 5 tasks, zero sections |
| transposes | 4 | `r3 [0,2,1]` both directions at `[1,375,1024]↔[1,1024,375]`, `r4 [0,2,1,3]` at `[1,8,375,128]` and `[1,375,256,16]`: 1 task each, zero sections — **the "non-foldable transpose" hole is Apple-lowerable** |
| slice_by_index | 1 | `[1,8,375,749]→[1,8,375,375]` last-dim window: 1 task — the rel-pos windowing slice is Apple-lowerable |
| broadcasts | 10 | per-channel `mul`/`add` on `[1,375,1024]`·`[1,1,1024]` and `[1,375,4096]`·`[1,1,4096]` (the layer_norm gamma/beta peel), on `[1,1024,375]`·`[1,1024,1]` (H-aligned), runtime `mul` at `[1,1024,375]` and `[1,375,4096]` (GLU/FFN products), runtime `add` at `[1,8,375,375]` (attention residual), per-head `add [1,8,375,128]·[1,8,1,128]` (pos bias) |
| convolutions | 12 | all nine encoder forms lower through rank-4 `same`/`valid` respells: pointwise `k1×1` (W-major `(1,1024,1,375)`), depthwise `k9×1` `same` `g1024`, grouped padconv as `k1×1 g8 valid` on the W-padded surface, stem convs `same`/`valid` with bias; square and rect controls included |
| FFN chain | 1 | `matmul→bias add→silu→matmul→bias add` at `d1024 s375`: **one program, 28 tasks**, 16.8 MB constant section — the `linear`-op spelling refuses, the matmul spelling decodes |

**Refused by Apple (recorded, with the planner consequence):**

| probe | verdict | consequence |
| --- | --- | --- |
| `layer_norm` with `gamma`/`beta` | refused | planner peels `y = normalize(x)·γ + β` into one norm program + two per-channel broadcasts (all three decoded) |
| `concat` (`x0/x1` and `values` spellings) | refused | ISA hole confirmed at encoder forms; eliminated structurally (the B=8 bmm respell removes the 72 head concats) |
| `linear`-op chains | refused | chains use the matmul spelling (decoded) |
| rank-3 conv spellings | refused | planner emits the rank-4 respells above |

## Row-padding fact the lowering needs

Every decoded elementwise/norm/broadcast surface uses
`row = alignUp(width·2, 64)` — width 375 takes a 768-byte row — not
`max(64, width·2)`. The WIP branch state that implements this
(`H13Program.cpp elementwiseTensor/matvecTensor`, regenerated
`H13NormTemplates.inc` incl. `softmax_1x8x375x375` = 5 tasks / 1024-byte
exp-table section, and the `mint_norm_probes` surface fix) is preserved in
`git stash` on this branch (`encoder-leftover WIP: row-formula + norm table +
emitter patches`). It was verified to lower `softmax_1x8x375x375` as one
program; the remaining dispatch work (rank-3 unary/broadcast acceptance, LN
argument gate, matvec rows + m375 packing, transpose/slice tables, rect conv
keys) is not started in plugin code.

## Campaign bug found and fixed mid-run

Rounds 2–3 initially refused every conv case: the campaign's `st`/`pd`/`dl`
const literals missed one closing bracket (`([1, 1]);` vs `([1, 1])];`).
Apple's tool refuses malformed MIL with `callback_status=1`, so the syntax bug
masqueraded as an envelope refusal until a square known-good control through
the same builder also refused. All conv conclusions come from the fixed
builder. The first `_idx` remint also compiled uniform payloads (index bytes
went to a sidecar instead of the compile input); the committed `_idx` records
are the corrected re-mints whose `weights.bin` description is
`uint16_le_index_plus_one_wrapping` and whose sections therefore expose the
bias region.

## Layout

- `mint_encoder_leftover{,2,3}.py` — campaigns (run with `--host macstudio`)
- `oracles-round1/` — 95 files: every decoded record JSON + `__TEXT,__const`
  sidecar for rounds 1–3, kept out of `research/oracles/h13/` so the parity
  suite stays at its qualified 846 cases until the lowerings land
- `oracles/` — raw round output as retrieved from the oracle host (includes
  the refused records: acceptance evidence for concat/LN-affine/linear-chain)

## Verification at commit time

- `make test-h13` PASS, `make test-h13-parity` PASS (846 cases, unchanged)
- `git stash` holds the WIP plugin/tree edits; `plugins/H13` is byte-identical
  to `b61de468`
