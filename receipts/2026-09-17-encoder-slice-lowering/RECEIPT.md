# H13 encoder slice_by_index last-dim lowering receipt (2026-09-17)

## Source

- Assignment: lower the decoded last-dim `slice_by_index` encoder forms to
  h13; everything the decoded corpus does not cover stays refused.
- Base: mil-hwx-compiler `agent/encoder-transpose-lowering` `bfa4aed`
  (the transpose lowering this mirrors; parity 878).
- Worktree: `~/.config/superpowers/worktrees/mil-hwx-compiler/slice-lastdim`,
  branch `agent/slice-lastdim-lowering`.
- Oracle: `encoder_slice_lastdim` from
  `receipts/2026-09-16-compiler-leftover/oracles-round1/` (minted on
  MacStudio.local against source commit `b61de468`, tool SHA-256
  `3d13fc85c2…`), copied verbatim (SHA-256 `ccca0d93…`) to
  `research/oracles/h13/encoder_slice_lastdim.json`. Host-only session: no
  ANE execution, no jwm1/jw16 contact, no compiler plugin ran on-device.

## What lowered

| form | count | before | after |
| --- | --- | --- | --- |
| `slice_by_index` last-dim window `[1,8,375,749] → [1,8,375,375]` (`begin [0,0,0,0]`, `end [1,8,375,375]`, unit strides, no squeeze; the `end_mask` spelling of the same ranges included) | 24 (encoder rel-pos windowing) | rewrote at the planner, 101 MB rewrite cost (per `receipts/2026-09-17-encoder-transpose-lowering/RECEIPT.md`); compiler had no lowering | one 1-task h13 program (`apple-parity-slice`), 504-byte task stream, zero constant section |

## What stays refused (unchanged codes)

| form | reason |
| --- | --- |
| same shape pair opened mid-row (`begin [0,0,0,374]`) | no decoded task stream — the recorded task copies from element 0 of each row; refuses `h13.noncontiguous-slice` |
| the encoder's other slice family, mask slicing `[1,8,750,375] → [1,8,749,375]` | narrows a non-last axis; no oracle; refuses `h13.noncontiguous-slice` (chunked message) |
| any other surface pair (e.g. `[1,4,375,749]`) | table row absent; refuses `h13.noncontiguous-slice` |
| strided windows | unit-stride ABI; refuses `h13.noncontiguous-slice` |
| non-last-axis narrows, multi-axis slices, dropped/squeezed axes | unchanged view-path refusals |
| constant-backed sources | stay on the view path, which names the exact blocker |

## Change

- `receipts/2026-09-17-encoder-slice-lowering/emit_slices.py` regenerates
  `plugins/H13/H13SliceTemplates.inc` from the 2026-09-16 oracle (1 row,
  126 words); word reconstruction re-verified against the recorded
  504-byte `size_bytes`.
- `H13Program.h/.cpp`: `OracleSliceTemplate`, `sliceTemplate` lookup,
  `supportsSliceParity`/`encodeSliceParity` — same whole-surface shape as
  the transpose encoders (`elementwiseTensor(5, input)` /
  `elementwiseTensor(4, output)`; the oracle TDs match the layout formula
  exactly: input strides `[4608000, 576000, 1536, 2]`, output
  `[2304000, 288000, 768, 2]`).
- `ANEH13Compiler.mm`: `sliceParityShapes` gates on fp16 rank-4
  `[1,C,H,W]`, resolved ranges full on axes 0–2 and `[0, resultWidth)` on
  axis 3, no stride/squeeze/mask-drop, table-covered pair; the rewrite pass
  emits the whole op before the view plan, `lowerOperation` and the
  schedule loop treat it as a whole-tensor `apple-parity-slice` program.
- `tests/test_h13_slice_cli.py` new (2 covered spellings, 4 refusals),
  wired into `make test-h13`.
- `tests/test_h13_layout_cli.py`: `slice-encoder-bias` pinned the old
  refusal contract for the now-covered form; it now asserts the compile
  and `apple-parity-slice` (same treatment the transpose commit gave
  `encoder-residual-add`). `slice-encoder-mask` keeps its refusal.
- `tests/test_h13_parity.py`: `encoder_structure` selection takes the
  `slice_lastdim` probe, encoder dispatch `apple-parity-slice`, family
  count 4 → 5.

## Verification (this worktree, Linux omp-studio-local)

| check | result |
| --- | --- |
| `make -j8 build/mil-hwxc` | 0 errors (`-Wall -Wextra -Werror`) |
| `python3 tests/test_h13_slice_cli.py build/mil-hwxc` | PASS (2 covered, 4 refusals) |
| `python3 tests/test_h13_parity.py build/mil-hwxc` | **PASS — 879 cases / 1758 artifacts** (was 878; +1 slice oracle, task stream word-identical to the Apple oracle, both anec and hwx) |
| `python3 tests/test_h13_layout_cli.py build/mil-hwxc` | PASS |
| `make test-h13` battery | PASS (17 CLI suites incl. the new slice suite) — see below |

## Notes

- The gate is deliberately table-driven: only the decoded geometry lowers.
  A different window width or surface pair needs a freshly minted Apple
  oracle before it can join `kSliceTasks`.
- Commit: `agent/slice-lastdim-lowering` `4587964`, parent `bfa4aed`
  (`agent/encoder-transpose-lowering`).
