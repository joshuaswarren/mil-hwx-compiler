# H13 layout stream-2 lowering receipt (transpose / slice_by_index / pad / tile)

## Source

- Compiler base: `83a4434810b0981d1764e6231f6a679df6f35241` (`origin/main`, the `ane-compiler.lock` pin)
- Branch: `feature/h13-layout-stream2`
- Encoder histogram: `receipts/2026-09-13-encoder-coverage/encoder-coverage.json` in `ane-linux-experiments/parakeet-mel-exact` — stream 2 is transpose 146, slice_by_index 48, pad 24, tile 1 (219 of 3351 ops)
- Encoder artifact analyzed: `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed)

## Lowering contract

All four ops now have H13 dispatch branches. Each lowers host-side in exactly
the forms the one-slice binding ABI can represent; every other form rejects
with a named code stating the exact IR blocker.

- **transpose**: a permutation whose row-major element order is unchanged
  (per free output axis, the non-unit suffix product equals the input
  dimension's input stride) lowers as a free manifest alias, like reshape.
  A trailing-two-dimension swap feeding exactly one `matmul` x/y operand
  folds into that matmul's `transpose_x`/`transpose_y` flag — verified
  byte-exact against the directly-written equivalent matmul for the y axis
  (both parity keys `rrmm_r3rr_m64_k64_n64_tx0_ty0` and `..._ty1` are in the
  decoded corpus). Everything else rejects `h13.nonfoldable-transpose`: the
  binding ABI reads storage row-major and the decoded corpus holds no encoder
  that permutes surface strides.
- **slice_by_index**: unit strides, at most one sliced dimension, unit
  dimensions before it, `begin_mask`/`end_mask`/negative bounds resolved —
  lowers as one offset view through the same `valueBaseOffsets` mechanism the
  split lowering established; consumers read the storage at
  `elementOffset`. Full-range slices are manifest aliases. `stride != 1`,
  two sliced axes, non-unit head dimensions, or dropped axes reject
  `h13.noncontiguous-slice`.
- **pad**: all-zero amounts are the identity alias. Any non-identity pad
  rejects `h13.unsupported-pad`: every decoded H13 encoder writes an output
  surface of its input's shape, so the padded region has no host-side
  producer.
- **tile**: all-ones reps are the identity alias; anything else rejects
  `h13.unsupported-tile` with the same blocker.

IR registry (`ANEOperationGraph.mm`) now classifies `slice_by_index`, `pad`,
and `tile` as `ANEOperationKindLayout` alongside `transpose`.

## Exact blockers for the encoder's own forms

All 146 encoder transposes use perms `[0,2,1,3]`, `[0,2,1]`, `[0,2,-3,-1]`.
The 48 feeding `matmul` y are rank-4 `[0,2,1,3]`-family: the permutation
moves non-unit leading dimensions, so it cannot fold into `transpose_y`, and
the underlying rank-4 batched matmul is itself outside the decoded envelope.
The 98 feeding `add`/`conv`/`reshape` have no consumer flag to absorb them.
All 48 `slice_by_index` ops slice across non-unit head dimensions (mask
slicing `[1,8,750,375] -> [1,8,749,375]`; bias slicing `[1,8,375,X] ->
[1,8,375,375]`). All 24 pads grow a row. The single tile replicates across
heads. **Net: 0 of the 219 encoder ops become supported for this package;
each now fails with its named per-op blocker instead of "no H13 dispatch
branch".** The host-expressible subsets (contiguous slices, layout-preserving
transposes, tail-swap folds, identity pad/tile) are emitted and tested.

## Commands and observed results

```text
make -j8 build/mil-hwxc            # 0 errors (GNUstep host toolchain)
python3 tests/test_h13_layout_cli.py build/mil-hwxc
h13 layout cli: PASS
python3 tests/test_h13_cli.py build/mil-hwxc
H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
python3 tests/test_h13_split_cli.py build/mil-hwxc
h13 split cli: PASS
```

`tests/test_h13_layout_cli.py` covers: alias emission with manifest checks
and inspector validation for every lowered form; the y-fold byte-identical
to the reference matmul (program and manifest); the x-fold handing its
flipped flag to the verified matmul envelope (no mirrored tx oracle exists,
so byte parity is impossible there today); recompile determinism for every
successful case in both `anec` and `hwx` formats; encoder-shaped rejection
contracts (`[1,375,8,128]`/`[0,2,-3,-1]`, `[1,8,750,375]` mask slice,
`[1,8,375,749]` bias slice, `[0,0,0,0,0,0,1,0]` pad, `[1,375,1]` tile
reps); negative-index and begin_mask/end_mask normalization; and the
dense-reference slice check the split test established.

Baseline note: `tests/test_h13_parity.py` fails
`binary_add_1x1024x1x1 task 0 words differ from the oracle` on the pinned
base commit `83a4434` with this work stashed — the failure pre-exists this
branch and is not a stream-2 regression.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or
Mac execution was performed. No files under `ane-linux-experiments` were
edited or executed; the encoder artifact was read only.
