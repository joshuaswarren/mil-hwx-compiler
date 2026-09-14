# Encoder H13 compile attempt 2 (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

Re-run of Phase 5 encoder compile after bool logical-result + bool `expand_dims` alias (`e0419ed`) and the main-land2 fp16 peel (`e0712aec`).

## Pins

| item | value |
| --- | --- |
| compiler branch | `feature/h13-bool-logical-result` |
| compiler HEAD | `e0419ed0d986bc664908b0f7d52db3ca3bcd1d05` |
| `build/mil-hwxc` sha256 | `b72c119764bc421f5d2728737a5f77e68ad29359e530c1aa66106c0a93e619ed` |
| overlay | mlx-omarchy `main-land2` `fa8654d022313790056cc4ed3915ae82d55fab3e` |
| fp16 peel | `e0712aeca2c03581de50efad552fd8d7956300d0` (ancestor of overlay HEAD) |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |
| cache verify | `verify_locked_encoder` on the cache original against `overlay/tools/coreml/parakeet-reference.lock` |

## Frontend (same as attempt 1)

Work copy: `/tmp/encoder-compile-attempt-2-20260913/encoder.mlpackage` (cache untouched).

| transform | result |
| --- | --- |
| `expand_lut_constants` | 194 `constexpr_lut_to_dense` → dense fp16 `const` (11.801 s) |
| `eliminate_pads` | 24 pads → depthwise identity conv; 144 helper consts reordered into SSA |
| `lower_mask_ops` | 27 rewritten (`logical_and` 1, `logical_not` 1, `reduce_min` 1, finite `select` 24) |
| non-finite `select` | 24 skipped (`0xFC00` −inf fill) |

Ops after rewrite: 3602. Raw emitted `model.mil` is byte-identical to attempt 1 (`675653` / `6b13ad66b1841cb512a8653abfe882253113bafacb96807366150ae8262a23ee`).

## Peel (`e0712aec`)

`coreml.output_peel.peel_fp16_outputs` on the emitted MIL.

| field | value |
| --- | --- |
| `model-fp16.mil` | `/tmp/encoder-compile-attempt-2-20260913/emitted/model-fp16.mil` |
| bytes / sha256 | 675365 / `c970fee136da2efa80eb74604a08107b1043dc49e7f8fc38b3ffb133c399e3e1` |
| lines | 3607 → 3605 (two trailing widening casts removed) |
| returns | `linear_217_cast_fp16` `tensor<fp16, [1, 375, 640]>`, `output_mask` `tensor<bool, [1, 375]>` |

GPU epilogue (not compiled): `encoder_hidden` fp16→fp32, `encoder_mask` bool→int32.

## Compiler run

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil /tmp/encoder-compile-attempt-2-20260913/emitted/model-fp16.mil \
  --model-root /tmp/encoder-compile-attempt-2-20260913/emitted/model-root \
  --output /tmp/encoder-compile-attempt-2-20260913/anec-out \
  --target H13 --format anec
```

| field | value |
| --- | --- |
| exit | 65 |
| stdout | empty |
| stderr | `155:5: error [h13.invalid-shape-alias]: H13 shape aliases require static fp16 or bool input and result shapes with equal element counts and constant shape parameters` |
| program emitted | **no** (`anec-out` empty) |

Line 155 is an int32 length broadcast, not a bool view:

```text
tensor<int32, [1, 1]> var_283 = expand_dims(axes = var_283_axes_0, x = lengths_cast_fp16_to_int32);
```

`[1] → [1, 1]`, const axis `1`, equal counts. The diagnostic now names fp16 **or bool**; int32 is still outside the alias path. The compiler fail-fasts here and does not walk remaining ops.

Bool `expand_dims` aliases on this graph did compile through the new path: six bool views before line 155 were accepted (`[1,1500]→[1,1,1500]→[1,1,1500,1]`, same for 750 and 375). Attempt 1's `h13.unsupported-logical-result-conversion` on the fp32/int32 function results did not fire (peeled away; bool `output_mask` is a legal return at `e0419ed`).

## Remaining reject list

### Observed (this run)

| op | count | code |
| --- | ---: | --- |
| `expand_dims` (int32 `[1]→[1,1]`) | 1 | `h13.invalid-shape-alias` |

### Present after frontend+peel, not reached

Counts from peeled MIL. Codes are the named refusal for that encoder form at `e0419ed`, not a second compiler print.

| op | count | code |
| --- | ---: | --- |
| `transpose` | 146 | `h13.nonfoldable-transpose` (1 bool: `h13.invalid-transpose-parameters`) |
| `slice_by_index` | 48 | not observed (layout view lowering exists; first gate stops before it) |
| `select` (−inf fill) | 24 | `h13.select-needs-decoded-encoder` |
| `expand_dims` (bool, after line 155) | 2 | not observed (bool alias path already accepted the six earlier views) |
| `cast` | 9 | not observed (6 fp16, 2 int32, 1 bool; line 154 int32 cast was accepted) |
| `less` | 4 | not observed (boolean encoder exists at `e0419ed`; `output_mask = less(...)` is line 156) |
| `floor` | 3 | not observed |
| `floor_div` | 3 | not observed |
| `tile` | 1 | `h13.unsupported-tile` |

Peeled-histogram `expand_dims` total 13: 4 fp16 accepted, 6 bool accepted, 1 int32 rejected, 2 bool unreached.

Frontend-eliminated (count 0 in peeled MIL): `constexpr_lut_to_dense`, `pad`, `logical_and`, `logical_not`, `reduce_min`. Trailing fp32/int32 result casts peeled (count 0 in function returns).

## Not established

- Any H13 program, ANEC package, or bundle for the public encoder.
- Geometry-envelope outcomes for `less` / `floor` / `floor_div` / `slice_by_index` / remaining `transpose` on this graph.
- Hardware execution.
