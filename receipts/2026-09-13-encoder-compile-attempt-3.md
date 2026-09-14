# Encoder H13 compile attempt 3 (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

Re-run of encoder compile after bool transpose coverage (`1f67253`) and the main-land2 int32 unit-view fold (`82c5806f`).

## Pins

| item | value |
| --- | --- |
| compiler branch | `feature/h13-bool-transpose` |
| compiler HEAD | `1f672539699ce32fa4da4a876404559aec88c36e` |
| `build/mil-hwxc` sha256 | `a3b03ecab14d70fe5e262c2f4066b9c5d3b102a6a92a09f0adb7d6b693a397dd` |
| overlay | mlx-omarchy `main-land2` `9bd1997103770f0012ed3241f92a7b682b4fc8d2` |
| fold | `82c5806f240d4ab5496561f7869e932ef80de0f3` (ancestor of overlay HEAD) |
| fp16 peel | `e0712aeca2c03581de50efad552fd8d7956300d0` (ancestor of overlay HEAD) |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |

Frontend + peel artifacts reused from attempt 2 after hash check (`model.mil` `6b13ad66…`, `model-fp16.mil` `c970fee1…`). Overlay `peel_fp16_outputs` re-ran on the raw MIL and matched `c970fee1…`.

## Fold (`82c5806f`)

`coreml.fold_unit_views.fold_unit_views` on the peeled MIL.

| field | value |
| --- | --- |
| `model-fp16-noview.mil` | `/tmp/encoder-compile-attempt-3-20260913/emitted/model-fp16-noview.mil` |
| bytes / sha256 | 675487 / `b75801f75e3574799dd822bad3959ec6eb9dc7f9826789e48120753e7185c10b` |
| folded | `var_283` (int32 `expand_dims` `[1]→[1,1]`) |
| refused | none |
| returns | `linear_217_cast_fp16` `tensor<fp16, [1, 375, 640]>`, `output_mask` `tensor<bool, [1, 375]>` |

Attempt 2's fail-fast (`h13.invalid-shape-alias` on that int32 view) did not fire. The `less` that consumed it is now `less(x = var_281_fold_const, y = lengths_cast_fp16_to_int32)`.

## Compiler run

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil /tmp/encoder-compile-attempt-3-20260913/emitted/model-fp16-noview.mil \
  --model-root /tmp/encoder-compile-attempt-3-20260913/emitted/model-root \
  --output /tmp/encoder-compile-attempt-3-20260913/anec-out \
  --target H13 --format anec
```

| field | value |
| --- | --- |
| exit | 65 |
| stdout | empty |
| stderr | `162:5: error [h13.nonfoldable-transpose]: H13 a tail-swap transpose moves the storage-fastest axis, so its view reads rows whose elements sit a non-unit storage stride apart: the NCHW descriptor carries contiguous rows and no surface interpretation expresses the permutation at any size; a consuming matmul's transpose flag is the only decoded mechanism that absorbs one, and this view has a 'mul' consumer with no such flag` |
| program emitted | **no** (`anec-out` empty) |

One diagnostic. The compiler fail-fasts here and does not walk remaining ops.

Line 162 is the bool mask transpose. Perm constant on line 161 is exact rank-matching `[0, 2, 1]`:

```text
tensor<int32, [3]> var_289_perm_0 = const()[..., val = tensor<int32, [3]>([0, 2, 1])];
tensor<bool, [1, 375, 375]> var_289 = transpose(perm = var_289_perm_0, x = attention_mask_3)[name = tensor<string, []>("transpose_144")];
```

This is **not** `h13.invalid-transpose-parameters`. The dtype gate opened; the perm classified as a fast-axis tail-swap.

Ops that compiled through before line 162 include: 4 `less`, 3 `floor`, 3 `floor_div`, 7 bool `expand_dims`, bool `tile` `attention_mask_3` `[1,1,375]→[1,375,375]`, and fp16 transpose `hidden_states_11` `[1,256,375,16]` perm `[0,2,1,3]` (fast axis 16 preserved).

## Remaining reject list

### Observed (this run)

| op | count | code |
| --- | ---: | --- |
| `transpose` (bool `[1,375,375]` perm `[0,2,1]`) | 1 | `h13.nonfoldable-transpose` |

### Present after frontend+peel+fold, not reached

Counts from folded MIL. Codes are the named refusal for that encoder form at `1f67253`, not a second compiler print.

| op | count | code |
| --- | ---: | --- |
| `transpose` | 144 | not observed (1 fp16 alias accepted before the fail; 1 bool printed above) |
| `slice_by_index` | 48 | not observed |
| `select` (−inf fill) | 24 | `h13.select-needs-decoded-encoder` |
| `cast` | 2 | not observed (7 earlier casts accepted) |
| `expand_dims` (bool, after line 162) | 1 | not observed (11 earlier expand_dims accepted, including 7 bool) |

`less` / `floor` / `floor_div` / `tile` counts in the folded MIL are 4 / 3 / 3 / 1; all sit before line 162 and produced no diagnostic.

Frontend-eliminated (count 0 in folded MIL): `constexpr_lut_to_dense`, `pad`, `logical_and`, `logical_not`, `reduce_min`. Trailing fp32/int32 result casts peeled. The one int32 `expand_dims` folded.

## Not established

- Any H13 program, ANEC package, or bundle for the public encoder.
- Geometry-envelope outcomes for remaining `transpose` / `slice_by_index` / `select` on this graph.
- Hardware execution.
