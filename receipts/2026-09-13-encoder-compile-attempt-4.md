# Encoder H13 compile attempt 4 (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute. No jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

Re-run after overlay `4f76d0ee` (`mask_layout` rewrite of the attempt-3 bool tail-swap). Compiler still `mil-hwxc` `1f67253`.

## Pins

| item | value |
| --- | --- |
| compiler branch | `feature/h13-bool-transpose` |
| compiler HEAD | `1f672539699ce32fa4da4a876404559aec88c36e` |
| `build/mil-hwxc` sha256 | `a3b03ecab14d70fe5e262c2f4066b9c5d3b102a6a92a09f0adb7d6b693a397dd` |
| overlay | mlx-omarchy `parakeet-mel-exact` `4f76d0ee37aab66b106e00a6c65b6c1152e963f5` |
| fold | `82c5806f240d4ab5496561f7869e932ef80de0f3` (ancestor of overlay HEAD; source MIL) |
| fp16 peel | `e0712aeca2c03581de50efad552fd8d7956300d0` (ancestor of overlay HEAD) |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |

Source MIL is attempt 3's folded file, hash-checked:

| field | value |
| --- | --- |
| `model-fp16-noview.mil` | `/tmp/encoder-compile-attempt-3-20260913/emitted/model-fp16-noview.mil` |
| bytes / sha256 | 675487 / `b75801f75e3574799dd822bad3959ec6eb9dc7f9826789e48120753e7185c10b` |

`overlay.tools.coreml.mask_layout.rewrite_mask_layout` on that file:

| field | value |
| --- | --- |
| `model-fp16-nolayout.mil` | `/tmp/encoder-compile-attempt-4-20260913/emitted/model-fp16-nolayout.mil` |
| bytes / sha256 | 675929 / `c67e29322e9d71f5c240c72145bc87fb449c0a58aaad61a5ea3d2f2b610e1d09` |
| rewritten | `var_289` |
| refused | none |

`var_289` is no longer a transpose. Inert perm const left in place:

```text
tensor<int32, [3]> var_289_perm_0 = const()[..., val = tensor<int32, [3]>([0, 2, 1])];
tensor<int32, [1]> var_289_layout_axes = const()[..., val = tensor<int32, [1]>(2)];
tensor<bool, [1, 375, 1]> var_289_layout_exp = expand_dims(axes = var_289_layout_axes, x = output_mask)[...];
tensor<int32, [3]> var_289_layout_reps = const()[..., val = tensor<int32, [3]>([1, 1, 375])];
tensor<bool, [1, 375, 375]> var_289 = tile(reps = var_289_layout_reps, x = var_289_layout_exp)[name = tensor<string, []>("var_289_layout_tile")];
tensor<fp16, [1, 375, 375]> attention_mask_5 = mul(x = attention_mask_3, y = var_289)[];
```

Zero bool `[1, 375, 375]` transposes remain.

## Compiler run

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil /tmp/encoder-compile-attempt-4-20260913/emitted/model-fp16-nolayout.mil \
  --model-root /tmp/encoder-compile-attempt-3-20260913/emitted/model-root \
  --output /tmp/encoder-compile-attempt-4-20260913/anec-out \
  --target H13 --format anec
```

| field | value |
| --- | --- |
| exit | 65 |
| stdout | empty |
| stderr | `204:5: error [h13.nonfoldable-transpose]: H13 transpose with several consumers or a returned value needs a materialized transposed surface, and the decoded corpus holds no data-movement encoder` |
| program emitted | **no** (`anec-out` empty) |

One diagnostic. The compiler fail-fasts here and does not walk remaining ops.

## Next reject (fail-fast)

| field | value |
| --- | --- |
| op | `transpose` |
| name | `query_states_1_cast_fp16` (`transpose_143`) |
| dtype / shape | fp16 `[1, 8, 375, 128]` |
| perm | `[0, 2, 1, 3]` (const `query_states_1_perm_0`, exact) |
| x | `var_332_cast_fp16` fp16 `[1, 375, 8, 128]` |
| consumers | `add` `query_1_cast_fp16`; `add` `query_states_with_bias_v_1_cast_fp16` (not a graph return) |
| code | `h13.nonfoldable-transpose` |

Full line 204:

```text
tensor<fp16, [1, 8, 375, 128]> query_states_1_cast_fp16 = transpose(perm = query_states_1_perm_0, x = var_332_cast_fp16)[name = tensor<string, []>("transpose_143")];
```

Perm constant on line 192:

```text
tensor<int32, [4]> query_states_1_perm_0 = const()[name = tensor<string, []>("query_states_1_perm_0"), val = tensor<int32, [4]>([0, 2, 1, 3])];
```

Consumers:

```text
tensor<fp16, [1, 8, 375, 128]> query_1_cast_fp16 = add(x = query_states_1_cast_fp16, y = var_345_to_fp16)[name = tensor<string, []>("query_1_cast_fp16")];
tensor<fp16, [1, 8, 375, 128]> query_states_with_bias_v_1_cast_fp16 = add(x = query_states_1_cast_fp16, y = var_348_to_fp16)[name = tensor<string, []>("query_states_with_bias_v_1_cast_fp16")];
```

This is the multi-consumer class: two `add`s, no matmul transpose flag on this view. It is **not** the attempt-3 bool tail-swap (`perm [0,2,1]`, `mul` consumer).

One fp16 transpose compiled through before line 204: `var_207_cast_fp16` `[1, 256, 375, 16] → [1, 375, 256, 16]` perm `[0, 2, 1, 3]` (`transpose_145`; fast axis 16 preserved). Bool `expand_dims`/`tile` for the rewritten mask also compiled through (lines 162–165 sit before the fail).

## Remaining reject list

### Observed (this run)

| op | shape | perm | consumers | count | code |
| --- | --- | --- | --- | ---: | --- |
| `transpose` | fp16 `[1, 8, 375, 128]` | `[0, 2, 1, 3]` | `add`, `add` | 1 | `h13.nonfoldable-transpose` |

### Cleared since attempt 3

| op | form | code |
| --- | --- | --- |
| `transpose` | bool `[1, 375, 375]` perm `[0, 2, 1]` `x=attention_mask_3`, consumer `mul` | `h13.nonfoldable-transpose` (rewritten to complementary `expand_dims`+`tile`; did not fire) |

### Present after frontend+peel+fold+mask_layout, not reached

Counts from rewritten MIL. Codes are the named refusal for that encoder form at `1f67253`, not a second compiler print.

| op | count | code |
| --- | ---: | --- |
| `transpose` | 143 | not observed (1 fp16 alias accepted before the fail; 1 fp16 printed above) |
| `slice_by_index` | 48 | not observed |
| `select` (−inf fill) | 24 | `h13.select-needs-decoded-encoder` |
| `cast` | 9 | not observed (some sit before line 204) |
| `expand_dims` | 13 | not observed (rewrite added one bool expand; attempt-3 bool expands already compiled) |

`less` / `floor` / `floor_div` / `tile` counts in the rewritten MIL are 4 / 3 / 3 / 2; all sit before line 204 and produced no diagnostic. Tile count is 2 because `var_289` is now a tile.

Frontend-eliminated (count 0): `constexpr_lut_to_dense`, `pad`, `logical_and`, `logical_not`, `reduce_min`.

## Not established

- Any H13 program, ANEC package, or bundle for the public encoder.
- Geometry-envelope outcomes for remaining `transpose` / `slice_by_index` / `select` on this graph.
- Hardware execution.
