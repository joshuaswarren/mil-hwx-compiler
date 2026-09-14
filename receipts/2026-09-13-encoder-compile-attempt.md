# Encoder H13 compile attempt (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Pins

| item | value |
| --- | --- |
| compiler branch | `feature/h13-boolean-and-round2` |
| compiler HEAD | `366eb15057c9806605e7341531545e753f7a8aec` (boolean headers, materialized tile, pad citations) |
| locked ancestor | `83a4434810b0981d1764e6231f6a679df6f35241` (ordered MIL results; `merge-base --is-ancestor` true) |
| `build/mil-hwxc` sha256 | `fd4754117da05b7cf67f284f4ff9d42e345adc64b94cebbcf6e5e0bdb259159e` |
| overlay | mlx-omarchy `main-land2` `aad63efb96c2eca8446790b0a30388bc1569a85a` |
| encoder | `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018` |
| cache verify | `mlx-omarchy-parakeet verify` → `OK: 12 files verified` |

## Frontend (depalettize + pad elimination + mask lowering)

Work copy: `/tmp/encoder-compile-attempt-20260913/encoder.mlpackage` (cache untouched).

| transform | result |
| --- | --- |
| `expand_lut_constants` | 194 `constexpr_lut_to_dense` → dense fp16 `const` (12.518 s) |
| `eliminate_pads` | 24 pads → depthwise identity conv; 144 helper consts reordered into SSA |
| `lower_mask_ops` | 27 rewritten (`logical_and` 1, `logical_not` 1, `reduce_min` 1, finite `select` 24) |
| non-finite `select` | 24 skipped (`0xFC00` −inf fill; `lower_mask_ops` raises unless guarded) |

Post-frontend histogram deltas: `constexpr_lut_to_dense` 194→0, `pad` 24→0, `conv` 77→101, `const` 1783→2175, `select` 48→24, `logical_and`/`logical_not`/`reduce_min` gone. Ops after rewrite: 3602.

## Emission

`coreml.mil_adapter._Adapter.emit` on the lowered package (lock verified on the cache original, not the mutated copy).

| field | value |
| --- | --- |
| `model.mil` | `/tmp/encoder-compile-attempt-20260913/emitted/model.mil` |
| bytes / sha256 | 675653 / `6b13ad66b1841cb512a8653abfe882253113bafacb96807366150ae8262a23ee` |
| operations | 3602 (24 splits / 48 split results) |
| `normalized_constexpr_count` | 0 |
| returns | `encoder_hidden` `tensor<fp32, [1, 375, 640]>`, `encoder_mask` `tensor<int32, [1, 375]>` |

## Compiler run

```text
/home/joshuawarren/src/mil-hwx-compiler/build/mil-hwxc \
  --mil /tmp/encoder-compile-attempt-20260913/emitted/model.mil \
  --model-root /tmp/encoder-compile-attempt-20260913/emitted/model-root \
  --output /tmp/encoder-compile-attempt-20260913/anec-out \
  --target H13 --format anec
```

| field | value |
| --- | --- |
| exit | 65 |
| stdout | empty |
| stderr | `3604:5: error [h13.unsupported-logical-result-conversion]: H13 logical result conversions require explicit hardware or GPU coverage` |
| program emitted | **no** (`anec-out` empty) |

Line 3604 is the returned fp32 cast:

```text
tensor<fp32, [1, 375, 640]> encoder_hidden = cast(..., x = linear_217_cast_fp16);
```

The compiler fail-fasts on the first non-fp16 function result (`ANEH13Compiler.mm` `h13.unsupported-logical-result-conversion`). It does not walk remaining ops.

## Remaining reject list

### Observed (this run)

| op | count | code |
| --- | ---: | --- |
| `cast` (returned `encoder_hidden` fp32) | 1 | `h13.unsupported-logical-result-conversion` |

The sibling returned `encoder_mask` int32 cast is the same gate; the compiler did not print a second diagnostic.

### Present after frontend, not reached

Counts from the lowered MIL. Codes are the 366eb15 named refusal for that encoder form, not a second compiler print.

| op | count | code |
| --- | ---: | --- |
| `transpose` | 146 | `h13.nonfoldable-transpose` (1 bool: `h13.invalid-transpose-parameters`) |
| `slice_by_index` | 48 | not observed (layout view lowering exists; first gate stops before it) |
| `select` (−inf fill) | 24 | `h13.select-needs-decoded-encoder` |
| `cast` (other) | 10 | not observed (6 fp16, 2 extra int32, 1 bool, plus the printed fp32) |
| `less` | 4 | not observed (boolean encoder exists at 366eb15) |
| `floor` | 3 | not observed (boolean encoder exists at 366eb15) |
| `floor_div` | 3 | not observed (boolean encoder exists at 366eb15; captured divisor is scalar 2.0) |
| `tile` | 1 | `h13.unsupported-tile` |

Frontend-eliminated (count 0 in lowered MIL): `constexpr_lut_to_dense`, `pad`, `logical_and`, `logical_not`, `reduce_min`.

## Not established

- Any H13 program, ANEC package, or bundle for the public encoder.
- Geometry-envelope outcomes for `less` / `floor` / `floor_div` / `slice_by_index` on this graph.
- Hardware execution.
