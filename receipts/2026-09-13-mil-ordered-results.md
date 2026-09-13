# Ordered MIL Results Receipt

## Source

- Repository: `mil-hwx-compiler`
- Base commit: `a0ce354cf800011a84420da4e12013eb8140b2a5`
- Branch: `agent/mil-ordered-results`
- Worktree: `/home/joshuawarren/.config/superpowers/worktrees/mil-hwx-compiler/mil-ordered-results`
- Model: `openai-codex/gpt-5.6-sol` through the configured `mlx-openai-deep` route. No fallback occurred.

## Implemented contract

Textual MIL accepts one typed result or an ordered parenthesized list of typed results. GraphIR stores every result as a separate SSA value in source order and assigns the same producer to each value. The importer registers all names without partial symbol-table mutation. It retains `program(1)` with `CoreML8` and `program(1.3)` with `ios18`; the verifier rejects other program and opset contracts.

The supported split contract requires each `axis` and `num_splits` SSA declaration and its typed constant literal to match the known rank-zero `tensor<int32, []>` form. It also requires one ranked tensor input, a positive divisible split dimension, result count equal to `num_splits`, and each ordered result type equal to the partition type. The pinned representative uses axis 1, count 2, two `[1, 1024, 375]` results, the first result as the `mul` input, and the second result as the `sigmoid` input.

H13, H14, and the H16G operation graph reject multi-result lowering with named diagnostics before they access a result. This is a semantic frontend change, not a claim that any hardware backend can encode `split`.

## Verification

Build commands:

```text
make -j4 build/test_mil_parser build/test_graph_import build/test_operation_graph
make -j4 build/test_graph_transforms build/test_structural_fusion
make -j4 build/test_graph_import build/mil-hwxc
```

Scoped test results:

```text
mil parser: PASS
graph import: PASS
operation graph: PASS
graph transforms: PASS
structural fusion: PASS
```

The real compiler CLI parsed the full faithful encoder and reached the expected adapter boundary:

```text
./build/mil-hwxc --mil /tmp/text-mil-encoder/encoder-faithful-proposed-multi-result.mil --model-root /tmp/text-mil-encoder --output /tmp/mil-ordered-results-faithful-final-20260913 --target H13 --format anec
125:120: error [mil.import.unsupported-type]: unsupported MIL type 'uint4'
Command exited with code 65
```

Line 125 contains `tensor<uint4, [1024, 4096]>` indices for `constexpr_lut_to_dense`. The parser processes the complete program before semantic import, so this result confirms that the ordered result grammar no longer stops the faithful file. It does not establish full H13 operation coverage.

The adapter-normalized pinned source has SHA-256 `ac8e9526154ac8b8d4c08b83e44dbc1a2208d7ec33e42a7838d57b698b14f133`. The same compiler advanced to the first split and stopped at the explicit lowering boundary:

```text
./build/mil-hwxc --mil /tmp/coreml-text-adapter-pinned-source-v2/model.mil --model-root /tmp/coreml-text-adapter-pinned-source-v2/model-root --output /tmp/mil-ordered-results-normalized-20260913 --target H13 --format anec
265:5: error [h13.unsupported-multi-result-operation]: H13 does not lower multi-result operations
Command exited with code 65
```

Line 265 is the pinned two-result split. It declares axis 1, count 2, and two `[1, 1024, 375]` results; the first result feeds `mul`, and the second feeds `sigmoid`. This run proves normalized text passes parsing, import, and split verification. It does not prove split lowering or full H13 encoder coverage.

The declared-type mismatch regression first failed before the fix:

```text
FAIL: mismatched split constant type is rejected
FAIL: mismatched split constant type has stable diagnostic
graph import: FAIL
```

After the fix, `graph import: PASS`. The exact review repro now stops in semantic verification rather than reaching H13 lowering:

```text
./build/mil-hwxc --mil /tmp/mil-ordered-review-mismatched-const-type.mil --model-root /tmp --output /tmp/mil-ordered-results-mismatch-fixed-20260913 --target H13 --format anec
5:5: error [ane.verify.split-constant-parameters]: split requires x plus constant axis and num_splits values
Command exited with code 65
```

Backend refusal probes used `/tmp/mil-ordered-results-split.mil`:

```text
H13: 5:5: error [h13.unsupported-multi-result-operation]: H13 does not lower multi-result operations
H14: 5:5: error [h14.unsupported-multi-result-operation]: H14 does not lower multi-result operations
```

Both commands exited with code 65. No SSH, hardware execution, inference, router change, pin change, push, or shared main-land2 write occurred.
