# H13 split lowering receipt

## Source

- Compiler base: `a1bfea0cd52208ac9f168791e8c6e9a83ddc8efb`
- Adapter source commit: `eac5eed9`
- Adapter artifact: `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`
- Artifact identity: textual `program(1)` with `main<CoreML8>` and ordered parenthesized split results
- Executing model: `openai-codex/gpt-5.6-sol` (configured route `mlx-openai-deep`; no fallback observed)

## Contract

H13 lowers an equal two-result split as two ordered views of one source buffer when all dimensions before the split axis are one. The second result adds one result-sized base offset to each consumer binding slice. No ANE split program, host tensor arithmetic, tensor copy, alternate graph, or descriptor field was added.

The package alias schema requires equal element counts and cannot describe a split subset. Split results therefore remain compiler-only views; consumer slices carry their offsets. A non-unit dimension before the axis would require disjoint slices, which the one-slice binding ABI cannot represent. The compiler rejects that geometry as `h13.noncontiguous-split-axis`. Counts other than two fail as `h13.unsupported-split-count`.

## Commands and observed results

Baseline reproduction:

```text
make -j4 build/mil-hwxc && python3 tests/test_h13_split_cli.py build/mil-hwxc
AssertionError: 5:5: error [h13.unsupported-multi-result-operation]: H13 does not lower multi-result operations
Command exited with code 1
```

Post-fix build and split contract:

```text
make -j4 build/mil-hwxc && python3 tests/test_h13_split_cli.py build/mil-hwxc
h13 split cli: PASS
```

The test compiles `split -> reshape(second) -> sigmoid -> mul(first, gate)`, checks source binding offsets 64 and 0 with counts 64, validates the package with `research/inspect_anec.py`, compares both slices against an independent dense reference, and checks the two named rejection contracts. Adding the shape-only alias first reproduced `tensor alias element count differs from its underlying tensor`; keeping split-derived aliases compiler-only made the same test pass while preserving offset 64.

Scoped regression checks:

```text
python3 tests/test_h13_cli.py build/mil-hwxc
H13 MIL-to-ANEC/HWX CLI: PASS (device-free)

python3 tests/test_h13_split_cli.py build/mil-hwxc
h13 split cli: PASS
```

Pinned encoder probe, bounded to 60 seconds:

```text
timeout 60 ./build/mil-hwxc --mil /tmp/ane-runtime-adapter-integrated-20260913/model.mil --model-root /tmp/ane-runtime-adapter-integrated-20260913/model-root --output /tmp/h13-split-probe-20260913-b --target H13 --format anec
3354:5: error [h13.unsupported-chain]: H13 chains must return only the last operation result
Command exited with code 65
```

The pinned MIL returns `(encoder_hidden, encoder_mask)` at line 3355. This is the next compiler boundary after removal of the blanket split precheck. The probe does not establish full encoder compilation or execution.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or Mac execution was performed.
