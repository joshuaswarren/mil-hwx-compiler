# Numerics, precision, quantization, and accuracy

Numerical support has four separate questions:

1. Can the compiler encode the data type and operation?
2. Will the runtime place that operation on the ANE?
3. Will the hardware execute it for the requested shape?
4. Is the result accurate enough for the model?

A positive answer to one question does not answer the others.

## FP16

The source-native paths documented in this repository use FP16 tensor surfaces for their established hardware cases. The H13 metadata element-type code for FP16 is `5`. **Evidence: high for repository behavior; medium for the decoded code meaning.** See the local [H13 field ledger](../../research/h13-hwx-fields.md) and [verification guide](../../docs/VERIFICATION.md).

IEEE binary16 has a narrow dynamic range and less precision than FP32. Overflow, underflow, rounding, and cancellation can therefore change results even when the compiled operation is structurally correct. **Evidence: high.** The format behavior follows [IEEE 754](https://standards.ieee.org/ieee/754/6210/); an applied ANE source records the practical finite range near ±65,504 and recommends preflight checks in the pinned [ANEMLL README](https://github.com/Anemll/Anemll/blob/ff3b97783a64e5e59bb9da8a21f79290a1611142/README.md).

Training adds a gradient-range problem. Loss scaling can keep small FP16 gradients representable, but scaling does not make every operator or update numerically stable. **Evidence: high for the numerical mechanism; medium for the private-ANE implementation.** The pinned [maderix/ANE repository](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29) documents its FP16 training constraints and scaling approach.

## BF16

BF16 preserves an FP32-sized exponent field but has fewer significand bits than FP16. That usually trades precision for dynamic range. **Evidence: high.** See Google Cloud's [bfloat16 format description](https://cloud.google.com/tpu/docs/bfloat16).

A request for BF16 in a model API does not prove native BF16 arithmetic in the generated ANE program. The compiler or runtime may reject, convert, partition, or use mixed precision. **Evidence: high as an interface distinction; open question for undocumented hardware paths.** The tracked oMLX evidence includes an open BF16 mixed-precision request rather than a settled native contract; see [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

**Open question:** Which H13 through H18 units support BF16 storage, multiply, accumulation, and conversion natively, and which compiler paths expose each capability?

## INT8 and weight-only quantization

“INT8” is incomplete without operand roles, scale granularity, zero points, accumulation type, and output conversion. W8A8 integer arithmetic, INT8 weights with FP16 activations, and an INT8 storage format followed by dequantization are different execution contracts. **Evidence: high as a quantization definition.** The distinction is visible in oMLX's per-output-channel requantization and mixed MLX/ANE paths recorded in [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

In the cited oMLX implementation, selected weights are requantized per output channel for ANE use. Q5, affine Q6, and affine Q8 paths landed in separate changes. This is implementation-specific evidence, not a general ANE format specification. **Evidence: high for the cited code history; medium for hardware interpretation.** See pull requests [#2756](https://github.com/jundot/omlx/pull/2756), [#2833](https://github.com/jundot/omlx/pull/2833), and [#2889](https://github.com/jundot/omlx/pull/2889), plus commits [`39baf93c3976a5513096dc57406574333d0e3b76`](https://github.com/jundot/omlx/commit/39baf93c3976a5513096dc57406574333d0e3b76) and [`fc6807c477d4d1b5be251e176da13b4a7bab8dc6`](https://github.com/jundot/omlx/commit/fc6807c477d4d1b5be251e176da13b4a7bab8dc6). The source context is preserved in [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

The pinned maderix benchmark reports a higher operation rate for its INT8 kernel than for its FP16 kernel. This demonstrates why a bare TOPS number needs a precision label; it does not establish that arbitrary quantized models receive that rate. **Evidence: medium.** See [maderix/ANE](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29).

## Accuracy failures are workload-specific

A quantization path can pass isolated matrix tests and still fail a recurrent model. In oMLX's recorded history, recurrent projection quantization produced deterministic long-context failures, so the implementation kept those projections on MLX while limiting ANE work to token-local rows. **Evidence: high for that project history; medium for the causal interpretation outside those tests.** See issue [#3069](https://github.com/jundot/omlx/issues/3069), pull request [#3133](https://github.com/jundot/omlx/pull/3133), and commit [`4c8e43a5b44f2aa877f728315d82188c39454fcf`](https://github.com/jundot/omlx/commit/4c8e43a5b44f2aa877f728315d82188c39454fcf). The evidence is preserved in [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

ANEMLL reports that simple LUT4 conversion can lose quality without appropriate block quantization. That is a project observation, not a universal threshold. **Evidence: medium.** See the pinned [ANEMLL README](https://github.com/Anemll/Anemll/blob/ff3b97783a64e5e59bb9da8a21f79290a1611142/README.md).

## Shape restrictions

Operation support can depend on rank, row width, channels, padding, and state shape. ANEMLL records a nonuniform state-shape restriction on M1/A14-class hardware, while oMLX uses fixed compile rows and handles wider or tail cases with tiling, padding, or another backend. **Evidence: medium.** See the pinned [ANEMLL README](https://github.com/Anemll/Anemll/blob/ff3b97783a64e5e59bb9da8a21f79290a1611142/README.md) and [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

A compiler should reject unsupported shapes with a named error before submission. It should not rely on a hardware crash, hang, or malformed tensor fixture to discover the boundary. **Evidence: high as a safety requirement.** The repository's valid-input hardware discipline and named error contracts are described in [verification](../../docs/VERIFICATION.md).

## Verification protocol

For each operation, data type, geometry, and generation:

1. generate finite, valid inputs within the intended range;
2. record logical and physical layouts;
3. compute a higher-precision reference where practical;
4. execute the actual compiled program on the named hardware;
5. compare absolute and relative error, not only a checksum;
6. test meaningful boundaries such as padded rows and quantized range limits;
7. record the hardware, operating-system build, compiler revision, and model identity.

**Evidence: high as a verification standard.** The repository applies hardware-vs-reference checks and records concrete tolerances in [verification](../../docs/VERIFICATION.md). The need to validate the input signal before trusting a metric follows from the repository's evidence policy.

## Open questions

- **Open question:** What accumulator widths and rounding modes apply to each PE and NE operation on each generation?
- **Open question:** Which denormal, NaN, infinity, saturation, and overflow behaviors are hardware, firmware, or compiler choices?
- **Open question:** What native quantized encodings exist beyond the paths independently reconstructed in current projects?
- **Open question:** Which accuracy thresholds predict end-to-end model quality rather than only local tensor agreement?
