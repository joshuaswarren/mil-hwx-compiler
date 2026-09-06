# Oracle parity: the method and its evidence boundary

This page states how the repository proves that a source-native encoder matches Apple's compiler, and exactly what that proof does and does not cover. It exists because "byte-identical to Apple" is easy to over-read.

## The claim

For a covered case, the compiler in this repository emits a task stream, program descriptor, tensor descriptors, and constant section whose decoded fields are word-for-word identical to those Apple's compiler produced for the same MIL, target, and weights. **Evidence: high; it is a repeatable repository test.** Run `make test-h13-parity` (342 H13 cases) and `make test-h14-parity` (137 H14 cases); the checks are described in the [README](../../README.md#h13-apple-parity-programs) and the [verification guide](../../docs/VERIFICATION.md).

## How an oracle is made

1. `research/mint_oracles.py` builds a MIL program and, where the case needs one, a weight blob. It changes one input at a time in the first campaign; `envelope_campaign()` adds the `env_*` cases that probe the outer edge of the accepted forms.
2. It invokes Apple's compiler for both targets on an arm64 Mac over SSH, parses the returned Mach-O load commands, splits every task of every program the object declares, and decodes each register stream with `research/h13_td.py`.
3. It writes one JSON record per attempt — MIL text, weight description, decoded task words, program and tensor descriptors, constant-section size and hashes, compiler status, and the SHA-256 of the compiler, the minting driver, and the decoder — then **deletes the HWX**. No Apple-generated container bytes enter the repository.
4. A task whose register stream cannot be split is stored with its header words and a `decode_error`, and the case still counts as accepted. A parser gap is therefore never recorded as an Apple rejection.

**Evidence: high for the checked-in procedure.** See the method sections of [oracle-diff.md](../../research/oracle-diff.md) and [oracle-envelope.md](../../research/oracle-envelope.md).

Constant sections are preserved as evidence without preserving Apple's bytes: every accepted record keeps the section size, whole-section SHA-256, nonzero-byte count, first-128-byte SHA-256, and tail nonzero count; sections up to 64 KiB also keep 2 KiB chunk hashes, and sections no larger than 256 bytes keep every nonzero fp16 word as an index/value table. That is enough to prove a byte-exact reconstruction and not enough to redistribute the original. **Evidence: high.** See [oracle-diff.md](../../research/oracle-diff.md).

## From oracle to encoder

A template generator reads the decoded records, groups them by a key that names the geometry (for example `(operation, input CHW, reduced-axis mask, keep_dims)` for normalization), and asserts that two records sharing a key agree on every decoded word, section hash, and descriptor before it writes a `.inc` file. The encoders in `plugins/H13` and `plugins/H14` then emit that stream for a covered geometry and refuse anything else with a named error such as `h13.norm-outside-envelope`, `h14.outside-parity-envelope`, or `h14.transpose-x-unsupported`. Nothing is interpolated between grid points. **Evidence: high for the generated templates and named refusals.** See [h13-td-fields.md](../../research/h13-td-fields.md), [h14-td-fields.md](../../research/h14-td-fields.md), and the [README](../../README.md#experimental-m2h14-compilation).

Where a formula was derivable it is implemented as a formula, not a table: the matvec weight permutation, the surface-address chain, the padded-row surface rule, and the mean divisor `fp32(1 / reduced extent)` are computed. Where no formula was found the decoded value is carried per case, and the field is listed as unresolved. **Evidence: high.** See [h13-td-fields.md](../../research/h13-td-fields.md).

## Uniform weights cannot prove a permutation

A campaign that compiles a uniform `fp16 0.5` weight cannot distinguish one constant-section permutation from another: every permutation of identical values has the same hash. Known-pattern probes were minted for exactly this reason — zero, mask, index, per-element one-hot, and row/column-stain payloads — and the permutation was then read out of the recorded hashes rather than guessed. 502 of 505 H13 probes and all 125 H14 probes reproduce Apple's section hashes; the three H13 mismatches are geometries outside the accepted envelope. **Evidence: high.** See the matvec section of [h13-td-fields.md](../../research/h13-td-fields.md) and the appendix of [h14-td-fields.md](../../research/h14-td-fields.md).

The same trap appears in reverse for the whole method: a score computed on an input that cannot carry the signal is not a pass. Validate that the fixture can distinguish the hypotheses before trusting agreement.

## What byte equality proves

Parity with an oracle proves this and only this:

- For one fixed MIL program, target, and compiler version, the emitted object matches what Apple's compiler produced for that input.
- Every field the decoder reads is reproduced, including fields whose meaning is unknown.

It does not prove:

- **Device execution.** No H13 or H14 program in this repository has been executed on an M1 or M2 ANE. Parity with Apple's emitted stream is not proof that the hardware accepts or computes it. **Evidence: high; the repository's own hardware receipts record the blocker rather than a run.** See the [handoff receipt](../../receipts/2026-09-05-ane-community/h13-handoff.json) and the [M1 Ultra runtime blocker](../../receipts/2026-09-05-ane-community/m1ultra-runtime-blocker.json).
- **Field understanding.** Unresolved words are copied, not understood: 54 of 70 H14 Kernel DMA words and 467 H13 normalization word slots remain unexplained. See [h14-td-fields.md](../../research/h14-td-fields.md) and [h13-td-fields.md](../../research/h13-td-fields.md).
- **Portability.** The bytes are specific to a generation, a compiler build, and an operating-system release. Two runs of Apple's compiler on the same MIL produce different HWX **file** hashes with identical decoded task words, so the decoded stream is the stable evidence and the file hash is only a record identifier. See [oracle-diff.md](../../research/oracle-diff.md).
- **Safe interpolation.** A covered grid point is not a covered neighborhood. `matmul` at M=128 K=4096 compiles and M=128 K=8192 is refused, with no size ceiling to explain it. See section 6 of [oracle-envelope.md](../../research/oracle-envelope.md).
- **Numerical agreement.** Parity is a byte contract; numerics are a separate contract with its own reference. See below.

## Parity is not numerics

Apple's softmax evaluates the exponential and the reciprocal through the fp16 lookup tables in its constant section, and `layer_norm`'s reciprocal square root is hardware-evaluated. `tools/h13_reference.py` accumulates sums, means, and variances in float32 and rounds once. The two therefore agree in form, not bit for bit: the reference is the numerical contract and the oracle bytes are the parity contract. A chunked matmul adds a third distinction, since partial sums are rounded to fp16 between chunks, which is why such tensors are marked `accumulation: chunked-fp16` and compared with a tolerance rather than for equality. **Evidence: high as a scope statement.** See the normalization section of [h13-td-fields.md](../../research/h13-td-fields.md) and the [README](../../README.md#h13-reference-and-linux-dispatch).

## The evidence ladder

Each rung adds something the rung below cannot supply:

1. **Decoded oracle.** Apple accepted this MIL and emitted these fields.
2. **Encoder parity.** A source-native encoder reproduces those fields exactly, from a formula where one is known.
3. **Container validation.** An independent inspector parses the emitted object and checks its structure, bounds, and identifiers.
4. **Device execution with valid inputs.** The hardware accepts the object and computes the tested result — for the exercised shape, data type, operation, and runtime only.
5. **Numerical qualification against a higher-precision reference**, with recorded tolerances, host, operating-system build, and compiler revision.

This repository is at rung 3 for H13 and H14. Rungs 4 and 5 are reached for H16G on an M4 and recorded in the [verification guide](../../docs/VERIFICATION.md). **Evidence: high as a status statement.**

## Rules this method depends on

- Never invent a fixture that the hardware or compiler would treat as invalid to make a case "pass"; a refusal is data, and an invalid tensor produces no usable evidence.
- Record the compiler, driver, and decoder identity per record, not once per report. The corpus deliberately carries two provenance sets because one campaign was not re-minted.
- Keep a rejection record as a rejection record. It documents the attempted MIL and the compiler's response; it is not a decoded oracle.
- State an absent write as absent. It is not a zero write.
- Re-derive claimed equality from the compiler's own output in a test, so that a stale document cannot become the evidence.

**Evidence: high as the repository's applied policy.** See [oracle-diff.md](../../research/oracle-diff.md), [oracle-envelope.md](../../research/oracle-envelope.md), and the [disclaimer](../../DISCLAIMER.md).

## Open questions

- **Open question:** Which unresolved fields would a device run prove irrelevant, and which would it prove load-bearing?
- **Open question:** How far does a parity template stay valid across Apple compiler releases on the same target?
- **Open question:** What is the smallest probe set that pins a constant-section layout without brute-forcing chunk hashes?
