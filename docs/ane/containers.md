# HWX and ANEC container anatomy

## Names and boundaries

**HWX** is the conventional research name for the compiled Mach-O object that carries ANE program data. **ANEC** is the wrapper used by runtime and driver paths to carry one or more compiled records plus metadata. Apple has not published either format as an application ABI. **Evidence: medium.** The names and parse flow appear in the fixed [coreml_to_ane_hwx parser](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump), the open [libane implementation](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c), and this repository's [disclaimer](../../DISCLAIMER.md).

A file extension alone does not identify a compatible generation or wrapper revision. Consumers must validate headers, bounds, architecture identifiers, section offsets, and record sizes before decoding. **Evidence: high as a parser safety requirement; medium as a format claim.** Compare the older `0x800` ANEC header in [libane](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c) with this repository's H13 `0x1000` wrapper in the [field ledger](../../research/h13-hwx-fields.md).

## HWX as Mach-O

Observed HWX objects are 64-bit little-endian Mach-O files. The Mach-O CPU type is `0x80`, and the CPU subtype selects the ANE generation. A load command carries the ANE ISA version. **Evidence: medium.** See the fixed [Mach-O parser](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m) and the local [H13 field ledger](../../research/h13-hwx-fields.md).

For the H13 objects currently decoded in this repository:

| Property | Observed value | Evidence |
|---|---:|---|
| CPU type | `0x80` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |
| CPU subtype | `4` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |
| ANE ISA | `7` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |
| `__TEXT` file offset | `0x4000` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |
| Observed `__text` size | `0x274` for the repository's native object; Apple output can use a different task size | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) and the task comparison recorded in repository research |
| Program descriptor size | `0x880` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |
| Metadata size | `0x728` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |
| FP16 element type code | `5` | **Medium**, [H13 field ledger](../../research/h13-hwx-fields.md) |

These values describe observed H13 files. They are not universal HWX constants. **Evidence: medium.** H14 and H16 register layouts differ in the pinned [generation maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

One Mac's compiler emits every generation from H13 through H17 for the same input, with the subtype changing per requested target. A single `add [1,64,1,1]` fp16 MIL compiled through the private entry point with `TargetArchitecture` set to `h13`, `h14`, `h15`, `h16`, and `h17` returned exit code 0 and CPU subtypes 4, 5, 6, 7, and 9; the `h11` request failed with exit code 1. The subtype-to-generation names still come from the reverse-engineered parser, not from this observation. **Evidence: medium; one host, one input, one compiler build.** See [`receipts/anecompile-cross-target.json`](../../receipts/2026-09-05-ane-community/anecompile-cross-target.json) and the [generation table](generations.md).

## Logical contents

A decoded object contains these logical layers:

1. Mach-O headers and load commands identify architecture and locate sections.
2. A text section contains a task descriptor encoded as register-write records.
3. Constant data carries program descriptors, metadata, and operation parameters.
4. Binding metadata describes input, output, and intermediate surfaces.

**Evidence: medium.** See the local [H13 field ledger](../../research/h13-hwx-fields.md), [H14 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md), and allbilly's pinned [H13 elementwise encoder](https://github.com/allbilly/ane/blob/e159e2d18ce6cea100e8f19bb27a7f07acaa9c24/examples/elementwise.py).

The task descriptor is not a CPU instruction stream. It is a sequence of register blocks interpreted by ANE control hardware. **Evidence: medium.** The record decoder and address maps are visible in the pinned [parser and maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump); Apple patents describe task assignment and tile-processing machinery at a higher architectural level in [US20190340490A1](https://patents.google.com/patent/US20190340490A1/en).

Two campaign results sharpen layers 3 and 4. The constant data is not a flat weight image: it concatenates packed weights, lookup tables, and bias rows, each in a family-specific packing — the convolution probes recovered dense, grouped, depthwise, and stride-2 layouts in which a bias is one extra row per plane and a stride-2 section's size depends on which weights are zero (conv section of [h13-td-fields.md](../../research/h13-td-fields.md), conv appendix of [h14-td-fields.md](../../research/h14-td-fields.md)). And the binding layer of a multi-operation chain declares **only the chain's inputs and its result**: no intermediate — not a feed-forward hidden state, not an attention score matrix — gets a tensor descriptor or a resource address; intermediates live in the `__DATA/__bss` scratch below the surfaces, based at `0x30000000`, reused across the blocks of a stack. **Evidence: high over the chain corpus.** See section 4 of [fusion-rules.md](../../research/fusion-rules.md) and [transformer-layer.md](transformer-layer.md).

## How many programs one object carries

Apple emitted exactly one program load command in every one of the 542 accepted objects of the envelope campaign, on both H13 and H14, including an object with 129 tasks and a 134,217,728-byte constant section. Work is partitioned into more tasks inside the single program, not into more programs. The minting driver parses multi-program objects and records `program_count` per object, so a second program would have been recorded rather than mis-parsed. **Evidence: high over that corpus; it does not prove Apple never emits two programs for some untested form.** See the partitioning section of [oracle-envelope.md](../../research/oracle-envelope.md).

The chain campaign extends the same result to multi-operation programs: every decoded chain object — including a 512-block stack at 10,240 tasks and a constant section of 402,786,304 bytes, and an 8,192-unit stack of 73,728 MIL operations — still carries exactly one program load command. **Evidence: high over the chain corpus; the two accepted-but-undecoded depth-4,096 and depth-8,192 objects are not counted.** See section 7 of [fusion-rules.md](../../research/fusion-rules.md).

An object's task stream is therefore the interesting structure. H13 links tasks by a next-offset field with zero padding between sections; H14 has no link and instead 16-byte-aligns each task after a zero-size 16-byte frame. A reader must follow the generation's own rule. **Evidence: high for the decoded records.** See [task descriptors](task-descriptors.md) and [h14-td-fields.md](../../research/h14-td-fields.md).

## ANEC wrapper

In the H13 wrapper decoded by this repository, the header occupies `0x1000` bytes, the task starts at file offset `0x1000`, and constants start at `0x1280`. The current channel assignments are output channel 4, required input channel 5, and optional input channel 6. The observed physical channel stride is 64 bytes. **Evidence: medium.** See the local [H13 field ledger](../../research/h13-hwx-fields.md).

The older open `libane` implementation constructs an ANEC buffer with a `0x800`-byte header and performs explicit tile/untile conversion around a `0x4000` tile size. It must not be assumed compatible with this repository's newer H13 wrapper without field-by-field validation. **Evidence: high for that implementation; medium for compatibility risk.** See the pinned [`libane/ane.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c).

The H14 form this repository emits keeps the same `0x1000`-byte header as H13, followed by the H14 task stream at `0x1000` and the constant section at `align_up(text_bytes, 64)`. Channel 4 is the output and channels 5 and 6 the inputs, as on H13. The first-task length field is informational for H14, because a reader walks the stream by task-header size and 16-byte alignment. **Evidence: high for the emitted contract; medium for how a future driver will consume it.** See the [README](../../README.md#h14-task-stream-anec-and-hwx-layout).

## Parsing rules

A safe inspector should:

- validate Mach-O magic, endianness, and architecture fields;
- bound every load command and section within the file;
- accept task lengths derived from section metadata rather than one fixture size;
- honour the generation's extended-header flag before splitting the register stream: a missed extra word raises an unaligned-address error on H13 and, worse, decodes silently wrong words on H14;
- validate each register-record header before consuming payload words;
- identify unknown addresses without treating them as known zero-filled blocks;
- keep logical tensor sizes separate from physical allocation and stride fields.

**Evidence: high as parser invariants.** The need for variable task lengths follows from observed H13 object differences documented in the task research and from the generation-specific maps in [coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

## Open questions

- **Open question:** What semantic contract covers every H13 program-descriptor and metadata field?
- **Open question:** Which ANEC header revisions correspond to which driver and firmware revisions?
- **Open question:** Does any current production wrapper carry more than one program? Apple emitted exactly one in all 542 accepted envelope objects and in every decoded chain object up to 10,240 tasks and 402 MB of constants, so the multi-program path is parsed but unobserved. See [oracle-envelope.md](../../research/oracle-envelope.md) and [fusion-rules.md](../../research/fusion-rules.md).
- **Open question:** How would a runtime select among architecture slices if an object ever carried more than one?
