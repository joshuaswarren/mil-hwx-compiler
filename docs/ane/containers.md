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

## Logical contents

A decoded object contains these logical layers:

1. Mach-O headers and load commands identify architecture and locate sections.
2. A text section contains a task descriptor encoded as register-write records.
3. Constant data carries program descriptors, metadata, and operation parameters.
4. Binding metadata describes input, output, and intermediate surfaces.

**Evidence: medium.** See the local [H13 field ledger](../../research/h13-hwx-fields.md), [H14 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md), and allbilly's pinned [H13 elementwise encoder](https://github.com/allbilly/ane/blob/e159e2d18ce6cea100e8f19bb27a7f07acaa9c24/examples/elementwise.py).

The task descriptor is not a CPU instruction stream. It is a sequence of register blocks interpreted by ANE control hardware. **Evidence: medium.** The record decoder and address maps are visible in the pinned [parser and maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump); Apple patents describe task assignment and tile-processing machinery at a higher architectural level in [US20190340490A1](https://patents.google.com/patent/US20190340490A1/en).

## ANEC wrapper

In the H13 wrapper decoded by this repository, the header occupies `0x1000` bytes, the task starts at file offset `0x1000`, and constants start at `0x1280`. The current channel assignments are output channel 4, required input channel 5, and optional input channel 6. The observed physical channel stride is 64 bytes. **Evidence: medium.** See the local [H13 field ledger](../../research/h13-hwx-fields.md).

The older open `libane` implementation constructs an ANEC buffer with a `0x800`-byte header and performs explicit tile/untile conversion around a `0x4000` tile size. It must not be assumed compatible with this repository's newer H13 wrapper without field-by-field validation. **Evidence: high for that implementation; medium for compatibility risk.** See the pinned [`libane/ane.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c).

## Parsing rules

A safe inspector should:

- validate Mach-O magic, endianness, and architecture fields;
- bound every load command and section within the file;
- accept task lengths derived from section metadata rather than one fixture size;
- validate each register-record header before consuming payload words;
- identify unknown addresses without treating them as known zero-filled blocks;
- keep logical tensor sizes separate from physical allocation and stride fields.

**Evidence: high as parser invariants.** The need for variable task lengths follows from observed H13 object differences documented in the task research and from the generation-specific maps in [coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

## Open questions

- **Open question:** What semantic contract covers every H13 program-descriptor and metadata field?
- **Open question:** Which ANEC header revisions correspond to which driver and firmware revisions?
- **Open question:** Are multiple tasks or architecture slices accepted in one current production wrapper, and how are they selected?
