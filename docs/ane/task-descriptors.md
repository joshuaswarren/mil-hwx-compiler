# Task descriptors and register-block maps

## Record encoding

A task starts with a fixed header in the currently decoded H13 layout. The remaining words form records. Each record begins with a header word whose upper bits encode record length and whose lower bits encode a register address, followed by the payload words written to that block. The working decoder uses:

```text
record_words = (header >> 26) + 1
address      = header & 0x03ffffff
```

**Evidence: medium.** This is an independently reconstructed encoding implemented by the fixed [coreml_to_ane_hwx parser](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump) and exercised by this repository's [HWX inspector](../../research/inspect_hwx.py). It is not an Apple-published ABI.

A record address names a hardware register region. Payload indices have meaning only within that region and generation. A value at NE word 1 must not be interpreted using a Tile DMA word-1 definition. **Evidence: medium.** The separate structures and writers in allbilly's [H13 sources](https://github.com/allbilly/ane/tree/e159e2d18ce6cea100e8f19bb27a7f07acaa9c24) and freedomtan's [generation maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump) support this boundary.

## H13 block map

| Base address | Name used in research code | Role supported by evidence | Evidence |
|---:|---|---|---|
| `0x00000` | Common | Task-wide control fields | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |
| `0x04800` | L2 | Shared-buffer and transfer configuration | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |
| `0x08800` | PE | Planar-engine configuration | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |
| `0x0c800` | NE | Neural-engine arithmetic configuration | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |
| `0x13800` | Tile DMA source | Source surface shape, stride, and addressing | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |
| `0x17800` | Tile DMA destination | Destination surface shape, stride, and addressing | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |
| `0x1f800` | Kernel DMA | Kernel or constant transfer configuration | **Medium**, [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md) |

Apple's patents describe a tiled neural processor with data buffers, buffer DMA, neural-engine circuitry, and a multi-mode planar engine. The patents support the broad roles above, but they do not prove that each patent block maps one-to-one to these H13 addresses. **Evidence: high for the disclosed concepts; low for direct register mapping.** See [US20190340491A1](https://patents.google.com/patent/US20190340491A1/en), [US20190340490A1](https://patents.google.com/patent/US20190340490A1/en), and [US20210103803A1](https://patents.google.com/patent/US20210103803A1/en).

## Generation changes

The map is not stable across generations. The H14 map uses compact legacy addresses and translated modern addresses. Its modern regions include Common `0x0000`, L2 `0x4100`, PE `0x4500`, NE `0x4900`, source DMA `0x4d00`, destination DMA `0x5100`, and Kernel DMA `0x5500`. **Evidence: medium.** See the pinned [H14 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md).

The same source adds a Cache DMA region at `0x5900` for H16 and later maps and expands several register structures. **Evidence: medium.** See the pinned [H16 register definitions](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

A compiler must therefore select a generation-specific encoder. Translating only block base addresses is insufficient when word counts and field meanings also change. **Evidence: medium.** The differing structures in the fixed [generation maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump) demonstrate both changes.

## Shape and stride fields

Logical shape is the tensor shape seen by the operation. Physical shape includes row padding, planes, batches, and allocation size required by transfer hardware. A narrow FP16 row can occupy a 64-byte physical row even when its logical byte count is smaller. **Evidence: medium.** See the local [H13 field ledger](../../research/h13-hwx-fields.md), [runtime boundary](../../README.md#runtime-boundary), and allbilly's [elementwise encoder](https://github.com/allbilly/ane/blob/e159e2d18ce6cea100e8f19bb27a7f07acaa9c24/examples/elementwise.py).

Do not infer logical dimensions by dividing allocation size by element size. Padding and tiling break that equivalence. **Evidence: high as a consequence of the distinct logical and physical fields; medium for the decoded field meanings.** See the same sources above.

## What byte equality proves

For one fixed input program, target, and compiler version, byte equality with an independently produced oracle proves that the encoded object matches that oracle. It does not prove that all fields are understood or that the bytes are portable to another operating-system or hardware generation. **Evidence: high as a scope statement.** This repository's H16G method stores native encoder data and verifies hashes and decoded values without distributing Apple-generated containers; see [verification](../../docs/VERIFICATION.md) and [provenance](../../README.md#provenance).

Hardware execution with valid inputs adds evidence that the object is accepted and computes the tested result. It still covers only the exercised shape, data type, operation, and runtime. **Evidence: high as a test-scope statement.** The repository records its hardware cases and numerical checks in [verification](../../docs/VERIFICATION.md).

## Open questions

- **Open question:** Which H13 task-header words encode task identity, generation, dependency, and scheduling state?
- **Open question:** What do all Kernel DMA words at `0x1f800` and later records mean?
- **Open question:** Which block fields are firmware-facing contracts and which are direct hardware register images?
- **Open question:** Which descriptor fields may vary without changing observable computation?
