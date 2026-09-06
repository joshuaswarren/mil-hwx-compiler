# Memory model and common myths

## Start with observable objects

The macOS paths in the cited projects allocate IOSurfaces for inputs, outputs, weights, and intermediates. Those surfaces have logical tensor shapes and physical byte layouts with row alignment and padding. **Evidence: high for the APIs and repository behavior; medium for undocumented ANE interpretation.** See Apple's [IOSurface documentation](https://developer.apple.com/documentation/iosurface), maderix/ANE's pinned [`ane_bridge.m`](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/bridge/ane_bridge.m), and this repository's [runtime boundary](../../README.md#runtime-boundary).

The Linux research paths allocate host-visible buffers, map them through DART/IOMMU, and pass device addresses to task machinery. **Evidence: high for the fixed source revisions.** See m1n1's [`fw/ane.py`](https://github.com/AsahiLinux/m1n1/blob/940439b9a407fbfc499bea933269219f3f62d4c7/proxyclient/m1n1/fw/ane.py) and `eiln/ane`'s [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c).

These facts establish shared-system-memory mappings. They do not reveal the full internal cache and SRAM hierarchy of a shipping ANE. **Evidence: high for the limitation.** The cited software observes mappings and transfers but does not expose a complete physical memory diagram.

## Logical size is not physical size

For a tensor with element size `E`, logical byte count is the product of logical dimensions times `E`. Physical allocation can be larger because of row alignment, plane stride, batch stride, tiling, and guard requirements. **Evidence: high as a layout rule; medium for decoded ANE fields.** See the local [H13 field ledger](../../research/h13-hwx-fields.md) and the older tile conversion in [`libane/ane.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c).

The current H13 evidence uses a 64-byte physical channel stride for narrow FP16 data. This is an observed format rule, not proof that every generation, rank, or transfer mode uses 64 bytes. **Evidence: medium.** See the [H13 field ledger](../../research/h13-hwx-fields.md).

## Data movement blocks

The reconstructed H13 task map contains source Tile DMA, destination Tile DMA, Kernel DMA, L2, PE, and NE register blocks. Later maps add a Cache DMA region. **Evidence: medium.** See [task descriptors](task-descriptors.md) and the fixed [generation maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

Apple patents describe data buffers, buffer-DMA engines, neural-engine tiles, and a multi-mode planar engine. They provide useful architectural vocabulary, but a patent drawing is not evidence that a named shipping chip implements the exact sizes or connections shown. **Evidence: high for the patent disclosure; low for direct implementation mapping.** See [US20190340491A1](https://patents.google.com/patent/US20190340491A1/en), [US20190340490A1](https://patents.google.com/patent/US20190340490A1/en), and [US20210103803A1](https://patents.google.com/patent/US20210103803A1/en).

## Synchronization and lifetime

A buffer must remain allocated and correctly mapped while hardware can access it. Producer writes must become visible before submission, and consumer reads must occur after completion. The exact cache-maintenance operations depend on the kernel, mapping attributes, and device coherency contract. **Evidence: high as a DMA safety requirement; open question for undocumented generation-specific details.** The mapping, queue, and completion paths are visible in [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c) and [`ane_tm.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_tm.c).

Removing waits or lifetime guards because unified memory “should be coherent” is unsafe. Unified physical memory does not remove asynchronous device execution, ownership, address translation, or object lifetime. **Evidence: high as an execution-order rule.** The open driver has explicit enqueue, execution, and completion paths in [`ane_tm.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_tm.c).

## Myths

### “Unified memory means zero-copy”

Unified memory means processors can access a shared physical memory system. A runtime may still copy, retile, pad, convert data types, allocate intermediates, or migrate ownership. **Evidence: high for observed copies and layout conversion.** The older userspace library explicitly tiles and untiles in [`libane/ane.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c), while the local binding manifest distinguishes logical and physical layouts in the [runtime boundary](../../README.md#runtime-boundary).

### “IOSurface memory is ANE SRAM”

IOSurface is a shareable allocation interface. It does not name the accelerator's internal storage. **Evidence: high.** See Apple's [IOSurface documentation](https://developer.apple.com/documentation/iosurface).

### “Resident model banks are stored in ANE SRAM”

A process can retain compiled bank files, FP16 IOSurfaces, and runtime objects in system memory without proving that the entire model is resident in on-chip SRAM. oMLX's reported bank blobs and neural-ledger memory are host-visible accounting, not an internal SRAM measurement. **Evidence: high for the reported objects; open question for internal residency.** See the preserved sources and findings in [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

### “Neural-engine memory in a system monitor is the hardware capacity”

A software ledger reports allocations attributed by that software stack. It can include weights, intermediates, compiled objects, and bookkeeping. It does not establish physical on-chip capacity. **Evidence: medium.** The oMLX receipt records neural-ledger, Metal-headroom, IOSurface, and bank-file observations without a cited SRAM claim; see [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

### “A warm compile cache means weights are on the ANE”

A warm cache proves that reusable compilation artifacts exist. It does not prove where weights reside during execution. **Evidence: high as a distinction.** The `aned` cache behavior and source-native runtime boundary are documented in [macOS stack](macos-stack.md), while oMLX records separate compile-cache and memory effects in [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

### “One allocation number is model memory”

Model execution can involve file-backed compiled banks, mapped weights, activation surfaces, backend-specific caches, temporary conversion buffers, and fallback-device allocations. A single counter needs a precise owner and lifetime before it can be interpreted. **Evidence: medium.** The distinct categories appear in the oMLX evidence ledger, [`receipts/omlx.json`](../../receipts/2026-09-05-ane-community/omlx.json).

## What can be measured

Useful measurements include IOSurface allocation sizes, row and plane strides, mapped I/O virtual addresses, wrapper and task sizes, compile-cache bytes, process resident memory, and end-to-end peak memory. Report each separately with the tool and sampling point. **Evidence: high as an observability rule.** The cited source paths expose these objects at different layers; none makes them interchangeable.

## Open questions

- **Open question:** What are the private cache and SRAM capacities for each ANE generation?
- **Open question:** Which transfers use coherent mappings, explicit cache maintenance, or firmware-managed copies?
- **Open question:** When can weights or activations remain in internal storage across tasks?
- **Open question:** How do Cache DMA and L2 fields map to physical storage and replacement behavior on H16 through H18?
