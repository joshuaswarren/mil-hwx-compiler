# The Linux and Asahi stack

Linux support has separate firmware, kernel, userspace, compiler, and model-runtime layers. Evidence for one layer does not establish support for the others.

## Platform and firmware

Asahi documents the ANE as an accelerator whose proprietary firmware is supplied with macOS installation assets in Preboot. **Evidence: high for Asahi's platform documentation; medium for details not published by Apple.** See Asahi's [accelerator documentation](https://asahilinux.org/docs/hw/soc/accelerators/).

Asahi's public M1 and M2 feature matrices list the Neural Engine as work in progress or out of tree rather than a generally supported distribution feature. **Evidence: high.** See the [M1](https://asahilinux.org/docs/platform/feature-support/m1/) and [M2](https://asahilinux.org/docs/platform/feature-support/m2/) feature pages. Check those live matrices before treating this statement as current.

## m1n1 research path

`m1n1` provides a low-level Apple Silicon research environment. Its ANE experiment powers the device, converts a compiled HWX object into the expected wrapper, supplies FP16 input buffers, enqueues a task, and reads the output. Its firmware helper talks through DART-backed device mappings. **Evidence: high for the code path at the fixed revision.** See the pinned [`ane.py` experiment](https://github.com/AsahiLinux/m1n1/blob/940439b9a407fbfc499bea933269219f3f62d4c7/proxyclient/experiments/ane.py) and [`fw/ane.py`](https://github.com/AsahiLinux/m1n1/blob/940439b9a407fbfc499bea933269219f3f62d4c7/proxyclient/m1n1/fw/ane.py).

This is a research execution path, not a Linux kernel ABI or an end-user model runtime. **Evidence: high.** The code runs inside the m1n1 proxy environment and directly manages device state in the sources above.

## Out-of-tree Linux driver and reviewed runtime

The `eiln/ane` project provides an out-of-tree DRM accelerator driver and a small userspace library. The driver allocates GEM buffers, maps them through the IOMMU/DART path, manages task queues, and exposes DRM ioctls. **Evidence: high for revision `0dcea9976fae0b500a236a62fca69cd4d39f0809`.** See [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c) and [`ane_tm.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_tm.c).

The reviewed H13 runtime uses a libane Python shared library that exports `pyane_init`, `pyane_free`, `__ane_src_size`, `__ane_dst_size`, `__ane_send`, `__ane_read`, and `ane_exec`. Before native transfer it resolves every symbol and validates the libane-reported surface sizes against the package allocations. The reviewed identity record pins the shared library and `libane.a` by SHA-256. It does not supply approved identity values.

## Compiler boundary

A Linux driver does not compile Core ML or MIL into HWX. It accepts already compiled program data in the format expected by its reviewed driver and firmware combination. **Evidence: high for the open driver architecture.** The ioctl and task paths in [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c) and [`ane_tm.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_tm.c) contain no graph compiler.

This repository's H13 encoder constructs target-specific ANEC packages without calling Apple's compiler at runtime. The native wrapper only accepts a package whose `compiler.sha256` marker matches the pinned reviewed `ANE_COMPILER_BIN`; `--package DIR` verifies that marker before dispatch. Dry-run validates the package and reference contract without loading libane or performing device calls. The [M1 native receipt](../../receipts/2026-09-06-m1-native-progress.json) records successful execution and timing for eight first-run models and both 512-element add-ReLU schedules using the reviewed ABI-1 driver and library. This does not qualify other SoCs or end-to-end model integration.

## Historical tinygrad work

Tinygrad once contained direct ANE compilation and execution research. The historical tree documents the macOS stack, private compiler calls, task descriptors, and direct execution. **Evidence: high for the historical revision.** See revision [`1dcaecacc476ca6369b427961c578dddf2eb9f35`](https://github.com/tinygrad/tinygrad/tree/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane).

Tinygrad later removed that ANE tree. Current tinygrad must not be cited as if it ships an ANE backend. **Evidence: high.** See removal commit [`641b1dbb40f95a478cddbdedd7ee312072e39254`](https://github.com/tinygrad/tinygrad/commit/641b1dbb40f95a478cddbdedd7ee312072e39254).

## Integration boundary

A complete Linux path needs reviewed provenance for the exact firmware, kernel,
module, libane artifacts, compiler source, and compiler binary; a stable
userspace buffer and submission ABI; a compiler for the exact HWX and ANEC
revisions; tensor-layout conversion and cache synchronization; valid-input
native numerical qualification; and a model runtime that partitions unsupported
operations and manages lifetime.

**Evidence: high as a decomposition of the cited implementations.** The m1n1
experiment supplies research forms of firmware, kernel, and userspace layers;
`eiln/ane` supplies out-of-tree kernel and userspace layers; this repository
supplies the compiler and parts of layout conversion and qualification. No cited
project establishes the complete current-generation stack.

## Open questions

- **Open question:** Which current Linux kernel interfaces and firmware versions are required for M2 through M5 systems?
- **Open question:** What cache-maintenance and synchronization rules are required on every supported memory path?
- **Open question:** Which source-native wrapper should connect this compiler to a maintained DRM accelerator ABI?
- **Open question:** Who owns compatibility testing across firmware, kernel, compiler, and SoC revisions?
