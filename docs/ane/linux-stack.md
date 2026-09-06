# The Linux and Asahi stack

Linux support has separate firmware, kernel, userspace, compiler, and model-runtime layers. Evidence for one layer does not establish support for the others.

## Platform and firmware

Asahi documents the ANE as an accelerator whose proprietary firmware is supplied with macOS installation assets in Preboot. **Evidence: high for Asahi's platform documentation; medium for details not published by Apple.** See Asahi's [accelerator documentation](https://asahilinux.org/docs/hw/soc/accelerators/).

Asahi's public M1 and M2 feature matrices list the Neural Engine as work in progress or out of tree rather than a generally supported distribution feature. **Evidence: high.** See the [M1](https://asahilinux.org/docs/platform/feature-support/m1/) and [M2](https://asahilinux.org/docs/platform/feature-support/m2/) feature pages. Check those live matrices before treating this statement as current.

## m1n1 research path

`m1n1` provides a low-level Apple Silicon research environment. Its ANE experiment powers the device, converts a compiled HWX object into the expected wrapper, supplies FP16 input buffers, enqueues a task, and reads the output. Its firmware helper talks through DART-backed device mappings. **Evidence: high for the code path at the fixed revision.** See the pinned [`ane.py` experiment](https://github.com/AsahiLinux/m1n1/blob/940439b9a407fbfc499bea933269219f3f62d4c7/proxyclient/experiments/ane.py) and [`fw/ane.py`](https://github.com/AsahiLinux/m1n1/blob/940439b9a407fbfc499bea933269219f3f62d4c7/proxyclient/m1n1/fw/ane.py).

This is a research execution path, not a Linux kernel ABI or an end-user model runtime. **Evidence: high.** The code runs inside the m1n1 proxy environment and directly manages device state in the sources above.

## Out-of-tree Linux driver

The `eiln/ane` project provides an out-of-tree DRM accelerator driver and a small userspace library. The driver allocates GEM buffers, maps them through the IOMMU/DART path, manages task queues, and exposes DRM ioctls. **Evidence: high for revision `0dcea9976fae0b500a236a62fca69cd4d39f0809`.** See [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c) and [`ane_tm.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_tm.c).

The driver revision matches `apple,t8103-ane` and `apple,t6000-ane`. Those identifiers cover M1-family device-tree targets represented by those compatibles. The source does not claim M2 or later support. **Evidence: high.** See the match table in [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c#L649-L650) and Asahi's [SoC codename table](https://asahilinux.org/docs/hw/soc/soc-codenames/).

The userspace library wraps DRM ioctls, allocates buffers, creates an older ANEC header, and tiles or untiles data in `0x4000`-byte units. **Evidence: high for that revision.** See [`libane/ane.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c).

## Compiler boundary

A Linux driver does not compile Core ML or MIL into HWX. It accepts already compiled program data in the format expected by its driver and firmware combination. **Evidence: high for the open driver architecture.** The ioctl and task paths in [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c) and [`ane_tm.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_tm.c) contain no graph compiler.

This repository addresses that missing source-native compiler layer. Its H13 and H16G encoders construct target-specific containers and task data without calling Apple's compiler at runtime. **Evidence: high.** See the repository [README](../../README.md), [disclaimer](../../DISCLAIMER.md), and [verification guide](../../docs/VERIFICATION.md).

The open driver's old `0x800`-byte ANEC header and this repository's observed H13 `0x1000` header are different contracts. Connecting the compiler to that driver requires an explicit wrapper adaptation and hardware validation. Copying bytes between them is not a supported integration. **Evidence: medium.** Compare [`libane/ane.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c) with the local [H13 field ledger](../../research/h13-hwx-fields.md).

## Historical tinygrad work

Tinygrad once contained direct ANE compilation and execution research. The historical tree documents the macOS stack, private compiler calls, task descriptors, and direct execution. **Evidence: high for the historical revision.** See revision [`1dcaecacc476ca6369b427961c578dddf2eb9f35`](https://github.com/tinygrad/tinygrad/tree/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane).

Tinygrad later removed that ANE tree. Current tinygrad must not be cited as if it ships an ANE backend. **Evidence: high.** See removal commit [`641b1dbb40f95a478cddbdedd7ee312072e39254`](https://github.com/tinygrad/tinygrad/commit/641b1dbb40f95a478cddbdedd7ee312072e39254).

## Integration checklist

A complete Linux path needs all of the following:

1. firmware acquisition and boot for the exact SoC;
2. kernel device, IOMMU, power, interrupt, queue, and buffer support;
3. a stable userspace buffer and submission ABI;
4. a compiler that targets the exact HWX and ANEC revisions;
5. tensor layout conversion and cache synchronization;
6. valid-input hardware tests for numerical correctness and named failures;
7. a model runtime that partitions unsupported operations and manages lifetime.

**Evidence: high as a decomposition of the cited implementations.** The m1n1 experiment supplies layers 1, 2, and a research form of 3; `eiln/ane` supplies an out-of-tree form of layers 2 and 3; this repository supplies layer 4 and parts of 5 and 6. No cited project establishes the complete current-generation stack.

## Open questions

- **Open question:** Which current Linux kernel interfaces and firmware versions are required for M2 through M5 systems?
- **Open question:** What cache-maintenance and synchronization rules are required on every supported memory path?
- **Open question:** Which source-native wrapper should connect this compiler to a maintained DRM accelerator ABI?
- **Open question:** Who owns compatibility testing across firmware, kernel, compiler, and SoC revisions?
