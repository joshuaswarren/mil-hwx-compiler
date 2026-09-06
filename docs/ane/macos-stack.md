# The macOS software stack

## Supported application boundary

Core ML is Apple's public model-deployment API. An application supplies a model, configuration, and inputs; Core ML selects an execution plan. `MLComputeUnits` lets an application permit the CPU, GPU, and Neural Engine, but it does not expose ANE instructions, HWX objects, or task descriptors. **Evidence: high.** See Apple's [Core ML documentation](https://developer.apple.com/documentation/coreml) and [`MLComputeUnits`](https://developer.apple.com/documentation/coreml/mlcomputeunits).

The setting `.cpuAndNeuralEngine` is permission to use those compute units. It is not proof that every operation ran on the ANE. Core ML can partition or reject work according to model support and runtime policy. **Evidence: high for the API contract; medium for observed partitioning.** Apple presents compute-unit selection in the [`MLComputeUnits` documentation](https://developer.apple.com/documentation/coreml/mlcomputeunits); Hollance documents symbolic-breakpoint methods for identifying the selected Espresso engine in [is-model-using-ane.md](https://github.com/hollance/neural-engine/blob/d0bb5305a595a59142df47a4581e8b0b765a7385/docs/is-model-using-ane.md).

## Reconstructed private path

The following path is a useful model, not a promised ABI:

```text
Core ML model and configuration
        |
        v
CoreML.framework
        |
        v
Espresso execution and compilation layers
        |
        +--> CPU engine
        +--> Metal/GPU engine
        `--> ANE runtime engine
                    |
                    v
        AppleNeuralEngine.framework
        _ANEClient, _ANEModel, _ANERequest,
        _ANEIOSurfaceObject
                    |
                    v
        ANEServices / aned helpers
                    |
                    v
        AppleH11ANEInterface IOKit user client
                    |
                    v
             ANE hardware
```

Historical tinygrad traces show Core ML entering Espresso, Espresso's `ANERuntimeEngine`, `AppleNeuralEngine.framework`, ANEServices, and `AppleH11ANEInterface`. **Evidence: medium.** The trace and component inventory are fixed at tinygrad revision [`1dcaecacc476ca6369b427961c578dddf2eb9f35`](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/README).

Hollance observed `ANERuntimeEngine` for ANE execution, `MPSEngine` or `MetalLowmemEngine` for GPU execution, and `BNNSEngine` for CPU execution in the tested operating-system versions. Those class names are diagnostic observations, not stable interfaces. **Evidence: medium.** See the pinned [runtime-engine notes](https://github.com/hollance/neural-engine/blob/d0bb5305a595a59142df47a4581e8b0b765a7385/docs/is-model-using-ane.md).

Project Zero found both a less-privileged `H11ANEIn` user client and a more privileged entitlement-gated ANE path while auditing Apple kernel attack surfaces. This establishes that the driver boundary has had multiple user-client surfaces; it does not establish their current names or availability. **Evidence: high for the audited release, low for current releases.** See Project Zero's [“Oops, I missed it again”](https://projectzero.google/2020/11/oops-i-missed-it-again.html).

## Compilation

Historical research code calls a private `ANECCompile` entry point with an Espresso-style network description and parameter dictionaries, then receives a compiled object. **Evidence: medium.** See tinygrad's pinned [`compile.m`](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/2_compile/compile.m).

A newer private route constructs `_ANEInMemoryModelDescriptor` from MIL text and a weights blob, then uses `_ANEInMemoryModel` to compile and load it. **Evidence: medium.** See maderix/ANE's pinned [`ane_bridge.m`](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/bridge/ane_bridge.m).

This repository deliberately does not call `ANECCompile` in its delivered compiler. It emits the container and task bytes from source-native encoders. **Evidence: high.** See the repository [disclaimer](../../DISCLAIMER.md) and [verification guide](../../docs/VERIFICATION.md).

## Loading and execution

Private runtimes commonly bind tensors through IOSurface objects. IOSurface provides shareable pixel or tensor storage across process and subsystem boundaries; it does not imply that the storage resides inside the ANE. **Evidence: high for IOSurface's public role; medium for private ANE binding.** See Apple's [IOSurface documentation](https://developer.apple.com/documentation/iosurface) and maderix/ANE's [`ane_bridge.m`](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/bridge/ane_bridge.m).

The source-native runtime in this repository creates IOSurfaces from a binding manifest and calls the private framework. The manifest distinguishes logical tensor dimensions from padded physical rows, planes, batches, and allocation sizes. **Evidence: high.** See the local [runtime boundary](../../README.md#runtime-boundary) and [verification guide](../../docs/VERIFICATION.md).

On the tested macOS 26.6.2 M1 Ultra host, direct load of a source-native H13 object missed the `aned` in-memory model cache, while the cache directory was protected against creation by the test process. No direct private API that accepts arbitrary compiled bytes was found in the bounded source search. This is a host-specific blocker, not proof that no such route exists. **Evidence: medium.** See [`receipts/m1ultra-runtime-blocker.json`](../../receipts/2026-09-05-ane-community/m1ultra-runtime-blocker.json).

## Entitlements and stability

Historical direct-I/O research used the private entitlement `com.apple.ane.iokit-user-access`. **Evidence: medium for the audited release.** See tinygrad's pinned [`entitlements.xml`](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/3_run/entitlements.xml).

Private class names, selectors, entitlements, cache paths, and accepted binary layouts can change without source compatibility. They are unsuitable as an application-distribution contract. **Evidence: high for the absence of a public contract; medium for the named private mechanisms.** Apple's public surface is [Core ML](https://developer.apple.com/documentation/coreml); the private mechanisms are independently observed in the pinned sources above.

## Open questions

- **Open question:** Which service owns compilation and cache population on each current macOS build?
- **Open question:** Is there a current, entitlement-accessible API that loads arbitrary source-native HWX bytes without an `aned` cache entry?
- **Open question:** Which Core ML graph partitions and data transfers occur for a specific mixed CPU/GPU/ANE model? Measure them per model and operating-system build.
