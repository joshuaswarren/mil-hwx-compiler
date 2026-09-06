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

That private compile route also cross-targets. On one M1 Ultra running macOS 26.6.2, the same `add [1,64,1,1]` fp16 MIL compiled with `TargetArchitecture` set to `h13`, `h14`, `h15`, `h16`, and `h17` returned five objects with CPU subtypes 4, 5, 6, 7, and 9; the `h11` request failed. Compilation for a generation therefore does not require that generation's hardware, which is how this repository holds an H14/M2 oracle corpus with no M2 device. **Evidence: medium; one host, one input, one compiler build.** See [`receipts/anecompile-cross-target.json`](../../receipts/2026-09-05-ane-community/anecompile-cross-target.json) and the [generation table](generations.md).

This repository deliberately does not call `ANECCompile` in its delivered compiler. It emits the container and task bytes from source-native encoders. **Evidence: high.** See the repository [disclaimer](../../DISCLAIMER.md) and [verification guide](../../docs/VERIFICATION.md).

## Loading and execution

Private runtimes commonly bind tensors through IOSurface objects. IOSurface provides shareable pixel or tensor storage across process and subsystem boundaries; it does not imply that the storage resides inside the ANE. **Evidence: high for IOSurface's public role; medium for private ANE binding.** See Apple's [IOSurface documentation](https://developer.apple.com/documentation/iosurface) and maderix/ANE's [`ane_bridge.m`](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/bridge/ane_bridge.m).

The source-native runtime in this repository creates IOSurfaces from a binding manifest and calls the private framework. The manifest distinguishes logical tensor dimensions from padded physical rows, planes, batches, and allocation sizes. **Evidence: high.** See the local [runtime boundary](../../README.md#runtime-boundary) and [verification guide](../../docs/VERIFICATION.md).

## The `aned` cache is a data vault on current builds

Compiling is not the blocker; loading is. On the tested macOS 26.6.2 M1 Ultra host, a direct load of a compiled object missed the `aned` in-memory model cache, and the cache directory could not be provisioned to fix that. The failure is not ordinary permissions:

- `sudo` works, but **root** cannot stat, list, or create entries under `/Library/Caches/com.apple.aned` or `com.apple.aneuserd`, even though `/Library/Caches` itself is `drwxrwxrwt`.
- `mkdir` of the model-cache path returns `Operation not permitted`. A neighbouring `com.apple.amsengagementd.classicdatavault` directory fails identically, which is what identifies the behaviour as a data vault rather than a mode bit.
- Without a cache entry the runtime reports `ANERuntimeErrorCacheMiss`; no model load or evaluation occurs.
- No private API that accepts arbitrary compiled bytes without a cache entry was found in the bounded source search.

**Evidence: medium; one host and operating-system build.** See [`receipts/m1ultra-runtime-blocker.json`](../../receipts/2026-09-05-ane-community/m1ultra-runtime-blocker.json).

Two routes past it are recorded, and both are owner decisions rather than code changes:

1. **Disable SIP from Recovery** on the 26.6 host, which is what the H13 macOS hardware gate documents as its prerequisite for the `InMemoryModelCache/<key>` path to be writable by root. **Evidence: medium.** See the [handoff receipt](../../receipts/2026-09-05-ane-community/h13-handoff.json).
2. **Use a build whose cache is provisionable**, which the repository recorded on macOS 26.3 build `25D125` — the build behind the H16G hardware results. None of the surveyed 26.6.x hosts behaves that way. **Evidence: high for the recorded H16G runs on 25D125; medium for the 26.6.x survey.** See the [verification guide](../../docs/VERIFICATION.md) and [`receipts/m1ultra-runtime-blocker.json`](../../receipts/2026-09-05-ane-community/m1ultra-runtime-blocker.json).

This is a host-and-build blocker, not proof that no load route exists. The Linux path avoids it entirely by talking to the open driver instead of `aned`; see the [Linux stack](linux-stack.md).

## Entitlements and stability

Historical direct-I/O research used the private entitlement `com.apple.ane.iokit-user-access`. **Evidence: medium for the audited release.** See tinygrad's pinned [`entitlements.xml`](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/3_run/entitlements.xml).

Private class names, selectors, entitlements, cache paths, and accepted binary layouts can change without source compatibility. They are unsuitable as an application-distribution contract. **Evidence: high for the absence of a public contract; medium for the named private mechanisms.** Apple's public surface is [Core ML](https://developer.apple.com/documentation/coreml); the private mechanisms are independently observed in the pinned sources above.

## Open questions

- **Open question:** Which service owns compilation and cache population on each current macOS build?
- **Open question:** Is there a current, entitlement-accessible API that loads arbitrary source-native HWX bytes without an `aned` cache entry?
- **Open question:** Which Core ML graph partitions and data transfers occur for a specific mixed CPU/GPU/ANE model? Measure them per model and operating-system build.
- **Open question:** Which macOS builds vault `/Library/Caches/com.apple.aned`, and what changed between 26.3 `25D125`, where provisioning worked, and 26.6.x, where root is refused?
