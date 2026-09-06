# Glossary

Each definition is scoped to this repository and its cited research sources.

**ANE, Apple Neural Engine.** Apple's dedicated machine-learning accelerator. Applications normally reach it through Core ML rather than a public instruction API. **Evidence: high.** [Core ML](https://developer.apple.com/documentation/coreml).

**ANEC.** A research name for the runtime or driver wrapper around compiled ANE program data and metadata. Wrapper layouts differ across observed implementations. **Evidence: medium.** [libane](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/libane/ane.c), [H13 field ledger](../../research/h13-hwx-fields.md).

**ANECCompile.** An observed private compilation entry point that accepts Espresso-style model and parameter dictionaries. It is not used by this repository's delivered compiler. **Evidence: medium.** [tinygrad compile.m](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/2_compile/compile.m), [disclaimer](../../DISCLAIMER.md).

**ANEServices.** A private service layer observed between higher-level frameworks and the ANE IOKit interface in historical traces. **Evidence: medium for the audited release.** [tinygrad ANE notes](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/README).

**`aned`.** A macOS helper associated with ANE compilation and cached in-memory models in observed releases. Its exact responsibilities are private and release-dependent. **Evidence: medium.** [tinygrad ANE notes](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/README), [M1 Ultra blocker receipt](../../receipts/2026-09-05-ane-community/m1ultra-runtime-blocker.json).

**AppleH11ANEInterface.** An IOKit-facing driver interface name observed across several Apple Silicon releases despite later hardware generations. The name does not imply H11 hardware. **Evidence: medium.** [Project Zero](https://projectzero.google/2020/11/oops-i-missed-it-again.html), [tinygrad h11ane.h](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/3_run/h11ane.h).

**BF16.** A 16-bit floating-point format with an 8-bit exponent and 7 explicit fraction bits. Native ANE support must be established separately for each operation and generation. **Evidence: high for the format; open question for complete ANE support.** See Google Cloud's [bfloat16 format description](https://cloud.google.com/tpu/docs/bfloat16).

**BLOBFILE constant.** A weight or bias supplied to the compiler as an external blob file rather than as an inline tensor literal in the MIL text. The distinction is load-bearing: inline fp16 tensor constants were refused in all 12 sampled binary cases and in 6 convolution-bias cases, while the same values in a BLOBFILE compiled. **Evidence: high over the sampled cases.** [oracle-diff.md](../../research/oracle-diff.md), [oracle-envelope.md](../../research/oracle-envelope.md).

**Binding manifest.** This repository's description of logical tensor bindings and their physical row, plane, batch, stride, and allocation properties. **Evidence: high.** [Runtime boundary](../../README.md#runtime-boundary).

**Cache DMA.** A register region named in reconstructed H16-and-later maps. Its complete hardware semantics remain undocumented. **Evidence: medium.** [coreml_to_ane_hwx maps](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

**Constant section.** The `__TEXT/__const` region of a compiled object, holding packed weights, lookup tables, and per-operation parameter blocks. Its size and content are part of the parity contract; oracle records keep its size, hashes, and nonzero-word tables, never Apple's bytes. **Evidence: high.** [oracle-diff.md](../../research/oracle-diff.md).

**Core ML.** Apple's public framework for model representation, conversion, configuration, and execution across supported compute units. **Evidence: high.** [Core ML](https://developer.apple.com/documentation/coreml).

**CPU subtype.** The Mach-O field used by observed HWX files to identify an ANE target generation. It is not a CPU-core subtype in the ordinary application-binary sense. **Evidence: medium.** [HWX parser](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m).

**DART.** Apple's device address resolution table, an IOMMU used to map memory for devices. **Evidence: high for the m1n1 and Linux driver paths.** [m1n1 ANE firmware helper](https://github.com/AsahiLinux/m1n1/blob/940439b9a407fbfc499bea933269219f3f62d4c7/proxyclient/m1n1/fw/ane.py), [ane_drv.c](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c).

**Espresso.** A private Apple model compilation and execution layer observed beneath Core ML. **Evidence: medium.** [tinygrad ANE notes](https://github.com/tinygrad/tinygrad/blob/1dcaecacc476ca6369b427961c578dddf2eb9f35/ane/README), [Hollance runtime notes](https://github.com/hollance/neural-engine/blob/d0bb5305a595a59142df47a4581e8b0b765a7385/docs/is-model-using-ane.md).

**Extended task header.** The one extra word between a fixed task header and the first register record, present when the last header word's two low bits are both set (H13 `header[9]`, H14 `header[7]`). The declared task size already includes it, so only the register stream shifts. It appears only in runtime-runtime matmuls and attention chains. **Evidence: high over the decoded corpus.** [oracle-envelope.md](../../research/oracle-envelope.md), [task descriptors](task-descriptors.md).

**GEM.** The Linux DRM Graphics Execution Manager memory-object infrastructure used by the open ANE driver for device buffers. **Evidence: high for that driver.** [ane_drv.c](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c).

**HWX.** The conventional research name for a Mach-O object containing compiled ANE program data. Apple has not published it as an application ABI. **Evidence: medium.** [coreml_to_ane_hwx](https://github.com/freedomtan/coreml_to_ane_hwx/tree/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump).

**H11 through H18.** Community labels associated with successive HWX architecture subtypes. Product associations come from reverse-engineered parsers, not an Apple-published naming table. **Evidence: medium.** [hwx_parsing.m](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m).

**IOSurface.** Apple's shareable allocation object used to exchange image or tensor storage across processes and hardware subsystems. It is not a synonym for ANE SRAM. **Evidence: high.** [IOSurface](https://developer.apple.com/documentation/iosurface).

**ISA version.** A target identifier stored in observed HWX load-command data. Its numeric value is not monotonic across H11 through H18. **Evidence: medium.** [hwx_parsing.m](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m).

**Kernel DMA.** The reconstructed task block that configures kernel or constant-data transfers. **Evidence: medium.** [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md).

**L2 block.** A reconstructed register block associated with shared-buffer and transfer configuration. It is not enough evidence to assign a physical cache capacity. **Evidence: medium.** [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md).

**MIL.** Model Intermediate Language, the textual intermediate representation accepted by observed private in-memory compilation paths and used as this compiler's input language. **Evidence: high for repository behavior; medium for the private path.** [Repository README](../../README.md), [maderix bridge](https://github.com/maderix/ANE/blob/d91c9845c0784dec7753048954fc6d0e8411fe29/bridge/ane_bridge.m).

**NE block.** The task register region associated with neural-engine arithmetic configuration. **Evidence: medium.** [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md).

**Oracle.** An independently generated output used for differential comparison. In this project, oracle bytes are research inputs and are not stored in the repository; decoded fields and hashes may be retained. **Evidence: high.** [Disclaimer](../../DISCLAIMER.md), [provenance](../../README.md#provenance).

**Oracle parity.** The repository's proof method: emit a source-native object and compare every decoded task word, descriptor field, and constant-section hash with a decoded Apple oracle for the same input. It is a compiler-agreement claim, not device execution. **Evidence: high as a repeatable test.** [Parity method](parity-method.md).

**Oracle envelope.** The set of geometries Apple's compiler accepted, together with its refusals, measured by the `env_*` campaign. A covered grid point is not a covered neighborhood: M=128 with K=4096 compiles and M=128 with K=8192 is refused. **Evidence: high.** [oracle-envelope.md](../../research/oracle-envelope.md).

**PE, planar engine.** A multi-mode processing block used for elementwise, pooling, reduction, and related data-plane operations in Apple's patent description and reverse-engineered maps. Direct field meanings remain generation-specific. **Evidence: high for the patent concept; medium for map association.** [US20210103803A1](https://patents.google.com/patent/US20210103803A1/en), [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md).

**Physical layout.** The padded byte representation submitted to hardware, including row, plane, batch, tile, and allocation strides. It can be larger than the logical tensor. **Evidence: medium.** [H13 field ledger](../../research/h13-hwx-fields.md).

**Scatter record.** An H14 register-record form with bit 31 set and a 16-bit mask in bits 30:15 selecting which following words are written. H13 has no equivalent; it uses dense records only. **Evidence: high for the decoded corpus.** [h14-td-fields.md](../../research/h14-td-fields.md).

**Task descriptor.** A header plus register-write records that configures one observed ANE task. It is not a host CPU instruction stream. **Evidence: medium.** [Task descriptor documentation](task-descriptors.md).

**Tile DMA.** Reconstructed source and destination blocks that describe tensor transfer shapes, strides, and addresses. **Evidence: medium.** [H13 register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h13_register_map.md).

**TOPS.** Trillion operations per second. A useful peak claim only when precision, operation counting, sparsity, utilization, and power conditions are stated. Apple's cited launch pages do not state that full contract. **Evidence: high for the literal unit and source omission; open question for Apple's detailed method.** [M4 announcement](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/).
