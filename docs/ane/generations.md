# Generations, SoCs, cores, ISA values, and TOPS

Three names must remain separate:

1. Apple markets chips as A12, M1, M4, and similar product names.
2. Apple device trees and drivers use SoC and interface names such as T8103 and H13ANE.
3. HWX files contain a Mach-O CPU subtype and an ANE ISA version.

A shared row does not prove that two products have identical clocks, memory systems, or enabled operators.

## Reconstructed generation table

| HWX generation | Product families associated by the cited source | Mach-O CPU subtype | ANE ISA | Public core count | Public peak claim | Evidence |
|---|---|---:|---:|---:|---:|---|
| H11 | A12 | 1 | 5 | 8 | 5 trillion operations/s | **Medium** for the H11 mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for A12 cores and peak from [Apple](https://www.apple.com/newsroom/2018/09/iphone-xs-and-iphone-xs-max-bring-the-best-and-biggest-displays-to-iphone/). |
| H12 | A13 | 3 | 6 | 8 | Not stated in the cited launch source | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for the 8-core claim from [Apple](https://www.apple.com/newsroom/2019/09/iphone-11-pro-and-iphone-11-pro-max-the-most-powerful-and-advanced-smartphones/). |
| H13 | A14 and M1 | 4 | 7 | 16 | 11 trillion operations/s | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for cores and peak from Apple's [A14](https://www.apple.com/newsroom/2020/09/apple-unveils-all-new-ipad-air-with-a14-bionic-apples-most-advanced-chip/) and [M1](https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/) announcements. |
| H14 | A15 and M2 | 5 | 11 | 16 | 15.8 trillion operations/s | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for cores and peak from Apple's [A15](https://www.apple.com/newsroom/2021/09/apple-introduces-iphone-13-and-iphone-13-mini/) and [M2](https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/) announcements. |
| H15 | A16 and M3 | 6 | 8 | 16 | Nearly 17 trillion operations/s for A16; no comparable number in the cited M3 launch | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for the A16 value from [Apple](https://www.apple.com/newsroom/2022/09/apple-debuts-iphone-14-pro-and-iphone-14-pro-max/). Apple's [M3 announcement](https://www.apple.com/newsroom/2023/10/apple-unveils-m3-m3-pro-and-m3-max-the-most-advanced-chips-for-a-personal-computer/) describes a faster Neural Engine but gives no directly comparable peak. |
| H16 | A17 Pro and M4 | 7 | 17 | 16 for M4 | 38 trillion operations/s for M4 | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for M4 cores and peak from [Apple](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/). The [A17 Pro announcement](https://www.apple.com/newsroom/2023/09/apple-unveils-iphone-15-pro-and-iphone-15-pro-max/) gives a relative speed claim, not a comparable TOPS value. |
| H17 | A18/A18 Pro and M5 | 9 | 19 | 16 | Not stated as a comparable TOPS value in the cited Apple sources | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for 16 cores from Apple's [A18](https://www.apple.com/newsroom/2024/09/apple-introduces-iphone-16-and-iphone-16-plus/) and [M5](https://www.apple.com/newsroom/2025/10/apple-unleashes-m5-the-next-big-leap-in-ai-performance-for-apple-silicon/) announcements. |
| H18 | A19/A19 Pro | 10 | 20 | 16 | Not stated as a comparable TOPS value in the cited Apple sources | **Medium** for mapping and values from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m); **high** for the A19 family and 16-core Neural Engine from Apple's [A19](https://www.apple.com/newsroom/2025/09/apple-debuts-iphone-17/) and [A19 Pro](https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/) announcements. |

The numeric subtype sequence has gaps and the ISA numbers are not monotonic. Treat both fields as identifiers selected by the compiler, not as generation counters. **Evidence: medium.** The parser cases above decode exactly the listed values; they do not define why Apple assigned them.

## One Mac cross-compiles five generations

The subtype values above are not only parser table entries. On an M1 Ultra running macOS 26.6.2 (build 25G83), one `add [1,64,1,1]` fp16 MIL program compiled through the private compile entry point with `TargetArchitecture` set per request produced:

| Requested target | Exit code | HWX bytes | Mach-O CPU subtype |
|---|---:|---:|---:|
| `h13` | 0 | 49,152 | 4 |
| `h14` | 0 | 49,152 | 5 |
| `h15` | 0 | 49,152 | 6 |
| `h16` | 0 | 65,536 | 7 |
| `h17` | 0 | 65,536 | 9 |
| `h11` | 1 | — | — |

The five accepted subtypes match the parser table rows for H13, H14, H15, H16, and H17, and the `h11` request failed. So a single Mac can mint oracles for generations it does not contain, which is how the H14/M2 corpus in this repository exists without M2 hardware. Two limits: this is one host, one input, and one compiler build, and it says nothing about whether the emitted object executes on the corresponding device. **Evidence: medium for the observation; medium for the generation names, which still come from the [fixed parser revision](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/hwx_parsing.m).** See [`receipts/anecompile-cross-target.json`](../../receipts/2026-09-05-ane-community/anecompile-cross-target.json).

The H14 records minted this way behave as a coherent generation rather than as relabelled H13 output: they use different header sizes, record encodings, and block bases, and 41 of 271 decoded pairs split work into different task counts. **Evidence: high over the corpus.** See [task descriptors](task-descriptors.md) and [h14-td-fields.md](../../research/h14-td-fields.md).

## SoC identifiers

Asahi's hardware table maps the mobile and base Mac SoCs below. Apple does not publish the H-label column as an application contract. **Evidence: high for the Asahi table; medium for using these labels as HWX-family boundaries.** See the live [SoC codename table](https://asahilinux.org/docs/hw/soc/soc-codenames/) and its [versioned source](https://github.com/AsahiLinux/docs/blob/a7d0a3dd31ac6c4238b6b4994d036a753db87824/docs/hw/soc/soc-codenames.md).

| HWX family | Mobile SoC | Mac SoC |
|---|---|---|
| H11 | A12, T8020 | Not listed |
| H12 | A13, T8030 | Not listed |
| H13 | A14, T8101 | M1, T8103; M1 Pro/Max/Ultra use T6000/T6001/T6002 |
| H14 | A15, T8110 | M2, T8112; M2 Pro/Max/Ultra use T6020/T6021/T6022 |
| H15 | A16, T8120 | M3, T8122; M3 Pro/Max/Ultra use T6030/T6031/T6032 |
| H16 | A17 Pro, T8130 | M4, T8132; M4 Pro/Max use T6040/T6041 |
| H17 | A18, T8140a; A18 Pro, T8140 | M5, T8142 |
| H18 | A19 and A19 Pro, T8150 | Not listed |

The table associates an HWX family with product SoCs. It does not claim that every product in a row has the same core count, frequency, memory bandwidth, or enabled instruction set. **Evidence: medium.** The family labels come from the cited Asahi table and the independent HWX parser; neither source asserts full microarchitectural identity.

The open Linux ANE driver at revision `0dcea9976fae0b500a236a62fca69cd4d39f0809` matches only `apple,t8103-ane` and `apple,t6000-ane`. That source does not establish support for T8112, T602x, or later chips. **Evidence: high.** See [`ane_drv.c`](https://github.com/eiln/ane/blob/0dcea9976fae0b500a236a62fca69cd4d39f0809/ane/src/ane_drv.c#L649-L650).

## What TOPS means here

Apple's launch pages use “trillion operations per second” as a peak hardware claim. They do not state, in those pages, the operand precision, sparsity assumptions, operation-counting convention, model utilization, or sustained power envelope. A TOPS number therefore does not predict latency for a given model and must not be read as FP16 FLOPS. **Evidence: high for what the Apple pages state; open question for the omitted measurement contract.** Compare Apple's [M4 claim](https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/) with a source-level benchmark that reports different FP16 and INT8 rates in the same project, [maderix/ANE](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29).

The M4 benchmark in maderix/ANE reports about 18.6 FP16 trillion operations/s and 35.1 INT8 trillion operations/s for its tested kernels. These are measured software results, not Apple specifications, and depend on the benchmark's counting and utilization. **Evidence: medium.** See the pinned [benchmark repository](https://github.com/maderix/ANE/tree/d91c9845c0784dec7753048954fc6d0e8411fe29).

## Open questions

- **Open question:** Which marketing products share identical ANE implementations rather than only the same HWX generation label?
- **Open question:** What exact operation-counting and precision contract underlies each Apple peak figure?
- **Open question:** Do future compiler releases preserve the H11 through H18 subtype and ISA associations?
- **Open question:** Why does the `h11` target fail on this compiler build while `h13` through `h17` succeed — dropped support, a different target string, or an unrelated error?
- **Open question:** Do `h15` and `h17` objects from this cross-compile decode with the same block bases and record forms their parser rows predict? No H15 or H17 oracle has been decoded here.
