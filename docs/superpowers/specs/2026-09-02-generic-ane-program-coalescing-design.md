# Generic ANE Program Coalescing

## Goal

Reduce ANE submissions by combining compatible scheduled operations into one HWX program. The decision must depend on decoded task capabilities, graph dependencies, storage, and measured hardware limits. It must not depend on workload names or recognized whole-graph patterns.

## Measured problem

The current research compiler emits eight HWX artifacts for FP16 flash attention and two for matmul followed by GELU. Warm M4 measurements put them at 1113.44 us and 225.04 us. Apple's compiler emits one HWX for each graph and measures 132.21 us and 105.64 us.

A five-second sample of the research path found 96.8 percent of FA2 samples and 97.6 percent of matmul-GELU samples inside Apple's synchronous direct-evaluation call. Less than 0.5 percent was spent preparing bindings and requests in our runtime. Caching request objects is therefore not the first optimization.

Decoded Apple artifacts also rule out simple file concatenation. Apple's FA2 program contains three transfer task endings and two tile starts. The research path exposes more intermediate operations across eight files. Apple's matmul-GELU program carries the activation within the tiled program instead of appending the standalone activation artifact.

## Compiler structure

The scheduler continues to produce `ANEScheduledGraph`. Task encoders continue to describe individual executable operations. A new program-planning step groups adjacent tasks when all of the following hold:

1. Their dependency edge is direct, and every consumer of a value retained internally is inside the partition.
2. The producer output and consumer input can use a supported tiled post-operation form.
3. The combined external inputs, outputs, task count, descriptor count, and SRAM use fit target facts.
4. A table-driven H16G composition capability describes the exact packet-family transition.
5. The combined program can be validated after assembly.

If any check fails, the planner ends the current partition and preserves the existing multi-artifact path.

## Neutral representation

`ANEProgramPartition` contains an ordered list of scheduled task indices and the graph values that remain external at its boundary. It contains no operation-name shortcut.

`H16GProgramCompositionCapability` describes a producer packet family, consumer packet family, permitted bridge storage, hardware-task contribution, and supported composition action. The implemented action adds a measured tiled post-operation to a matmul program. Unknown transitions decline.

Task capability rows also record standalone and composed task counts and semantic requirements. The assembler checks transpose flags and operand order before selecting the matmul row. The target owns the conservative input, task, SRAM, and descriptor bounds. Composition declines when a required bound is unavailable.

## Assembly

Standalone task assembly remains byte-stable. Multi-task assembly uses the partition and capability rows to:

- retain only partition-boundary buffers as external bindings;
- keep the producer result inside a supported tiled program;
- add the consumer as a decoded post-operation;
- produce one valid HWX image with deterministic ordering.

The result is parsed again before it is accepted. Parsing checks section, symbol-table, and relocation ranges and section ordinals. The writer rejects unsupported surface ranks, arithmetic overflow, excess resources, and relocation addends outside the kernel table. Invalid or unsupported compositions fall back to separate artifacts.

The previous `assembleSingleTileOnlineReduction` path is removed. The assembler now receives partitions and task capabilities for every scheduled graph. If composition declines, it first tries the original standalone task encoders. A task containing multiple stages can be lowered to standalone primitives. This includes the general algebraic form `mul(x, reciprocal(y))`, which uses the decoded `real_div(x, y)` packet pair. The fallback is assembled and parsed through the same object path.

## Partition choice

M4 profiling showed that synchronous ANE evaluation dominated the original multi-artifact runtime. The implementation therefore uses a deterministic longest-valid-prefix partitioner over the scheduled graph. It joins tasks only when the target has measured packet and transition rows for the full candidate. Candidates that exceed input, task, SRAM, descriptor, or surface limits are rejected. The planner does not estimate unsupported task costs or insert spills.

## Validation

Host tests cover the tiled post-operation, unsupported transitions, fan-out, external-input limits, hardware-task limits, SRAM limits, deterministic output, and object bounds. Generated programs round-trip through the HWX parser. FA2 remains on the eight-program primitive path because the corresponding internal packet stream has not been decoded into field-level encoders suitable for this release.

M4 validation must include:

- numerical comparison for FP16 FA2 and matmul-GELU;
- one-program matmul-GELU composition and eight-program FA2 execution;
- the eight-program affine state scan path;
- artifact and ANE submission counts;
- 20 warmups followed by two independent alternating A/B runs;
- power evidence from `powermetrics` during the compiled path.

The matmul-GELU performance goal is at most 1.25 times Apple's latency. FA2 is reported with its existing eight-program latency. Missing a target does not justify a workload-specific emitter.

No push or publication happens until the user reviews the new numbers.
