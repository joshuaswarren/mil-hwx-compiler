# Generic ANE Program Coalescing Implementation Plan

> Execute this plan in the main checkout on `optimize/generic-ane-runtime`. Keep all changes local until the benchmark review.

**Goal:** Reduce ANE dispatch overhead through target-neutral graph partitioning and table-driven H16G task composition.

**Architecture:** Insert a program-partition planner between scheduling and H16G assembly. The planner uses graph dependencies and target capability rows. The assembler consumes partitions, emits supported tiled post-operations, validates the resulting HWX, and preserves separate artifacts on decline.

**Technology:** Objective-C++, command-line tests, decoded H16G task descriptors, M4 ANE runtime benchmarks.

---

### Task 1: Record the current performance and artifact evidence

**Files:**
- Add: `docs/evidence/2026-09-02-generic-program-coalescing-baseline.md`

1. Record exact commits, host, OS, ANECompiler version, shapes, dtype, warmup and iteration counts.
2. Record runtime sample attribution and decoded marker counts for Apple and research artifacts.
3. Check that the evidence document contains no proprietary artifact bytes or private cache paths.
4. Commit the design, plan, and evidence document.

### Task 2: Add target-owned composition capabilities

**Files:**
- Modify: `plugins/H16G/H16GTarget.h`
- Modify: `plugins/H16G/H16GTarget.mm`
- Modify: `tests/test_structured_td_encoding.mm`

1. Add failing tests for the supported tiled post-operation and an unknown transition.
2. Add a named composition action and immutable capability-row type.
3. Put all H16G limits and transition rows in the target table.
4. Run the focused test and confirm it passes.

### Task 3: Plan generic program partitions

**Files:**
- Add: `lib/Planning/ANEProgramPartition.h`
- Add: `lib/Planning/ANEProgramPartition.mm`
- Add: `tests/test_program_partition.mm`
- Modify: `Makefile`

1. Add failing tests for a linear chain, fan-out boundary, unknown transition, external-input limit, task limit, SRAM limit, and deterministic partitions.
2. Implement partition boundary inputs and outputs from scheduled graph values.
3. Implement deterministic greedy grouping using target capability rows and measured limits.
4. Confirm unsupported graphs produce the same one-task partitions as before.

### Task 4: Expose composable encoded-task metadata

**Files:**
- Modify: `plugins/H16G/Encoding/H16GEncodedTask.h`
- Modify: `plugins/H16G/Encoding/H16GEncodedTask.mm`
- Modify: `plugins/H16G/Encoding/H16GTaskEncoder.mm`
- Modify: `tests/test_structured_td_encoding.mm`

1. Add failing tests for packet family, resources, relocations, bridge eligibility, and standalone byte stability.
2. Add metadata derived from encoded task contents, not graph names.
3. Keep standalone program output byte-identical.

### Task 5: Assemble and validate multi-task programs

**Files:**
- Modify: `plugins/H16G/Encoding/H16GProgramAssembler.h`
- Modify: `plugins/H16G/Encoding/H16GProgramAssembler.mm`
- Add: `tests/test_program_composition.mm`
- Modify: `Makefile`

1. Add failing tests for tiled-post-operation composition.
2. Add tests for partition-boundary bindings, task markers, deterministic bytes, parser round-trip, and invalid-composition decline.
3. Implement generic assembly from `ANEProgramPartition` and capability rows.
4. Parse and validate each combined image before accepting it.

### Task 6: Integrate the planner and remove the workload shortcut

**Files:**
- Modify: `lib/Driver/ANEStagedCompiler.mm`
- Modify: `plugins/H16G/Encoding/H16GProgramAssembler.mm`
- Modify: `tests/test_compiler_e2e.mm`
- Modify: `tests/test_no_pattern_shortcuts.sh`

1. Add failing end-to-end tests for reduced partitions on two unrelated graph shapes and unchanged decline behavior.
2. Route scheduled graphs through the partition planner.
3. Delete `assembleSingleTileOnlineReduction` and any graph-name or fixed-sequence selection.
4. Strengthen the shortcut guard against workload names and fixed operation sequences in planner and assembler code.
5. Run all host tests.

### Task 7: Validate and profile on M4

**Files:**
- Modify if needed: `tests/hardware/benchmark_compiler_ab.mm`
- Modify if needed: `tests/run_compiler_ab_hardware.sh`
- Modify: `README.md`
- Update: `docs/evidence/2026-09-02-generic-program-coalescing-baseline.md`

1. Build the reviewed source set on M4 and record its base commit plus implementation patch fingerprint.
2. Run exact parity for the eight-program FA2 path, one-program matmul-GELU, a non-attention chain, and affine scan partition or decline behavior.
3. Record emitted artifact counts and actual runtime submission counts.
4. Run 20 warmups and two independent alternating A/B runs for each benchmark.
5. Profile any remaining gap by compiler stage and runtime submission.
6. Capture `powermetrics` evidence while the research output runs on ANE.
7. Run the full host suite and repeat the key M4 correctness cases.
8. Update the README table only with measured results.
9. Present the numbers and remaining gap to the user. Do not push.
