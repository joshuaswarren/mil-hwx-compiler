# Generic Staged MIL-to-HWX Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace whole-graph dispatch and descriptor patching with one generic graph-to-scheduled-HWX pipeline for the decoded H16G primitive surface.

**Architecture:** Preserve the MIL frontend and bundle/runtime. Transform the typed SSA graph through six passes into a scheduled HWX graph, then emit task descriptors and the object container from zeroed buffers through primitive field encoders. The three existing workloads are regression oracles, never dispatch keys.

**Tech Stack:** Objective-C++, Foundation, Make, command-line unit tests, M4 H16G hardware verification.

**Spec:** `docs/GENERIC_STAGED_HWX_COMPILER_DESIGN.md`

## Global Constraints

- No workload-named IR kinds, passes or emitters.
- No exact operation-count recognition.
- No recovered TD-row insertion.
- No HWX skeleton patching.
- H16G numbers live in one target-fact table.
- Unsupported operations and shapes fail closed.
- Every behavior begins with a failing test.
- Existing v1 files and reference artifacts are not deleted.

---

### Task 1: Graph-shaped primitive annotations and pass dumps

**Files:**
- Modify: `lib/IR/ANEGraphIR.h`
- Modify: `lib/IR/ANEGraphIR.mm`
- Create: `lib/IR/ANEOperationGraph.h`
- Create: `lib/IR/ANEOperationGraph.mm`
- Create: `lib/Transform/ANEPassDump.h`
- Create: `lib/Transform/ANEPassDump.mm`
- Create: `tests/test_operation_graph.mm`

**Interfaces:**
- Produces `ANEOperationGraph`, `ANEOperationNode`, `ANEGraphRegion` and deterministic textual dumps.
- Consumes the existing `ANEGraphFunction` without changing the MIL parser contract.

- [ ] Add failing tests proving arbitrary producer-consumer topology imports without a workload enum.
- [ ] Run `make build/test_operation_graph` and confirm failure because the graph API is absent.
- [ ] Implement graph nodes, use lists, regions and deterministic dumps.
- [ ] Run the focused test and full local suite.

### Task 2: Normalize and decompose passes

**Files:**
- Create: `lib/Transform/ANENormalizePass.h`
- Create: `lib/Transform/ANENormalizePass.mm`
- Create: `lib/Transform/ANEDecomposePass.h`
- Create: `lib/Transform/ANEDecomposePass.mm`
- Create: `tests/test_graph_transforms.mm`

**Interfaces:**
- Consumes and mutates `ANEOperationGraph`.
- Produces canonical primitive nodes drawn from Conv, Matmul, ALU, LUT, Reduce and Layout.

- [ ] Add failing normalize tests for identity reshape, inverse transpose, dead operations and equivalent scalar spellings.
- [ ] Add failing decomposition tests for softmax and layer normalization.
- [ ] Verify the failures identify missing transformations rather than fixture errors.
- [ ] Implement normalization and primitive decomposition.
- [ ] Assert stable before/after pass dumps and run the full suite.

### Task 3: Structural fusion

**Files:**
- Create: `lib/Transform/ANEFusionPass.h`
- Create: `lib/Transform/ANEFusionPass.mm`
- Create: `tests/test_structural_fusion.mm`

**Interfaces:**
- Consumes a canonical `ANEOperationGraph` and `H16GTargetFacts` capability queries.
- Produces `ANEGraphRegion` membership and explicit materialization boundaries.

- [ ] Add failing tests for Conv→ReLU, primitive attention, and quantized Conv-chain fusion without pattern names.
- [ ] Add barriers for external uses, unsupported primitives, incompatible layouts and working-set overflow.
- [ ] Implement deterministic producer-consumer region growth.
- [ ] Run focused tests and confirm all three fixture graphs are formed by the same algorithm.

### Task 4: H16G facts and legalization

**Files:**
- Replace: `plugins/H16G/H16GTarget.h`
- Replace: `plugins/H16G/H16GTarget.mm`
- Create: `lib/Transform/ANEH16GLegalizePass.h`
- Create: `lib/Transform/ANEH16GLegalizePass.mm`
- Create: `tests/test_h16g_legalization.mm`

**Interfaces:**
- Produces layouts, physical numeric modes and legal split descriptions on graph nodes.
- Exposes all hardware literals through `H16GTargetFacts` and capability rows.

- [ ] Add failing fact-provenance, unsupported-shape and no-silent-default tests.
- [ ] Add failing W8A8 boundary/packed/output propagation tests.
- [ ] Implement layout, capability and numeric legalization.
- [ ] Run focused tests and the full suite.

### Task 5: Scheduled HWX graph and deterministic planner

**Files:**
- Create: `lib/IR/ANEScheduledGraph.h`
- Create: `lib/IR/ANEScheduledGraph.mm`
- Create: `lib/Planning/ANETilePlanner.h`
- Create: `lib/Planning/ANETilePlanner.mm`
- Create: `lib/Planning/ANEMemoryPlanner.h`
- Create: `lib/Planning/ANEMemoryPlanner.mm`
- Create: `lib/Planning/ANETaskScheduler.h`
- Create: `lib/Planning/ANETaskScheduler.mm`
- Create: `tests/test_hwx_planning.mm`

**Interfaces:**
- Consumes legalized graph regions and target facts.
- Produces surfaces, allocations, DMA commands, compute commands, waits and task dependencies.

- [ ] Add failing tile-law tests using the recovered Conv and Matmul sweeps.
- [ ] Add failing liveness/reuse, 16-byte alignment and 64-bank mapping tests.
- [ ] Add failing DMA Load/Inter/Store and dependency tests for branched graphs.
- [ ] Implement deterministic tiling, first-fit reuse and conservative scheduling.
- [ ] Run focused tests and the full suite.

### Task 6: Structured TD command encoders

**Files:**
- Create: `plugins/H16G/Encoding/H16GTDWriter.h`
- Create: `plugins/H16G/Encoding/H16GTDWriter.mm`
- Create: `plugins/H16G/Encoding/H16GConvEncoder.h`
- Create: `plugins/H16G/Encoding/H16GConvEncoder.mm`
- Create: `plugins/H16G/Encoding/H16GALUEncoder.h`
- Create: `plugins/H16G/Encoding/H16GALUEncoder.mm`
- Create: `plugins/H16G/Encoding/H16GLUTEncoder.h`
- Create: `plugins/H16G/Encoding/H16GLUTEncoder.mm`
- Create: `plugins/H16G/Encoding/H16GReduceEncoder.h`
- Create: `plugins/H16G/Encoding/H16GReduceEncoder.mm`
- Create: `plugins/H16G/Encoding/H16GMatmulEncoder.h`
- Create: `plugins/H16G/Encoding/H16GMatmulEncoder.mm`
- Create: `tests/test_structured_td_encoding.mm`

**Interfaces:**
- Consumes one scheduled command at a time.
- Produces TD bytes from a zeroed buffer using named, bounds-checked fields.

- [ ] Add failing tests for exact decoded fields across two shapes per primitive.
- [ ] Add a guard that rejects descriptor-row-sized resource insertion.
- [ ] Implement the writer and primitive encoders incrementally.
- [ ] Reparse every emitted TD and run the full suite.

### Task 7: HWX object writer from empty buffers

**Files:**
- Create: `lib/HWX/HWXObjectWriter.h`
- Create: `lib/HWX/HWXObjectWriter.mm`
- Create: `tests/test_hwx_object_writer.mm`

**Interfaces:**
- Consumes encoded TDs, constants, bindings and target metadata.
- Produces a complete `0xBEEFFACE` H16G object without reading an HWX input.

- [ ] Add a failing test that succeeds with all resource HWX files unavailable.
- [ ] Add section alignment, bounds, load-command and deterministic-hash tests.
- [ ] Implement header, segment, section, symbol and content construction.
- [ ] Reparse generated objects and run the full suite.

### Task 8: Driver migration and legacy-path exclusion

**Files:**
- Modify: `lib/Driver/ANECompiler.mm`
- Modify: `Makefile`
- Create: `tests/test_no_pattern_shortcuts.sh`
- Modify: `tests/test_compiler_e2e.mm`

**Interfaces:**
- Makes the six-pass staged pipeline the only production compiler route.
- Leaves v1 sources and resources on disk as test oracles but excludes their registration and linkage.

- [ ] Add a failing layering guard for workload names, TD-row resources and skeleton reads in production sources.
- [ ] Add arbitrary supported-topology and unsupported-primitive CLI tests.
- [ ] Route the driver through normalize, decompose, fuse, legalize, plan and emit.
- [ ] Disable legacy registrations without deleting their files.
- [ ] Run the guard, focused tests and full suite.

### Task 9: M4 hardware gates

**Files:**
- Modify: `tests/run_m4_hardware.sh`
- Modify: `docs/VERIFICATION.md`

**Interfaces:**
- Verifies Conv, Conv→ReLU, attention and W8A8 objects emitted exclusively by the staged compiler.

- [ ] Build and structurally inspect two legal variants per primitive route.
- [ ] Confirm production compilation reads no oracle HWX or TD resource.
- [ ] Execute each article workload twice on M4.
- [ ] Report element count, mismatches, max absolute error and relative L2 against independent CPU references.
- [ ] Record pass traces, emitted hashes, target identity and hardware results.
