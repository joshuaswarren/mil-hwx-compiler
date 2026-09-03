# H16G Composed Lowerings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add FP16 FlashAttention-2, chunked affine-scan DeltaNet, and matmul followed by GELU through one scheduled primitive path.

**Architecture:** Extend the existing scheduled graph with primitive stages, carried surfaces, tile iteration shapes, and dependency waves. Replace exact mixed-graph sequence emission with capability-selected task encoders and a program assembler. Keep compound names in frontend decomposition and structural planning only.

**Tech Stack:** Objective-C++, Foundation, Make, MIL fixtures, H16G field encoders, Apple M4 hardware tests.

**Spec:** `docs/COMPOSED_LOWERING_DESIGN.md`

## Global Constraints

- FP16 is the only numeric mode added by this plan.
- FlashAttention-2 covers forward inference, causal and noncausal masks.
- Dropout and backward propagation are rejected.
- Existing primitive routes remain byte-identical until their replacement gate passes.
- New target facts are table rows with explicit rejection for unknown forms.
- No workload name may reach an H16G encoder.
- No opaque HWX image or complete descriptor row may be added.
- Each production behavior begins with a failing test.
- Floating-point hardware gates use relative L2 and maximum absolute error.
- Repository history is rewritten only after implementation and verification.

---

### Task 1: Scheduled stages, waves, and carried surfaces

**Files:**
- Modify: `lib/IR/ANEScheduledGraph.h`
- Modify: `lib/IR/ANEScheduledGraph.mm`
- Modify: `lib/Planning/ANETaskScheduler.mm`
- Modify: `lib/Planning/ANEMemoryPlanner.mm`
- Modify: `tests/test_hwx_planning.mm`

**Interfaces:**
- Produces `ANEScheduledStage`, `ANEScheduledBridgeStorage`, and `ANEScheduledTopology`.
- Extends `ANEScheduledTask` with `stages` and `waveIndex`.
- Extends `ANEScheduledSurfaceRole` with `ANEScheduledSurfaceRoleCarry`.
- Extends `ANETilePlan` with `iterationShape` and `tileShape`.

- [ ] Add a failing planner test for two tasks in the same wave and a dependent task in the next wave.
- [ ] Run `make build/test_hwx_planning` on the M4 and confirm the missing stage API causes the expected compile failure.
- [ ] Add a failing lifetime test for a carried surface spanning every iteration.
- [ ] Implement stage construction from each task group and compute waves from dependencies.
- [ ] Extend memory planning so carried surfaces cannot be reused before their final iteration.
- [ ] Run the focused planning test and the existing suite.

### Task 2: Structural composed-region plans

**Files:**
- Create: `lib/Planning/ANEComposedRegionPlanner.h`
- Create: `lib/Planning/ANEComposedRegionPlanner.mm`
- Modify: `lib/Planning/ANETaskScheduler.mm`
- Modify: `Makefile`
- Modify: `tests/test_hwx_planning.mm`
- Create: `tests/fixtures/matmul_gelu_128.mil`
- Create: `tests/fixtures/matmul_gelu_256.mil`

**Interfaces:**
- Produces `ANEScheduledTopologyOnlineReduction` for legal QK, scale, mask, softmax, PV regions.
- Produces `ANEScheduledTopologyAssociativeScan` for legal affine transitions.
- Leaves ordinary connected operations as `ANEScheduledTopologyDirect`.
- Returns a diagnostic reason for every rejected candidate.

- [ ] Add failing tests showing both matmul-GELU fixtures form the same two-stage direct task region.
- [ ] Add failing structural tests for online attention and affine scan without function-name checks.
- [ ] Add negative tests for fanout, unsupported dtype, dropout, malformed mask, and unsafe state aliasing.
- [ ] Implement topology selection from operation kinds, edges, shapes, and attributes.
- [ ] Assert that planner output contains no original function name.
- [ ] Run the focused tests and the existing suite.

### Task 3: H16G capability rows and encoded task fragments

**Files:**
- Modify: `plugins/H16G/H16GTarget.h`
- Modify: `plugins/H16G/H16GTarget.mm`
- Create: `plugins/H16G/Encoding/H16GEncodedTask.h`
- Create: `plugins/H16G/Encoding/H16GEncodedTask.mm`
- Create: `plugins/H16G/Encoding/H16GTaskEncoder.h`
- Create: `plugins/H16G/Encoding/H16GTaskEncoder.mm`
- Modify: `plugins/H16G/Encoding/H16GMatmulEncoder.mm`
- Modify: `plugins/H16G/Encoding/H16GALUEncoder.mm`
- Modify: `plugins/H16G/Encoding/H16GLUTEncoder.mm`
- Modify: `plugins/H16G/Encoding/H16GReduceEncoder.mm`
- Modify: `tests/test_structured_td_encoding.mm`
- Modify: `Makefile`

**Interfaces:**
- `H16GTarget` answers a capability row query using stage sequence, FP16 mode, geometry, bridge storage, and mask mode.
- `H16GTaskEncoder` returns `H16GEncodedTask` with TD bytes, relocations, constants, bindings, task count, record count, format code, and scratch bytes.

- [ ] Add failing table tests for two matmul-GELU geometries and two reduction geometries.
- [ ] Add failing rejection tests for unknown stage sequences and nearby unsupported geometry.
- [ ] Adapt existing primitive encoders to return task fragments through one interface.
- [ ] Preserve byte-identical TD output for every existing standalone primitive case.
- [ ] Run structured encoding tests and compare existing hashes.

### Task 3A: Decode missing composed primitive forms

**Files:**
- Create: `research/mint_composed_oracles.m`
- Create: `research/analyze_composed_oracles.py`
- Modify: `plugins/H16G/H16GTarget.mm`
- Modify: `plugins/H16G/Encoding/H16GTaskEncoder.mm`
- Modify: `tests/test_structured_td_encoding.mm`

**Interfaces:**
- The research mint writes controlled Apple-compiler outputs under a caller-provided temporary directory.
- The analyzer reports changed task fields for one-factor geometry and operation sweeps.
- Production capability rows contain only decoded field values and provenance labels.

- [ ] Mint controlled N64 and N128 matmul, row broadcast ALU, row reduction, causal mask, and carried-state update cases on the M4.
- [ ] Change one semantic or geometry variable per pair and identify every affected TD field.
- [ ] Add failing field-level tests with hand-recorded expected values for two geometries per admitted form.
- [ ] Implement the minimum field encoders needed by the planned lowerings.
- [ ] Build compiler-created objects from zeroed buffers and execute each new primitive form twice on the M4.
- [ ] Confirm no minted HWX object, complete TD row, or temporary output is tracked by Git.

### Task 4: Program assembly and bundle dispatch

**Files:**
- Create: `plugins/H16G/Encoding/H16GProgramAssembler.h`
- Create: `plugins/H16G/Encoding/H16GProgramAssembler.mm`
- Modify: `plugins/H16G/Encoding/H16GProgramEncoder.mm`
- Modify: `lib/Driver/ANEStagedCompiler.mm`
- Modify: `lib/Runtime/ANEExecutableBundle.h`
- Modify: `lib/Runtime/ANEExecutableBundle.mm`
- Modify: `lib/Runtime/ANEProvisionedRuntime.h`
- Modify: `lib/Runtime/ANEProvisionedRuntime.mm`
- Modify: `tests/test_compiler_e2e.mm`
- Modify: `tests/test_runtime_contract.mm`
- Modify: `tests/test_no_pattern_shortcuts.sh`
- Modify: `Makefile`

**Interfaces:**
- `H16GProgramAssembler` consumes an operation graph and scheduled graph and returns all emitted artifacts plus their dispatch order.
- `ANEExecutableBundle` records shared surface identifiers across artifacts.
- `ANEProvisionedRuntime` loads the artifact identities, allocates one IOSurface per shared identifier, and executes the dispatch plan.

- [ ] Add a failing compiler test where matmul and GELU produce a deterministic two-artifact dispatch plan.
- [ ] Add a failing runtime test proving the intermediate IOSurface is shared by identifier.
- [ ] Add malformed-plan tests for missing producers, repeated writes, cycles, and incompatible surface layouts.
- [ ] Implement task-fragment assembly and multi-artifact bundle serialization.
- [ ] Implement runtime loading and ordered evaluation for shared surfaces.
- [ ] Remove whole-graph task-count dispatch from `H16GProgramEncoder`.
- [ ] Run bundle, runtime, guard, and existing regression tests.

### Task 5: FP16 online attention plan

**Files:**
- Modify: `lib/Planning/ANEComposedRegionPlanner.mm`
- Modify: `lib/Planning/ANETaskScheduler.mm`
- Modify: `plugins/H16G/H16GTarget.mm`
- Modify: `plugins/H16G/Encoding/H16GTaskEncoder.mm`
- Create: `tests/fixtures/fa2_fp16_s128_d128.mil`
- Create: `tests/fixtures/fa2_fp16_s256_d128.mil`
- Modify: `tests/test_hwx_planning.mm`
- Modify: `tests/test_staged_conv_compiler.mm`

**Interfaces:**
- Produces query-tile and key/value-tile iterations with carried row maximum, row sum, and output accumulator surfaces.
- Admits causal or noncausal masks through a capability row.
- Emits no full score surface for sequence lengths larger than one key tile.

- [ ] Add a failing S128 plan test for one key tile.
- [ ] Add a failing S256 plan test for two key tiles and carried online state.
- [ ] Add failing tests for FP32, dropout, backward use, malformed scale, and unsupported dimensions.
- [ ] Implement online maximum, rescale, sum, and accumulator update stages.
- [ ] Assert route count, tile count, waves, carried lifetimes, and absence of a full score allocation.
- [ ] Compile both fixtures twice and check deterministic bundle bytes.

### Task 6: Generic affine scan and chunked gated delta lowering

**Files:**
- Modify: `lib/IR/ANEOperationGraph.mm`
- Modify: `lib/Transform/ANEDecomposePass.mm`
- Modify: `lib/Planning/ANEComposedRegionPlanner.mm`
- Modify: `lib/Planning/ANETaskScheduler.mm`
- Modify: `plugins/H16G/H16GTarget.mm`
- Modify: `plugins/H16G/Encoding/H16GTaskEncoder.mm`
- Create: `tests/fixtures/affine_scan_vector.mil`
- Create: `tests/fixtures/chunked_gated_delta_fp16.mil`
- Modify: `tests/test_graph_transforms.mm`
- Modify: `tests/test_hwx_planning.mm`
- Modify: `tests/test_staged_conv_compiler.mm`

**Interfaces:**
- Decomposes `gated_delta_rule` into primitive affine transition stages.
- Uses `ANEScheduledTopologyAssociativeScan` for both vector and matrix-state fixtures.
- Records transition composition waves and final carried state.

- [ ] Add a failing decomposition test with hand-derived primitive edges.
- [ ] Add a failing vector affine scan test with two composition levels.
- [ ] Add a failing matrix-state gated delta test with exact dependency waves.
- [ ] Add rejection tests for state aliasing, non-FP16 inputs, non-associative update order, and unsupported chunk geometry.
- [ ] Implement semantic decomposition, pair composition, and state carry.
- [ ] Compile both fixtures twice and check deterministic bundle bytes.

### Task 7: M4 numerical execution gates

**Files:**
- Create: `tests/hardware/run_matmul_gelu.mm`
- Create: `tests/hardware/run_fa2.mm`
- Create: `tests/hardware/run_chunked_delta.mm`
- Create: `tests/hardware/prepare_composed.mm`
- Create: `tests/run_composed_hardware.sh`
- Modify: `Makefile`
- Modify: `docs/VERIFICATION.md`

**Interfaces:**
- Each runner reads the emitted bundle and reports route, finite status, relative L2, maximum absolute error, and evaluation count.
- References use FP32 or FP64 host calculations with the same recurrence and mask conventions.

- [ ] Run matmul-GELU at N128 and N256 against an independent GELU reference.
- [ ] Run noncausal and causal FP16 attention at S128 and S256.
- [ ] Run vector affine scan and matrix-state chunked gated delta.
- [ ] Execute every case twice and record numerical metrics.
- [ ] Run all prior H16G hardware sweeps to catch primitive regressions.
- [ ] Update verification receipts with exact commands and results.

### Task 8: Remove specialization, verify release, and rewrite history

**Files:**
- Delete: `plugins/H16G/Encoding/H16GMixedTaskEncoder.h`
- Delete: `plugins/H16G/Encoding/H16GMixedTaskEncoder.mm`
- Modify: `Makefile`
- Modify: `tests/test_no_pattern_shortcuts.sh`
- Modify: `tests/test_release_hygiene.sh`
- Modify: `README.md`
- Modify: `docs/VERIFICATION.md`

**Interfaces:**
- Production emission contains no exact mixed-attention task sequence matcher.
- The final repository has one clean reachable history rooted at the tested source tree.

- [ ] Add a failing guard for graph-name checks, exact source-node sequences, and fixed mixed-task stream lengths in H16G emission.
- [ ] Remove `H16GMixedTaskEncoder` after replacement gates pass.
- [ ] Run `tests/run_all.sh` and all available M4 hardware scripts.
- [ ] Confirm the worktree contains only intended files and no generated HWX objects.
- [ ] Create one new root commit from the final tree.
- [ ] Verify removed HWX paths and blob identities are absent from `git rev-list --objects --all`.
- [ ] Expire local reflogs and prune unreachable objects after verifying the new root.
- [ ] Perform exactly one `git push --force-with-lease origin main`.
