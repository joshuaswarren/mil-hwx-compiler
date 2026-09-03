# MIL-to-HWX Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Build and hardware-verify a typed, plugin-based MIL-to-HWX compiler for Conv+ReLU, composed attention and a four-layer W8A8 convolution chain.

**Architecture:** A grammar-driven MIL frontend produces typed SSA Graph IR. Named passes lower Graph IR through ANE Primitive IR and H16G Machine IR into deterministic HWX images and binding manifests. Existing Stage 1 emitters are used only through typed target adapters while structured descriptor encoders are completed.

**Tech Stack:** Objective-C, Objective-C++17, Foundation, XCTest-free command-line tests, Make, existing H16G Stage 1 library, M4 hardware runner.

**Spec:** `docs/MIL_TO_HWX_COMPILER_DESIGN.md`

## Global Constraints

- No regex-based MIL parsing.
- No invocation of ANECCompile from production compiler or runtime code.
- Every production behavior begins with a failing test.
- Exact unsupported forms fail closed with structured diagnostics.
- Hardware success means two runs and element-wise comparison with an independent CPU reference.
- Conv+ReLU, attention and W8A8 use the exact article fixtures and shapes.

---

### Task 1: Compiler shell, diagnostics and source manager

**Files:**
- Create: `Makefile`
- Create: `include/ANEDiagnostic.h`
- Create: `lib/Support/ANEDiagnostic.mm`
- Create: `tests/test_diagnostics.mm`

**Interfaces:**
- Produces `ANESourceLocation`, `ANESourceRange`, `ANEDiagnostic` and `ANEDiagnosticEngine`.

- [x] Write tests for ordered diagnostics, source ranges and error counting.
- [x] Run the test and verify the missing interface fails the build.
- [x] Implement the minimal support layer.
- [x] Run the support test and full suite.

### Task 2: MIL lexer

**Files:**
- Create: `lib/MIL/MILToken.h`
- Create: `lib/MIL/MILLexer.h`
- Create: `lib/MIL/MILLexer.mm`
- Create: `tests/test_mil_lexer.mm`

**Interfaces:**
- Consumes UTF-8 `NSData` and `ANEDiagnosticEngine`.
- Produces immutable tokens with kind, spelling and source range.

- [x] Write failing tests using fragments from all three article fixtures.
- [x] Verify failures for strings, hex floats, comments and malformed input.
- [x] Implement character-driven tokenization.
- [x] Run lexer tests and the full suite.

### Task 3: MIL syntax parser and printer

**Files:**
- Create: `lib/MIL/MILSyntax.h`
- Create: `lib/MIL/MILSyntax.mm`
- Create: `lib/MIL/MILParser.h`
- Create: `lib/MIL/MILParser.mm`
- Create: `lib/MIL/MILPrinter.h`
- Create: `lib/MIL/MILPrinter.mm`
- Create: `tests/test_mil_parser.mm`
- Create: `tests/fixtures/conv_relu.mil`
- Create: `tests/fixtures/attention.mil`
- Create: `tests/fixtures/w8a8_conv_chain.mil`

**Interfaces:**
- Produces `MILProgramSyntax` with typed syntax nodes and source ranges.

- [x] Add the exact three fixtures and failing parse tests.
- [x] Add malformed-program diagnostics and parse-print-parse tests.
- [x] Implement recursive-descent parsing for the specified MIL subset.
- [x] Run parser tests and the full suite.

### Task 4: Typed Graph IR and MIL semantic import

**Files:**
- Create: `lib/IR/ANEGraphIR.h`
- Create: `lib/IR/ANEGraphIR.mm`
- Create: `lib/IR/ANEGraphVerifier.h`
- Create: `lib/IR/ANEGraphVerifier.mm`
- Create: `lib/MIL/MILGraphImporter.h`
- Create: `lib/MIL/MILGraphImporter.mm`
- Create: `tests/test_graph_import.mm`

**Interfaces:**
- Produces SSA `ANEGraphModule`, typed values, typed operation attributes and constants.

- [x] Write failing import tests for the three fixtures.
- [x] Write negative tests for duplicate names, unknown operands and type disagreement.
- [x] Implement symbol resolution, type import and use-def construction.
- [x] Verify all graphs and run the full suite.

### Task 5: Pass manager and plugin registry

**Files:**
- Create: `include/ANEPlugin.h`
- Create: `lib/Pass/ANEPassManager.h`
- Create: `lib/Pass/ANEPassManager.mm`
- Create: `lib/Plugin/ANEPluginRegistry.h`
- Create: `lib/Plugin/ANEPluginRegistry.mm`
- Create: `tests/test_pass_manager.mm`
- Create: `tests/test_plugin_registry.mm`

**Interfaces:**
- Produces deterministic pass execution and capability-based plugin selection.

- [x] Write failing ordering, decline and ambiguity tests.
- [x] Implement stable registration and structured match results.
- [x] Add verifier hooks around each pass.
- [x] Run registry, pass and full tests.

### Task 6: Primitive IR and three semantic recognizers

**Files:**
- Create: `lib/IR/ANEPrimitiveIR.h`
- Create: `lib/IR/ANEPrimitiveIR.mm`
- Create: `plugins/H16G/H16GPatternPlugin.mm`
- Create: `tests/test_pattern_recognition.mm`

**Interfaces:**
- Produces exact `ConvRelu`, `AttentionH4S64D64` and `W8A8ConvChain4` region laws.

- [x] Write one accepted and at least two near-miss tests per pattern.
- [x] Implement Conv+ReLU recognition and verify its dataflow.
- [x] Implement attention recognition including transposes, scale and softmax axis.
- [x] Implement W8A8 recognition including scales, zero points and boundary conversions.
- [x] Run recognizer and full tests.

### Task 7: H16G target and Machine IR

**Files:**
- Create: `lib/IR/ANEMachineIR.h`
- Create: `lib/IR/ANEMachineIR.mm`
- Create: `plugins/H16G/H16GTarget.h`
- Create: `plugins/H16G/H16GTarget.mm`
- Create: `plugins/H16G/H16GLowering.mm`
- Create: `tests/test_h16g_lowering.mm`

**Interfaces:**
- Produces typed surfaces, constants, tasks, dependencies, DMA and compute commands.

- [x] Write failing lowering tests for all three primitive programs.
- [x] Add target-fact availability and fail-closed legality tests.
- [x] Implement target selection, scheduling and binding construction.
- [x] Run lowering and full tests.

### Task 8: Conv+ReLU HWX backend

**Files:**
- Create: `plugins/H16G/H16GConvReluEmitter.mm`
- Create: `lib/HWX/HWXBuilder.h`
- Create: `lib/HWX/HWXBuilder.mm`
- Create: `lib/HWX/HWXVerifier.mm`
- Create: `tests/test_conv_relu_emission.mm`
- Create: `tests/hardware/run_conv_relu.mm`

**Interfaces:**
- Produces deterministic H16G HWX and a three-surface binding manifest.

- [x] Write the failing structural golden and unsupported-shape tests.
- [x] Implement C64/S64 descriptor and weight emission from recovered grammar.
- [x] Reparse and structurally verify the emitted file.
- [x] Run twice on M4 and compare every result with CPU Conv+ReLU.

### Task 9: Attention HWX backend

**Files:**
- Create: `plugins/H16G/H16GAttentionEmitter.mm`
- Create: `tests/test_attention_emission.mm`
- Create: `tests/hardware/run_attention.mm`

**Interfaces:**
- Produces an explicit task graph and HWX image set for H4/S64/D64 attention.

- [x] Write failing task-graph, binding and structural tests.
- [x] Implement QK, scale, softmax, PV and layout task lowering.
- [x] Verify every intermediate surface contract.
- [x] Run twice on M4 and compare with stable CPU softmax attention.

### Task 10: W8A8 HWX backend

**Files:**
- Create: `plugins/H16G/H16GW8A8Emitter.mm`
- Create: `tests/test_w8a8_emission.mm`
- Create: `tests/hardware/run_w8a8.mm`

**Interfaces:**
- Produces the four-task boundary/middle/final mode sequence and packed constants.

- [x] Write failing tests for mode words, DMA configuration, scales and task count.
- [x] Implement first, middle and final descriptor forms.
- [x] Verify int8 weight packing and inter-layer activation bindings.
- [x] Run twice on M4 and compare every fp16 output with a quantization-aware CPU reference.

### Task 11: Compiler API, CLI and bundle runtime

**Files:**
- Create: `include/ANECompiler.h`
- Create: `lib/Driver/ANECompiler.mm`
- Create: `tools/mil-hwxc.mm`
- Create: `lib/Runtime/ANEExecutableBundle.mm`
- Create: `tests/test_compiler_e2e.mm`

**Interfaces:**
- Produces `ANEExecutableBundle` from MIL data and model root.

- [x] Write failing CLI and API end-to-end tests for all fixtures.
- [x] Wire frontend, passes, target, backend and bundle serialization.
- [x] Verify no production symbol imports ANECompiler.framework.
- [x] Run all local and M4 hardware tests.

### Task 12: Reproducibility and final receipts

**Files:**
- Create: `tests/run_all.sh`
- Create: `tests/run_m4_hardware.sh`
- Create: `docs/VERIFICATION.md`

- [x] Build twice and compare emitted hashes for all three fixtures.
- [x] Run all malformed and unsupported tests.
- [x] Run all three programs twice on M4 with element-wise references.
- [x] Record host, OS build, target, hashes, timings and numerical metrics.
- [x] Audit production dependencies and confirm no ANECCompile call path.

