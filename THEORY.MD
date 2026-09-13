# Theory: Ordered MIL Results as a Semantic Boundary

## Problem

The textual MIL frontend preserved operation order, program version, and function opset, but its semantic graph collapsed every operation to one result and dropped the metadata. The result model both rejected faithful Core ML text at the grammar boundary and risked a more serious downstream error: accepting a multi-result operation while compiling only its first value. The pinned encoder exposes the loss through 24 two-result `split` operations whose outputs have distinct consumers.

## Operating Theory

An operation owns an ordered list of SSA results. Each result is an independent typed value with the same defining operation, not a tuple wrapper or synthetic aggregate. The text grammar therefore keeps the unparenthesized typed binding for one result and adds a parenthesized, comma-separated list for multiple results. The importer reserves and registers every name as one operation so duplicate names cannot partially mutate the symbol table.

Program version and function opset are semantic compatibility claims. The graph retains them, and the verifier accepts the two contracts present in this repository and the faithful input: `program(1.3)` with `ios18`, and `program(1)` with `CoreML8`. Other pairings fail explicitly.

The semantic verifier accepts the known equal-partition split form. It requires each axis and partition-count SSA declaration and its typed constant literal to match rank-zero `tensor<int32, []>`, plus a ranked tensor input, divisible positive geometry, one result per partition, matching result types, and correct producer identity. It rejects type mismatches and unknown parameter forms instead of trusting the literal alone.

## Lowering Boundary

The semantic graph can represent split without claiming that a hardware backend can encode it. H13 and H14 consume GraphIR directly, while H16G first converts it to a one-node-per-operation graph. Each path now rejects a multi-result operation with a named diagnostic before indexing a result. There is no singular result compatibility API that could hide later outputs.

This boundary keeps numerical and hardware claims honest. Adding ordered SSA results proves frontend representation and validation. It does not prove a split encoding, full H13 operation coverage, or encoder execution.

## Verified Outcome

The parser, printer, importer, verifier, operation graph, graph transforms, and structural fusion scoped tests pass. A representative split with axis 1, count 2, two `[1, 1024, 375]` results, first-to-`mul`, and second-to-`sigmoid` imports and verifies. A declared FP16 axis carrying an int32 literal now fails semantic verification. H13 and H14 reject a valid multi-result graph as unsupported rather than truncating it.

The real compiler CLI parses the raw faithful encoder and then stops at line 125 with `mil.import.unsupported-type` for `uint4`. The adapter-normalized source replaces that form with ordinary FP16 constants. The compiler imports the normalized program, validates its split metadata, and stops at the first split on line 265 with the named H13 lowering refusal. These two probes separate grammar acceptance, adapter normalization, semantic split validation, and hardware lowering. No hardware result follows from them.

## Repository Shape

The implementation lives in the isolated worktree for source commit `a0ce354cf800011a84420da4e12013eb8140b2a5`. The similarly named research mirror has different history and is not a valid base. The coordinated adapter syntax is `(type first, type second)` in protobuf output order, with downstream uses bound by the original SSA names.
