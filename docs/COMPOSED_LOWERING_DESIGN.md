# Composed Lowerings for H16G

Status: implemented for the bounded geometries listed below
Target: Apple M4 H16G
Numeric mode: FP16

## Scope

This change adds three compiler paths:

- matmul followed by GELU;
- FlashAttention-2 forward inference;
- a chunked gated DeltaNet graph expressed with ordinary primitive operations.

The compiler continues to accept a typed MIL graph. The frontend may expand a
compound semantic operation, and the planner may recognize a legal connected
region. H16G emission receives scheduled primitive stages. It does not receive
the original workload name.

The implemented attention path covers unmasked FP16 forward inference at
S128/D128. Dropout, backward propagation, causal masking, and multi-tile
emission are outside the current hardware path. The planner records larger
tiled forms, but the assembler rejects them until the required tile-copy and
carried-state packets have been decoded.

## Previous limitation

The earlier implementation emitted one HWX object for every primitive in the
S128/D128 attention graph. Eight synchronous submissions made host and daemon
latency dominate the small fixture. The compiler also lacked reusable task
forms for values passed through SRAM within one program.

## Scheduled representation

Each scheduled task contains an ordered list of `ANEScheduledStage`
objects. A stage records:

- primitive operation kind and semantic operation;
- input and output surface identifiers;
- tile coordinates and tile shape;
- bridge storage class;
- carried input and output surfaces;
- dependency wave;
- whether the stage materializes an external result.

`ANEScheduledSurface` has a carried-state role. Carried surfaces remain live
across loop iterations and dispatches. The same role is used by the online
softmax state and the affine scan state.

`ANETilePlan` records iteration and tile shapes. Existing direct, row-tiled, and
layout plans continue to use their current fields. Composed plans use the same
type with an online-reduction or associative-scan topology.

## Target capability rows

H16G support is described by rows keyed by:

- primitive stage sequence;
- numeric mode;
- input and output geometry;
- tile geometry;
- bridge storage;
- mask mode when a mask is part of the semantic operation.

A row provides the packet family, field values, scratch requirements, and
program limits. Unknown combinations are rejected. Planner and emitter code do
not contain fallback values for unknown rows.

## Program assembly

Primitive task encoders return an `H16GEncodedTask` containing task descriptor
bytes, relocation records, constant data, binding requirements, and scratch
requirements. `H16GProgramAssembler` combines encoded tasks according to the
scheduled dependency order.

When the decoded packet grammar cannot place several tasks in one HWX object,
the assembler emits several compiler-created artifacts and an explicit dispatch
plan. Every artifact still comes from field-level encoders. No compiler-minted
HWX image or descriptor row is stored in the repository.

The production path uses the scheduled primitive assembler. It has no
attention-specific program encoder.

## Matmul followed by GELU

The operation graph contains an ordinary matmul node followed by an ordinary
GELU node. Fusion forms one region. Scheduling retains both stages and the
intermediate surface. The assembler chooses one program when the capability
table has a legal internal bridge. Otherwise it emits two ordered artifacts
whose shared surface is explicit in the bundle.

No GELU-specific matmul emitter is added.

## FlashAttention-2

The planner recognizes the connected primitive graph:

```text
Q times transpose(K)
  -> scale
  -> optional causal mask
  -> row softmax
  -> times V
```

It then creates an online-reduction plan. The current emitted geometry contains
one 128x128 query tile and one 128x128 key/value tile. The compiler emits three
programs. The first contains matmul and scale. The second contains row maximum,
subtraction, exp, row sum, reciprocal, and multiply. The final program contains
the output matmul. Larger plans record carried maximum, sum, and output surfaces
but are not emitted yet.

The middle program is assembled from task forms selected by operation, tensor
geometry, input bridge, output bridge, and retained-value requirements. Each
form declares the SRAM state it consumes and produces. The same mechanism is
tested on smaller non-attention chains, including subtract then exp, exp then
reduce-sum, reduce-sum then reciprocal, and reciprocal then multiply.

The operation stages are matmul, ALU, LUT, reduction, and carried-state update.
The H16G emitter only sees those stages and their scheduled geometry.

## Chunked DeltaNet

The test fixture supplies a complete C128/D128 block as ordinary matrix and
elementwise operations. The planner partitions that graph through the same
operation, geometry, dependency, and lifetime rules used for other graphs.

The backend receives matmul, reduction, ALU, layout, and state-copy stages. It
does not receive a DeltaNet packet kind.

The affine-scan gate separately uses four N128 FP16 matrix-state transitions of
the form `state = state * factor + update`. Each multiply and add is emitted as
a normal square-ALU task.

## Verification

Software tests cover:

- stage formation from structural dataflow;
- exact dependency waves and carried-surface lifetimes;
- capability admission and rejection;
- measured and rejected geometries for shared paths;
- no workload-name checks below decomposition and structural planning;
- no full score surface in an admitted online-attention plan;
- deterministic bundle serialization and dispatch order.

M4 tests compare against independent FP32 or FP64 references. Floating-point
gates report relative L2 error, maximum absolute error, and finite-output
status. They also verify that the intended route fired. The S128/D128 attention
graph passes as both a three-program composed bundle and an eight-program
forced fallback. Scalar-fold tests exercise the same composition machinery on
non-attention graphs. The matmul-GELU, affine-scan, and complete C128/D128
Chunked DeltaNet gates also pass on M4.

## Repository history

After implementation and all available tests pass, the final source tree is
committed as a new root commit. Removed HWX files and their blobs are checked
against every reachable object. The private `main` branch is then updated with
one `git push --force-with-lease`. No earlier force push is permitted.
