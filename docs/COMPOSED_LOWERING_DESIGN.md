# Composed Lowerings for H16G

Status: implemented for the bounded geometries listed below
Target: Apple M4 H16G
Numeric mode: FP16

## Scope

This change adds three compiler paths:

- matmul followed by GELU;
- FlashAttention-2 forward inference;
- chunked gated DeltaNet through an affine scan.

The compiler continues to accept a typed MIL graph. The frontend may expand a
compound semantic operation, and the planner may recognize a legal connected
region. H16G emission receives scheduled primitive stages. It does not receive
the original workload name.

The implemented attention path covers unmasked FP16 forward inference at
S128/D128. Dropout, backward propagation, causal masking, and multi-tile
emission are outside the current hardware path. The planner records larger
tiled forms, but the assembler rejects them until the required tile-copy and
carried-state packets have been decoded.

## Existing problem

`ANETaskScheduler` already records tasks, surfaces, commands, and dependencies.
`H16GMixedTaskEncoder` then reconstructs an exact attention sequence from
`sourceNodeIdentifiers`. It assumes fixed head counts, fixed dimensions, one
fixed task stream length, and fixed scratch offsets. This prevents the planned
graph from being the source of emission decisions.

## Scheduled representation

Each scheduled task will contain an ordered list of `ANEScheduledStage`
objects. A stage records:

- primitive operation kind and semantic operation;
- input and output surface identifiers;
- tile coordinates and tile shape;
- bridge storage class;
- carried input and output surfaces;
- dependency wave;
- whether the stage materializes an external result.

`ANEScheduledSurface` gains a carried-state role. Carried surfaces remain live
across loop iterations and dispatches. The same role is used by the online
softmax state and the affine scan state.

`ANETilePlan` gains iteration and tile shapes. Existing direct, row-tiled, and
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

The existing mixed-attention encoder remains available for its older measured
H4/S64/D64 form. New composed regions use the scheduled primitive assembler.

## Matmul followed by GELU

The operation graph contains an ordinary matmul node followed by an ordinary
GELU node. Fusion forms one region. Scheduling retains both stages and the
intermediate surface. The assembler chooses one program when the capability
table has a legal internal bridge. Otherwise it emits two ordered artifacts
whose shared surface is explicit in the bundle.

No GELU-specific matmul emitter is added.

## FlashAttention-2

The planner recognizes the connected semantic graph:

```text
Q times transpose(K)
  -> scale
  -> optional causal mask
  -> row softmax
  -> times V
```

It then creates an online-reduction plan. The current emitted geometry contains
one 128x128 query tile and one 128x128 key/value tile. The compiler emits eight
ordered primitive artifacts for matmul, scale, row maximum, subtraction, exp,
row sum, division, and the final matmul. Larger plans record carried maximum,
sum, and output surfaces but are not emitted yet.

The operation stages are matmul, ALU, LUT, reduction, and carried-state update.
The H16G emitter only sees those stages and their scheduled geometry.

## Chunked DeltaNet

The frontend expands a gated delta state update into ordinary matrix and
elementwise operations plus affine transition pairs. The planner schedules a
connected chain of those pairs as an associative scan. It records dependency
waves and carried state in the same scheduled representation used by online
attention.

The backend receives matmul, reduction, ALU, layout, and state-copy stages. It
does not receive a DeltaNet packet kind.

The hardware gate uses four N128 FP16 matrix-state transitions of the form
`state = state * factor + update`. Each multiply and add is emitted as a normal
square-ALU task. Query projection and the rest of a complete DeltaNet block are
outside this compiler change.

## Verification

Software tests must prove:

- stage formation from structural dataflow;
- exact dependency waves and carried-surface lifetimes;
- capability admission and rejection;
- two different geometries for each shared path;
- no workload-name checks below decomposition and structural planning;
- no full score surface in an admitted online-attention plan;
- deterministic bundle serialization and dispatch order.

M4 tests compare against independent FP32 or FP64 references. Floating-point
gates report relative L2 error, maximum absolute error, and finite-output
status. They also verify that the intended route fired.

## Repository history

After implementation and all available tests pass, the final source tree is
committed as a new root commit. Removed HWX files and their blobs are checked
against every reachable object. The private `main` branch is then updated with
one `git push --force-with-lease`. No earlier force push is permitted.
