# Theory: H13 Split Results as Buffer Views

## Problem

The textual MIL frontend now preserves ordered operation results, but H13 still
rejected every multi-result operation before lowering. The pinned CoreML 8
encoder uses two-result channel splits whose first half feeds a multiply and
whose second half feeds a sigmoid. Treating both values as the whole source
buffer would compile the graph but bind the wrong data.

## Operating Theory

A split needs no ANE program when each result is one contiguous interval of the
source buffer. For an equal two-way split, this is true only when every logical
dimension before the split axis is one. Result order then determines exact
base offsets: result zero starts at the source base, and result one starts one
result-sized interval later. Consumers keep the source tensor name and add the
view base to their existing operation-local slice offset.

The package schema has one slice per binding. It cannot represent the disjoint
chunks produced when a non-unit prefix precedes the split axis. Such shapes
must fail with a named diagnostic rather than acquire guessed strides or host
materialization. Split values also do not become tensor aliases in the
manifest: that alias contract requires the alias and underlying tensor to have
the same element count. Their semantics live in consumer binding slices.
A view must also retain its storage provenance. A typed alias object does not
carry a producer, so treating an alias of a constant as an ordinary runtime
surface silently bypasses blob resolution. The compiler follows the original
GraphIR producer chain through reshape, squeeze, expand_dims, and split before
creating a view. Constant-backed split sources are rejected until the package
ABI can represent exact constant subranges.

## Lowering Boundary

H13 accepts only an exact rank-zero tensor<int32, []> axis, an exact rank-zero
tensor<int32, []> partition count of two, two equal positive static FP16
results, and a contiguous axis. It maps both ordered SSA results to typed views,
propagates view offsets through shape-only aliases, and validates offset
addition before recording a binding. Liveness checks every ordered result.
Other multi-result operations retain the existing refusal.
The qualified package contract is strict rather than inferred by consumers.
Every sliced binding records physicalElements even when it equals the logical
count, and every program records scratchBytes from its encoded H13 program.
The inspector requires both fields. It does not substitute count or zero when
a producer omits them.

The function-output contract remains separate. H13 still requires one return
that names the last encoded operation result. Supporting split views does not
imply support for multiple function outputs, unsupported operations, full
encoder compilation, or device execution.

## Verified Outcome

The host-only split contract compiles a valid channel split followed by sigmoid
and multiply. The manifest contains only the two encoded operations. Their
source bindings select offsets 64 and 0 respectively from the same 128-element
input, and the package inspector accepts the result. A dense independent
reference confirms that those intervals are the ordered second and first
halves. Valid noncontiguous geometry fails with
h13.noncontiguous-split-axis, and a valid three-result split fails with
h13.unsupported-split-count.
Direct and reshape-chained constant split sources now fail with
h13.unsupported-constant-split-source before blob access or alias creation. A
runtime-produced 256-element intermediate remains an intermediate across an
outer split, reshape, and nested splits; its four consumer windows are
128/64, 192/64, 0/64, and 64/64. This proves the provenance rejection does not
turn valid intermediate views into inputs.

The pinned adapter output at
/tmp/ane-runtime-adapter-integrated-20260913/model.mil passes the former blanket
split precheck. Its next compiler boundary is the two-value function return at
line 3355, rejected by h13.unsupported-chain. No hardware command was run.

## Repository Shape

The integration branch starts from scalar-folding receipt commit
7da065bd7c94aafdd7dd4f3e47c8ff16645910fb, merges ordered split commit
e8b8520bb75d9ae30a4926de54936b1cd9f105a7, and merges qualified H13 commit
f4ad09066b560818a9dcc1f5f4f53273d941bd1e. The adapter source remains
rootadapter commit eac5eed9 with program(1), CoreML8, and ordered parenthesized
result lists.
