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
Every sliced ANEC program binding records physicalElements even when it equals
the logical count, and every program records scratchBytes from its encoded H13
program. The inspector requires both fields and does not substitute defaults.

Package schema v2 keeps runtime buffers separate from function returns.
physicalOutputs describes each produced FP16 buffer once. logicalResults keeps
MIL return order, name, type, shape, and an identity view with only tensor,
elementOffset, and elementCount. Storage capacity comes from the referenced
physical output, not from duplicated view metadata. Identity requires equal
logical and physical dtypes, elementCount equal to the logical shape product,
and a view range within physical storage. Duplicate read-only views are valid.
A physical output remains storage even when a later program consumes it. The
producer stays in physicalOutputs, the consumer binding refers to the same
tensor, and dispatchPlan must keep the producer before every consumer. This
permits an early MIL return without misclassifying that buffer as a transient
intermediate.

HWX and ANEC must encode the same bound H13 task stream. ANEC binding normalizes
surface selectors and task header flags after task selection; HWX therefore
embeds that normalized task slice from the encoded ANEC rather than the raw
pre-binding task bytes. Extracting ANEC from HWX must reproduce the explicit
ANEC artifact byte for byte.
Mixed FP32 or int32 returns fail with
h13.unsupported-logical-result-conversion until an explicit hardware or GPU
conversion path exists.

## Verified Outcome

The host-only split contract compiles a valid channel split followed by sigmoid
and multiply. The manifest contains only the two encoded operations. Their
source bindings select offsets 64 and 0 respectively from the same 128-element
input, and the package inspector accepts the result. A dense independent
reference confirms that those intervals are the ordered second and first
halves. Valid noncontiguous geometry fails with
h13.noncontiguous-split-axis, and a valid three-result split fails with
h13.unsupported-split-count.

Direct and reshape-chained outputs, split subslices, and distinct view outputs
compile as ordered identity results. One split-backed physical output can expose
differently shaped logical views at offsets 0 and 64 in either MIL return order.
Independent producers keep physical output order while logicalResults follows
the declared return order. A returned producer may also feed a later program;
the manifest keeps it as physical output storage and records the later consumer
dependency. The strict inspector accepts these mappings, requires topological
dispatch coverage, and rejects a logical physicalElements field because that
field belongs only to ANEC program binding geometry.

Direct and reshape-chained constant split sources fail with
h13.unsupported-constant-split-source before blob access or alias creation. A
runtime-produced 256-element intermediate remains an intermediate across an
outer split, reshape, and nested splits; its four consumer windows are
128/64, 192/64, 0/64, and 64/64. This proves the provenance rejection does not
turn valid intermediate views into inputs.

The pinned adapter output at
/tmp/ane-runtime-adapter-integrated-20260913/model.mil reaches its ordered FP32
and int32 returns and fails with h13.unsupported-logical-result-conversion. The
full faithful source still stops earlier on the known uint4 importer boundary.
No hardware command was run.

## Repository Shape

The integration branch starts from scalar-folding receipt commit
7da065bd7c94aafdd7dd4f3e47c8ff16645910fb, merges ordered split commit
e8b8520bb75d9ae30a4926de54936b1cd9f105a7, and merges qualified H13 commit
f4ad09066b560818a9dcc1f5f4f53273d941bd1e. The adapter source remains
rootadapter commit eac5eed9 with program(1), CoreML8, and ordered parenthesized
result lists.
