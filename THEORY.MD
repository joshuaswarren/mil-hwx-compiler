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

## Lowering Boundary

H13 accepts only an exact rank-zero tensor<int32, []> axis, an exact rank-zero
tensor<int32, []> partition count of two, two equal positive static FP16
results, and a contiguous axis. It maps both ordered SSA results to typed views,
propagates view offsets through shape-only aliases, and validates offset
addition before recording a binding. Liveness checks every ordered result.
Other multi-result operations retain the existing refusal.

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

The pinned adapter output at
/tmp/ane-runtime-adapter-integrated-20260913/model.mil passes the former blanket
split precheck. Its next compiler boundary is the two-value function return at
line 3355, rejected by h13.unsupported-chain. No hardware command was run.

## Repository Shape

This change starts from ordered-result commit
a1bfea0cd52208ac9f168791e8c6e9a83ddc8efb in an isolated worktree. The
adapter input came from rootadapter commit
eac5eed9 and retains program(1), CoreML8, and ordered parenthesized result
lists.
