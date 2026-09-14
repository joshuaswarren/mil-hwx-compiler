# Theory: H13 Layout Views and the Concat Hole

## Problem

The encoder's heads_layout stack is concat of independent `[1,1,375,128]`
fp16 sources on axis 1. Split of one contiguous buffer is already a view.
Concat looks like the inverse, but the sources are distinct buffers, so the
result is a new surface. Treating concat as a view would bind the wrong data.
Inventing a copy program would fake an encoder the decoded corpus does not
hold.

## Operating Theory

A split needs no ANE program when each result is one contiguous interval of
the source buffer. For an equal two-way split, this is true only when every
logical dimension before the split axis is one. Result order then determines
exact base offsets.

Concat of independent sources is the opposite dataflow: N buffers into one.
The package schema has one slice per binding. That ABI can *read* a
contiguous window of one tensor. It cannot *write* N sources into one dest
without a program that copies. Tile is the only decoded data-movement
encoder, and it replicates one input. Pad is refused because Apple's own
tool rejects it. No concat or identity-copy capture exists.

Liveness used to inspect only `operands`, which are Value-kind arguments.
Concat's `values = (a, b)` is a tuple, so a and b were invisible and a
standalone concat failed as unused inputs before the hole could be named.
Use-tracking now walks tuple and list argument elements.

## Lowering Boundary

Valid concat is static fp16 sources of equal rank, an exact rank-zero
`tensor<int32, []>` axis, matching off-axis extents, and a result whose axis
extent is the sum of the sources. That form still fails with
`h13.unsupported-concat`: H13 has no concat encoder. Invalid parameters and
shapes fail first with `h13.invalid-concat-parameters`,
`h13.invalid-concat-shape`, or `h13.unsupported-concat-axis`.

Split, reshape-family aliases, layout-preserving transpose, and identity
pad/tile stay host-side views. Bool transpose uses the same exact-perm
contract as fp16. Mixed FP32 or int32 returns still fail with
`h13.unsupported-logical-result-conversion`.

## Strategy

Name the hole exactly, then stop. Do not emit mul-by-1 copies into output
slices, do not alias independent sources, and do not mint a task stream
without a capture. A concat encoder exists only if a decoded Apple program
for this geometry appears.

## Key Discoveries

Split is a view because one buffer already holds both halves. Concat of the
encoder's stacked heads is not a view because the heads are independently
produced. The first compile of a 2-way `[1,1,375,128]` axis-1 concat failed
`H13 function inputs must all be used` until tuple uses counted. After that,
the named hole is the fail-fast for 2-way, 8-way, and the 8-way binary tree.

## Open Questions

Whether Apple ever emits a concat or DMA-copy program for this stack. Until
a capture exists, the ISA cannot express it. Adjacent split halves of one
tensor concatenated back along the same axis could alias the source; the
encoder does not emit that form, so it stays unlowered.
