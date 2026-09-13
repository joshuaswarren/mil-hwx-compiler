# H13 channel-plane composite receipt (graph decompositions, stream 1)

## Source

- Compiler base: `8be0bda` (`feature/h13-transpose-absorb`)
- Branch: `feature/h13-graph-decompose`
- Encoder artifact analyzed read-only: `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed)

## Item 1 — the 24 reshape+linear composites: SUPPORTED

A `[0,2,1,3]` transpose of `[1,C,P,D]` feeding a reshape that merges the
transposed pair into one reduction (`[1,P,C*D]`), consumed by exactly one
`linear` with a constant weight, now lowers through the channel-plane
decomposition the transpose-absorb receipt identified:

- each channel `c` is one contiguous `[P,D]` plane of the input storage at
  element offset `c*P*D`;
- the linear becomes C matmuls, each reading its plane through the existing
  offset-view binding (`valueBaseOffsets`), with the weight's `[c*D,(c+1)*D)`
  column slice delivered as a synthesized row-major `[N,D]` constant;
- the partials accumulate in C-1 adds and the bias lands as one expanded
  per-row constant add — the same chunked-accumulation shape the >512
  reduction path already emits.

Zero data movement: every program reads contiguous storage that already
exists; the weight slices are host-side constant reordering (compile-time,
like the existing transpose_y=false host transpose).

### Byte-equality oracle

The direct reference is the per-plane MIL a user writes: nested axis-1
splits expose each plane as the offset view the split lowering already
emits (axis-1 splits are legal because axis 0 is unit), then per-plane
matmuls, accumulation adds, and the expanded-bias add. Verified:

- every `program-*.anec` byte-identical between folded and direct forms —
  C=2/P=4/D=128/N=512 (41 programs), C=4/P=3/D=16/N=512 (the subsampling
  D=16 class), and the no-bias variant;
- manifests structurally equal after erasing synthesized value names
  (`physicalOutputs`, `logicalResults` equal literally);
- recompile deterministic in both `anec` and `hwx` formats;
  inspector-validated (`research/inspect_anec.py`) on both sides;
- C=8 (the encoder's head count) pinned by manifest structure: 16 plane
  matmuls read the input at exactly the 16 offsets `c*P*D + row*D`.

### Measured on the true encoder shapes

`[1,8,375,128] → [1,375,1024]`, N=1024, with bias: **54 000 programs
(6 000 matvec + 48 000 add), 3.4 GB payload, 41.5 s compile** — mechanism
complete; program-count economics are a scheduling-envelope question (the
parent owns scheduling). The subsampling form `[1,256,375,16] → [1,375,4096]`
would emit ~1.7 M programs by the same arithmetic and is compiled only at
test scale; it needs a batched or L2-resident accumulation plan before it is
package-practical.

Net for the encoder: **25 of the 98 add/conv/reshape-feeding transposes
become supported** (24 o-projection + 1 subsampling).

## Item 2 — general strided/non-unit-head slice_by_index: documented upgrade

The valueBaseOffsets generalization (chunked views) is not deliverable under
the byte-equality bar: a chunked read (H chunks of L elements at stride S)
has no MIL-expressible direct reference — no view op exposes mid-range head
slices (slice_by_index supports prefix ranges only, split is binary on one
axis, and result assembly at chunked offsets is likewise inexpressible), so
there is nothing compilable to byte-prove a chunk decomposition against. The
rejection now names the exact chunk geometry instead:

- mask slicing `[1,8,750,375] → [1,8,749,375]`: "8 chunks of 280875 elements
  spaced 281250 apart" (24 ops);
- bias slicing `[1,8,375,749] → [1,8,375,375]`: "3000 chunks of 375 elements
  spaced 749 apart" (24 ops) — the mid-range end is now resolved correctly in
  the message (the plan records the slice upper bound).

Code unchanged: `h13.noncontiguous-slice`; tests pin code and fragment.

## Item 3 — pad and tile: documented upgrade

Same reasoning: the grown/replicated region lands interleaved at the padded
or tiled stride, not as one appendable tail, and no MIL direct reference
exists for a program writing a larger surface from strided input reads. The
rejections now carry the exact amounts vector:

- pad `[0,0,0,0,0,0,1,0]` on `[1,8,375,749]` (24 ops) —
  `h13.unsupported-pad` with amounts;
- tile `reps [1,375,1]` on the bool mask (1 op) — `h13.unsupported-tile`
  with reps; the operand is bool, so the fp16-only view contract rejects it
  anyway.

A latent gap worth naming for the next stream: slice_by_index only lowers
prefix ranges (`[b, extent)`); the bias slice's mid-range `[0,375)` of 749
would reject `h13.invalid-slice-shape` even with unit heads.

## Commands and observed results

```text
make -j8 build/mil-hwxc                    # 0 errors (GNUstep host toolchain)
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS
  h13 composite cli: PASS                  # new suite, wired into the target
python3 tests/test_h13_parity.py build/mil-hwxc
  AssertionError: binary_add_1x1024x1x1 task 0 words differ from the oracle
  # pre-exists on the pinned base (ParityTriage owns it); unchanged here
```

Encoder-shape measurement (receipt above): rc=0, 54 000 programs, 41.5 s.

## Implementation notes

- `lowerOperation`'s constant-weight matmul path accepts
  `synthesizedConstants` weights (row-major `[columns, reduction]` bytes)
  alongside BLOBFILE payloads; the runtimeWeight classification already
  treated synthesized constants as constant.
- Near-miss composites (non-linear consumer, two linears, returned reshape)
  reject `h13.nonfoldable-transpose` with the exact near-miss reason; the
  previous stream's generic reshape-class message no longer fires for
  merge-shaped composites (layout tests updated to the new contract).
- Rank-4 batched matmul stays out of scope; the 48 matmul-feeding
  transposes and the attention chains around items 2/3 still gate on it.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or
Mac execution was performed. No files under `ane-linux-experiments` were
edited or executed; the encoder artifact was read only.
