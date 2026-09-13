# H13 transpose-absorb receipt (consumer-side absorption classification)

## Source

- Compiler base: `9c2a448` (`feature/h13-layout-stream2`, itself on `83a4434`)
- Branch: `feature/h13-transpose-absorb`
- Encoder artifact analyzed read-only: `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed)

## Assignment result: 0 of the 98 become supported; all 146 exactly classified

The task asked for transpose-absorbing input flags on the H13 add and conv
lowering paths, mirroring matmul `transpose_x`/`transpose_y`. Investigation of
the decoded corpus falsifies the mechanism for every one of the 98 blockers.
The flags were not added because no sound implementation exists; the exact
per-class classification below replaces them, and the rejection contracts now
name each class's real blocker.

### Why an add/conv absorption flag cannot exist in this harness

1. **The matmul flag mechanism does not generalize.** `transpose_y` selects a
   different *decoded task stream* whose engine reads operand B transposed; it
   is not a manifest-stride trick. For an elementwise or conv consumer the
   equivalent would be a task stream reading one operand with permuted plane
   and row strides. Every decoded add/conv/broadcast template bakes its
   operand strides per geometry (`H13Program.cpp` `binaryTask` stride words,
   `elementwiseTensor` row/plane formula); the corpus holds no encoder that
   permutes surface strides. Inventing one has no decoded reference and so
   cannot meet the byte-exactness bar this repo compiles against.
2. **The proof oracle cannot exist.** "Byte-equality vs the un-folded direct
   path" requires a compilable MIL form of the folded consumer reading the
   pre-transpose operand. MIL cannot express a strided operand read — the
   direct path *is* the materialized transpose, which no encoder lowers. For
   the matmul y-fold the direct form exists (`matmul(transpose_y=true)`); for
   add/conv it does not, at any shape.
3. **Fast-axis swaps are inexpressible even as a descriptor.** The NCHW
   binding carries shape + row/plane/batch strides with the W axis contiguous
   by construction. A permutation whose output-fastest axis maps to a
   non-fast storage axis (every rank-3 `[0,2,1]`) cannot be written as any
   surface interpretation, at any size.
4. **Independently, every consumer geometry in the 98 is outside every
   decoded envelope.** The decoded elementwise corpus tops out at 4096 flat
   elements and spatial {8,16,32,224}; the broadcast corpus has no 375
   extent; the two attention adds sit at 384000 flat elements / CHW
   (8,375,128); the conv takes a rank-3 input with a rank-3
   `[2048,1024,1]` weight and a 1-D kernel. Even a hypothetical absorption
   would leave these consumers rejecting.

### Exact classification of the 146 encoder transposes

All perms normalize to `[0,2,1,3]` (rank 4) or `[0,2,1]` (rank 3).

| count | perm | shapes | consumers | blocker class |
|------:|------|--------|-----------|---------------|
| 48 | `[0,2,-3,-1]` | `[1,375,8,128]→[1,8,375,128]` | matmul y, 1 use | fast-axis-stable *middle* swap: the trailing-swap fold does not apply; no decoded matmul reads an operand with permuted leading strides; rank-4 batched matmul is outside the envelope (separate decision, not expanded here) |
| 24 | `[0,2,1,3]` | `[1,375,8,128]→[1,8,375,128]` | add x, **2 uses** each, const bias `[1,8,1,128]` | two consumers need a materialized surface; single-use absorption would additionally need permuted-stride reads (none decoded) or a source-layout C-broadcast against `[1,1,8,128]` (no decoded geometry) |
| 25 | `[0,2,1,3]` | 24× `[1,8,375,128]→[1,375,8,128]`, 1× `[1,256,375,16]→[1,375,256,16]` | reshape → linear | composite pure-view collapse: **0 of 25** (see below) |
| 24 | `[0,2,1]` | `[1,375,1024]→[1,1024,375]` | conv x | fast-axis swap: inexpressible in the NCHW descriptor; the conv is also outside the envelope on its own (rank-3 input, rank-3 weight, 1-D kernel) |
| 24 | `[0,2,1]` | `[1,1024,375]→[1,375,1024]` | add y (x plain) | fast-axis swap: inexpressible; mixed operand layouts also block algebraic commutation |
| 1 | `[0,2,1]` | `[1,375,375]` bool | logical_and | fp16-only lowering (`h13.invalid-transpose-parameters` at the view); `logical_and` has no H13 encoder |

### The reshape-feeding composite, classified exactly

`reshape` (like `squeeze`/`expand_dims`) is a flat-order-preserving relabel,
so `transpose∘reshape` is a pure view exactly when the transpose's own flat
map is identity — the reshape never rescues a permutation. For all 25 the
perm swaps two non-unit axes (8↔375 ×24, 256↔375 ×1), so the composite is a
genuine permutation, and the consuming linear's reduction rows (K=1024 as
8×128 runs; K=4096 as 256×16 runs) are chunked with non-unit inter-chunk
stride — no contiguous binding slice covers a row. **Pure-view collapses: 0
of 25.**

Follow-up finding (not built here): the source storage of the 24 attention
cases decomposes into 8 *contiguous* `[375,128]` channel planes, so a
chunked-accumulation rewrite over those planes is a genuine zero-movement
lowering for this class — but it is a producer-side graph decomposition, not
a consumer flag, and it has no compilable direct MIL reference to byte-prove
against, so it belongs to a separate stream with its own oracle work.

### Updated stream-2 coverage numbers

- 219 layout ops: transpose 146, slice_by_index 48, pad 24, tile 1 —
  supported count unchanged at **0** for this package (the full model also
  gates earlier on non-fp16 function results: `encoder_hidden` fp32 /
  `encoder_mask` int32 casts reject `h13.unsupported-logical-result-conversion`
  before op lowering).
- The 146 transposes now reject with per-class named reasons instead of one
  generic message: fast-axis swap (48 rank-3 ops), multi-consumer
  materialization (24), shape-view composite (25), middle-swap with no
  decoded permuted-stride consumer (48 matmul + the general case), and the
  fp16-only contract (1 bool). Code stays `h13.nonfoldable-transpose`
  (existing contracts pin it) except the bool case, which stays
  `h13.invalid-transpose-parameters`.
- Rank-4 batched matmul: still out of scope; the 48 matmul-feeding
  transposes counted and named above.

## Commands and observed results

```text
make -j8 build/mil-hwxc                    # 0 errors (GNUstep host toolchain)
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS                     # includes the new encoder-shaped class tests
python3 tests/test_h13_reference.py build/mil-hwxc
  ValueError: slice has incorrect fields   # in
  test_intermediate_physical_buffer_composition; pre-exists identically on
  base 9c2a448 with this work stashed; unrelated — this stream emits no
  binding fields
python3 tests/test_h13_deadline.py build/mil-hwxc
  H13_DEADLINE_OK
python3 tests/test_h13_parity.py build/mil-hwxc
  AssertionError: binary_add_1x1024x1x1 task 0 words differ from the oracle
  # pre-exists on the pinned base (ParityTriage owns it); unchanged here
build/mil-hwxc --mil /tmp/ane-runtime-adapter-integrated-20260913/model.mil \
  --model-root .../model-root --output encoder-out --target H13 --format anec
  3353:5: error [h13.unsupported-logical-result-conversion]   # whole-model gate, pre-existing

`tests/test_h13_layout_cli.py` additions: encoder-shaped cases for all six
classes above with exact shapes and perms from the blocker list, each pinned
to its code *and* its distinguishing message fragment; plus a positive
control that a layout-preserving transpose feeding a reshape stays the free
alias (`t` and `r` both `aliasOf: a`, inspector-validated, recompile
deterministic). No new supported encoder form is claimed.

## Corpus evidence for the falsification

- Decoded broadcast geometries (`H13EnvelopeTemplates.inc`
  `kBroadcastTasks`): no extent 375 anywhere; H/W ∈ {1,8,16,32,224}.
- Every decoded binary/broadcast task stream bakes per-geometry operand
  strides; none permutes them (`H13Program.cpp` `binaryTask`,
  `encodeBroadcast`).
- Manifest logical results carry `conversion: "identity"` only
  (`ANEH13Compiler.mm` logicalResults assembly), so a permuted-layout result
  cannot be declared.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or
Mac execution was performed. No files under `ane-linux-experiments` were
edited or executed; the encoder artifact and its coverage receipts were read
only.
