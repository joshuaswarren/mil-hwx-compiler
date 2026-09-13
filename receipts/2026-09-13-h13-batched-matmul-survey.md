# H13 batched matmul envelope survey receipt (needs-new-primitive verdict)

## Source

- Compiler base: `4154cb8` (`feature/h13-graph-decompose`)
- Branch: `feature/h13-batched-matmul-survey`
- Encoder artifact analyzed read-only: `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed)

## Verdict: the decoded corpus holds NO batched matmul form — needs new primitive

### Survey evidence (programmatic, over the decoded template tables)

- `OracleMatmulTemplate` and `MatmulShape` carry exactly
  `{rows, reduction, columns, transposeX, transposeY, runtimeWeight}` — no
  batch or leading-dimension field exists anywhere in the corpus structures
  (`plugins/H13/H13Program.cpp`, `H13Program.h`).
- All 145 `kMatmulEnvelopeTasks` geometries enumerate over
  rows ∈ {1,16,32,48,64,96,128,144,160,176,192,224,256,384,512},
  reduction ∈ {32,64,96,128,256,512,1024,2048,4096,8192},
  columns ∈ {32,64,96,128,256,512,1024,2048,4096,8192}.
  **No 375 or 749 extent appears in any dimension of any decoded matmul.**
- The 33 `kMatvecTasks` (legacy) enumerate rows ∈ {1,2,8,64},
  reduction/columns ∈ {256,512,1024} — likewise no batch axis, no
  attention-shaped geometry.
- Runtime-operand surfaces are always single 2-D planes:
  `encodeMatmulParity` builds both runtime operands as
  `matvecTensor(index, rows, width)` → `[1,1,rows,width]` with one plane
  stride — a batch-strided second-operand surface is unrepresentable in the
  derived layouts, which is why `runtimeMatmulOperand` in the compiler
  requires every axis before the trailing pair to be unit.
- Flattening the encoder's `[1,8,375,128]` x into 3000 rows does not
  reproduce the graph's math: a flattened matmul multiplies every row
  against ONE shared y, while the attention chains carry a per-batch y
  (`[1,8,128,749]` runtime KV, `[1,8,375,128]` runtime V). The encoder's
  48 `[0,2,-3,-1]` transposes exist precisely to build those per-batch
  operands.

### What the encoder needs

- scores: per-batch `[375,128] × [128,749]`, transpose_x/y = false → needs
  (375, 128, 749) per batch;
- attn-output: per-batch `[375,375] × [375,128]`, flags false → needs
  (375, 375, 128) per batch;
- the KV operand also appears as a batched CONSTANT (`var_355`
  `[1,8,128,749]` BLOBFILE) — the matvec program model carries one constant
  section per program, so per-batch weights need per-batch programs.

None of these geometries is decoded, with or without batching.

### Recommended primitive

A **batched matvec**: a task-stream template parameterized by
`(B, rows, reduction, columns, transposeX, transposeY)` that iterates B GEMMs
over batch-major contiguous operand slices — `x[b]` at element base
`b·rows·reduction`, `y[b]` at its transpose-shaped base, `out[b]` at
`b·rows·columns`. The encoder's rank-4 attention operands are already laid
out batch-major contiguously (the same contiguity the channel-plane
decomposition exploits), so the primitive needs no data movement — only new
decoded oracle captures, since inventing task words without an Apple oracle
breaks the byte-exact methodology. Equivalently: per-batch GEMM programs
over contiguous slices with new decoded (375, 128, 749) / (375, 375, 128)
geometries — same capture requirement, more programs.

## Rejection-contract upgrade

Both batched paths now reject `h13.matmul-outside-envelope` with the exact
structure instead of the generic envelope message:

- runtime per-batch y: names the absent template shape, the wrong-product
  flattening hazard, the contiguous batch-slice primitive, and B;
- batched constant weight: names the one-constant-section model, the
  per-batch program decomposition, and the missing geometries.

Encoder-shaped tests pin both (scores `[1,8,375,128]×[1,8,128,749]`,
attn-output `[1,8,375,375]×[1,8,375,128]`, batched const y).

## Carried latent gap (per Main)

slice_by_index only lowers PREFIX ranges (`[b, extent)`); a mid-range end
such as the bias slice's `[0,375)` of 749 rejects
`h13.invalid-slice-shape` even with unit heads. Named in the graph-decompose
receipt; still open.

## Commands and observed results

```text
make -j8 build/mil-hwxc                    # 0 errors (GNUstep host toolchain)
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS
  h13 composite cli: PASS                  # includes the survey pins
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or
Mac execution was performed. No files under `ane-linux-experiments` were
edited or executed; the encoder artifact was read only.
