# H14 equivalent-shape rewrites: proposals, not model changes

Date: 2026-09-30. Status: PROPOSALS ONLY — the staged Qwen MIL is unchanged.
Actor: H14Integ on `agent/h14-integ` (merge of `agent/h14-shape-ext`).

## Scope

Apple's own H14 compiler refuses a small set of reduce shapes with
`callback_status=1` while accepting sibling shapes. The 24 `gxreduce_*`
records in `research/oracles/h14/` hold both sides: the accepted points the
norm encoder already replays, and the rejected points this file proposes to
rewrite around. Every proposal here keeps the MIL semantics and the staged
model graph untouched; each rewrite is a suggestion for the harness or a
future compiler pass, gated on Apple-side verification that has not run.

## Apple-accepted vs Apple-refused reduce points

Accepted today (in the H14 norm template set, verified against the merged
build): reductions over flat `[1, C, 1, 1]` for C in 64..4096, the sequence
and spatial shapes, and — relevant to Qwen — `(1, 16, 128, 1)` at
`axes=[2]`, `(1, 16, 1, 128)` at `axes=[2]`/`[3]`, and `(1, 2048, 1, 1)`
at `axes=[2]`/`[3]`/`[2,3]` (per the `gxreduce_*` table).

Refused with `callback_status=1` (Apple's own compiler, records checked in):

- `reduce_sum` / `reduce_mean` / `reduce_max` at `(1, 16, 128, 1)`
  `axes=[2,3]` and `axes=[3]`
- `reduce_sum` / `reduce_mean` / `reduce_max` at `(1, 2048, 1, 1)`
  — `axes=[2,3]` is decoded for `reduce_max`/`reduce_mean` only;
  `reduce_sum` at that point is refused.

The Qwen staged graph needs `(1, 16, 128, 1) axes=[2,3]` (l2-norm
statistics: sum of squares per head) and `(1, 2048, 1, 1) axes=[2,3]`
(rms_norm sum of squares). Both land on refused points today.

## Rewrite 1 — rank collapse: fold trailing 1-sized axes away

`(1, 16, 128, 1) axes=[2,3]` asks one reduce to swallow the fastest axis
too. Apple accepts the same reduce at `axes=[2]` over the rank-3 twin
`(1, 16, 128)` → `(1, 16, 1)`.

Proposal: drop all trailing size-1 axes at the harness level, run the
reduce at the rank-3 form, and reshape back to `keep_dims` form on the
host (the staged runtime owns the surface layout anyway; ANE tensor
descriptors already record `nchw = [1, 16, 1, 1]` with a 64-wide channel
tile, so the extra axis is free at the storage level).

Numerical risk: none — the reduction order over the surviving axis is
unchanged; fp16 summation order inside Apple's kernel is whatever the
decoded point uses, which is exactly what parity pins.

Verification status: NOT RUN on Apple's compiler. Requires one mint
session: `reduce_sum(x = x, axes = [1], keep_dims = true)` at
`tensor<fp16, [1, 16, 128]>` through `ane-compile-hwx target=h14`. If
Apple accepts the rank-3 form, the parity oracle lands in
`research/oracles/h14/` and the norm template extends by one row.

## Rewrite 2 — matmul against a ones vector

`(1, 2048, 1, 1) axes=[2,3]` is a whole-tensor sum of 2048 values. Apple
accepts `matmul` at `[1, 1, 2048] x [1, 2048, 1]` (the decoded matvec
grid has K=2048 points and rank-3 M=1 is the same geometry the staged
decode already uses).

Proposal: express the whole-tensor sum as `matmul(x_rank3, ones)`
where `ones` is a constant `[2048, 1]` fp16 tensor of 1.0. The staged
MIL's `reduce_sum(axes=[1,2,3], keep_dims=true)` becomes one matvec
task.

Numerical risk: MODERATE and material. Apple's reduce kernels and matmul
kernels do not share an accumulation order; over 2048 fp16 terms the
reduction tree differs and the last-ulp tail differs with it. rms_norm
tolerates that (the envelope class is already fp16-envelope, not
bit-exact), but any consumer that diffs the sum bit-exactly would need
the same tolerance class declared. The ones vector also costs a
`2048 x 2 = 4 KiB` constant surface vs the reduce's zero-byte section.

Verification status: NOT RUN on Apple's compiler. Requires a mint
session with the `ones` BLOBFILE constant at a known offset.

## Rewrite 3 — transpose then reduce on an accepted axis

The l2-norm shape `(1, 16, 128, 1)` could also ride a transpose to
`(1, 16, 1, 128)` and reduce on `axes=[3]`, which is a decoded point for
all three reduce ops. But H14 has no transpose encoder (no decoded
transpose oracle exists — 1668 H14 oracle records, zero transpose
rows), so this rewrite needs a transpose decode FIRST and is strictly
worse than Rewrite 1 (which needs only a reduce at an already-99%-decoded
geometry).

Verdict: parked behind the transpose decode; do not pursue while
Rewrite 1 is open.

## Rewrite 4 — replace Qwen linear with a 1x1 convolution

The refused Qwen `linear` consumes the channel surface `[1, 2048, 1, 1]` with
weights `[N, 2048]`, where `N` is 6144 or 4096. Apple accepted these exact
convolution forms: `conv` with input `[1, 2048, 1, 1]`, weights
`[N, 2048, 1, 1]`, `groups=1`, kernel and stride 1, and `valid` padding. The
two decoded records are `qwen_conv1x1_k1_c2048_n6144_s1_valid` and
`qwen_conv1x1_k1_c2048_n4096_s1_valid`; each is one H14 task.

For every output channel `n`, linear computes
`y[n] = sum(k=0..2047, x[k] * W[n,k])`. The 1x1 convolution computes
`y[n,0,0] = sum(k=0..2047, x[0,k,0,0] * W[n,k,0,0])`. Mapping
`x[k]` to `x[0,k,0,0]` and `W[n,k]` to `W[n,k,0,0]` makes the real-number
expressions identical. The convolution output is `[1,N,1,1]`, not the linear
output `[1,N]`; a rewrite must preserve or explicitly adapt the consumer
shape.

Numerical limitation: both are fp16 dot products, but Apple's convolution and
linear kernels may use different accumulation and rounding orders. This mint
checked compiler acceptance and task decoding only; it did not execute either
program or compare outputs. Validate the actual rewritten model against its
current numerical tolerance before treating the forms as interchangeable.

Status: Apple-decoded rewrite proposal; staged Qwen MIL remains unchanged.
## Scope and remaining work

- The staged Qwen MIL remains unchanged. The conv1x1 rewrite is a proposal;
  its Apple-decoded template is available, but no model conversion or device
  execution is claimed.
- Add decoder rows only from Apple-decoded task streams. Do not infer task words
  from equivalent tensor shapes.
- The reduce rewrites above remain unverified on Apple's compiler. The next
  reduce batch should test rank-3 `reduce_sum` at `[1,16,128]` and the
  rank-3 matmul-ones spelling at `[1,1,2048]` × `[1,2048,1]`.
