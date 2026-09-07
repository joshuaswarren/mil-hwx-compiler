# MIL-to-HWX compiler

This repository contains a research compiler for the H16G Apple Neural Engine
in the M4, an experimental source-native H13/M1 backend, and an
oracle-parity H14/M2 elementwise, matvec, and normalization backend. It reads textual MIL and
emits H16G HWX objects, or H13 and H14 ANEC and HWX packages, without Apple's
compiler.

The project is a canary for the compiler pipeline recovered in *Inside the M4
Apple Neural Engine*, Part 4b. It shows which parts of that pipeline are
understood well enough to reproduce in code and verify on hardware.

## Linux build

This fork builds on Linux with GNUstep Foundation. H16G emits HWX; the
experimental H13 and H14 backends emit ANEC by default and HWX with
`--format hwx`. H13 and H14 performance and mlx-omarchy integration remain
unqualified.

Install Clang and LLD, CMake, Ninja, Make, pkg-config, Git, Python 3, and development packages for libffi, libxml2, ICU, OpenSSL, and zlib. Then run:

```sh
git clone https://github.com/joshuaswarren/mil-hwx-compiler.git
cd mil-hwx-compiler
scripts/verify-linux-compiler.sh all
```

The script builds pinned libobjc2 and GNUstep sources under `$HOME/.local/mil-hwx-gnustep`, builds the compiler, runs the Linux software checks, and emits and inspects Conv1x1/ReLU and W8A8 HWX files. Set `GNUSTEP_PREFIX` to choose another install directory. The bootstrap does not install system packages or require root. An explicit `CXX` path selects the compiler for both bootstrap and subsequent builds.

Use this verifier on Linux. The macOS hardware suites below require Apple runtime interfaces and do not run on Linux.

## Experimental M1/H13 compilation

The H13 path constructs descriptors from named register fields and packs the
model's own weights. It does not rename H16G output, patch a binary template,
or invoke Apple's compiler. The encoded device layouts stay fixed while logical
tensors may use any positive static size:

| Operation | MIL contract |
| --- | --- |
| add, mul, maximum, minimum | Two fp16 inputs, which may name the same value, or one fp16 input plus a same-shape fp16 const tensor on either side. `mul` also expands one inline fp16 scalar. Input, output, and tensor-constant shapes must match and be positive and static. The compiler emits 64-element programs and zero-pads the last program. `add` and `mul` over two rank-4 NCHW operands additionally reach Apple's own one- or two-task program when the geometry is decoded, including a per-channel, spatial, scalar-tensor or batch broadcast and a per-channel BLOBFILE constant; see the broadcast envelope below. |
| relu | One fp16 input and matching output with any positive static shape. Lowers each 64-element slice to `maximum(x, 0)`; the final slice is zero-padded. |
| clip | One fp16 input and matching output with any positive static shape, plus finite fp32 scalar `alpha` and `beta` attributes that are exactly representable in fp16 and satisfy `alpha <= beta`. Lowers each slice to `minimum(maximum(x, alpha), beta)` as two operation-major program groups. |
| sub | One fp16 input `x` and a same-shape fp16 const tensor `y`. The host negates each constant slice and emits the verified add descriptor. Two runtime inputs and `const - input` are rejected. |
| real_div | One fp16 input `x` and a same-shape fp16 const tensor `y`. Every constant element must be a finite, nonzero power of two with an fp16-exact reciprocal. The host stores each reciprocal slice and emits the verified multiply descriptor. |
| matmul | With `transpose_x=false`, fp16 x has shape `[..., K]`, `K >= 1`, and the product of the leading dimensions is the row count M. `transpose_x=true` accepts more than one logical row only where the decoded corpus covers it. The second operand is either a constant rank-2 tensor `[K,N]` with `transpose_y=false` or `[N,K]` with `transpose_y=true`, or a runtime fp16 surface of the same rank as x whose leading axes are 1 — attention's own form, which reaches Apple's own program for the decoded geometries and is refused with `h13.matmul-outside-envelope` otherwise, since only a constant weight can be sliced. The output is `[..., N]`. Outside the parity envelope, N is emitted in 512-output programs with the last program zero-padded, each reduction of at most 512 elements is zero-extended to the 256- or 512-lane descriptor, and larger reductions emit one matvec per 512-element reduction chunk summed with 64-lane add programs. The final tensor then records `"accumulation": "chunked-fp16"`, which is not bit-identical to one unbroken accumulation because each partial sum rounds to fp16. |
| linear | fp16 x has shape `[..., K]`, constant weight has shape `[N,K]`, and the output is `[..., N]`. The compiler lowers `linear(x, weight)` to `matmul(x, weight, transpose_y=true)`. An optional constant fp16 bias vector `[N]`, encoded as an inline list or BLOBFILE, adds one folded 64-lane add group after the matmul. |
| reshape, squeeze, expand_dims | Positive static fp16 input and result shapes with equal element counts. The operation emits no program: its result aliases the input's underlying tensor with a new logical shape. Shape and axes inputs must be constants. Returning an alias of a function input fails with `h13.returned-input-alias` because no program produces an output. |
| softmax | One fp16 input and matching output, and a constant `int32` axis. Emits Apple's own 4-to-8-task program with the exponential (and, for a channel- or height-axis reduction, reciprocal) fp16 lookup table in the constant section. Only the decoded shapes lower; see the envelope table below. |
| layer_norm | One fp16 input and matching output, a constant rank-1 `int32` axes tensor, and `epsilon = fp32(1e-5)`. Emits Apple's 3-to-6-task program with an all-zero 16384-byte constant section. `gamma` and `beta` are rejected: Apple's compiler rejects every affine form in this harness, so no oracle covers one. |
| reduce_sum, reduce_max, reduce_mean | One fp16 input, a constant rank-1 `int32` axes tensor, and `keep_dims` either way. Emits Apple's 1-to-3-task program; a `keep_dims = false` result uses Apple's dense rank-reduced output surface. Reduction over the batch axis is rejected. |
| conv | One fp16 rank-4 `[1, Cin, H, W]` input, a constant `[Cout, Cin/groups, k, k]` fp16 weight, an optional constant fp16 bias `[Cout]`, unit `dilations`, an all-zero explicit `pad`, and `pad_type` `same` or `valid`. Emits Apple's own single-task program whose whole constant section is the permuted weight, with the bias folded in as one row per plane. Kernel 1 and 3, stride 1 and 2, groups 1, 4 and depthwise, `Cin` and `Cout` over 64 to 1024 and spatial 1 to 64 lower where the decoded corpus covers them; anything else is refused with `h13.conv-outside-envelope` (`h14.conv-outside-envelope` on H14). See the convolution envelope table below. |

The MIL contracts follow coremltools commit
[`9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4`](https://github.com/apple/coremltools/commit/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4).
Its
[`reshape` definition](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_transformation.py#L161-L203)
preserves values and element count, its
[`expand_dims` definition](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_transformation.py#L77-L129)
inserts singleton dimensions, and its
[`squeeze` definition](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/tensor_transformation.py#L877-L935)
removes singleton dimensions. The same commit's
[`matmul` definition](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/linear.py)
broadcasts the operands' leading dimensions before appending the matrix
dimensions. Its
[`linear` definition](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/linear.py)
defines `linear(x, weight, bias)` as `x * weight^T + bias`, requires constant
weight and bias inputs, and gives bias the shape `[N]`.
The same commit's
[`conv` definition](https://github.com/apple/coremltools/blob/9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4/coremltools/converters/mil/mil/ops/defs/iOS15/conv.py)
takes a `[Cout, Cin / groups, kh, kw]` weight, an optional `[Cout]` bias,
square `strides` and `dilations` vectors, and either an explicit `pad` with
`pad_type="custom"` or the `same`, `same_lower` and `valid` spellings: `same`
covers `ceil(D / stride)` output pixels with the smaller half of the padding in
front and `valid` covers `floor((D - dilation * (kernel - 1) - 1) / stride) + 1`
with none.

Exactly one function and at least one encoded operation are required. The
compiler lowers operations in their verified MIL order; it does not re-sort
the graph. Any input or intermediate may feed multiple later operations or
multiple operands of one operation. Every non-returned operation result must
feed a later operation. The function must return exactly the last operation's
result. A returned shape alias of that result promotes the underlying tensor to
the output and applies the alias's logical shape.

Programs are ordered by operation and then by logical slice. Producer slices
must exactly tile every stored intermediate without overlapping physical write
ranges. Repeated and fan-out consumer slices are valid, but every consumer
physical range must be fully covered by producer writes and dispatched after
each overlapping producer. Padding beyond a consumer's logical count is
ignored. Unsupported retiling fails with `h13.unsupported-chain`.

The default `--schedule=per-op` keeps these intermediate transfers explicit.
`--schedule=chain` currently accepts only two operations: a supported producer
followed directly by the final `relu`, with all function inputs consumed by
the producer. The producer must use the existing Apple-parity matvec or
whole-tensor elementwise lowering. This emits one program without an
intermediate buffer and enables ReLU only on the producer's final task.
Oracle checks cover add at `[1,512,1,1]` and two-task matmul at
`[1,64,256] × [256,256]`. Other chain edges fail with
`h13.chain-unrepresentable-edge` or `h14.chain-unrepresentable-edge`;
non-straight-line graphs and incompatible boundary inputs fail with
`chain-outside-envelope`. General intermediate DMA/L2 routing is unresolved.
Descriptor parity and host simulation do not qualify native M1 execution.

Run the host-only checks with `make test-h13` (set `GNUSTEP_PREFIX` on Linux).
They cover encoding, coefficient packing, serialization, and the MIL CLI;
they do not execute ANE commands. A minimal compilation example is:

```sh
cat > /tmp/h13-add.mil <<'MIL'
program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1,64,1,1]> a, tensor<fp16, [1,64,1,1]> b) {
    tensor<fp16, [1,64,1,1]> y = add(x = a, y = b)[name = string("sum")];
  } -> (y);
}
MIL
./build/mil-hwxc --target H13 --mil /tmp/h13-add.mil --model-root /tmp --output build/h13-add
```

The output directory must be absent or empty. By default it receives one
`program-N.anec` file per operation slice. `--format hwx` instead writes
`program-N.hwx` with the same task and constant content. The v1 manifest always includes a
`programs` array, numeric `dispatchPlan`, and `tensors` object. Each stored
tensor records its full logical shape, byte count, and input, output,
intermediate, or constant role. A shape-only result has its own tensor record
with `aliasOf` naming the underlying tensor. Program bindings always name that
underlying tensor. `intermediates` lists every non-returned operation result,
including aliases. A sliced binding records its tensor name, element offset,
and logical element count. A padded slice also records `physicalElements`;
omitting it means the physical and logical counts match. Pack operations
zero-fill physical padding, and unpack operations discard it. An unsliced
binding keeps the original binding fields. A single-program package also keeps
the original top-level program fields for existing readers. Each program record
contains its local logical shape, physical strides, buffer indices, allocation
sizes, and constant bindings.

ANEC uses a 0x1000-byte header, a 0x274-byte descriptor, and constants at
content offset 0x280. Older libane readers expecting a 0x800-byte header cannot
consume this format. The schema remains `mil-hwxc.h13-anec-package.v1`, not an
`ANEExecutableBundle` or an mlx-omarchy bundle. The device-free reader validates
every container, tensor, binding, and slice. It also checks exact intermediate
tiling and every producer-to-consumer dispatch dependency. Packing and unpacking
convert dense little-endian fp16 bytes to and from the physical channel layout.

```bash
python3 research/inspect_anec.py build/h13-add
python3 research/inspect_anec.py build/h13-add --pack-input a input.fp16 --output input.buffer
python3 research/inspect_anec.py build/h13-add --unpack-output y output.buffer --output output.fp16
```

Conversion refuses incorrect byte counts and existing output paths. When one
program binding references the selected tensor, `--output` names one buffer.
When several bindings reference it, packing writes one
`program-N.NAME.buffer` file per binding under the `--output` directory;
unpacking reads the same directory form and reassembles one dense output file.
Inspection reports encoded command, input, and output allocation totals,
excluding driver and runtime overhead. It does not validate task instructions
or establish device safety. Device dispatch is handled by the validated runner below.

### H13 reference and Linux dispatch

`tools/h13_reference.py` evaluates the accepted H13 MIL subset without NumPy.
It reads and writes dense little-endian fp16 files. Matmul accumulates in
float32 and rounds once to fp16. Reductions larger than 512 elements on H13
round each chunk before the add chain, so tensors marked `chunked-fp16` use
`|device-reference| <= 0.02 + 0.02 * |reference|`; other outputs must be
byte-for-byte equal to the fp16 reference.

```bash
python3 tools/h13_reference.py model.mil --model-root models \
  --input x=input.fp16 --output y=reference.fp16
python3 tools/h13_run_linux.py build/h13-package --mil model.mil \
  --model-root models --input x=input.fp16 --output y=device.fp16 --dry-run
```

The Linux runner imports `research/inspect_anec.py` and validates the complete
package, dense inputs, MIL returns, bindings, dispatch plan, and reference before
it opens libane. Dry-run performs those checks without loading libane, writing an
output, or reporting native timings. Native execution resolves the selected
`--libane-library` ABI: `pyane_init`, `pyane_free`, `__ane_src_size`,
`__ane_dst_size`, `__ane_send`, `__ane_read`, and `ane_exec`. It validates
the returned surface sizes before the first transfer and writes outputs only after
every declared output passes.

`--benchmark-json PATH` enables a correctness-gated native benchmark;
`--warmup N` and `--iterations N` require that flag and are rejected with
`--dry-run`. The defaults are three validated warmups and ten measured
iterations. The report separates setup from the timed work. Per-program samples
cover transfer, submission, readback, and their total; end-to-end samples span
the first transfer through the last readback and include Python intermediate
composition between programs. Final output unpacking, numerical checks, reference evaluation and
program setup are outside every measured window. See
`tests/h13_first_run/RUNBOOK.md` for reviewed identity and exact-package
provenance requirements.

### H13 Apple-parity programs

`make test-h13-parity` compiles all 635 decoded H13 oracles in both formats
and compares each emitted task's header words and register writes, the program
descriptor, the tensor descriptors, and the constant-section SHA-256 with
`research/oracles/h13/*.json`. 87 are elementwise, scalar-constant, and unary
cases (`"encoder": "h13-oracle-parity"`); 216 are matmul, 113 are broadcast
(`"encoder": "apple-parity-broadcast"`), and 219 are softmax, layer_norm, and
reductions (`"encoder": "apple-parity-norm"`).

The matmul cases split by where the second operand lives:

| Form | Encoder | Cases | Envelope |
|---|---|---:|---|
| Constant weight, `transpose_y=true` | `apple-parity-matvec` | 144 | M in {1, 2, 8, 16, 32, 64, 128, 144, 160, 176, 192, 224, 256, 512}, K and N in {256, 512, 1024, 2048, 4096, 8192}, rank 2 and rank 3, plus `transpose_x=true` at M in {1, 16, 32, 128, 256, 512} |
| Both operands runtime | `apple-parity-matmul` | 72 | M in {16, 32, 48, 64, 96, 128, 192, 256, 384, 512}, K and N in {32, 64, 128, 256, 512, 1024, 2048, 4096}, all four transpose combinations, rank 2 and rank 3 |

Both operands runtime is attention's own shape — `QK^T` and `PV` — and it is
the form the first campaign never tested. Apple stages the second operand as
its own surface: the object carries three surfaces instead of two, the
constant section is 16 KiB of zeros rather than a weight, and the package
binds the second operand on ANEC channel 5 and `x` on channel 6, matching the
order Apple declares them in. `transpose_y=false` costs one task fewer than
`transpose_y=true` at the same geometry, the opposite of the constant-weight
case, where `transpose_y=false` is refused outright: three fresh probes
(`rrmm_r2rb_m1_k256_n256_tx0_ty0`, `..._m32_k2048_n2048_tx0_ty0`,
`..._m128_k2048_n2048_tx0_ty0`) still return `ANE internal validation error:
Metadata data type does not match requested type.`, so the compiler keeps
lowering a `transpose_y=false` constant weight by transposing it on the host
and emitting the `transpose_y=true` program Apple accepts.

A parity matvec program differs from the source-qualified one in four ways:
it runs Apple's 126-word preparation task before the 157-word compute task,
its surfaces are dense `[1, 1, M, W]` fp16 rows (padded up to the 64-byte row
floor) instead of 64-byte lanes, its object carries Apple's `__DATA/__bss`
scratch with the output surface laid out below the inputs, and its constant
section is the `K * N` fp16 weight in Apple's packing permutation instead of a
512-row padded block. Above 128 x rows — the row count at which Apple starts
partitioning one program into `1 + K * N / 2^19` tasks, up to 129 of them —
that permutation also caps its row group at `32768 / K` halfwords: M = 128
with K = 4096 still interleaves 16 weight rows per group and M = 144
interleaves 8. `research/mint_rrmm_probes.py` mints the runtime-runtime,
extended-matvec, and broadcast probes, emits
`plugins/H13/H13EnvelopeTemplates.inc`, and refuses to emit a template whose
surfaces, scratch, binding order, or constant section it cannot reproduce.

Geometries outside the decoded grid keep the source-qualified encoder, and a
matmul with both operands runtime outside it is refused with
`h13.matmul-outside-envelope` rather than sliced: the chunked plan can only
pack a constant weight. Chunked and column-sliced plans are still built on top
of parity programs when a slice lands on a decoded geometry.

#### Broadcast elementwise envelope

`add`, `mul`, and `sub` over two NCHW surfaces lower to Apple's own one- or
two-task program, so a convolution or linear bias add and a per-channel or
spatial scale are each a single program instead of a tiled 64-lane plan:

| Operand form | Example | Tasks | Constant section |
|---|---|---:|---|
| Identical shapes, runtime | `[1, 3072, 1, 1] add [1, 3072, 1, 1]` | 1 | 16 KiB zero |
| Per-channel, runtime | `[1, 768, 16, 16] add [1, 768, 1, 1]` | 2 | 16 KiB zero |
| Spatial, runtime | `[1, 96, 8, 8] mul [1, 1, 8, 8]` | 2 | 16 KiB zero |
| Scalar tensor, runtime | `[1, 64, 16, 16] add [1, 1, 1, 1]` | 2 | 16 KiB zero |
| Batch broadcast, runtime | `[8, 64, 8, 8] add [1, 64, 1, 1]` | 2 | 16 KiB zero |
| Inline fp16 scalar | `[1, 768, 8, 8] mul fp16(0.5)` | 1 | 16 KiB zero |
| Per-channel BLOBFILE constant | `[1, 768, 16, 16] mul const [1, 768, 1, 1]` | 2 | `4 * C` bytes |

Any shape difference costs one extra task; the channel count does not change
it. Channel counts 64, 96, 200, 256, 300, 512, 768, 1024, 3072, 8192 and
16384 are covered, batches 1, 2, 4 and 8, and spatial 1x1, 8x8, 16x16 and
32x32. A per-channel constant is folded into the constant section as one fp16
bias per channel followed by one fp16 scale per channel: `add` stores the
constant as the bias and 1.0 as the scale, `mul` leaves the bias zero and
stores the constant as the scale. A batched surface records one batch
element's span as its tensor-descriptor size while the object reserves and
spaces the whole batch, so a runner must size buffers from
`allocationBytes`, never from the descriptor.

#### Softmax, layer_norm, and reduction envelope

`softmax`, `layer_norm`, `reduce_sum`, `reduce_max`, and `reduce_mean` lower
directly to Apple's own multi-task programs instead of decomposing into the
tiled elementwise path. The encoder selects a program by
`(operation, input CHW, reduced-axis mask, keep_dims)`, and 186 templates in
`plugins/H13/H13NormTemplates.inc` cover the whole decoded corpus:

| Operation | Reduced axes | Tasks | Constant section | Shapes |
|---|---|---|---|---|
| `softmax` | C | 4-5 | 256 / 1152 / 2176 B, exp + reciprocal LUT | flat `[1, C, 1, 1]`, sequence `[1, C, 1, W]`, spatial `[1, C, S, S]` |
| `softmax` | H | 5 | 640 / 1152 / 2176 B, exp + reciprocal LUT | spatial `[1, C, S, S]` |
| `softmax` | W (last axis) | 5-8 | 128 / 1024 / 1536 / 2048 B, exp LUT | flat, sequence, spatial, and attention scores `[1, H, S, S]` |
| `layer_norm` | C | 5-6 | 16384 B zero | flat, spatial |
| `layer_norm` | W, HW, CHW | 3-5 | 16384 B zero | flat, sequence, spatial |
| `reduce_sum`, `reduce_mean` | C | 1-2 | 16384 B zero | flat, sequence, spatial |
| `reduce_max` | C | 1 | 16384 B zero | flat, sequence, spatial |

### Convolution envelope

Every row is proven byte-for-byte against decoded Apple oracles on both H13 and
H14: the task stream comes from the oracle and the constant section from the
packing `research/mint_conv_probes.py` derives, whose permutation is read off
known-weight probes rather than inferred from a hash. 284 decoded convolutions
per target lower inside the envelope.

| Form | Covered | Constant section |
| --- | --- | --- |
| Dense, stride 1 | kernel 1 and 3, `Cin` and `Cout` over 64, 128, 256, 512, 768 and 1024, spatial 1, 8, 16, 32 and 64, `pad_type` `same` and `valid` | 16 plane groups of planes that interleave `lanes` output channels at halfword granularity; `lanes` is the largest power of two at or below `Cout / 16`, capped at 32 for an input of 8 or fewer pixels per side and 16 above it, and the remaining channels take the next chunk down |
| Grouped, stride 1 | groups 4 over the same kernels and channel counts | 64 planes of `Cout / 64` channels, each plane padded to 64 bytes |
| Depthwise, stride 1 | groups = `Cin` = `Cout` over the same kernels and channel counts | 16 lanes, each holding `Cout / 16` channels' taps back to back, the lane padded to 64 bytes |
| Bias | any covered form, `[Cout]` BLOBFILE bias | one bias row per plane ahead of the weight rows, so the section grows by a plane row rather than by `Cout * 2` |
| Stride 2 | kernel 1, groups 1, up to 8 interleaved lanes | zero-skipping planes: an optional dense bias prefix, a 16-bit body count, then per reduction step a mask byte per eight lanes followed by the nonzero halfwords. Both signed zeros are skipped, so the section size depends on the weight values |

Outside those rows the compiler refuses by name. The campaign measured, but did
not resolve, four boundaries:

| Unresolved | What the corpus shows |
| --- | --- |
| Multi-task convolutions | 57 grid points per target where Apple partitions one convolution into 2 to 32 tasks; the per-task constant partition is not derived |
| Stride 2 with a 3x3 kernel | The taps reorder by stride phase — `(0,2)`, `(6,8)`, `(1,7)`, `(3,5)`, then the centre tap — and eight slots share one mask byte, but the per-row overhead is 2 bytes short of the measured body count |
| Stride 2 with 16 or more lanes | The mask and value layout matches the 8-lane form, and the measured body count is 2 bytes below the model's |
| Stride 2, grouped and depthwise | 68 grid points per target, unprobed at byte level |
| all three reductions | H | 3 | 16384 B zero | spatial |
| all three reductions | W, HW, CHW | 1 | 16384 B zero | sequence, spatial |

The swept extents are C in {64, 128, 256, 512, 1024, 2048, 4096} for the flat
form, W in {32, 64, 128, 512} for the sequence form, (C, S) in
{(32, 4), (64, 8), (128, 16)} for the spatial form, and
(H, S) in {(1, 64), (1, 256), (8, 64), (8, 128), (12, 64)} for attention
scores. `keep_dims = false` is covered as its own template because Apple writes
a rank-reduced result to a dense `[1, 1, 1, N]` surface rather than the
channel-major one.

MIL rank is normalized before lookup, so the rank-2 and rank-3 forms a
transformer emits reach the same program as the rank-4 surface they collapse
onto: `softmax(axis = -1)` on `[1, 512]` emits the same bytes as `axis = 3` on
`[1, 1, 1, 512]`. Anything else is refused with `h13.norm-outside-envelope`:
a shape off the sweep, an epsilon other than 1e-5, a non-constant axes
operand, a reduction over the batch axis, and `gamma`/`beta` — Apple's compiler
rejects every affine `layer_norm` form in this harness, in seven variants
tried, so no oracle exists to match.

Parity is on bytes; numerics are a separate contract. Apple evaluates softmax's
exponential and reciprocal through the fp16 lookup tables in the constant
section, so device output differs from `tools/h13_reference.py`, which
accumulates reductions, softmax sums, and layer_norm statistics in float32 and
rounds once to fp16. `research/mint_norm_probes.py` mints the sweep and emits
the template table; [`research/h13-td-fields.md`](research/h13-td-fields.md)
documents the constant-section layout, the surface formula, the resolved
scaling words, and the words that stay unresolved.

### H13 HWX and macOS aned gate

Pass `--format hwx` to emit a loadable H13 Mach-O object for each program
instead of an ANEC container. The manifest schema stays
`mil-hwxc.h13-anec-package.v1` and records `"artifactFormat": "hwx"`.
`research/inspect_hwx.py` validates the H13 subtype, program descriptor,
tensor channels, relocations, and embedded task/constant layout. It can also
reconstruct the matching ANEC bytes:

```bash
./build/mil-hwxc --target H13 --format hwx --mil model.mil \
  --model-root models --output build/h13-hwx
python3 research/inspect_hwx.py build/h13-hwx/program-0.hwx
python3 research/inspect_hwx.py --extract-anec \
  build/h13-hwx/program-0.hwx /tmp/program-0.anec
```

On an arm64 Mac, the hardware gate compiles the package, computes dense fp16
reference outputs, provisions every HWX object under the current aned
`InMemoryModelCache/h13_exec` key, and loads and runs each program through
`ANEProvisionedRuntime`. It stops before compiling or dispatching when that
cache directory is not writable.

```bash
tests/run_h13_hardware.sh model.mil models x=input.fp16
```

On an aarch64 Linux host with `ane.ko` loaded and the `omarchy` branch of
`joshuaswarren/omarchy-ane` built (libane and `bindings/python/dylib`), the
Linux gate compiles the ANEC package and dispatches it through
`tools/h13_run_linux.py`, comparing every output with the same reference.
It refuses to run until the device node, library, branch, and ANEC header
contract all check out.

```bash
ANE_CHECKOUT=~/src/omarchy-ane tests/run_h13_linux_hardware.sh model.mil models x=input.fp16
```

Elementwise outputs must match the reference bit for bit. A tensor marked
`chunked-fp16` uses
`abs(device-reference) <= 0.02 + 0.02 * abs(reference)` because partial
matmul results are rounded to fp16 between chunks. The decoded H13 fields and
the remaining unknown field meanings are listed in
[`research/h13-hwx-fields.md`](research/h13-hwx-fields.md).

Before hardware use, qualify the matching driver, BAR/kernel binding contract,
and DMA spans on an ANE-enabled base M1. Compilation on an M1 Ultra host is
not proof that the emitted program executes on a base M1. General model
coverage, program fusion, and device latency optimization remain unfinished.
The 512-wide encoder omits 512 KiB of unreachable coefficient padding:
the sixteen KDMA ranges address only 512 KiB. Both reductions therefore
emit 529,024-byte ANEC files, with 33 command tiles.
This is a file-size/allocation reduction, not a measured device speedup.

`python3 research/inspect_hwx.py FILE.hwx` identifies H14/M2 (ISA 11) and
H16G/M4 (ISA 17) from their headers, and decodes H14 task streams against the
published register ranges and record lengths. Run `make test-hwx-inspection`.

## Experimental M2/H14 compilation

`--target H14` emits H14 task streams, ANEC containers, and HWX objects for
the elementwise, matmul, normalization, and reduction families Apple's own
compiler was sampled on. Every emitted task is word-for-word identical to the
decoded oracle for that case: `make test-h14-parity` compiles 434 cases in
both formats, decodes the emitted stream with `research/h13_td.py`, and
compares each task's header words and register writes, the program descriptor,
the tensor descriptors, and the constant-section sha256 with
`research/oracles/h14/*.json`. 165 are elementwise, scalar-constant, unary,
and broadcast cases (`"encoder": "h14-oracle-parity"`); 36 are every decoded
`matmul` and 14 are known-weight matmul probes
(`"encoder": "apple-parity-matvec"`); 105 softmax and layer_norm and 114
reduction cases collapse to 186 templates
(`"encoder": "apple-parity-norm"`).

```bash
./build/mil-hwxc --target H14 --format hwx --mil /tmp/h14-add.mil \
  --model-root /tmp --output build/h14-add
python3 research/inspect_hwx.py build/h14-add/program-0.hwx
```

The parity envelope is exactly the decoded points, and nothing is
interpolated between them:

- `add`, `mul`, `maximum`, `minimum`, `sub` with two fp16 tensor operands at
  `1x64x1x1`, `1x128x1x1`, `1x256x1x1`, `1x512x1x1`, `1x1024x1x1`,
  `1x2048x1x1`, `1x4096x1x1`, `1x64x8x8`, `1x128x16x16`, and `1x3x224x224`,
  plus the envelope campaign's channel counts and spatial shapes.
- `add` and `mul` broadcasting a runtime `1x1x1x1`, `1xCx1x1`, or `1x1xHxW`
  operand against a `1xCxHxW` result: 36 decoded broadcast programs whose
  second input surface differs from the result surface.
- The same operations plus `real_div` against the fp16 scalar constant `0.5`
  at `1x64x1x1` and `1x512x1x1`. Scalar `real_div` lowers to two tasks.
- `abs` at C = 64, 128, 256, 512, 1024, 2048, and 4096, and `exp`, `gelu`
  (`EXACT`), `leaky_relu` (`alpha = 0.125`), `relu`, `rsqrt`
  (`epsilon = 1e-6`), `sigmoid`, `silu`, `sqrt`, `tanh` at C = 64 and 512,
  plus `gelu` and `silu` at the envelope's spatial shapes. `gelu` is three
  programs, not one: `EXACT`, `SIGMOID_APPROXIMATION`, and
  `TANH_APPROXIMATION` each carry their own decoded task stream.
- `matmul`, and `linear` without a bias, at M in {1, 2, 8, 64} and K, N in
  {256, 512, 1024}, detailed below.
- `softmax`, `layer_norm` without gamma or beta, `reduce_sum`, `reduce_max`,
  and `reduce_mean`, detailed below.

Anything else is rejected with `h14.outside-parity-envelope`, and a
normalization or reduction outside the decoded geometries with
`h14.norm-outside-envelope`. Tensors lower through their literal NCHW
geometry, or their channel-flattened form when that is the decoded point.

`plugins/H14/H14ElementwiseTemplates.inc`,
`plugins/H14/H14MatvecTemplates.inc`, and
`plugins/H14/H14NormTemplates.inc` hold the decoded task words,
constant-section runs, and per-case program-descriptor metadata. They are
generated, never edited by hand:

```bash
python3 research/generate_h14_templates.py           # regenerate
python3 research/generate_h14_templates.py --check   # fail when stale
python3 research/mint_h14_norm_probes.py --emit-templates           # regenerate
python3 research/mint_h14_norm_probes.py --emit-templates --check   # fail when stale
```

`tests/test_h14_parity.py` runs the norm staleness check itself, so a
hand-edited `H14NormTemplates.inc` fails `make test-h14-parity`.

### H14 Apple-parity matvec programs

`matmul` covers the decoded grid M in {1, 2, 8, 64} and K, N in
{256, 512, 1024}: 36 two-task programs whose x surface is one dense
`[1, 1, M, K]` fp16 plane, whose output is the same form at N columns, and
whose constant section is the `K * N` fp16 weight in Apple's packing
permutation. Inputs and outputs keep ANEC channels 5 and 4, and each program
carries its own `constantOffset` at `align_up(text_bytes, 64)`. Leading batch
dimensions collapse into M, because the surface is the same contiguous plane
however the MIL shape spells the batch.

`transpose_y=true` is the decoded form and encodes natively.
`transpose_y=false` is rejected by Apple's compiler, so its `[K, N]` weight is
transposed exactly on the host and fed to the same encoder; the test proves
both forms of one non-square matrix emit a single identical artifact.
`linear` without a bias lowers to the identical program. A `linear` bias is
rejected with `h14.linear-bias-unsupported`, because this backend emits one
program per artifact and a bias needs a second elementwise program.
`transpose_x=true` is rejected with `h14.transpose-x-unsupported`, and any
geometry off the grid with `h14.outside-parity-envelope`.

The campaign's uniform fp16 `0.5` weight cannot tell one permutation from
another, so `research/mint_h14_matvec_probes.py` mints 125 known-weight H14
matmul oracles (`research/oracles/h14/h14mv_*.json`, all 125 decoded by
Apple) that pin it. Its `--verify` mode, which `make test-h14-parity` also
runs, rebuilds every recorded constant section byte-for-byte, and the 14
probes whose geometry the encoder covers are compiled and compared like any
other oracle, which proves the packing over all nine (K, N) grid points
against a nonuniform weight. The permutation is the H13 one unchanged: every
uniform-weight H14 section has the same sha256 as its H13 twin. The appendix
in [`research/h14-td-fields.md`](research/h14-td-fields.md) records the
permutation, the blob-header bytes Apple's compiler packs ahead of the
payload, the 1 KiB minimum section, and what stays unresolved.

```bash
python3 research/mint_h14_matvec_probes.py --host macstudio  # re-mint (macOS host)
python3 research/mint_h14_matvec_probes.py --verify          # offline re-check
```

### H14 Apple-parity normalization and reduction programs

`softmax`, `layer_norm`, `reduce_sum`, `reduce_max`, and `reduce_mean` encode
Apple's own multi-task streams for the decoded geometries: the flat
`[1, C, 1, 1]` sweep for C in 64..4096, the sequence shapes, the
`[1, H, S, S]` attention scores, the spatial shapes, and reductions over the
channel, height, width, and all-non-batch axes, with and without `keep_dims`.
219 decoded oracles collapse to 186 `(operation, input CHW, axis mask,
keep_dims)` templates. A program takes one input on ANEC channel 5 and writes
channel 4, and its task count follows the reduced axis rather than the shape:
one task for a channel-axis or last-axis reduction, three for `layer_norm`
over every non-batch axis, five or six for `softmax`.

`layer_norm` must carry no gamma or beta and `epsilon` must be `1e-5`: Apple's
compiler rejects every affine form on H14 exactly as it does on H13, so no
oracle covers one. All 17 rejected `h14norm_*_affine` records are checked in
with their `callback_status=1` error.

Softmax constant sections are the same LUT words the H13 encoder already
carries, so `plugins/H14/H14Program.cpp` includes
`plugins/H13/H13ElementwiseConstants.inc` instead of restating them. That
reuse is proven, not assumed: the generator resolves every decoded section by
sha256 against those words before emitting a kind, and refuses a section it
cannot rebuild. A last-axis softmax carries the exponential table alone; a
channel-axis or height-axis softmax also carries the reciprocal table in its
final 128 bytes; `layer_norm` and every reduction leave the section zero.

```bash
python3 research/mint_h14_norm_probes.py --host macstudio  # re-mint (macOS host)
python3 research/mint_h14_norm_probes.py --emit-templates  # regenerate the table
```

### H14 task stream, ANEC, and HWX layout

An H14 task has eight header words; `header[0]` carries
`task_words << 16 | task_id`, and there is no H13-style next-task pointer.
Register records use word-index bases at the old non-Common block addresses
(Common `0x0000`, L2 `0x0500`, PE `0x0900`, NE `0x0d00`, TileDMA source
`0x1100`, TileDMA destination `0x1500`, KernelDMA `0x1900`); dense records
carry a count minus one in bits 15:20 and scatter records set bit 31 plus a
16-bit following-word mask in bits 15:30. `__TEXT/__text` holds one
16-byte zero-size task frame and then every task 16-byte aligned, with the
last task unpadded, which is what the program descriptor's text word count
at command offset `0x824` measures.

The H14 ANEC form is the same 0x1000-byte header H13 uses, followed by the
H14 task stream at `0x1000` and the constant section at
`align_up(text_bytes, 64)` after it. The header fields are content size,
first task byte length, task count, task-stream byte length, constant byte
length, input count, output count, 32 tile counts, and 32 six-field tensor
layouts, with channel 4 the output and channels 5 and 6 the inputs. An H14
reader walks the stream by task-header size and 16-byte alignment, skipping
zero-size frames; the first-task field is informational. A future
omarchy-ane M2 libane consumes this form. `research/inspect_anec.py` stays
H13-only.

The HWX container is Mach-O `cpusubtype` 5 (ISA 11) with the H13-style
layout: surfaces as `__FVMLIB` segments from `0x30000000` in 16 KiB tiles,
then `__TEXT/__text` and `__TEXT/__const`. Its program descriptor is a
0x890-byte kind-4 command carrying the text address at `0x10`, the constant
address at `0x20`, resource slots 0 to 3 zero with `0x30000000` in slot 4,
the text word count at `0x824`, the task count at `0x830`, and a 32-word
trailer whose function name is `main`. Tensor descriptors match H13
byte-for-byte in every field the oracle campaign decodes.

### Unresolved H14 words and envelope limits

The oracles retain decoded fields and hashes, never HWX bytes, so these parts
of the container are not evidence-backed:

- The single 16-byte zero-size frame is required by the text word count, but
  the oracles cannot say whether Apple places it before the first task or
  after the last; both decode identically. This emitter places it first.
- The descriptor word at `0x860` (`0x10`, `0x11`, or `0x15` here; `0x15` in
  every matvec case) and the word at `0x880` (22 distinct values here, 15 of
  them matvec) have no resolved formula. Parity carries the decoded value per
  case. `0x880` tracks surface allocation and `0x860` tracks task structure,
  but no sampled formula reproduces either across shapes: matvec `0x880` is
  mostly a function of (K, N), yet M = 64 shifts every value and
  M = 8, K = 512, N = 1024 is `0x1f` where the same (K, N) is `0x1e` at every
  other row count.
- The descriptor word at `0x858` is zero in every decoded elementwise,
  matvec, convolution, softmax, and reduction program, and nonzero in exactly
  13 of the 219 normalization records: the three-task `layer_norm` over every
  non-batch axis on a multi-channel surface, where it equals the input surface
  byte count. The same form on a one-channel surface leaves it zero, and it
  shifts no surface address. Parity carries the decoded value per template,
  and the elementwise and matvec generators refuse an oracle whose `0x858` is
  nonzero rather than dropping it silently.
- Two decoded envelope forms have no encoder: 20 batched `env_bcast_*`
  records (`N = 2` and `N = 8`), whose descriptor records one batch element's
  bytes against a rank-4 shape, and 8 `BLOBFILE`-operand broadcast records,
  whose constant section is not the operand payload (a `[1, 64, 1, 1]` fp16
  `0.5` operand yields a 256-byte section with 72 nonzero bytes) and for which
  no packing rule is proven.
- The descriptor word at `0xc` is not decoded at all; this emitter writes
  `0x222`, the value the other kind-4 descriptor uses.
- The campaign records no `__text` relocation table for H14, so none is
  emitted. Buffer-reference commands, surface segments, symbol and string
  tables, and the compiler-metadata command follow the H13 container.
- Per-register meanings and the unresolved word counts per block are in
  [`research/h14-td-fields.md`](research/h14-td-fields.md): Common 7 of 19,
  L2 15 of 25, PE 2 of 5, NE 2 of 5, TileDMA source 6 of 53, TileDMA
  destination 3 of 10, and KernelDMA 54 of 70 remain unresolved.
- 44 matvec register words stay unresolved and are carried verbatim per
  (M, K, N) template: t0 `0x001c`, `0x0028`, `0x0038`, `0x0508`, `0x0510`,
  `0x0514`, `0x0518`, `0x0538`, `0x111c`, and t1 `0x0038`, `0x050c`, `0x0534`,
  `0x1514` plus the 31 KernelDMA words `0x195c`..`0x19d4`, together with
  header words t0:h1, t0:h6, and t1:h1. The 16 coefficient base/size words
  dominate the gap, so nothing is interpolated between grid points.

No H14 program in this repository has been executed on M2 hardware. Parity
with Apple's emitted stream is not proof of execution.

## macOS build and hardware tests

### Requirements

- An Apple silicon Mac. Hardware results in this repository were measured on an
  M4 (`Mac16,10`).
- macOS and the Xcode Command Line Tools, which provide `clang++`, `make`, the
  Foundation SDK, and the IOSurface SDK.
- macOS 26.3 build `25D125` for the recorded H16G hardware results. Other
  releases may use different private interfaces or descriptor layouts.
- Administrator access for hardware tests. Compilation and software tests do
  not require `sudo`.

There are no third-party package dependencies.

Install the command-line tools if needed:

```sh
xcode-select --install
```

Clone the repository and run the software test suite:

```sh
git clone https://github.com/maderix/mil-hwx-compiler.git
cd mil-hwx-compiler
make test -j4
```

Build only the compiler:

```sh
make build/mil-hwxc -j4
```

Compile the included Conv1x1 and ReLU fixture:

```sh
./build/mil-hwxc \
  --mil tests/fixtures/conv_relu.mil \
  --model-root tests/models/conv_relu \
  --target H16G \
  --output build/conv-bundle
```

The output directory contains one or more `program-N.hwx` files and a
`manifest.json` file. The manifest records dispatch order, tensor bindings,
physical strides, and shared intermediate surfaces.

To compile and run the Chunked DeltaNet block on an M4:

```sh
sudo -v
bash tests/run_chunked_deltanet_hardware.sh
```

The script builds the compiler and runner, emits 58 HWX programs, provisions
them in the macOS `aned` cache, runs two numerical cases, and prints a warm
latency measurement. The script exits with a failure if compilation,
provisioning, execution, or either output comparison fails.

To compile and run the FP16 attention graph:

```sh
sudo -v
bash tests/run_online_reduction_hardware.sh
```

This emits three programs and checks all 16,384 output elements against the
CPU reference. To compare it with the output of Apple's compiler on the same
MIL, inputs, and synchronous runtime path:

```sh
ANE_BENCHMARK_WARMUP=50 \
ANE_BENCHMARK_ITERATIONS=5000 \
ANE_BENCHMARK_BATCHES=5 \
bash tests/run_fa2_ab_hardware.sh
```

## Scope

The repository provides:

- A compiler path from textual MIL to fresh H16G HWX objects.
- Hardware tests with independent CPU references for every supported family.
- Encoders for the decoded HWX container and Task Descriptor fields.
- A small runtime that loads provisioned objects and submits them through
  `_ANEClient`.

Coverage is limited to the operation and shape rows listed below. The target
tables were measured on one M4 and one macOS build. This compiler is not a Core
ML or MLX replacement, and the private runtime is unsuitable for App Store
software.

## Compiler pipeline

```text
MIL source
  -> lexer and parser
  -> typed SSA graph
  -> normalization and decomposition
  -> structural fusion
  -> H16G legality and numeric-mode selection
  -> tiling, liveness, SRAM, and DMA planning
  -> H16G task encoding and program composition
  -> HWX object and binding manifest
```

The production path does not load an Apple-compiled HWX file, choose code from
a fixture name, or patch an existing container. `HWXObjectWriter` creates each
object from compiler data structures. The compiler reparses the completed
object before accepting it.

Multi-operation graphs are partitioned according to target capability tables.
Compatible adjacent tasks can share one HWX program. Other tasks remain
separate programs connected through manifest-managed IOSurfaces. Unsupported
operations, shapes, data types, and axes fail compilation. A transition that
cannot be composed stays on the standalone program path.

Composition has two forms. Simple elementwise operations can be folded into a
field of an adjacent task. Longer chains use operation-specific SRAM input and
output forms so an intermediate remains inside one program. The planner selects
these forms from operation, shape, data type, bridge state, and value lifetime.
It does not select them from a model or function name.

## Coverage and measured results

### Supported operation families

| Family | Measured H16G coverage |
|---|---|
| Conv1x1 | 16 FP16 channel and spatial geometries, with optional ReLU |
| Regular convolution | C64/C128, S32/S64, K3/K5 |
| Depthwise convolution | C64/C128/C256/C512, S64, K3 |
| Matmul | Square N128/N256/N512 and tiled multiples of 128 through N4096 |
| Binary ALU | Add at N128 through N2048; multiply at N128/N512; max/min at N512 |
| Unary and LUT | ReLU, sigmoid, tanh, GELU, SiLU, exp, log, sqrt, rsqrt, reciprocal |
| Reduction | Sum, mean, and max over measured channel, height, and width axes |
| Layout | S2D and D2S B4/B8, including 64-byte physical row padding |
| Fused layout | S2D, Conv1x1, and D2S at natural C8/C16/C24/C32 |
| W8A8 | Four-layer C64/S64 Conv1x1 chain with packed middle blocks |
| Attention | H4/S64/D64 decomposition; unmasked FP16 S128/D128 forward graph in three programs |
| State updates | Four-step N128 affine scan; FP16 Chunked DeltaNet block at C128/D128 |

The Chunked DeltaNet fixture uses ordinary matmul, add, multiply, and exp
operations. The caller supplies normalized Q and K tensors, the transposed K
layout, and the fixed triangular matrices. No DeltaNet operation name reaches
the planner or H16G encoders.

### M4 multi-operation results

| Graph | HWX programs | CPU-reference result | Research latency | Apple compiler | ANE power capture |
|---|---:|---|---:|---:|---|
| FP16 attention, S128/D128 | 3 | 16,384/16,384 elements passed; maximum error `0` | `369.458 us` | `127.208 us` | 61/140 active; 601.9 mW active average; 681 mW peak |
| Matmul, reshape, GELU, N256 | 1 | Maximum error `0.00610352` | `66.94 us` | `76.60 us` | 8/60 active; 36-134 mW; GPU 0 mW |
| Four-step FP16 affine scan, N128 | 8 | Every stage within 1 FP16 ULP; final output within 3 ULP | Not measured | `InvalidMILProgram` | 60/60 active; 37-152 mW |
| Chunked DeltaNet block, C128/D128 | 58 | Output relative L2 `0.004214`; final-state relative L2 `0.004443`; maximum error `3.69e-05` | `22728.792 us` | `InvalidMILProgram` | 73/80 active; 19-403 mW; 86.34 mW average; GPU 0 mW |

The attention graph is split at its two external-surface boundaries. Program 0
runs matmul and scale. Program 1 runs reduce-max, subtraction, exponentiation,
reduce-sum, reciprocal, and multiply while keeping its intermediate values in
SRAM. Program 2 runs the final matmul.

The attention A/B used 50 warmups and five alternating batches of 5,000
evaluations per compiler, for 25,000 measured samples each. Both implementations
used the same MIL, inputs, IOSurfaces, QoS, and synchronous completion boundary.
The research compiler took 2.986 times the Apple compiler median. A separate
research-only profile measured median submission times of 114.458, 110.583,
and 113.146 microseconds for the three programs. The median complete chain was
369.417 microseconds without profiling instrumentation.

The matmul-GELU values are medians from two earlier runs. Each run used 20
warmups and five alternating batches of 2,000 evaluations per compiler.

The Chunked DeltaNet result used 10 warmups and five batches of 50 evaluations.
Its 58-program schedule is a correctness result and currently carries
substantial dispatch and intermediate-surface cost. Apple's compiler rejected
the same 14-input MIL program, so an equivalent latency comparison is not
available.

Powermetrics sampled the system ANE and GPU rails every 100 ms while the
hardware tests ran. The attention power capture ran only research-generated
programs. These values confirm ANE activity during the tests. They are
system-wide estimates and do not measure per-process energy.

The current attention result is recorded in a
[benchmark and profile receipt](docs/evidence/fa2-three-program-m4-2026-09-03.txt)
and a [powermetrics screenshot](docs/evidence/fa2-three-program-powermetrics.png).
Earlier compiler A/B results are in the
[original baseline](docs/evidence/compiler-ab-m4-2026-09-02.txt) and the
[first optimization receipt](docs/evidence/compiler-ab-m4-2026-09-02-optimized.txt).
The current Chunked DeltaNet run is recorded in
[docs/evidence/chunked-deltanet-m4-2026-09-03.txt](docs/evidence/chunked-deltanet-m4-2026-09-03.txt).

## Hardware tests

Each hardware script compiles its MIL fixture, provisions the emitted object,
runs it on the M4 ANE, and checks the output. Useful entry points include:

```sh
bash tests/run_m4_hardware.sh
bash tests/run_matmul_hardware.sh
bash tests/run_unary_hardware.sh
bash tests/run_reduce_hardware.sh
bash tests/run_layout_hardware.sh
bash tests/run_online_reduction_hardware.sh
bash tests/run_online_reduction_fallback_hardware.sh
bash tests/run_affine_scan_hardware.sh
bash tests/run_matmul_gelu_hardware.sh
bash tests/run_chunked_deltanet_hardware.sh
bash tests/run_compiler_ab_hardware.sh
```

Stock macOS loads these objects from
`/Library/Caches/com.apple.aned/<build>/InMemoryModelCache/<executable>/`.
The scripts use `sudo -n` to create that cache entry and install the generated
HWX file. Run `sudo -v` first if the current shell has no valid credential
timestamp. The software test suite writes only to `build/`.

## Runtime boundary

`ANEProvisionedRuntime` creates IOSurfaces from the binding manifest and calls
the private `AppleNeuralEngine.framework` runtime. The manifest keeps logical
tensor sizes separate from physical row, plane, batch, and allocation sizes.
This is required for narrow tensors whose rows are padded to 64 bytes.

The compiler does not provide a kernel-driver path or bypass the normal `aned`
cache requirement. Private interfaces and accepted HWX layouts can change with
macOS releases.

## Provenance

See the [source-cited ANE knowledge base](docs/ane/README.md) for generation,
software-stack, container, register-map, numerical, and memory-model references.

The HWX container, Task Descriptor fields, and compiler stages were recovered
by compiling author-written MIL with Apple's compiler, comparing generated
objects, tracing compiler execution, and testing edited objects on hardware.
No Apple source code was available or used, and no Apple binary code is
distributed here.

Some `plugins/H16G/Encoding/*EncoderData.inc` files contain measured Task
Descriptor words for operation and geometry rows whose field grammar is still
incomplete. They are indexed target measurements rather than copied program
containers. Tests store hashes and decoded field values, not Apple-generated
HWX files.

H13 register-field evidence comes from allbilly/ane revision
`e159e2d18ce6cea100e8f19bb27a7f07acaa9c24`, specifically its
[elementwise](https://github.com/allbilly/ane/blob/e159e2d18ce6cea100e8f19bb27a7f07acaa9c24/examples/elementwise.py)
and [GEMM](https://github.com/allbilly/ane/blob/e159e2d18ce6cea100e8f19bb27a7f07acaa9c24/examples/gemm.py)
examples. That source revision has no LICENSE file; these are attributed
register-layout facts, not an imported dependency or relicensed source files.

H14 inspection follows the [register map](https://github.com/freedomtan/coreml_to_ane_hwx/blob/ce54664e787976b646c450ceabed1731b506a4cd/hwx_dump/h14_register_map.md)
and parser at freedomtan/coreml_to_ane_hwx revision
`ce54664e787976b646c450ceabed1731b506a4cd`.

`plugins/H14/H14ElementwiseTemplates.inc` and
`plugins/H14/H14MatvecTemplates.inc` are generated by
`research/generate_h14_templates.py` from the decoded oracle JSON in
`research/oracles/h14`, and the H14 weight-packing permutation is pinned by
the known-weight probes `research/mint_h14_matvec_probes.py` mints.
`plugins/H14/H14NormTemplates.inc` is generated the same way by
`research/mint_h14_norm_probes.py --emit-templates`, whose softmax LUT kinds
are resolved by sha256 against the words already in
`plugins/H13/H13ElementwiseConstants.inc`. They store
task register words, constant-section halfword runs verified against the
recorded sha256, and per-case program-descriptor metadata. No HWX container
bytes are read or stored, and the checked-in oracles keep decoded fields and
hashes only.

See [DISCLAIMER.md](DISCLAIMER.md) for the full scope and private-API notes.

## License

The original code and documentation in this repository are available under the
[MIT License](LICENSE).
