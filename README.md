# MIL-to-HWX compiler

This repository contains a research compiler for the H16G Apple Neural Engine
in the M4 and an experimental source-native H13/M1 backend. It reads textual
MIL and emits H16G HWX objects or H13 ANEC and HWX packages without Apple's
compiler.

The project is a canary for the compiler pipeline recovered in *Inside the M4
Apple Neural Engine*, Part 4b. It shows which parts of that pipeline are
understood well enough to reproduce in code and verify on hardware.

## Linux build

This fork builds on Linux with GNUstep Foundation. H16G emits HWX; the
experimental H13 backend emits ANEC by default and HWX with `--format hwx`.
H13 performance and mlx-omarchy integration remain unqualified.

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
| add, mul, maximum, minimum | Two fp16 inputs, which may name the same value, or one fp16 input plus a same-shape fp16 const tensor on either side. `mul` also expands one inline fp16 scalar. Input, output, and tensor-constant shapes must match and be positive and static. The compiler emits 64-element programs and zero-pads the last program. |
| relu | One fp16 input and matching output with any positive static shape. Lowers each 64-element slice to `maximum(x, 0)`; the final slice is zero-padded. |
| clip | One fp16 input and matching output with any positive static shape, plus finite fp32 scalar `alpha` and `beta` attributes that are exactly representable in fp16 and satisfy `alpha <= beta`. Lowers each slice to `minimum(maximum(x, alpha), beta)` as two operation-major program groups. |
| sub | One fp16 input `x` and a same-shape fp16 const tensor `y`. The host negates each constant slice and emits the verified add descriptor. Two runtime inputs and `const - input` are rejected. |
| real_div | One fp16 input `x` and a same-shape fp16 const tensor `y`. Every constant element must be a finite, nonzero power of two with an fp16-exact reciprocal. The host stores each reciprocal slice and emits the verified multiply descriptor. |
| matmul | With `transpose_x=false`, fp16 x has shape `[..., K]`, `K >= 1`, and the product of the leading dimensions is the row count M. With `transpose_x=true`, only one logical row is accepted. W is a constant rank-2 tensor `[K,N]` with `transpose_y=false` or `[N,K]` with `transpose_y=true`, and the output is `[..., N]`. N is emitted in 512-output programs with the last program zero-padded. Each reduction of at most 512 elements is zero-extended to the 256- or 512-lane descriptor. Larger reductions emit one matvec per 512-element reduction chunk and sum the partial tensors with 64-lane add programs. The final tensor records `"accumulation": "chunked-fp16"`. Chunked results are not bit-identical to one unbroken accumulation because each partial sum adds one fp16 rounding step. |
| linear | fp16 x has shape `[..., K]`, constant weight has shape `[N,K]`, and the output is `[..., N]`. The compiler lowers `linear(x, weight)` to `matmul(x, weight, transpose_y=true)`. An optional constant fp16 bias vector `[N]`, encoded as an inline list or BLOBFILE, adds one folded 64-lane add group after the matmul. |
| reshape, squeeze, expand_dims | Positive static fp16 input and result shapes with equal element counts. The operation emits no program: its result aliases the input's underlying tensor with a new logical shape. Shape and axes inputs must be constants. Returning an alias of a function input fails with `h13.returned-input-alias` because no program produces an output. |

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
instead round each chunk before the add chain, so their device results can
differ by the extra fp16 rounding.

```bash
python3 tools/h13_reference.py model.mil --model-root models \
  --input x=input.fp16 --output y=reference.fp16
python3 tools/h13_run_linux.py build/h13-package --mil model.mil \
  --model-root models --input x=input.fp16 --output y=device.fp16 --dry-run
```

The Linux runner imports `research/inspect_anec.py` and validates the complete
package before it opens libane. Dry-run mode performs package, input, reference,
binding, and dispatch-plan checks without loading libane or writing an output. A
hardware run uses the `omarchy` branch Python binding library from
`~/src/omarchy-ane`, forwards intermediate slices as raw physical buffers, and
compares elementwise output at exact fp16 equality. A tensor marked
`chunked-fp16` uses
`abs(device-reference) <= 0.03125 + 0.01 * abs(reference)` to allow the extra
partial-sum rounding. Override the binding path with `--libane-library`. Run the
host-only reference and dry-run checks with `make test-h13-reference`.

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
H16G/M4 (ISA 17) from their headers. H14 task-stream inspection checks the
published register ranges and record lengths. Its tests use synthetic,
source-derived records, not a real H14 artifact; this is not an H14 emitter
or a claim of M2 execution support. Run `make test-hwx-inspection`.

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

See [DISCLAIMER.md](DISCLAIMER.md) for the full scope and private-API notes.

## License

The original code and documentation in this repository are available under the
[MIT License](LICENSE).
