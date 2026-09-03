# Verification receipt

Date: 2026-09-03
Host: `Mac16,10`, Apple M4, arm64  
macOS: 26.3, build `25D125`  
Target: H16G

## Compiler pipeline exercised

Each hardware executable starts from its MIL fixture and model root. The path
executed in the final gate is:

```text
MIL bytes
  -> character lexer
  -> recursive-descent syntax parser
  -> typed SSA Graph IR import and verification
  -> typed operation graph
  -> normalize and decompose
  -> structural graph fusion
  -> H16G legality and numeric-mode selection
  -> task grouping, tiling, SRAM/liveness and DMA planning
  -> H16G register-packet encoding into a zeroed TD stream
  -> complete HWX object construction from a zeroed buffer
  -> HWX structural verification and binding manifest
  -> ANEProvisionedRuntime
  -> _ANEClient load, evaluate and unload
  -> IOSurface output
  -> independent CPU comparison
```

No production source or binary references `ANECCompile`, a named workload
emitter, a descriptor-row loader, a container skeleton, or
`ANECompiler.framework`. A build guard checks this property. The compiler CLI
links Foundation only. The hardware
executables dynamically open `AppleNeuralEngine.framework` in the runtime
layer; they do not use Apple's compiler.

## Clean software gate

Command:

```sh
make test
```

Result: PASS. The clean `-Wall -Wextra -Werror` build passed diagnostics,
lexer, parser, parse/print, operation Graph IR, normalization, decomposition,
structural fusion, H16G legalization, task/SRAM/DMA planning, field encoders,
object writer, compiler API, malformed external constants, runtime contracts,
the CLI, the release-hygiene guard, and the production-route guard. The CLI
compiled each fixture twice and produced byte-identical output. It also
compiled a mixed graph using absolute input and output paths with no
descriptor-resource tree in its working directory.

The current suite also checks exact unary and reduction TD identities, padded
tensor metadata, all measured standalone layout packet families, and rejection
of nearby unmeasured forms. The production-route guard still passes.

## Extended primitive sweeps

The following compiler-created objects were executed on the same M4. Each case
ran twice and compared semantic outputs with independent CPU code.

| Family | Cases | Result |
|---|---:|---|
| Conv1x1 shape table | 16 geometries | zero mismatches |
| Depthwise 3x3 | 4 geometries | zero mismatches |
| Regular Conv 3x3/5x5 | 6 geometries | zero mismatches |
| Square matmul | N128, 256, 512, 768, 1152, 2176, 4096 | zero mismatches |
| Binary ALU | 9 operation/geometry forms | zero mismatches |
| Unary/LUT | 14 operation/geometry forms | within declared PWL envelopes; ReLU exact |
| Reduction | 24 operation/geometry forms | bit-exact, zero mismatches |
| Standalone S2D | 13 shapes | bit-exact, zero mismatches |
| Standalone D2S | 8 shapes | bit-exact, zero mismatches |
| Fused S2D-Conv-D2S | C8, C16, C24, C32 | bit-exact, zero mismatches |

Reduction covers sum, mean, and max over channel, height, and width axes. The
first hardware run exposed an incorrect dense input assumption. The independent
object describes narrow rows with 64-byte physical strides. After the object
writer and runtime manifest carried separate semantic and physical layouts, all
24 forms passed twice with maximum absolute error zero.

The same rule unlocked the narrow layout families. An independently minted
C32/S64/B4 S2D object uses a logical 32-byte output row and a physical 64-byte
row. The compiler now emits that contract directly. The 25-case standalone and
fused layout sweep then passed twice per case.

## Emitted hardware images

```text
991aa7a3ec28f5603beb090f2ca244921370d4081948bea30ad5074499ca1266  Conv1x1+ReLU
90d014ab1cdb6288d490780b8d9d8565bc84037efade13d506d91067126678d0  H4/S64/D64 mixed graph
38f1858486a352de2145d888601b9c50526459f75775377a22080b80a8627013  four-layer W8A8 numerical control
```

The W8A8 structural differential test separately uses broad signed int8
weights and emits
`fce4ce6a1378d19bf637c189ec8d273e9e997e39e815a7a05e26c7f981dab853`.
The numerical test uses four distinct permutation matrices with quantized
value 8 and scale 0.125. This avoids activation saturation while exercising
all four dynamic weight packs and all three int8 inter-layer boundaries.

## M4 hardware gate

Command:

```sh
bash tests/run_m4_hardware.sh
```

The script compiles the three MIL programs with the standalone CLI, verifies
the hashes above, and provisions those exact files in each test executable's
`aned` cache namespace. Separate runtime executables then deserialize the
on-disk bundles and run every program twice; they do not contain or invoke the
MIL frontend, pattern plugins, H16G lowering or HWX emitters.

| Pattern | Run | Elements compared | Mismatches | Max absolute error | Evaluation time |
|---|---:|---:|---:|---:|---:|
| Conv1x1+ReLU | 1 | 262,144 | 0 | 0 | 212.0 us |
| Conv1x1+ReLU | 2 | 262,144 | 0 | 0 | 211.0 us |
| Mixed matmul/softmax graph | 1 | 16,384 | 0 | 0 | 225.9 us |
| Mixed matmul/softmax graph | 2 | 16,384 | 0 | 3.10903e-05 | 184.1 us |
| W8A8 chain | 1 | 262,144 | 0 | 0 | 214.0 us |
| W8A8 chain | 2 | 262,144 | 0 | 0 | 379.1 us |

The timings are single request wall-clock samples from the correctness gate,
not a performance benchmark.

## Composed hardware gates

The scheduled primitive assembler was also tested with multi-artifact bundles.
Every case was compiled from MIL, provisioned, and executed twice.

| Region | Geometry | Result |
|---|---|---|
| FP16 attention forward | S128/D128, unmasked | 3 artifacts; zero mismatches; maximum absolute error `9.92883e-06` |
| FP16 affine state scan | 4 N128 multiply-add transitions | every stage within 1 FP16 ULP; final result within 3 ULP |
| Matmul -> reshape -> GELU | N128 and N256 | one HWX program; matmul element-exact; GELU maximum absolute error `0.00610352` |
| Chunked DeltaNet block | C128/D128 | 58 HWX programs; output relative L2 `0.004214`; final-state relative L2 `0.004443`; maximum error `3.69e-05` |

The attention result covers one query tile and one key/value tile. Its first
program runs matmul and scale. Its second program keeps the row-normalization
chain in SRAM across reduce-max, subtraction, exponentiation, reduce-sum,
reciprocal, and multiply tasks. Its third program runs the final matmul. The
forced eight-program fallback also passed twice. The compiler does not yet
emit the S256 plan or causal masking.

The Chunked DeltaNet fixture is a complete C128/D128 block expressed as
ordinary matmul, add, multiply, and exp operations. It does not introduce a
DeltaNet operation or a workload-specific lowering path.

The Chunked DeltaNet hardware run used 10 warmups and five batches of 50
evaluations. Its sample median was `22728.792 us` and its batch median was
`22828.750 us`. Apple's compiler
rejected the same 14-input MIL program as `InvalidMILProgram`, so no equivalent
latency comparison is available. The complete output is stored in
`docs/evidence/chunked-deltanet-m4-2026-09-03.txt`.

## Attention performance

The compiler A/B used 50 warmups and five alternating batches of 5,000
evaluations, for 25,000 measured samples per compiler. The MIL, input values,
IOSurfaces, QoS, and synchronous completion boundary were the same.

| Implementation | Median | Batch median | p95 |
|---|---:|---:|---:|
| Apple compiler output | `127.208 us` | `126.625 us` | `197.958 us` |
| Research compiler output | `369.458 us` | `378.125 us` | `3602.875 us` |

The research median was 2.986 times the Apple compiler median. A separate
research-only profile measured median submissions of `114.458 us`, `110.583
us`, and `113.146 us`. The uninstrumented three-program chain median was
`369.417 us`. Powermetrics sampled the ANE rail every 100 ms during that
research-only run and reported activity in 61 of 140 samples, with a 601.9 mW
active-sample mean and a 681 mW peak. Powermetrics reports system-wide rail
estimates rather than per-process energy.

## Runtime boundary

`ANEProvisionedRuntime` is the userspace controller. It validates the compiler
bundle, per-artifact I/O indices, shared-surface layouts, and dispatch order. It
allocates each IOSurface from the binding manifest, checks buffer identity and
allocation sizes, looks up one provisioned cache identity per artifact, and
owns ordered load, evaluate, and unload. It fails closed for missing cache
entries, wrong bindings, invalid dispatches, and evaluation before load.

Stock macOS does not accept these raw programs through the direct user-client
program-creation route observed in the accompanying RE work. The reproducible
test therefore provisions the compiler-produced HWX under the executable's
`aned` in-memory cache namespace before loading it. That provisioning step is
the remaining operating-system boundary; no numerical or compiler work is
delegated to Apple during the tested path.
