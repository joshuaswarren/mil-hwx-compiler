# H13 qualified split integration receipt

## Source identity

- Worktree: `/home/joshuawarren/.config/superpowers/worktrees/mil-hwx-compiler/h13-qualified-split-integration`
- Branch: `agent/h13-qualified-split-integration`
- Combined compiler commit: `a3665cd531d3bfe5c686418bbbb94eb7f68f6a5e`
- Executing model: `openai-codex/gpt-5.6-sol`
- Configured route: `mlx-openai-deep`; no routing fallback was observed.

The combined history is explicit:

1. `40624cd1900a3f5049b9b3946c80a7a54941b7ef` merges scalar-fold base `7da065bd7c94aafdd7dd4f3e47c8ff16645910fb` with split lowering `e8b8520bb75d9ae30a4926de54936b1cd9f105a7`. The split commit contains ordered-result commit `a1bfea0cd52208ac9f168791e8c6e9a83ddc8efb`.
2. `30236d78d716053183b3bcdb7eb5327b718b9581` merges `40624cd1900a3f5049b9b3946c80a7a54941b7ef` with qualified H13 tip `f4ad09066b560818a9dcc1f5f4f53273d941bd1e`, retaining the `17664fa` through `f4ad090` qualification series.
3. `a3665cd531d3bfe5c686418bbbb94eb7f68f6a5e` adds constant-storage provenance checks and the strict package fields required by the real consumer.

Git reported no textual merge conflicts. The semantic integration retained ordered parenthesized MIL results, split aliases, scalar folding, qualified task binding, the driver-derived kernel-window guard, scratch allocation, unsigned-zero behavior, and strict package inspection.

## Changed source at the combined tip

`a3665cd` changes:

- `plugins/H13/ANEH13Compiler.mm`
- `research/inspect_anec.py`
- `tests/test_h13_split_cli.py`
- `THEORY.MD`

The compiler follows `reshape`, `squeeze`, `expand_dims`, and `split` aliases back to their storage producer. A split whose storage origin is `const` now fails as `h13.unsupported-constant-split-source`; a direct constant-backed shape view fails as `h13.unsupported-constant-view-source`. Runtime-backed nested aliases remain valid. Every slice record contains `physicalElements`, and every program contains `scratchBytes`; neither consumer field has a default.

## Compiler verification

The scoped command was:

```text
make -j4 build/mil-hwxc && python3 tests/test_h13_split_cli.py build/mil-hwxc
h13 split cli: PASS
```

Direct and reshape-chained constant split reproducers both returned exit 65 with `h13.unsupported-constant-split-source`. The nested runtime-backed split compiled seven artifacts. Its `whole` tensor remained an intermediate, and the four consumer windows were `(offset,count,physical) = (128,64,64), (192,64,64), (0,64,64), (64,64,64)`.

An independent compiler review at `a3665cd` repeated the official split CLI, both constant-source rejections, the runtime intermediate split, and the nested sub-split. It reported no remaining compiler finding. The pinned encoder probe still stops at line 3354 with `h13.unsupported-chain`; this proves only the current two-return boundary, not prior-operation coverage.

A standard `make -j4 test` was attempted. This Linux host cannot complete the repository-wide target unchanged: the Makefile applies Objective-C flags to `test_benchmark_stats.cpp`, four older tests include Apple-only `CommonCrypto`, two fixtures use clang-14-unsupported `_Float16`, and `test_runtime_contract` imports Apple-only `IOSurface`. After temporary command-line/header workarounds exposed deeper targets, the qualified ANEC task rebinding also differed from the unbound parity oracle at the two fields intentionally rewritten by `ea903c4`, and a reference fixture lacked newly strict `physicalElements`. No guessed descriptor or oracle change was retained. The worktree was restored to the reviewed `a3665cd` source before this receipt.

## Strict consumer and loader verification

Fresh package:

- Package: `/tmp/h13-qualified-split-a3665cd/package`
- Graph: `/tmp/h13-split-negative-y0ydew77/nested-split.mil`
- Compiler binary SHA-256: `04739b99a6d72f98b74ec39ef544541dcc4ef0e918b6610184c200f15456ab39`
- Graph SHA-256: `9499b8e1487f7efca2889d1344ad405feeecbe044b13d80e4506f83b9fbeaef2`
- Manifest SHA-256: `619627d7c2c976a550a64c3047d2eb98d2a1949fc84f0e9b425635f40e936953`
- `program-0.anec`: `93e3af5314e899781fd269a91142ed1330ed973035b41a6aa3344ad8679e14f3`
- `program-1.anec`: `62595e4a61db24066b83a4f7c077d459c69a6a740109098ee0122b4b46dcd3cc`

`research/inspect_anec.py` accepted the package. It reported sigmoid input `(offset,count,physical) = (64,64,64)`, mul input `(0,64,64)`, `scratchBytes: 0` on both programs, tensors `input`, `gate`, and `output`, and dispatch order `0,1`.

The actual mlx-omarchy adapter command used the consumer worktree at commit `3f4d06079c4fdc36d147875b29291aceb9eaf24e`, compiler receipt `source.json`, compiler source `a3665cd`, and source-model commit `57cc36a2e8ece78dd979b5344752d04c76b4d49d`. It returned:

```text
h13_package_to_bundle: PASS programs=2 payloads=2 output=/tmp/h13-qualified-split-a3665cd/converted/bundle
```

The supplied host loader returned:

```text
[receipt] anec input input: channel=5 logical_bytes=128 allocation_bytes=16384 element_offset=64 element_count=64 physical_elements=64 nchw=[1,64,1,1,64,64]
[receipt] anec input input: channel=5 logical_bytes=128 allocation_bytes=16384 element_offset=0 element_count=64 physical_elements=64 nchw=[1,64,1,1,64,64]
[receipt] compiler: host_build=Linux x86_64 qualified split integration verification toolchain=mil-hwxc a3665cd531d3bfe5c686418bbbb94eb7f68f6a5e sha256:04739b99a6d72f98b74ec39ef544541dcc4ef0e918b6610184c200f15456ab39 target=h13
[receipt] OK: bundle valid
```

An independent consumer review regenerated `/tmp/h13-adapter-independent-Bo3BbswQ/package`. Its manifest and payload hashes matched the package above. The strict adapter passed, the supplied loader returned OK, only `input`, `gate`, and `output` were materialized, and dispatch `0 -> 1` preserved gate production and consumption.

## Proposed mixed-return cross-repository contract

The pinned textual CoreML8 function returns an ordered pair:

1. `encoder_hidden`: logical `tensor<fp32,[1,375,640]>`, produced by a cast from `linear_217_cast_fp16`.
2. `encoder_mask`: logical `tensor<int32,[1,375]>`, produced by a cast from boolean `output_mask`.

The package must not pretend these logical values are existing H13 FP16 surfaces. The proposed contract separates storage from API results:

- `physicalOutputs` describes actual ANE buffers only: physical dtype, physical shape, allocation bytes, and any slice `(tensor, elementOffset, elementCount, physicalElements)`.
- `logicalResults` is an ordered list matching the MIL function return list. Each record contains the SSA name, logical dtype, logical shape, a reference to one physical output, and an explicit boundary conversion.
- The only initially accepted conversion is `identity`. `fp16 -> fp32` and `bool -> int32` stay explicit unsupported conversions until an evidenced ANE conversion program or an explicitly selected MLX GPU conversion implements them. The adapter must never use a CPU tensor fallback and must never relabel FP16 or boolean bytes as FP32 or int32.
- A physical output can back more than one ordered logical result only through validated, non-overlapping slice metadata. Logical return order never changes physical dispatch order.
- `physicalOutputs` remains the runtime allocation and loader contract. `logicalResults` remains the caller-visible type and ordering contract. The loader validates both layers and the mapping between them; it does not infer one from the other.
- Until the cross-repository choice is implemented, the compiler continues to reject the two terminal casts and the two-result chain with named errors. Unsupported `less`, boolean storage, and other hardware operations remain separate explicit lowering gaps; the result-layout contract does not claim those operations are supported.

This proposal keeps one textual MIL graph, one ANEC package format, explicit GPU execution if selected, and no direct library API, alternate graph, hidden cast, or CPU fallback.

## Hardware boundary

No SSH, Mac, jwm1, hardware inference, ANE command, router edit, model pin, push, or shared main-land2 write was performed.
