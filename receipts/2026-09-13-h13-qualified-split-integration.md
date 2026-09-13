# H13 qualified split integration receipt

## Source identity

- Worktree: `/home/joshuawarren/.config/superpowers/worktrees/mil-hwx-compiler/h13-qualified-split-integration`
- Branch: `agent/h13-qualified-split-integration`
- Feature tip before upstream integration: `f91bc92ed892d03d20b6c945dfaa0a5bf50752af`
- Executing model: `openai-codex/gpt-5.6-sol`
- Configured route: `mlx-openai-deep`; no routing fallback was observed.

The combined history is explicit:

1. `40624cd1900a3f5049b9b3946c80a7a54941b7ef` merges scalar-fold base `7da065bd7c94aafdd7dd4f3e47c8ff16645910fb` with split lowering `e8b8520bb75d9ae30a4926de54936b1cd9f105a7`. The split commit contains ordered-result commit `a1bfea0cd52208ac9f168791e8c6e9a83ddc8efb`.
2. `30236d78d716053183b3bcdb7eb5327b718b9581` merges `40624cd1900a3f5049b9b3946c80a7a54941b7ef` with qualified H13 tip `f4ad09066b560818a9dcc1f5f4f53273d941bd1e`, retaining the `17664fa` through `f4ad090` qualification series.
3. `a3665cd531d3bfe5c686418bbbb94eb7f68f6a5e` adds constant-storage provenance checks and the strict package fields required by the real consumer.
4. `b12b03f17619c76b77c48a83510847de2245ea1a` emits strict v2 physical outputs and ordered identity logical results.
5. `600dc3e606886f67d111c011e8c91e9a05a4d6be` preserves distinct ordered logical views across one sliced physical output and independent physical producers.
6. `b7a59f07c45e9fd4e753b0b25004fe6a18180d81` preserves a returned physical output as later-consumed storage, validates its dispatch dependency, and makes HWX embed the normalized ANEC task stream.

Git reported no textual merge conflicts. The semantic integration retained ordered parenthesized MIL results, split aliases, scalar folding, qualified task binding, the driver-derived kernel-window guard, scratch allocation, unsigned-zero behavior, and strict package inspection.

## Changed source at the combined tip

`b12b03f` changes:

- `plugins/H13/ANEH13Compiler.mm`
- `research/inspect_anec.py`
- `tests/test_h13_split_cli.py`
- `tests/test_h13_cli.py`
- `tests/hardware/run_h13.mm`
- `README.md`
- `THEORY.MD`

The compiler follows `reshape`, `squeeze`, `expand_dims`, and `split` aliases back to their storage producer. A split whose storage origin is `const` fails as `h13.unsupported-constant-split-source`; a direct constant-backed shape view fails as `h13.unsupported-constant-view-source`. Runtime-backed nested aliases remain valid. ANEC program slices retain `physicalElements`, and every program contains `scratchBytes`; neither binding field has a default. Logical return views deliberately omit `physicalElements` because their capacity comes from the referenced physical output.

## Compiler verification

The scoped command was:

```text
make -j4 build/mil-hwxc && python3 tests/test_h13_split_cli.py build/mil-hwxc
h13 split cli: PASS
```

Direct and reshape-chained constant split reproducers both returned exit 65 with `h13.unsupported-constant-split-source`. The nested runtime-backed split compiled seven artifacts. Its `whole` tensor remained an intermediate, and the four consumer windows were `(offset,count,physical) = (128,64,64), (192,64,64), (0,64,64), (64,64,64)`.

Fresh v2 producer packages were emitted under `/tmp/h13-v2-b12b03f`. The compiler and strict inspector accepted all four:

- `ordinary-package/manifest.json`: `5a6b5627ff171e918e11ee0fd3c275c1e1f01075def0410e9dd7f339e29f91ce`
- `duplicate-package/manifest.json`: `adb26c403598c4048ba9502f7b34560be005aa832dec80614444b90ff579d1bc`
- `sliced-package/manifest.json`: `9d5bca2e30f505e6cacdaed713bf5d05b8d93332da920d078e8afb5acd8e357c`
- `reshaped-package/manifest.json`: `9a6bf068a55d4c277a23a3e90b27eaeb7f44287b19842d94d6bb32eab6e2d0a9`

The producing `build/mil-hwxc` SHA-256 was `b3587e4d87bad93d95c1b1e69c3164d7f445787c65f9bdf2d81978e17d128a9e`.

Each package used the same commands, with `NAME` set to `ordinary`, `duplicate`, `sliced`, or `reshaped`:

```text
build/mil-hwxc --mil /tmp/h13-v2-b12b03f/NAME.mil --model-root /tmp/h13-v2-b12b03f --output /tmp/h13-v2-b12b03f/NAME-package --target H13 --format anec
python3 research/inspect_anec.py /tmp/h13-v2-b12b03f/NAME-package
```

The source SHA-256 values are `747a5cad80437a9cc9aacbf66144ec2bd8e33a047772a7956c3a49531716210f` for ordinary, `fe6a6dd165c47f3cd39efb9e7ec5764d395a64fab11dc1c46aa7410c607bc9bf` for duplicate, `c93b84cfed35e01d04bf1fb33eeac6bcb39af5bbaef5b3489b064deed6eea1e1` for sliced, and `8488b0bb027f8b0fcd3cd9019ba4893bbc594831ffa870bf6be3392a9fc31704` for reshaped.

An independent compiler review at `a3665cd` repeated the original split CLI and provenance checks without a remaining compiler finding. At `b12b03f`, the integrated adapter source reached its ordered FP32 and int32 returns and returned exit 65 with `h13.unsupported-logical-result-conversion` at line 3353. The full faithful source still reaches the previously recorded `mil.import.unsupported-type` boundary for `uint4`.

The device-free full verifier now completes successfully. `scripts/verify-linux-compiler.sh all` rebuilt the compiler, passed operation graph, HWX object writer, program composition, H13 encoding, H13 ANEC, H13 MIL-to-ANEC/HWX CLI, H13 split CLI, and HWX inspection checks, emitted both Linux compiler smoke artifacts, and ended with `linux compiler build: PASS`, `linux compiler software tests: PASS`, `linux compiler emission: PASS`, and `linux compiler hygiene: PASS` in 64.33 seconds.

## Pre-v2 strict consumer and loader verification

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

## Implemented compiler v2 result contract

Commit `b12b03f` emits schema `mil-hwxc.h13-anec-package.v2`. A single textual CoreML8 function can return the same read-only FP16 value more than once without creating another physical buffer. The manifest contains:

- `physicalOutputs[]`: exactly the produced runtime FP16 tensor name, physical shape, dtype, and logical byte capacity.
- `logicalResults[]`: ordered MIL name, dtype, independent logical shape, `conversion: identity`, and `physical {tensor, elementOffset, elementCount}`.

The strict inspector requires `elementCount == product(logical shape)`, equal identity dtypes, and `elementOffset + elementCount` within the physical output's capacity. It requires every produced physical output to be declared and referenced. Duplicate and overlapping read-only result views are valid. It rejects extra logical mapping fields, including the ANEC-binding-only `physicalElements` field.

The scoped fixture covers three accepted mappings: two direct views of one 64-element output, two reshape views with logical shape `[1,1,8,8]` over physical shape `[1,64,1,1]`, and two split-subslice views at offset 64 over a 128-element physical output. The same test rejects a forged identity dtype and rejects extra logical `physicalElements` metadata.

The pinned adapter's `encoder_hidden` FP32 and `encoder_mask` int32 returns are not relabeled as FP16 and do not receive a CPU fallback. The compiler rejects them with `h13.unsupported-logical-result-conversion` until explicit hardware or GPU conversion coverage exists. This preserves one textual graph and one package format; it adds no direct library API or alternate IR.

## Hardware boundary

No SSH, Mac, jwm1, hardware inference, ANE command, router edit, model pin, or shared main-land2 write was performed. The two completed compiler checkpoints were pushed without force to the named feature branch; no default branch or release pin changed.

## Ordered storage and HWX parity closure

Commit `b7a59f07c45e9fd4e753b0b25004fe6a18180d81` changes `plugins/H13/ANEH13Compiler.mm`, `plugins/H13/H13ANEC.cpp`, `plugins/H13/H13Program.h`, `research/inspect_anec.py`, `tests/test_h13_cli.py`, and `THEORY.MD`. Its built compiler SHA-256 is `249601e59ff509eba2bcaa11b504df839a8c417d1d1eab2d74e570257d0ff680`.

A MIL return may name an earlier physical producer that a later program consumes. The compiler now retains that value in `physicalOutputs`, emits the later binding against the same storage, and preserves physical producer order independently of ordered `logicalResults`. The inspector validates exact producer and consumer tiling plus topological dispatch coverage for both intermediate and returned physical storage. A forged `dispatchPlan: [1,0,2]` is rejected. An entirely dead terminal operation still fails `h13.unsupported-chain`; a partially used multi-result split remains valid.

The HWX writer previously embedded raw selected task bytes while explicit ANEC normalized surface selectors and task header flags. HWX now embeds the normalized task slice from the already encoded ANEC. `tests/test_h13_cli.py` extracts ANEC from the emitted HWX and requires byte-for-byte equality with explicit ANEC output.

Scoped verification after the fix returned:

```text
python3 tests/test_h13_cli.py build/mil-hwxc
H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
python3 tests/test_h13_split_cli.py build/mil-hwxc
h13 split cli: PASS
```

The GitHub feature branch receipt is `https://github.com/joshuaswarren/mil-hwx-compiler/tree/agent/h13-qualified-split-integration`. The first push created it at `600dc3e606886f67d111c011e8c91e9a05a4d6be`; the second advanced it to `b7a59f07c45e9fd4e753b0b25004fe6a18180d81`. Both used explicit `HEAD-or-commit:refs/heads/agent/h13-qualified-split-integration` refspecs without force.
## Constant-input-free ordered split artifacts

The runtime adapter requested add-based split views without materialized constant inputs. A detached Linux checkout at exact compiler commit `600dc3e606886f67d111c011e8c91e9a05a4d6be` built `/tmp/mil-hwxc-600dc3e-build/build/mil-hwxc` with SHA-256 `6e08cb08194b452894dab985c93f8dfe96974587f956a87f5ba89163debdedae`. It generated new paths without mutating the earlier package metadata:

- Forward MIL: `/tmp/h13-ordered-add-splits/split-forward.mil`, SHA-256 `1f17595221a889ad04a785e5ac9e465a2abf30b3f407d9a00effe0e38de1cdf5`.
- Forward package: `/tmp/h13-ordered-add-splits/split-forward-package`; manifest SHA-256 `6723c39c55e30485ba3c0f64afff7e33e17534e01a84558ead1d85f69ececfc5`.
- Reverse MIL: `/tmp/h13-ordered-add-splits/split-reverse.mil`, SHA-256 `258f94e03a09a134e2ae2d6546533cf42f877c70ead382784d09d8d40dda2ba0`.
- Reverse package: `/tmp/h13-ordered-add-splits/split-reverse-package`; manifest SHA-256 `ed77acc0ba9933890dbf415c25e2bc5100b59aa6efab975de4715dd51cd6a27e`.
- Both packages contain `program-0.anec` SHA-256 `70b9e497556ce990c697fa86316f5292384bd27bbe51911ce24df14df7735a23`.

Both compiler invocations succeeded and `research/inspect_anec.py` accepted each package. Each manifest has `constantInputs: {}` and operation `add`. Forward logical results are `tall` at offset 0 then `wide` at offset 64; reverse logical results are `wide` at offset 64 then `tall` at offset 0.

## Upstream main integration

The feature tip was merged with `origin/main` at `5271ab0b2bc1d9c28b1eb9d3bcfef9773902b57c`. The resolution preserves ordered result indexing, upstream native encoder selection, normalized ANEC bytes inside HWX, strict v2 storage dependencies, reviewed-host preflight, and the native runtime's mandatory finite whole-device-phase deadline. The linked ANEC constant region begins at the exact 64-byte-aligned task length; the 80-byte linked-task fixture therefore starts constants at byte 128.

The focused device-free verification command was:

```text
make -j4 build/test_h13_anec
build/test_h13_anec
python3 tests/test_h13_cli.py build/mil-hwxc
python3 tests/test_h13_split_cli.py build/mil-hwxc
python3 tests/test_h13_deadline.py
python3 tests/test_h13_linux_runtime.py
python3 tests/test_h13_preflight.py
python3 tests/test_h14_parity.py build/mil-hwxc
```

The command passed. It reported H13 CLI PASS, H13 split CLI PASS, `H13_DEADLINE_OK`, H13 Linux runtime PASS for 15 cases with one real-libane skip, all preflight cases PASS, and H14 parity PASS for 718 cases and 1,436 artifacts. `build/test_h13_anec` completed successfully without diagnostic output. No hardware command or model inference ran.
