# H13 finite fp16 scalar add/sub compiler proof

## Scope

Host/compiler proof only. No SSH, ANE hardware execution, model inference, compiler pin change, release change, or push was performed.

- Base commit: `a0ce354cf800011a84420da4e12013eb8140b2a5`
- Source and regression commit: `af9e1a74df756f7066c248883ae776c6b755dd44`
- Branch: `ane-unblock/h13-scalar-constant-fold`
- Actual agent model: `openai-codex/gpt-5.6-sol`

`git show --stat --oneline af9e1a74df756f7066c248883ae776c6b755dd44` reports only:

```text
plugins/H13/ANEH13Compiler.mm |  7 ++++---
tests/test_h13_cli.py         | 28 ++++++++++++++++++++++++++--
2 files changed, 30 insertions(+), 5 deletions(-)
```

No H13 encoder, binary format, object writer, or ABI source changed.

## Regression cycle

Before the source change:

```text
$ python3 tests/test_h13_cli.py build/mil-hwxc
AssertionError: 5:5: error [h13.invalid-constant-input]: H13 constants must be a matching fp16 tensor; only mul accepts an inline fp16 scalar broadcast
Command exited with code 1
```

After the source change:

```text
$ make build/mil-hwxc -j4
$ python3 tests/test_h13_cli.py build/mil-hwxc
H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
```

The permanent CLI regression compiles scalar and repeated-tensor `fp16(1.25)` constants for both add and runtime-minus-constant subtraction at shape `[130]`. It compares the complete manifests and all three emitted `.anec` programs. It also checks constant-first scalar subtraction and fp16 overflow rejection.

## Direct CLI evidence

Command:

```text
python3 /tmp/h13_scalar_cli_proof.py
```

The proof invokes `build/mil-hwxc --target H13` separately for scalar and repeated-tensor MIL files. Observed results:

- Scalar add: `artifacts=3`; repeated add: `artifacts=3`; manifests and all paired program bytes matched.
- Scalar subtraction: `artifacts=3`; repeated subtraction: `artifacts=3`; manifests and all paired program bytes matched. Each manifest operation is `add`, preserving the existing exact sign-negation transform.
- Both operations emitted dispatch plan `[0, 1, 2]` and runtime shapes `a=[130]`, `y=[130]`.
- Both operations emitted input slices `(offset=0,count=64)`, `(offset=64,count=64)`, and `(offset=128,count=2,physical=64)`.
- The three paired program SHA-256 values were `ac97b46653e5f0dd85ce0f45338b35aabd6049f9612f46ae756e4c224bd0bcb5`. Constants remain manifest-backed, so equal binary-program hashes across add and transformed subtraction are expected.
- Constant-first scalar subtraction returned `65`, emitted `h13.nonfoldable-binary`, and created no output directory.
- Scalar source value `65520.0`, which rounds to nonfinite fp16, returned `65`, emitted `h13.invalid-constant-payload`, and created no output directory.

## Hardware boundary

ANE hardware execution was not run. This receipt establishes Linux host compilation, emitted-package equivalence, slicing, runtime shape preservation, finite-value rejection, and operand-order rejection only.
