# H13 bool transpose under the fp16 perm contract

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler base: `e0419ed0d986bc664908b0f7d52db3ca3bcd1d05` (`feature/h13-bool-logical-result`)
- Branch: `feature/h13-bool-transpose`

## What shipped

`transposeViewPlan` accepted only static fp16 `x`, so the encoder's
`attention_mask` transpose `bool [1,375,375] perm [0,2,1]` failed
`h13.invalid-transpose-parameters` before the perm was classified.

Bool transpose now uses the same exact-perm contract as fp16: rank-matching
const perm, unique in-range axes, result shape and dtype match the permutation.
A layout-preserving perm is a free alias. The encoder perm is a tail-swap of
two non-unit 375 axes and moves the storage-fastest axis, so it is **not** a
view; it classifies as `h13.nonfoldable-transpose` ("moves the storage-fastest
axis"), same as fp16 `[1,64,64] perm [0,2,1]`.

## Tests

`tests/test_h13_layout_cli.py`:

- Encoder-shaped `bool [1,375,375] perm [0,2,1]` feeding `select` (logical_and
  already folded) pins `h13.nonfoldable-transpose` / fast-axis.
- Layout-preserving bool transpose after `less [1,64,1,1]` aliases and
  compiles as a bool logical result.

```text
sha256sum build/mil-hwxc
a3b03ecab14d70fe5e262c2f4066b9c5d3b102a6a92a09f0adb7d6b693a397dd

make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS
  h13 composite cli: PASS
  h13 batched cli: PASS
  h13 registry cli: PASS
  h13 boolean cli: PASS
  h13 tile cli: PASS
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
