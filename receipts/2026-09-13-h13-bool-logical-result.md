# H13 bool logical results and shape aliases

Host-only. No ANE, no SSH, no `ane-linux-experiments` edit or execute.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler base: `366eb15057c9806605e7341531545e753f7a8aec` (`feature/h13-boolean-and-round2`)
- Branch: `feature/h13-bool-logical-result`
- Ground truth: `research/oracles/h13/boolean/less_bool_rr_1500x1x1.json` (Apple out6 / ANEC in4)

## What shipped

1. **Bool graph output.** `less` already encoded a 1-byte result surface. Function returns rejected it with `h13.unsupported-logical-result-conversion` because only fp16 identity results were allowed. Bool returns are identity: `physicalOutputs` / `logicalResults` emit `dtype: bool` and 1-byte `logicalBytes`. FP32/int32 still hit the conversion diagnostic.

2. **Bool shape aliases.** `expand_dims` `[1,1500]→[1,1,1500]` (static, equal counts, const axes) failed `h13.invalid-shape-alias` only because `tensorElementCount` was fp16-only. Bool views use the same alias path. Alias tensor records add `"dtype": "bool"` so inspector byte counts stay 1-byte.

3. **Whole-surface boolean bindings.** Opening element counts for bool would have let the 64-lane split bind a 1500-element less result as 64 lanes. Boolean programs keep the whole-tensor spans already sized above that split.

4. **Inspector.** Physical outputs accept `float16` or `bool`. Tensor records allow `{shape, logicalBytes, role, aliasOf, dtype}`.

## Tests

Encoder-shaped cases in `tests/test_h13_boolean_cli.py`:

- `output_mask = less(...)` on `[1, 1500]`, byte-exact against `less_bool_rr_1500x1x1`
- bool `expand_dims` `[1,1500]→[1,1,1500]` as the function result, same task stream, alias `output_mask` of `mask`

Unskipped `less_bool_rr_*` captures now compile as graph outputs.

```text
sha256sum build/mil-hwxc
b72c119764bc421f5d2728737a5f77e68ad29359e530c1aa66106c0a93e619ed

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

python3 tests/test_h13_parity.py build/mil-hwxc
  H13 oracle parity: PASS (846 cases, 182 matmul, 79 broadcast, 105 softmax/layer_norm, 114 reduction, 284 convolution, 1692 artifacts)
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution. No files under `ane-linux-experiments` were edited or executed.
