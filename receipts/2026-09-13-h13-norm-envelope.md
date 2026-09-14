# H13 layer_norm/softmax envelope (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `feature/h13-concat` `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` plus this envelope pin
- Island receipt: mlx-omarchy `receipts/2026-09-13-encoder-split-compile.md`
  (worktree `~/.config/superpowers/worktrees/mlx-omarchy/main-land2/receipts/2026-09-13-encoder-split-compile.md`)
- Encoder MIL: `receipts/2026-09-13-slice-layout-rewrite/model-fp16-noslice.mil`
  sha256 `58694e04d605c0973f411a77e8b3ad3fc5426c4f15dd3ad12fbcca377c53cce1`
- Encoder: `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018`

Measured before this pin (split compile leftover table): 120 `layer_norm` and 24 `softmax` at encoder form, all `h13.norm-outside-envelope`.

## Envelope

`kNormTasks` is exact-match on `(operation, input CHW, output CHW, axisMask, keepDims)`. Layer_norm must be non-affine (`gamma`/`beta` absent) and `epsilon = fp32(1e-5)`. Apple's compiler in this harness rejects every affine form, so no oracle covers one.

Host compile of nearby decoded points (one program, `apple-parity-norm`):

| MIL | result |
| --- | --- |
| `layer_norm` `[1,1024,1,1]` axes=`(1,)` epsilon 1e-5, no affine | 1 program, 5 tasks |
| `softmax` `[1,512,1,1]` axis=`1` | 1 program, 5 tasks |
| `softmax` `[1,8,128,128]` axis=`-1` | 1 program, 6 tasks |

Encoder surfaces are not those keys:

| Encoder form | Canonical CHW / mask | In `kNormTasks`? |
| --- | --- | --- |
| `layer_norm` `[1,375,1024]` axes=`(-1,)` | `{1,375,1024}` `0x08` | no |
| same, plus gamma/beta `[1024]` | same, plus affine | no (affine refused even on `[1,512,1,1]`) |
| `softmax` `[1,8,375,375]` axis=`-1` | `{8,375,375}` `0x08` | no |

Closest decoded neighbors that are **not** this geometry: layer_norm `{1024,1,1}` `0x02` (one 1024-vector, not 375 of them) and softmax `{8,128,128}` `0x08` (attention width 128, not 375). There is no 375 extent in the softmax table.

Encoder epsilon is a `tensor<fp16,[]>` BLOBFILE (`var_5_to_fp16`), not `fp32(1e-5)`. That is a third independent miss; the shape miss already fires first on a rewritten `fp32(1e-5)` form.

## Hole

`h13.norm-outside-envelope` for encoder `layer_norm` at `[1,375,1024]` (120 ops, gamma/beta present) and encoder `softmax` at `[1,8,375,375]` (24 ops).

Last-axis layer_norm on `[1,375,1024]` is 375 independent 1024-vectors, and those 1024-element slices are contiguous in row-major. Emitting 375 copies of `{1024,1,1}` would still be a new lowering, not a decoded program, and affine `mul`/`add` of `[1024]` over `[1,375,1024]` would then 64-lane-split. Do not emit it. Softmax of length 375 has no tile at all.

The diagnostic name is already `h13.norm-outside-envelope`. This pin does not invent a second code.

## Verification

```text
make -C ~/src/mil-hwx-compiler build/test_h13_encoding
./build/test_h13_encoding
# H13_ENCODING_OK

python3 tests/test_h13_norm_envelope_cli.py build/mil-hwxc
# h13 norm envelope cli: PASS
```

`build/mil-hwxc` sha256 after this pin (unchanged; no encoder change): `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`.

Did not compile the full encoder. Did not emit 375 vector layer_norms or pad softmax 375→512.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
