# H13 conv envelope (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `feature/h13-concat` `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` plus this envelope pin
- Island receipt: mlx-omarchy `receipts/2026-09-13-encoder-split-compile.md`
- Encoder MIL: `receipts/2026-09-13-slice-layout-rewrite/model-fp16-noslice.mil`
  sha256 `58694e04d605c0973f411a77e8b3ad3fc5426c4f15dd3ad12fbcca377c53cce1`
- Encoder: `mweinbach1/parakeet-tdt-0.6b-v3-coreml` revision `b650695c2322ee5281dff48d7345b2f3a58ff018`

Measured before this pin (split compile leftover table): 101 `conv`; subsample (custom pad, stride 2) and 1×1 valid `[1,256,750,32]` were `h13.conv-outside-envelope`; 99 unprobed.

## Envelope

`kConvTasks` is exact-match on `(kernel, stride, groups, bias, input CHW, output CHW)`. `convParityPlan` also requires rank-4 batch-1, square `[Cout, Cin/groups, k, k]` weight, unit dilations, zero explicit `pad`, `pad_type` `same` or `valid`, and `int32` scalar `groups`. 175 decoded programs. Spatial extents are square 1/8/16/32/64 (valid k3 shrinks by 2; stride-2 halves). Kernel 1 and 3. No 375/749/750/1500/3000, no rank-3, no kernel 9, no `pad_type` custom. k3 stride-2 is an unresolved packing hole even on 16×16.

Host compile of a nearby decoded point (one program, `apple-parity-conv`):

| MIL | result |
| --- | --- |
| `conv` k1 c256 n256 `[1,256,32,32]` same, bias, stride 1 | 1 program, 1 task, `constantBytes` 132096 |

That is `kH13ConvText49`. Encoder 1×1 uses the same channels and a 32-wide last axis, not this square.

## Encoder forms (101 ops, 9 unique)

None are in `kConvTasks`. All refuse `h13.conv-outside-envelope`.

| Encoder form | n | Canonical ConvShape | In table? | First miss |
| --- | ---: | --- | --- | --- |
| subsample k3 st2 custom `[1,1,3000,128]→[1,256,1500,64]` g1 bias | 1 | `{3,2,1,true,{1,3000,128},{256,1500,64}}` | no | custom pad `[1,1,1,1]`; k3 st2 unresolved; spatial 3000×128 |
| dw k3 st2 custom `[1,256,1500,64]→[1,256,750,32]` g256 bias | 1 | `{3,2,256,true,{256,1500,64},{256,750,32}}` | no | custom pad; k3 st2 grouped unresolved |
| 1×1 valid `[1,256,750,32]` g1 bias | 1 | `{1,1,1,true,{256,750,32},{256,750,32}}` | no | CHW `{256,750,32}` (pad/valid/rank-4/k1/st1 would pass) |
| dw k3 st2 custom `[1,256,750,32]→[1,256,375,16]` g256 bias | 1 | `{3,2,256,true,{256,750,32},{256,375,16}}` | no | custom pad; k3 st2 grouped |
| 1×1 valid `[1,256,375,16]` g1 bias | 1 | `{1,1,1,true,{256,375,16},{256,375,16}}` | no | CHW `{256,375,16}` |
| pad-as-conv 1×1 custom `[1,8,375,749]→[1,8,375,750]` g8 | 24 | `{1,1,8,false,{8,375,749},{8,375,750}}` | no | `pad_type` custom, pad `[0,0,1,0]`, C=8 |
| 1D pointwise `[1,1024,375]×[2048,1024,1]` valid | 24 | n/a (rank-3) | no | rank-3 input/weight; 1-length stride/pad |
| 1D depthwise k9 `[1,1024,375]` custom g1024 bias | 24 | n/a (rank-3, k=9) | no | rank-3; kernel 9; pad `[4,4]` custom |
| 1D pointwise `[1,1024,375]×[1024,1024,1]` valid | 24 | n/a (rank-3) | no | rank-3 |

The MIL also spells `pad_type` as `tensor<string, []>` and `groups` as `tensor<int32, []>` (padconv: `tensor<int32, [1]>`). Corpus probes use `string` / `int32` scalars. That spelling miss fires first on the encoder graph; the CHW miss remains after rewriting 1×1 to corpus spelling (CLI `enc-1x1-750`).

Closest decoded neighbors that are **not** these geometries: `{256,32,32}` and `{256,16,16}` 1×1 bias, and k1 stride-2 `{64,16,16}→{64,8,8}`. k3 stride-2 `{64,16,16}→{64,8,8}` is refused. `750/32` is not an integer, so 32-high strips are not a tiling of `[1,256,750,32]`. Do not emit them.

## Hole

`h13.conv-outside-envelope` for every encoder conv (101). The diagnostic name is already that. This pin does not invent a second code.

Host compile of encoder-shaped isolates (corpus spelling, dummy blobs):

```text
12:5: error [h13.conv-outside-envelope]: H13 conv needs a decoded geometry: ...
```

exit 65, no output dir. Same code for subsample, padconv, both 1D forms, and k3 stride-2 16×16.

## Verification

```text
make -C ~/src/mil-hwx-compiler build/test_h13_encoding
./build/test_h13_encoding
# H13_ENCODING_OK

python3 tests/test_h13_conv_envelope_cli.py build/mil-hwxc
# h13 conv envelope cli: PASS
```

`build/mil-hwxc` sha256 after this pin (unchanged; no encoder change): `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`.

Did not compile the full encoder. Did not tile 750×32 into 32×32 programs. Did not fake concat.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
