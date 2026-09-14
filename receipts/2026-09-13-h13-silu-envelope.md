# H13 silu/sigmoid envelope (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `feature/h13-concat` `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` plus this envelope pin
- Island receipt: mlx-omarchy `receipts/2026-09-13-encoder-split-compile.md`
  (worktree `~/.config/superpowers/worktrees/mlx-omarchy/main-land2/receipts/2026-09-13-encoder-split-compile.md`)
- Island: encoder GLU `split → sigmoid/silu → mul` at fp16 `[1,1024,375]`
- Measured before this pin: `h13.unsupported-program` / `H13 has no source-qualified encoder for 'silu'` (same for `sigmoid`); 64-CHW compiles

## Envelope

`kElementwiseTasks` LUT unaries (`silu`, `sigmoid`, `exp`, `gelu`, `leaky_relu`, `rsqrt`, `sqrt`, `tanh`) and `relu` are CHW `(64,1,1)` and `(512,1,1)` only. `abs` is the power-of-two vector `(64..4096,1,1)`.

`parityPlan` does not invent a rank-4 NCHW for rank-3 `[1,1024,375]`. It flattens to `{384000,1,1}`. That key is not in the table.

Host compile of the envelope (one program, `h13-oracle-parity`):

| MIL shape | silu | sigmoid |
| --- | --- | --- |
| `[1,64,1,1]` | 1 program | 1 program |
| `[1,512,1,1]` | 1 program | 1 program |
| `[512]` | 1 program | not re-run |

Nearby misses, same reject:

| Candidate | In unary table? |
| --- | --- |
| silu CHW `(128,1,1)` | no — 128 is abs-only |
| silu CHW `(1024,1,1)` | no — 1024 is abs-only |
| silu CHW `(1024,375,1)` / `(1,1024,375)` | no |
| flattened 384000 | no |

Spatial `env_act_silu_*` oracles (`1x64x8x8`, `1x128x16x16`, `1x256x32x32`, `1x768x16x16`, `1x3072x1x1`) live under `research/oracles/h13/` and H14 templates. They are not in `H13ElementwiseTemplates.inc`. None is `[1,1024,375]`. Importing them would not close the GLU.

## Hole

`h13.unary-outside-envelope` for `silu`/`sigmoid` at `[1,1024,375]`.

The 64-lane split is `binaryNames` only (`add`/`mul`/`maximum`/`minimum`/`sub`/`real_div`). LUT unaries have no slice path. `384000 / 512 = 750` would be contiguous flat tiles, but that is a new lowering (750 copies of the 2 KiB table), not a decoded program. Do not emit it. Relu at this size is a different path: it rewrites to `maximum` against zero and then 64-lane splits.

The GLU island therefore stays uncompiled: split is a view, sigmoid/silu is the hole, mul never reached.

## Verification

```text
make -C ~/src/mil-hwx-compiler build/test_h13_encoding build/mil-hwxc
./build/test_h13_encoding
# H13_ENCODING_OK

python3 tests/test_h13_silu_envelope_cli.py build/mil-hwxc
# h13 silu envelope cli: PASS
```

`build/mil-hwxc` sha256 after this pin: `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`.

Did not compile the full encoder. Did not emit 750 or 6000 LUT programs.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
