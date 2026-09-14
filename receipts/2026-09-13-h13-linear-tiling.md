# H13 encoder linear tiling (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `feature/h13-concat` `c2cf32e4aa0200d72cecbce203bf6b47f50f729e`
- Island receipt: mlx-omarchy `receipts/2026-09-13-encoder-split-compile.md`
  (worktree `~/.config/superpowers/worktrees/mlx-omarchy/main-land2/receipts/2026-09-13-encoder-split-compile.md`)
- Island: `query_states_1_cast_fp16_h0_lin`
  fp16 `[1,375,1024] × [128,1024] + bias[128] → [1,375,128]`
- Measured: 1875 programs (750 matmul + 1125 add), 1875 `.anec`, 412169327 bytes

## Cause

Not a missed select. No decoded constant-weight program matches this geometry.

| Candidate | In corpus? |
| --- | --- |
| `(375, 1024, 128, tx=0, ty=1, const W)` | no |
| `(1, 1024, 128)` | no |
| `(1, 512, 128)` | no |
| `(1, 256, 128)` envelope | yes — K=256, would need 4 K-chunks (worse) |
| `(1, 1024, 256/512/1024)` matvec | yes — N≠128; selecting them needs N-padding, not a match |

Native `encodeMatvec` only accepts reduction 256 or 512 and always writes 512 output columns. K=1024 therefore splits into two 512-chunks. N=128 pads to 512. M=375 is coprime with every multi-row envelope M in `{2,8,16,32,64,128,256,512}`, so the only covered row count is 1.

Bias plus `rows > 1` lowers to one M=1 linear per row. Per row:

| Op | Tile | Programs |
| --- | --- | ---: |
| matmul | native K=512 × N-pad-512 | 2 |
| K-accum add `[1,128]` | `{128,1,1}` runtime elementwise envelope | 1 |
| bias add `[1,128]` const | no 128-wide constant/scalar add; 64-lane | 2 |

`375 × 5 = 1875` (750 matmul + 1125 add). The 412 MB package is 1875 ANEC files, not one large program.

## Hole

`h13.linear-outside-envelope` for constant-weight `(M=375, K=1024, N=128)`: ISA/oracle has no one-program form. Nearby larger tiles are different geometries. Do not pad N to steal `(1,1024,256)` without a capture of that packing.

## Verification

```text
make -C ~/src/mil-hwx-compiler build/test_h13_encoding
./build/test_h13_encoding
# H13_ENCODING_OK

python3 tests/test_h13_linear_tiling_cli.py build/mil-hwxc
# h13 linear tiling cli: PASS
```

1-row and 3-row compiles of the same K/N/bias lock `2` matmul + `3` add per row, source-qualified matmul slices of 512. Did not recompile the 375-row / 412 MB island.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
