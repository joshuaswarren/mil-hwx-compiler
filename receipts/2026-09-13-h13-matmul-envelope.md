# H13 matmul envelope (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `feature/h13-concat` `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` plus this envelope pin
- Linear island receipt: `receipts/2026-09-13-h13-linear-tiling.md`
- Batched primitive: `plugins/H13/H13BatchedMatmulTemplates.inc` `kBatchedTasks`

## Attention: one program

`[1,8,375,128] × [1,8,128,375]`, `tx=0`, `ty=0`, runtime y.

This is the decoded batched form `(B=8, rows=375, reduction=128, columns=375, runtime)`. Rank-3 `[8,375,128]` and rank-4 `[1,8,375,128]` share that entry. The compiler emits one program: `apple-parity-batched-matmul`, `taskDescriptors = 208` (`26×8`).

It is not a flattened 3000-row matmul. Per-batch y stays per-batch.

Nearby batched entries `(375,128,749)` and `(375,375,128)` are other attention ops (scores with 749, attn-output). They are not this GEMM and are not a missed select for it.

## Linear: named hole

Constant-weight `(M=375, K=1024, N=128)`, `ty=1`.

No one-program form. `supportsMatmulParity` is false for `(375,1024,128)` and `(1,1024,128)`. `supportsBatchedMatmul` is false for any B at K=1024 N=128.

The 145-entry matmul envelope has no M=375 and no `(*, 1024, 128)`. Const-weight N=128 exists only as `(1,256,128)` and `(32,256,128)` — both K=256.

Those cannot compose over K=1024 without a strided x: `[1,375,1024]` row-major makes a 32-row × 256-K tile 32 discontiguous 256-element segments. The binding ABI is one contiguous slice per binding. Full-K 32-row tiles would be contiguous, but `(32,1024,128)` is absent. Native `encodeMatvec` is K∈{256,512}, N-pad 512, and the bias+`rows>1` path therefore emits M=1: `375 × (2 matmul + 3 add) = 1875`.

Do not pad N to steal `(1,1024,256)`. Do not flatten the 8-head attention GEMM. Do not emit mul-by-1 copies.

Named hole: `h13.linear-outside-envelope` for constant-weight `(375,1024,128)`. Attention `[1,8,375,128]×[1,8,128,375]` is inside `kBatchedTasks`, not this hole.

## Verification

```text
make -C ~/src/mil-hwx-compiler build/test_h13_encoding
./build/test_h13_encoding
# H13_ENCODING_OK

python3 tests/test_h13_matmul_envelope_cli.py build/mil-hwxc
# h13 matmul envelope cli: PASS
```

Did not recompile the 375-row / 412 MB linear island.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
