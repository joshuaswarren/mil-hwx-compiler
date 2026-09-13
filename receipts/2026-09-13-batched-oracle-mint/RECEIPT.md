# H13 batched-matmul oracle mint receipt

## Source

- Compiler base: `12d5f05bae67e239251e5e1afd54344987d03de1` (`feature/h13-batched-matmul-survey`)
- Campaign: `receipts/2026-09-13-batched-oracle-mint/mint_batched.py`
  SHA-256 `c3557393fd8957cff8256fb40ea3e472e92ffcb2ac088c5215a0215d6d91b14a`
- Decoder/driver (recorded in each JSON `compiler` block):
  - `research/mint_oracles.py` SHA-256 `a6e8f70d438973e60913815be66dbd6de19d7200df9053a9efc2a615823d8074`
  - `research/h13_td.py` SHA-256 `125da0249ca3c3ba0ca1f9f0cabfd1db54a965abb75f9f4cdcc2598e255c8fbc`
- Apple tool rebuilt in `/tmp` on MacStudio (previous `/tmp/h13-oracle/bin/ane-compile-hwx` was gone):
  - source `ane-compile-hwx.mm` SHA-256 `55c6ae904e65c115066e980dcf92d4072f33dbc9acef71c151f30c5c69b610e8` (3009 B)
  - binary 52336 B, SHA-256 `3d13fc85c2a6baa0b7628f0848269b16fb99f742180f0085f608c371bb9da6e0`
  - build: `xcrun clang++ -std=c++17 -fblocks -framework Foundation -F/System/Library/PrivateFrameworks -framework ANECompiler ane-compile-hwx.mm -o bin/ane-compile-hwx`
- Host: `MacStudio.local`, macOS 26.6.2 arm64, build 25G83
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback)
- No ANE execution, no reboot, no system change. `ane-linux-experiments` was not edited or executed; only the `.mm` source was copied into `/tmp` to rebuild the compile tool. Compiler plugin code was not modified.

## Command

```text
python3 receipts/2026-09-13-batched-oracle-mint/mint_batched.py --host macstudio
# 23 cases, 22 decoded, 1 rejected
```

Oracles land in `research/oracles/h13/batched/`. Compact index: `capture_index.json`.

## Geometries minted

Rank-3 `(B, rows, reduction) @ (B, reduction, columns)`, `tx=false`, `ty=false`:

| chain | rows | reduction | columns | B |
|---|---:|---:|---:|---|
| scores | 375 | 128 | 749 | 2, 4, 8, 16 |
| attn-output | 375 | 375 | 128 | 2, 4, 8, 16 |

Each geometry × `{runtime y, batched BLOBFILE y}`.

Fold flags at encoder B=8, scores, runtime-runtime only: `ty=true` and `tx=true`.

Rank-4 encoder forms at B=8, both y variants:

- `[1,8,375,128] × [1,8,128,749] → [1,8,375,749]`
- `[1,8,375,375] × [1,8,375,128] → [1,8,375,128]`

Mixed layout `[1,375,8,128] × [1,8,128,749]` was minted and **rejected** (`callback_status=1`). Null for that pairing, not a flatten.

## Batch-structure verification — not flattened

Every accepted capture is **one program** (`program_count=1`) whose tensor descriptors keep B in NCHW as `[1, B, rows, width]`. Task count scales exactly with B for `tx=ty=false`:

| B | tasks | notes |
|---:|---:|---|
| 2 | 52 | 26 tasks/batch |
| 4 | 104 | |
| 8 | 208 | rank-3 and rank-4 heads match this count |
| 16 | 416 | |
| 8, ty=1 or tx=1 | 209 | one extra task |

Tile-DMA `src_off` (`0x13808`) for `ty=false` takes **B distinct values**, stride one y-plane:

- scores: stride `0x30000` = 196608 = `128 * 1536` (W=749 padded to 768 fp16)
- attn-output const/runtime y: stride `0x17700` = 96000 = `375 * 128 * 2` (no pad on 128)

`ty=true` B=8 keeps shape `[1,8,749,128]` but only `src_off=0` — batch is still in the descriptor; the y operand is the transposed kernel path, not a flatten to `B*rows`.

Surface math (scores B=2 runtime, representative):

| tensor | shape | batch stride | total_bytes |
|---|---|---:|---:|
| y | `[1,2,128,749]` | 196608 | 393216 |
| x | `[1,2,375,128]` | 96000 | 192000 |
| out | `[1,2,375,749]` | 576000 | 1152000 |

`total_bytes = B * plane`. Per-batch bases recorded in each JSON `parameters`:

- `batch_x_base_elements = rows * reduction`
- `batch_y_base_elements = reduction * columns`
- `batch_out_base_elements = rows * columns`

Apple then pads the 749-wide planes to row stride 1536.

## Batched-const y packing

**One constant section holding all B batches**, not per-batch programs.

| geometry | B | constant_bytes | formula |
|---|---:|---:|---|
| scores blob | 2/4/8/16 | 393216 / 786432 / 1572864 / 3145728 | `B * 128 * 1536` |
| attn-output blob | 2/4/8/16 | 192000 / 384000 / 768000 / 1536000 | `B * 375 * 128 * 2` |

Runtime-runtime keeps the usual empty 16384-byte constant section (`4fe7b59a…`). Const-y programs bind two runtime surfaces (x, out); y lives in `__const`. Rank-4 heads const-y uses the same byte counts as rank-3 at B=8 (1572864 / 768000) with different HWX hashes.

## Rank-4 vs rank-3

`[1,8,M,K]` and rank-3 `[8,M,K]` both emit 208 tasks at B=8. HWX SHA-256s differ, so the captures are not byte-identical; tensor descriptor shapes after Apple's NCHW rewrite are the same `[1,8,…]`.

## Files

22 decoded + 1 rejected JSON under `research/oracles/h13/batched/`. No HWX bytes retained. Index: `capture_index.json`.

## For TransposeAbsorb

- Template key still `{rows, reduction, columns, tx, ty, runtimeWeight}`; add `B`.
- Encode B GEMMs as **one task stream**, `taskCount = 26*B` (`tx=ty=false`), with per-batch DMA bases (tile-DMA `src_off` stride = padded y plane).
- Const y: **one packed weight section**, `constantBytes` as above, permutation is Apple's usual matvec packing plus leading batch of B planes.
- Do not lower `[1,375,8,128] × [1,8,128,D]` — Apple rejects it.
