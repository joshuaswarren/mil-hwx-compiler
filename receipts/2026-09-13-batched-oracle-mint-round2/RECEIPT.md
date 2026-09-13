# H13 oracle mint receipt — batched round 2 + boolean + pad/tile

## Source

- Compiler base: `7fd06ac89f63694571795f2f9e26e75b78e58550` (`feature/h13-batched-matmul`)
- Branch: `research/oracles-batched-round2` (worktree; implementation checkout was not touched)
- Campaigns:
  - `receipts/2026-09-13-batched-oracle-mint-round2/mint_batched_round2.py`
    SHA-256 `113f93c8562625cb679f60b83116c03bb3360058a60356e496646421db6df1b6`
  - `receipts/2026-09-13-batched-oracle-mint-round2/mint_boolean_ops.py`
    SHA-256 `594e4f1cb62eb5e7628a49eb187c65be862021e8d56467450c566681ab57ebfe`
  - `receipts/2026-09-13-batched-oracle-mint-round2/mint_pad_tile.py`
    SHA-256 `0f856fd41a2e4923d9cba8f89ba4b4e801139064ae77be254b89a6f60d16489e`
- Decoder/driver (recorded in each JSON `compiler` block):
  - `research/mint_oracles.py` SHA-256 `a6e8f70d438973e60913815be66dbd6de19d7200df9053a9efc2a615823d8074`
  - `research/h13_td.py` SHA-256 `125da0249ca3c3ba0ca1f9f0cabfd1db54a965abb75f9f4cdcc2598e255c8fbc`
- Apple tool on MacStudio (reused from round 1, not rebuilt):
  - binary `/tmp/h13-oracle/bin/ane-compile-hwx` 52336 B,
    SHA-256 `3d13fc85c2a6baa0b7628f0848269b16fb99f742180f0085f608c371bb9da6e0`
  - source `/tmp/h13-oracle/ane-compile-hwx.mm` 3009 B,
    SHA-256 `55c6ae904e65c115066e980dcf92d4072f33dbc9acef71c151f30c5c69b610e8`
- Host: `MacStudio.local`, macOS 26.6.2 arm64, build 25G83
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback)
- No ANE execution, no reboot, no system change. `ane-linux-experiments` was not edited or executed. Compiler plugin code was not modified.

## Command

```text
python3 receipts/2026-09-13-batched-oracle-mint-round2/mint_batched_round2.py --host macstudio
# 18 cases, 18 decoded, 0 rejected

python3 receipts/2026-09-13-batched-oracle-mint-round2/mint_boolean_ops.py --host macstudio
# 35 cases, 20 decoded, 15 rejected (callback_status=1)

python3 receipts/2026-09-13-batched-oracle-mint-round2/mint_pad_tile.py --host macstudio --force
# 10 cases, 4 decoded (all tile), 6 rejected (all pad, callback_status=1)
```

Indexes: `capture_index_batched.json`, `capture_index_boolean.json`, `capture_index_layout.json`.

## Gap 1 — packed-const y, value = f(index)

Input payload is little-endian `uint16(index+1)` stored as fp16 bits: never zero, includes subnormals (`0x0001..0x03ff`), wraps at 65536. Distinct from round 1's uniform `0x3800`.

Every const-y case retains `{name}.weights.bin` (input) and `{name}.const.bin` (Apple `__TEXT,__const`).

| case | tasks | const bytes | prefix uniform 0x3800 | unique u16 in first 4096 |
|---|---:|---:|---|---:|
| scores r3 blob B=2/4/8/16 `_idx` | 52/104/208/416 | 393216..3145728 | False | 3941 |
| attn-output r3 blob B=2/4/8/16 `_idx` | 52/104/208/416 | 192000..1536000 | False | 4036 |
| r4heads blob B=8 both geometries `_idx` | 208 | 1572864 / 768000 | False | 3941 / 4036 |
| scores ty=1 blob B=8 `_idx` | 209 | 1533952 | False | 4036 |
| scores tx=1 blob B=8 `_idx` | 209 | 1572864 | False | 3941 |

Smoke (scores B=2): input starts `0x0001,0x0002,0x0003,…`; packed const starts `0x0001,0x0000,0x0002,0x0000,…`. Packing is directly checkable.

**PASS** all 12 const-idx remints (10 geometry remints + 2 fold-flag const).

## Gap 2 — fold-flag tx=1 / ty=1 at B=8 (209-task form)

Scores runtime fold flags already existed from round 1 (`bmm_r3rr_m375_k128_n749_tx{0,1}_ty{1,0}_b8`, 209 tasks). Not reminted.

New:

| case | status | tasks |
|---|---|---:|
| attn-output ty=1 runtime B=8 | decoded | 209 |
| attn-output tx=1 runtime B=8 | decoded | 209 |
| scores ty=1 blob idx B=8 | decoded | 209 |
| scores tx=1 blob idx B=8 | decoded | 209 |

**PASS** all four. 209 = 26·8+1.

## Gap 3 — V-projection 375×128×375

| case | ty | storage | status | tasks | const bytes |
|---|---:|---|---|---:|---:|
| `bmm_r3rr_m375_k128_n375_tx0_ty0_b8` | 0 | runtime | decoded | 208 | 16384 |
| `bmm_r3rr_m375_k128_n375_tx0_ty1_b8` | 1 | runtime | decoded | 209 | 16384 |
| `bmm_r3rb_m375_k128_n375_tx0_ty0_b8_idx` | 0 | blob idx | decoded | 208 | 786432 |
| `bmm_r3rb_m375_k128_n375_tx0_ty1_b8_idx` | 1 | blob idx | decoded | 209 | 768000 |

ty=0 is 208 tasks (26·8); ty=1 is the 209-task fold form. Transposed const packing is smaller (768000 vs 786432).

**PASS** all four.

## Boolean ops (parent add)

Mint list from `receipts/2026-09-13-h13-registry-boolean-ops.md`. Apple `callback_status=1` is a finding.

### less

fp16-typed result (`tensor<fp16,…> z = less(...)`) **rejected** at every shape: 375/750/1500 flat, `[1,375]`, `[1,64,1,1]`, `[1,512,1,1]`, and const-y idx forms. Error: `callback_status=1`.

bool-typed result **decoded**:

| case | tasks | const |
|---|---:|---:|
| `less_bool_rr_375x1x1` | 3 | 3072 |
| `less_bool_rr_750x1x1` | 3 | 4096 |
| `less_bool_rr_1500x1x1` | 3 | 5120 |
| `less_bool_rr_1x64x1x1` | 3 | 3072 |
| `less_bool_cast_1x375` (bool then cast to fp16) | 4 | 192 |

**PASS** bool forms. **REJECT** fp16-result forms (Apple requires bool output).

### floor

| case | status | tasks | notes |
|---|---|---:|---|
| `floor_r_1` | decoded | 4 | Apple rewrites `[1]` → `[1,1,1,1]` |
| `floor_r_1x64x1x1` | decoded | 4 | |
| `floor_r_1x512x1x1` | decoded | 4 | |
| `floor_b_1_idx` | decoded | 4 | raw const 448 B retained |

Task sizes 628/628/628/504. **PASS** all four, including `[1]`-shaped.

### select

fp16 0/1 cond **rejected** (`callback_status=1`): `-inf` scalar-a, `-inf` tensor-a, runtime-a at `[1,8,375,375]`, `[1,1024,375]`, `[1,64,1,1]`.

bool cond **decoded**:

| case | tasks | const |
|---|---:|---:|
| `select_rrb_1x8x375x375` runtime-a | 5 | 2048 |
| `select_rrb_1x64x1x1` | 5 | 2048 |
| `select_ninf_bool_1x8x375x375` (`-inf` a blob) | 5 | 2258048 |

`-inf` packing retained as `select_ninf_bool_1x8x375x375.const.bin` / `.weights.bin`.

**PASS** bool-cond forms including `-inf` a and runtime-a. **REJECT** fp16-cond forms.

### floor_div native vs composition

Inline `tensor<fp16,[1]>([fp16(2.0)])` **rejected** (`callback_status=1`). Scalar `fp16(2.0)` **decoded**.

| case | form | tasks | hwx vs pair |
|---|---|---:|---|
| `floor_div_native_1_rr` | native runtime | 6 | ≠ composed |
| `floor_div_comp_1_rr` | real_div+floor | 6 | const hash equal |
| `floor_div_native_1x64x1x1_rr` | native runtime | 6 | ≠ composed |
| `floor_div_comp_1x64x1x1_rr` | real_div+floor | 6 | const hash equal |
| `floor_div_native_1_s2` | native scalar 2.0 | 5 | ≠ composed |
| `floor_div_comp_1_s2` | real_div+floor scalar 2.0 | 5 | const hash equal |
| `floor_div_native_1x64x1x1_s2` | native scalar 2.0 | 6 | ≠ composed |
| `floor_div_comp_1x64x1x1_s2` | real_div+floor scalar 2.0 | 6 | const hash equal |

Apple accepts a native `floor_div` and a `real_div`+`floor` composition. They share constant-section hashes and task counts but **not** HWX hashes — native is not a byte-identical rewrite of the composition.


## Pad and tile (parent add)

MIL matches `tests/test_h13_layout_cli.py` (`pad(mode = string("constant"), …)` then relu; `tile(reps=…)` then relu).

### pad — Apple rejects every form

| case | status |
|---|---|
| identity `[1,64,1,1]` amounts all-zero | **REJECT** `callback_status=1` |
| encoder `[1,1,749,375]→[1,1,750,375]` amounts `[0,0,0,0,0,0,1,0]` runtime + idx | **REJECT** |
| scores-shaped `[1,8,375,749]→[1,8,375,750]` amounts `…10` and `…01` runtime + idx | **REJECT** |

Cannot observe materialized vs descriptor for pad: Apple's `ane-compile-hwx` does not compile `pad` (including the identity view).

### tile — materialized, not a descriptor trick

| case | status | tasks | tensor bytes |
|---|---|---:|---|
| identity reps `[1,1,1,1]` | decoded | 1 | 4096 → 4096 (two surfaces) |
| encoder `[1,1,375]` reps `[1,375,1]` runtime | decoded | 1 | **768 → 288000** |
| encoder same, blob idx | decoded | 1 | output 288000; const 768 (unexpanded input) |
| heads `[1,8,1,375]` reps `[1,1,375,1]` | decoded | 1 | **6144 → 2304000** |

Apple rewrites `[1,1,375]` to NCHW `[1,1,1,375]`. Output extent is the full tiled plane (`375×375×2 = 288000`). Input and output are distinct bindings. Const-x tile keeps the **unexpanded** 768-byte payload in `__const` and writes a 288000-byte runtime output — tile is executed, not pre-expanded and not a view.

**PASS** tile encoder + identity. **REJECT** every pad form.


## Files

- Batched oracles + raw bytes: `research/oracles/h13/batched/` (`*_idx`, V-proj `*n375*`, attn-output fold flags)
- Boolean oracles + raw bytes: `research/oracles/h13/boolean/`
- Pad/tile oracles + raw bytes: `research/oracles/h13/layout/`
- This receipt directory: campaigns, indexes, structure summaries

HWX bytes were not retained. Round-1 uniform const JSONs were not overwritten.
