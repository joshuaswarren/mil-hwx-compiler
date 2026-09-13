# H13 batched matmul envelope receipt (primitive implemented from minted oracles)

## Source

- Compiler base: `12d5f05` (`feature/h13-batched-matmul-survey`)
- Branch: `feature/h13-batched-matmul`
- Oracles: `research/oracles/h13/batched/` (22 decoded + 1 Apple-rejected),
  minted by OracleMint on MacStudio via Apple's own ANECompiler — receipt
  `receipts/2026-09-13-batched-oracle-mint/RECEIPT.md`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback)

## What shipped: the batched matvec primitive, runtime path fully verified

Rank-3 `[B,rows,K] × [B,K,N]` and rank-4 `[1,B,rows,K] × [1,B,K,N]` matmuls
with both transpose flags false now lower as **one program whose task stream
is Apple's own**: batch zero's 26-task group stamped once per batch with
per-batch word rules, linked exactly as decoded.

### Envelope (16 entries, generator-verified)

`research/generate_batched_matmul_tables.py` reconstructs each capture's raw
task words from the decoded register-stream JSON, classifies every word as
affine-in-batch (base + b·delta: x-tile addresses, y base, output base, link
pointers) or per-batch literal (first-task markers, task headers), verifies
the model reproduces **every batch of every capture exactly**, and emits
`plugins/H13/H13BatchedMatmulTemplates.inc`. Rank-3 and rank-4 captures are
byte-identical at equal (geometry, storage, B) — one entry serves both.

Covered: (375,128,749) and (375,375,128), B ∈ {2,4,8,16}, runtime y — 16
entries; the two fold-flag captures (tx=1 / ty=1, 209 tasks = 26·8+1) are
not yet in the envelope (different group structure, needs its own analysis).

### Byte-exactness (Main's acceptance bar)

`tests/test_h13_batched_cli.py` compiles **the capture's own MIL** and
asserts the emitted ANEC task stream equals the capture reconstruction
(after the link-marker rewrite `bindTasks` always applies):

- **10/10 runtime captures byte-equal**, rank-3 and rank-4, all four B
  values, both geometries;
- 10/10 packed-const captures byte-equal **at the task-stream level**;
- deterministic in both `anec` and `hwx`; inspector-validated;
- manifest: one program, `taskDescriptors = 26·B`, encoder
  `apple-parity-batched-matmul`, whole-tensor `[1,B,rows,width]` bindings.

### Deliberately gated: packed-const constant section

The const captures carry **uniform fp16 weights**, so their recorded
constant-section hashes cannot discriminate the packing (any permutation of
equal bytes hashes alike) — and the nonzero-byte counts prove the section is
NOT plain padded planes (55 zero bytes appear in a 192000-byte all-0x3800
section; the tail is exactly half zero after a ~119-byte head). Rather than
emit unverified constant bytes, the compiler rejects batched constant y with
the exact reason until OracleMint re-mints with **non-uniform weights**
(value = f(index)) — ideally retaining the raw constant section bytes. The
encoder-side effect: the scores matmul (y = const `[1,8,128,749]` BLOBFILE)
stays blocked; the V-projection and attn-output matmuls (runtime y) ship.

### Known-open items for the next mint/analysis round

1. Non-uniform-weight const re-mint (unlocks scores + verifies packing).
2. Fold-flag captures (tx=1/ty=1 at B=8, 209 tasks) — the V-projection form
   `(375,128,375,ty=1)` also has no capture at all.
3. HWX-format surface-address consistency: task words carry Apple's absolute
   DMA addresses verbatim (no relocations), which is exact for the ANEC
   artifact; whether my HWX writer's binding-order address plan matches
   Apple's for these programs is unasserted (hwx_sha256 comparison left as
   a follow-up once the const section lands).
4. The 48 `[0,2,-3,-1]` transposes: the mixed pre-transpose layout
   `[1,375,8,128] × [1,8,128,749]` is **rejected by Apple's own tool**
   (r4mixedrr, callback_status=1), so the transposes cannot fold away by
   pointing the batched matmul at the pre-transpose operand on macOS either.
   They remain blockers needing the c-plane materialization decomposition;
   what the envelope unlocks is their consumers.

### Supporting changes

- `HWXObjectWriter` task-stream cap raised 0x20000 → 0x60000: the decoded
  batched streams run 324,212 bytes at B=16 (the old cap cited the
  single-batch maximum).
- `research/inspect_anec.py` `surface_layout` now pads rows up to the 64-byte
  stride (`alignUp`) instead of flooring at 64 — the batched descriptors
  carry 1536-byte rows for 749-wide planes; identical for every previously
  decoded surface. New encoder names recognized with a 26-per-batch check.
- Rejection upgrades: uncaptured batched forms (e.g. B=3, flag variants)
  reject naming the exact envelope; the survey-era "needs a primitive"
  message updated to the post-envelope state; mixed layouts reject at the
  geometry gate exactly as before.

## Commands and observed results

```text
python3 research/generate_batched_matmul_tables.py
  # verified 16 entries (fatal on any model violation)
make -j8 build/mil-hwxc                    # 0 errors
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS
  h13 composite cli: PASS                  # survey pins removed (now supported)
  h13 batched cli: PASS                    # new suite: 10/10 byte-equal + contracts
python3 tests/test_h13_parity.py build/mil-hwxc
  AssertionError: binary_add_1x1024x1x1 ...  # pre-exists (ParityTriage), unchanged
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or
Mac execution was performed by this agent (OracleMint ran the Mac mint).
No files under `ane-linux-experiments` were edited or executed.
