# H13 slice mid-range end receipt (prefix-only gap closed)

## Source

- Compiler base: `b4f4da9` (`feature/h13-registry-boolean-ops`)
- Branch: `feature/h13-slice-mid-range`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback)

## Change

`slice_by_index` computed its contiguous count as `extent - begin`, so any
range whose end fell inside the extent rejected
`h13.invalid-slice-shape` even with unit head dimensions — the latent gap
named in the graph-decompose receipt. The count now uses the resolved
`[begin, end)` pair (the upper bound the plan already normalizes from
`end`/`end_mask`/negative indices), making every contiguous range one
offset view through the same `valueBaseOffsets` mechanism the split
lowering established. One line of semantics; no new ABI surface.

## Verification

- Encoder-shaped half range `[0,375)` of 750 on the head axis
  (`[1,1,750,375] → [1,1,375,375]`): **byte-equal to the compilable
  nested-split direct form** — `split(axis=2, num_splits=2)` first half
  feeding the same relu — plus recompile determinism and inspector
  validation.
- Suffix half `[375,750)`: reads at element offset `375·375` exactly.
- The bias range `[0,375)` of 749 with every head dimension unit
  (`[1,1,1,749] → [1,1,1,375]`): one contiguous view; the per-program
  64-element slices tile `[0,375)` exactly.
- The encoder's real bias shape `[1,8,375,749] → [1,8,375,375]` keeps its
  exact chunked rejection — 375 rows precede the sliced axis, so the read
  is 3000 interleaved chunks, unchanged by this fix (pinned already).
- All prior slice contracts (prefix ranges, negative ends, begin/end masks,
  identity aliases) unchanged; full `make test-h13` green.

## Commands and observed results

```text
make -j8 build/mil-hwxc                    # 0 errors
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS                     # includes the new mid-range cases
  h13 composite cli: PASS
  h13 batched cli: PASS
  h13 registry cli: PASS
python3 tests/test_h13_parity.py build/mil-hwxc
  AssertionError: binary_add_1x1024x1x1 ...  # pre-exists (ParityTriage), unchanged
```

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change,
or Mac execution was performed. No files under `ane-linux-experiments`
were edited or executed.
