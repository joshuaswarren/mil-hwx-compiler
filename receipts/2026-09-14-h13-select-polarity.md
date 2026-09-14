# H13 select polarity (2026-09-14)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

Yes: Apple template cond polarity is opposite MIL `select(a, b, cond)`
when a is bound to template 5. MIL is `cond ? a : b`. The kernel is
`cond ? template4 : template5`.

Fix: select `taskSurfaceChannels` `{7, 5, 4, 6}` → `{7, 4, 5, 6}`.
Operand swap, not cond negate, not an L2 patch. `{64,1,1}` is then
exact without host invert.

## Source

- Compiler: `feature/h13-concat` `8782cf16c9378911e980a8bb7193a7e1ffc6e3e5` plus this remap
- Oracle: `research/oracles/h13/boolean/select_rrb_1x64x1x1.json`
- Sibling device fact (not executed here): at `8f97f50` (DMA remap,
  a-on-5), `{64,1,1}` was `exact_fp16_mil=false` and
  `exact_invert_cond=true`, leftover 0 vs invert. Anecs `5f8c326f`.

## Remap

Canonical ANEC: y ch4, a ch5, b ch6, cond ch7.

| | `taskSurfaceChannels` | Apple 4 | Apple 5 | Apple 6 |
| --- | --- | --- | --- | --- |
| DMA remap | `{7, 5, 4, 6}` | b | a | cond |
| polarity | `{7, 4, 5, 6}` | a (cond-true) | b (cond-false) | cond |

Cond stays template 6 (375-wide DMA row 384). a and b are both fp16
(row 768). Swapping them does not reopen the bool-stride bug.

Emitted `{64,1,1}` anec
`732301a3cae256dca2a8483e49d92f3831c484f9873fdd9ee35e359e85ff508d`
(9216 bytes):

```text
task0 src1=7
task1 src2=5   # template 4 → MIL a
task3 src2=6   # template 5 → MIL b
task4 dst=4
const u16 0x8001 0x0001
```

375-wide DMA unchanged: task0 src1=7 row 384 / plane 144000, L2
`0x0480c=384` `0x04810=3072`. Named hole `h13.select-first-l2-tile`
stands. Did not patch L2.

## Why operand swap

Invert-cond and swapping a/b are the same select. Negating cond packing
would also invert `less` results consumed as cond. The remap is local
to select.

## Verification

```text
python3 tests/test_h13_boolean_cli.py build/mil-hwxc
# h13 boolean cli: PASS

python3 tests/test_h13_select_envelope_cli.py build/mil-hwxc
# h13 select envelope cli: PASS
```

`assert_select_64_polarity` locks `{64,1,1}` selectors above.
`SELECTOR_REMAP["select"] = {7: 4, 4: 5, 5: 6, 6: 7}`.

`build/mil-hwxc` sha256:
`77521fb36681f8ace6e4aff97ef7ca856b57137e3808680f85dfd4742248de61`.

No ANE execute. Did not change worker 0x01/0x00 packing.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
