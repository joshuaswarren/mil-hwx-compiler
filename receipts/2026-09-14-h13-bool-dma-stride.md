# H13 bool cond DMA stride (2026-09-14)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

Compiler bug, not a named ISA hole. The 375-wide select *header* was already
right (ch7 bool NCHW row=384). `bindTasks` remapped Apple's bool-stride DMA
onto fp16 `a` (ch5) and Apple's fp16-stride DMA onto cond (ch7).

Fix: select `taskSurfaceChannels` `{7, 6, 4, 5}` → `{7, 5, 4, 6}`.
Template ch6 DMA is row 384 / plane 144000 (1-byte). Template ch5 DMA is
row 768 / plane 288000 (fp16). Cond is the 1-byte surface.

Upstream packing receipt (mlx-omarchy, not landed on origin):
`receipts/2026-09-14-select-bool-packing.md`. Anecs `860de06c` was the
pre-fix emit. Header was not lying.

## Source

- Compiler: `feature/h13-concat` `738ce0a0eeea2650a696da1af0591f973e5d3741` plus this remap
- Oracle: `research/oracles/h13/boolean/select_rrb_1x8x375x375.json`
- Apple task 0: src1 template 6, `0x1380c=384`, `0x13810=144000`, depth `1152000`
- Apple task 1: src2 template 5, `0x13820=768`, `0x13824=288000`, depth `2304000`

`bindTasks` rewrites selectors only. Strides stay with the template slot.

## Before / after

Canonical ANEC: y ch4, a ch5, b ch6, cond ch7.

| | `taskSurfaceChannels` | task0 src1 | task1 src2 |
| --- | --- | --- | --- |
| before | `{7, 6, 4, 5}` | ch5 (`a`) row 384 | ch7 (`cond`) row 768 |
| after | `{7, 5, 4, 6}` | ch7 (`cond`) row 384 | ch5 (`a`) row 768 |

Header ch7 stays `[1, 8, 375, 375, 144000, 384]`. Header ch5 stays
`[1, 8, 375, 375, 288000, 768]`.

Pre-fix anec `860de06c53e3fdfed9452859042df59902fb733bef0a364b134dd9d9d6374cdd`
(9216 bytes). Post-fix anec
`27dc7f35fc9efc9431e4d05847798390fe0cfe79f930c3040f32dd76ef8e8e73`
(9216 bytes). Selectors differ; constant section unchanged.

`1x64x1x1` cannot show this: both dtypes pad to row 64.

## Verification

```text
python3 tests/test_h13_boolean_cli.py build/mil-hwxc
# h13 boolean cli: PASS

python3 tests/test_h13_select_envelope_cli.py build/mil-hwxc
# h13 select envelope cli: PASS
```

`assert_select_375_bool_dma` locks header ch7 row 384 against task0 src1=7
DMA 384/144000, and task1 src2=5 DMA 768/288000. Byte-exact vs the oracle
uses `SELECTOR_REMAP["select"] = {7: 4, 5: 5, 4: 6, 6: 7}`.

`build/mil-hwxc` sha256 after this pin:
`38c4d2cbc279624b2a151a0137775ce14d9fa2ac1adbbf0b5d8ff8ec787f1ef1`.

No ANE execute. Did not change worker 0x01/0x00 packing. Did not grow the
cond allocation to fp16.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
