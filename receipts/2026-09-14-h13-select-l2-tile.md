# H13 select first L2 tile (2026-09-14)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Verdict

Named hole `h13.select-first-l2-tile`. Not a compiler rewrite.

Leftover 5412 `-inf` after invert-cond (mlx-omarchy
`receipts/2026-09-14-select-tile-edge.md`) sits on the decoded kernel's
first 8-row L2 cond tile: Apple task 0 already has L2 `0x0480c=0x180`
and `0x04810=0xc00` (3072=8×384). Constant section starts
`0x8001, 0x0001`. `bindTasks` remaps selectors only. The 48-period
right wedge (`384/8`) plus H=8 is that first-tile cond compare-to-`0x0001`,
not host `tile_layout` packing.

Do not patch L2 or the immediate. A different first-tile form needs a
new capture.

## Source

- Compiler: `feature/h13-concat` `8f97f504edb4ed525767fad6d90f814ff1c13241`
- Oracle: `research/oracles/h13/boolean/select_rrb_1x8x375x375.json`
- Anec: post-DMA-remap `27dc7f35fc9efc9431e4d05847798390fe0cfe79f930c3040f32dd76ef8e8e73` (9216 bytes)
- Const: `6d07d38cdeb1a95b7b7acac96697b12a2e633349e554a8bf3057d2f2125acdd8` (2048 bytes, matches `select_rrb_1x8x375x375.const.bin`)
- Header ch7: `[1, 8, 375, 375, 144000, 384]`, tiles 71

## L2 vs 64-wide

| | task 0 L2 `0x0480c` | `0x04810` | const u16[1] |
| --- | ---: | ---: | ---: |
| `select_rrb_1x8x375x375` | 384 | 3072 | `0x0001` |
| `select_rrb_1x64x1x1` | 16 | 1024 | `0x0001` |

Task 2 on the 375-wide capture repeats row 384 / tile 3072 (last L2 slot 0).
Tasks 1/3/4 are fp16 tiles (`0x40` / `0x210`). Compare-to-`0x0001` is in
both captures; only the 375-wide program has the 8×384 first tile.

Emitted task 0 selectors are `0x04823027` (src1=ch7). Apple's capture
was `0x04823026`. DMA strides stay on the template slot.

## Why not a compiler fix

`encodeBooleanOp` copies `kBooleanTasks` words and constants verbatim.
`bindTasks` rewrites word 8 selectors and the `0x00400000` flag. L2
`0xc00` and const `0x0001` are Apple's. Retuning them without a capture
is inventing ISA.

Host compile of the runtime-a MIL:

```text
anec sha256 27dc7f35fc9efc9431e4d05847798390fe0cfe79f930c3040f32dd76ef8e8e73
task0 L2 0x0480c=384 0x04810=3072 0x04814=3072 0x04818=3072
const u16 0x8001 0x0001
```

`assert_select_375_bool_dma` locks those words.

No MIL reject: the 375-wide runtime-a form stays the decoded envelope.
The leftover is device first-tile cond read of that form.

## Verification

```text
python3 tests/test_h13_boolean_cli.py build/mil-hwxc
# h13 boolean cli: PASS
```

`build/mil-hwxc` sha256:
`38c4d2cbc279624b2a151a0137775ce14d9fa2ac1adbbf0b5d8ff8ec787f1ef1`.

No ANE execute. Did not change worker packing. Did not grow cond to fp16.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
