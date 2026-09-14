# H13 chain envelope for attn+select (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` (`feat(h13): name concat as an ISA hole`)
- `build/mil-hwxc` sha256: `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`
- Island MIL: four boundary inputs `x` fp16 `[1,8,375,128]`, `w` fp16 `[1,8,128,375]`, `ninf_rt` fp16 `[1,8,375,375]`, `cond` bool `[1,8,375,375]`; `matmul` then `select(a=ninf_rt, b=product, cond)`
- Sibling island: mlx-omarchy `receipts/2026-09-13-attn-select-island.md`

## Package envelope

Default `--schedule=per-op` is the envelope. Two programs, intermediate `product`, `dispatchPlan [0, 1]`, schema `mil-hwxc.h13-anec-package.v2`.

```text
compiled target=H13 artifacts=2 format=anec
```

| program | op | encoder | TDs | sha256 | bytes |
| --- | --- | --- | ---: | --- | ---: |
| 0 | matmul | `apple-parity-batched-matmul` | 208 | `a3aa2fe1333a48eba3e7fcae5b848563e2bfa747a232dadf9ed2d568821c5627` | 182400 |
| 1 | select | `apple-parity-boolean` | 5 | `860de06c53e3fdfed9452859042df59902fb733bef0a364b134dd9d9d6374cdd` | 9216 |

Hashes match the standalone programs. Not fused.

## Chain hole

`--schedule=chain` is `h13.chain-outside-envelope`. Four boundary inputs. rc=65, no output dir.

```text
7:5: error [h13.chain-outside-envelope]: H13 composed scheduling needs at least two operations and at most two boundary inputs
```

That gate is load-bearing for this island: runtime-a select cannot drop `ninf_rt` or `cond` without becoming `h13.select-needs-decoded-encoder`.

The decoded chain envelope is elsewhere, and it is not this pair:

| graph | schedule | result |
| --- | --- | --- |
| add+relu `[1,512,1,1]`, 2 inputs | chain | 1 program, `composed-chain`, 1 TD, sha256 `8be2a7e3fae166f007196d5fe9808c14481764dcfc15d60a37e2e264e907b4bf` |
| attn matmul+relu, 2 inputs | chain | `h13.chain-unrepresentable-edge`: `apple-parity-batched-matmul` has no decoded post-operation field |
| attn+select, 4 inputs | chain | `h13.chain-outside-envelope` (this island) |

`PostOperation` is only `Relu`. Fusion writes one clamp bit on `apple-parity-matvec` or `h13-oracle-parity`. `composePrograms` returns a single already-fused program and refuses a multi-program relink: task DMA is wired to boundary channels, with no addressing for a declared intermediate.

Do not stitch the 208-task batched matmul to the 5-task select. That would invent intermediate DMA.

## Verification

```text
python3 tests/test_h13_chain_envelope_cli.py build/mil-hwxc
# h13 chain envelope cli: PASS
```

Did not compile the full encoder. Did not emit a fused attn+select program.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
