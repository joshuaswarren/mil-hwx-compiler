# H13 select envelope (2026-09-13)

Host-only. No ANE, no SSH, no `ane-linux-experiments`, no jwm1.

Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed).

## Source

- Compiler: `feature/h13-concat` `c2cf32e4aa0200d72cecbce203bf6b47f50f729e` plus this envelope pin
- Boolean table: `plugins/H13/H13BooleanTemplates.inc` `kBooleanTasks`
- Oracle: `research/oracles/h13/boolean/select_rrb_1x8x375x375.json` (runtime-a, 5 tasks, 2048 const bytes)
- Oracle: `research/oracles/h13/boolean/select_ninf_bool_1x8x375x375.json` (tensor `-inf` a, 5 tasks, 2258048 const bytes)
- Leftover count: encoder compile attempts, 24 `select` (`0xFC00` −inf fill) after `lower_mask_ops` rewrote the 24 finite selects

## Envelope

`kBooleanTasks` select rows are exact-match on `(constInput, CHW)`:

| constInput | CHW | Capture | In table? |
| --- | --- | --- | --- |
| false | `{8,375,375}` | `select_rrb_1x8x375x375` | yes |
| false | `{64,1,1}` | `select_rrb_1x64x1x1` | yes |
| true | `{8,375,375}` | `select_ninf_bool_1x8x375x375` | yes |
| false | `{1024,375,1}` / `{1,1024,375}` | — | no |
| true | `{64,1,1}` | — | no |

Bool cond is load-bearing. fp16-cond forms were rejected by Apple (`callback_status=1`). Scalar `-inf` a was rejected the same way; the only decoded const-a form is a full `[1,8,375,375]` blob.

Host compile of the encoder-shaped runtime-a point (one program, `apple-parity-boolean`):

```text
python3 - <<'PY'
# select(a,b,cond) fp16/bool [1,8,375,375], all runtime
=== enc-rrb rc 0
programs 1 op select enc apple-parity-boolean tasks 5
PY
```

`[1,64,1,1]` runtime-a compiles the same way (1 program, 5 tasks). `[1,1024,375]` runtime-a is `h13.boolean-outside-envelope`.

## Hole

The leftover 24 encoder selects are constant `-inf` a at `[1,8,375,375]`. The geometry is inside the envelope. The MIL gate refuses every constant-a form before table lookup:

```text
6:5: error [h13.select-needs-decoded-encoder]: H13 select with a constant a belongs to the frontend, which materializes the fill as a runtime constant input and routes the +0.0-fill family through its exact mul rewrite: the captured constant-a form carries uniform -inf values whose retained section cannot discriminate the packing for arbitrary constants, so this path lowers only runtime-a forms with a bool cond
```

exit 65, no output dir. Same code for scalar `tensor<fp16,[]>` `-inf` and tensor `tensor<fp16,[1,8,375,375]>` blob a.

`supportsBooleanOp({Select, true, 8, 375, 375})` is true. The ninf capture's constant section is 2258048 bytes, prefix `0x0001, 0xfc00, …`, 9 unique u16 in the first 8192. That packing is not a patch map for an arbitrary fill. Do not emit it. Do not invent a concat or copy of `-inf` into the output.

The close is frontend materialization of the fill as runtime-a, which is already decoded.

## Verification

```text
make -C ~/src/mil-hwx-compiler build/test_h13_encoding
./build/test_h13_encoding
# H13_ENCODING_OK

python3 tests/test_h13_select_envelope_cli.py build/mil-hwxc
# h13 select envelope cli: PASS
```

`build/mil-hwxc` sha256 after this pin (unchanged; no encoder change): `82f1d4ce44dc8f2cb5eefd843eea41c47527af3fbff2def0ed1fc154ccd3ce71`.

Did not compile the full encoder. Did not emit the 2.2 MiB ninf packing. Did not fake concat.

## Hardware boundary

No SSH, hardware inference, ANE command, router change, or Mac execution.
No files under `ane-linux-experiments` were edited or executed.
