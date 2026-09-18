# Encoder conv integration: F4 padconv + F1 colgroup order + python tooling
# on one tip (2026-09-17)

Merge of `agent/f1-u32-colgroups` (`bbe5520`, F1 round 3) into
`agent/h13-py-tooling-fixes` (`a3bf9cb`, padconv 80c7b4d + tooling repair),
on branch `agent/encoder-conv-integrated`. Main-directed integration; no new
lowering work here.

## Conflicts and their resolution

- `research/mint_conv_probes.py`: single hunk — `MULTI_TASK_CASES` resolved
  by union (2 padconv rows + F1's wmaj row alongside the three pre-existing).
  The auto-merged dense/depthwise dispatches verified present on both sides
  (`pack_padconv_columns` + its depthwise-branch dispatch; F1's
  `pack_dense_colgroups` + dense-branch dispatch + lane_cap max-extent fix).
- `plugins/H13/H13ConvTemplates.inc`: regenerated from the combined record
  set per protocol (`--emit-templates --targets h13 h14`), never hand-merged.
  H14 output idempotent on top of the emitter fix.
- `tests/test_h13_conv_envelope_cli.py`: both positive blocks kept
  (`enc-padconv` 2-task; `enc-1d-pw-inproj` 4-task); refusal set drops both
  promoted ops and keeps `k3-st2-square` + the stem/strided forms; blob line
  at F1's 4 MiB (subsumes the 2 MiB the padconv cases need).
- Everything else auto-merged (ANEH13Compiler.mm untouched by F1;
  H13Program.cpp disjoint branches; encoding.cpp additive; oracle/receipt
  files additive).

## Verification (merged tip, pre-landing)

- `make test-h13`: PASS (17 PASS lines, zero FAIL/Error).
- `make test-h13-parity`: **PASS 891 cases / 1782 artifacts / 296
  convolution** — exactly the prediction (base 888 + 2 padconv + 1 F1 row).
- `make test-h13-reference`: PASS 8 H13 reference tests + H13_DEADLINE_OK.
- `make test-h13-simulation`: ALL SIM CHECKS PASSED.
- `make test-h14-parity`: PASS (718 cases / 1436 artifacts).
- `make test-hwx-inspection`: PASS.

## State of the encoder conv leftovers on this tip

F3 24, F2 24, F4 24, F1 24 lowered — 96 of 101 instances. Refused: F5/F6/F8
(3, stride-2 sections carry transformed halfwords). Refusals keep their named
bounds in the conv reject message and the CLI/encoding tests.
