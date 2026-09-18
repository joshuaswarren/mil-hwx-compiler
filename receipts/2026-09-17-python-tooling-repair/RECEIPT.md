# Python tooling repair: inspect_anec NameError, slice fixture, H14 norm
# emitter staleness, and the masked unsigned-zero simulator gap (2026-09-17)

Assigned off `80c7b4d` after the F4 round surfaced three pre-existing suite
failures that reproduce byte-identically on pristine `073c7b8`
(`worktrees/mil-hwx-compiler/slice-lastdim`, clean status): test-h13-reference,
test-h13-simulation, test-h14-parity. All four files fixed here are python-side
or generated tables; no compiler/C++ change, no oracle record touched, no
assertion weakened.

## 1. `inspect_anec.py` NameError (`destination` at :134)

`validate_task_headers` evaluated `destination == 0x000000c0 and
destination_channel == 0` in the channel-1 fallthrough, but the two locals
were deleted by `5ac214c` ("h13: name every declared surface in the task
stream") while restructuring the branch — the references date to `4252d76`,
where the definitions sat immediately above. Every H13 manifest reaching that
require crashed with NameError, which is what test-h13-simulation died on.
Fix: restore the two definition lines verbatim from `4252d76` (`registers.
get(0x17800, H13_DMA_DISABLED)` and `(words[8] >> 12) & 0x1f`) with a comment.

## 2. test-h13-reference "slice has incorrect fields"

`test_intermediate_physical_buffer_composition`'s consumer binding carried a
slice record without `physicalElements`, while `binding_interval` requires
exactly `{tensor, elementOffset, elementCount, physicalElements}` — and real
manifests always carry it (`test_h13_cli.py:1140`). The fixture was stale,
not the validator: the consumer's 32768-byte allocation is 512 x 64-byte
lanes, so the fixture now reads `{"tensor": "h", "elementOffset": 0,
"elementCount": 512, "physicalElements": 512}` (>= elementCount as the
validator demands; `_intermediate_buffer` uses the 64-byte-lane nchw stride,
which is what the expected permutation asserts).

## 3. test-h14-parity "H14NormTemplates.inc is stale"

`2fa23c3` ("h13: qualify encoder leftover lowerings — parity 846 to 861")
changed the shared `axis_mask` convention in `research/mint_norm_probes.py`
(canonical CHW axis = MIL axis shifted by leading/trailing unit dimensions,
plus the NCHW batch axis) and regenerated `H13NormTemplates.inc` — whose emit
passes the record's shape into `axis_mask(axes, shape)` — but missed the H14
sibling emitter: `mint_h14_norm_probes.emit` still called
`axis_mask(axes)` with no shape, and the new signature silently applies a
+1 shift even unshifted. Any regeneration therefore rewrote all 186 masks
(e.g. 0x0e -> 0x1c) — wrong, because the checked-in table's masks already
agree with the shape-aware formula (verified: `axis_mask((1,2,3),
(1,1,1,64)) == 0x0e` = the checked-in value).

Fix: pass `record['parameters']['shape']` in the H14 emit, exactly like the
H13 emit. The regenerated table is byte-identical to the checked-in one —
`--check` reports up to date, 186 entries, **zero content change, no oracle
expectation altered**. Regenerating the table instead (as the staleness
message suggests) would have corrupted all 186 masks; the emitter was the
bug.

## 4. Masked fourth failure: simulator missed the measured unsigned-zero rule

With the NameError gone, test-h13-simulation ran past its old crash point for
the first time since `5ac214c` and failed: `scalar_mul mismatch []` —
0.0 x (-1.0) byte-mismatched with no element off by more than 1e-6, i.e. a
sign-of-zero disagreement. The reference has modeled Apple's measured H13
datapath since `aa688df` ("fix(h13): model unsigned zero products",
`test_h13_mul_zero_sign`: 0.0 x -1.0 = +0.0) in `tools/h13_reference.py`, but
the simulation model `simulate_binary`/`binary_op` in
`tests/test_h13_package_simulation.py` still used raw Python IEEE semantics
(-0.0). Fixed `binary_op` to the measured rule (`0.0 if product == 0.0`),
which covers both the elementwise and `simulate_broadcast` paths. Hardware
semantics unchanged elsewhere; NaN passes through the `== 0.0` guard.

## Verification (this tip, `agent/h13-py-tooling-fixes`)

- `make test-h13-reference`: PASS 8 H13 reference tests + H13_DEADLINE_OK.
- `make test-h13-simulation`: ALL SIM CHECKS PASSED.
- `make test-h14-parity`: PASS (718 cases / 1436 artifacts; 186 norm
  templates, byte-rebuilt sections).
- Regression: `make test-h13` PASS (full battery), `make test-h13-parity`
  PASS (890 cases / 1780 artifacts — unchanged from 80c7b4d),
  `make test-hwx-inspection` PASS.

## Known merge points with sibling branches

- `agent/f1-u32-colgroups` (bbe5520) also edits
  `tests/test_h13_conv_envelope_cli.py` (enc-1d-pw positive, 4 MiB blob —
  take 4 MiB) and touches `mint_conv_probes.py`/`H13Program.cpp` in the
  dense branch only; this branch touches none of those files, so the only
  real overlap is the CLI test file.
- After any sibling merge that adds conv rows, re-run
  `mint_conv_probes.py --emit-templates --targets h13` (symbol renumbering
  churns the diff; `--check` is the gate).
