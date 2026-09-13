# H13 boolean-encoder validation, materialized tile, and pad citations

## Source

- Compiler base: `5229cc4` (`feature/h13-boolean-and-round2`, batched round 2)
- Branch: `feature/h13-boolean-and-round2`
- Ground truth: `research/oracles/h13/boolean/*.json` (15 implementable
  captures + const twins), `research/oracles/h13/layout/tile_*.json` (4),
  `research/oracles/h13/layout/pad_*.json` (6 Apple rejections)
- Encoder artifact read-only: `/tmp/ane-runtime-adapter-integrated-20260913/model.mil`
- Executing model: `openai-codex/gpt-5.6-sol` (assigned route; no fallback observed)

## What shipped

### 1. Boolean task-header validation (inspector + encoder)

The boolean task streams were byte-exact against the captures but had never
passed `research/inspect_anec.py` task-header validation. Four gaps, each
closed against capture evidence:

- **Blob-x floor writes through channel 5.** The oracle's `floor_b` task[3]
  selects DST channel 5, so its result binds the first-input slot — the same
  slot the blob tile twin uses. `encodeBooleanOp` now emits the blob-floor
  output as `booleanTensor(5, ...)` (runtime form stays on 4);
  `encodeANEC` places the output layout/tiles at `program.output.index`
  (gated to 4|5, with a no-sharing guard); the inspector checks outputs at
  their own binding index and requires inputs on 5.. in order.
- **Folded-constant reads on channel 1.** The boolean encoder streams folded
  constants/blobs on channel 1 with runtime operands on 4..7 (the inspector
  already knew the 4..7 plan). The channel-1 exception now covers
  `apple-parity-boolean` with `constantBytes > 0` at 0x13800/0x13804.
- **Scratch backing for channel-3 staging.** The 64-wide floor_div forms and
  the 8x375x375 selects stage through channel 3 (template word8 scan over
  all 15 templates; only those forms select 3). `encodeBooleanOp` detects
  template channel-3 use in its copy loop — channel 3 passes `bindTasks`
  through untouched — and backs it with the output allocation: the staging
  moves operand/result data so it never exceeds one surface, and the
  captures record no scratch size, so the allocator's own exact number errs
  toward hardware safety by construction. The inspector allows channel 3
  exactly when `tiles[3] > 0`.
- **Zero-input floor branch.** The blob twin binds no runtime surface; the
  elementwise shape block now accepts the 0-input form only as the
  constant-backed boolean floor.

Test fix owned here: the registry `floor_div` pin used a rank-1 `[1]`
divisor, but the encoder carries the divisor as scalar `tensor<fp16, []>`
(verified in model.mil: `var_23_promoted_to_fp16` and the int32 length
form are both scalars). The pin now uses `fp16(2.0)`.

### 2. Materialized tile (`apple-parity-tile`)

- Runtime-operand forms lower as one whole-surface program: whole-tensor
  input/output records (the tile is excluded from the 64-lane record
  split), encoder `apple-parity-tile`, inspector-validated.
- **64-lane tail rule** (compiler check + inspector, same wording): programs
  read in 64-element lanes, so a consumer may reach the end of the lane
  containing the last producer-covered element — the readable surface
  padding — but never beyond it. This is what lets 64-lane chained consumers
  read a whole-surface producer's padded tail; genuine holes still reject.
- Inspector `tile` branch: one runtime input, whole-surface bindings, output
  a whole-multiple replicate of the input.
- Blob-x tile **rejects** (`h13.unsupported-tile`, "no byte-exact
  reproducer"): the decoded blob template scatters its operand into
  constant-section lanes (value→lane map derived from the capture is
  ambiguous — config/operand collisions at lanes 0/2/64/65, 55 of 375
  values with no lane at all), so no reproducer exists for a general blob.
  Binding the blob as a runtime surface would need a new manifest field plus
  runner support — a separate stream, named here.
- New gate suite `tests/test_h13_tile_cli.py`: the two runtime captures
  byte-prove task stream + const section, the all-ones capture pins the
  host-side alias, the blob capture pins the rejection. (`verified == 4`.)

### 3. Pad citations (`tests/test_h13_layout_cli.py`)

All six Apple pad captures assert `error == callback_status=1` in-test, and
each pins our matching behavior: the zero-amounts identity view stays the
host-side alias, the five materializing forms (blob and runtime,
encoder-shaped 749→750 / 375x749→375x750) keep the Apple-citing
`h13.unsupported-pad`. The message's "(6/6 callback_status=1)" is now
machine-checked, including the count.

## Manifest contract note (for the mlx consumer)

Per-binding dtype keeps the existing name **`dtype`** (`float16`/`bool`;
inspector-enforced, gate-pinned). If the mlx side standardizes on
`elementDtype` camelCase, the mapping belongs to the consumer — this
harness does not rename. Confirmed, proceeding.

## Stream-2 coverage numbers (updated)

- 219 layout ops: transpose 146 (0 newly supported here — see the
  transpose-absorb receipt: 0 of 98, all 146 classified), slice_by_index 48,
```text
make -j8 build/mil-hwxc                    # 0 errors (GNUstep host toolchain)
make test-h13
  H13_ENCODING_OK
  H13 MIL-to-ANEC/HWX CLI: PASS (device-free)
  h13 split cli: PASS
  h13 layout cli: PASS                     # + six pad-citation pins
  h13 composite cli: PASS
  h13 batched cli: PASS
  h13 registry cli: PASS                   # scalar-2.0 divisor pin
  h13 boolean cli: PASS                    # task-header validation green
  h13 tile cli: PASS                       # 2 byte-proofs + alias + rejection
python3 tests/test_h13_reference.py build/mil-hwxc
  ValueError: slice has incorrect fields   # test_intermediate_physical_buffer_composition;
                                           # fails identically on the clean tree (stash-verified);
                                           # pre-existing, unrelated
python3 tests/test_h13_deadline.py build/mil-hwxc
  H13_DEADLINE_OK
python3 tests/test_h13_parity.py build/mil-hwxc
  H13 oracle parity: PASS (846 cases, 182 matmul, 79 broadcast, 105 softmax/layer_norm, 114 reduction, 284 convolution, 1692 artifacts)
```

### Regression caught and fixed in-session: dropped `allocationByteLength`

The session's boolean-dtype edit of `objectBinding` accidentally deleted
`binding.allocationByteLength = layout.allocationBytes`. Batch-1 HWX writes
still passed (the writer default covers one batch element), but any batch-N
surface failed `bindingLayoutsValid` — parity caught it at
`env_bcast_add_2x64x8x8` ("binding shapes and strides must be nonzero").
Clean-tree parity passed 846/846, isolating the regression to this diff;
restoring the line returns parity to 846/846. Lesson: adjacent-line edits
around dtype plumbing need a parity run, not just the gate (the gate's HWX
cases are batch-1).

## Hardware boundary

No SSH, hardware inference, ANE command, router change, model pin change, or
Mac execution was performed. No files under `ane-linux-experiments` were
edited or executed; the encoder artifact, its coverage receipts, and the
oracle captures were read only.
