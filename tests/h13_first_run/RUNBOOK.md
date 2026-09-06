# H13 first-run kit — the first ANE execution on jwm1

Eight rungs, in order, for the first time a package from this compiler is
submitted to real hardware: base M1 (`t8103`) on Omarchy Linux, the
`eiln/ane` driver from the `omarchy` branch of `~/src/omarchy-ane`, ANEC
header `0x1000`, `/dev/accel/accel0`.

Nothing in this repository has executed on an ANE. Byte parity with Apple's
compiler is not device execution (`docs/ane/parity-method.md`). Artifact
readiness is not permission to submit: the hardware handoff for jwm1 is the
owner's call, and this kit does not grant it.

**Stop rule.** After every rung: if it did not pass, stop. Do not climb. Each
rung is chosen so that a failure isolates one contract, and a failure carried
upward is unattributable. Rung order is also an escalation of blast radius —
rung 1 sends 32 KiB and one task descriptor, rung 8 sends 4.36 MiB of
package bytes across 77 dispatches.

## The ladder

| # | Rung | Op / shape | Encoder | Programs | Tasks | First failure implicates |
|---|---|---|---|---|---|---|
| 1 | `01-add-legacy` | `add` `[1,64,1,1]`, folded constant | `h13-source-qualified` | 1 | 1 | the driver contract itself: header, channel 4/5/6 binding, 0x4000 tiles |
| 2 | `02-add-parity` | `add` `[1,64,1,1]`, two runtime | `h13-oracle-parity` | 1 | 1 | Apple's descriptor form and `ane_bind_kernel` (the constant section) |
| 3 | `03-mul-scalar` | `mul` `[1,64,1,1]` by fp16 `0.5` | `h13-oracle-parity` | 1 | 1 | the constant section's scalar bias/scale blocks |
| 4 | `04-matvec-k256-n512` | `matmul` K=256 N=512 | `apple-parity-matvec` | 1 | 2 | the two-task linked stream, 256 KiB weight DMA, weight permutation |
| 5 | `05-softmax-512` | `softmax` `[1,512,1,1]` axis 1 | `apple-parity-norm` | 1 | 5 | five linked tasks in one program, fp16 exp/reciprocal lookup tables |
| 6 | `06-chain-add-mul` | `add` → `mul` `[1,64,1,1]` | `h13-oracle-parity` ×2 | 2 | 2 | intermediate handoff: program 0's channel 4 becomes program 1's channel 5 |
| 7 | `07-runtime-matmul-64` | `matmul` M=K=N=64, both runtime | `apple-parity-matmul` | 1 | 2 | Apple's reversed operand order (`y` on 5, `x` on 6) and `__DATA`/`__bss` scratch |
| 8 | `08-mlp-768-1024-768` | MLP block, relu + residual | 76 source-qualified + 1 parity | 77 | 77 | sustained dispatch: 77 submissions, chunked-fp16 partial sums, 4.02 MiB of constant sections |

Measured on Linux 2026-09-06 (see [Dry-run evidence](#dry-run-evidence)).

## Preflight, every session

```sh
bash tests/h13_first_run/preflight.sh
```

Checks and prints, without touching the device: host and kernel; the
`/proc/device-tree/soc/ane@*` node with its `compatible` and `status`; the
loaded `ane` module's `srcversion` and parameters; the bound platform device,
its driver and its runtime-PM state (must be pinned `on` — autosuspend
invalidates DART TLBs about a second later, including the `dart0` that
`apple-dart` owns, and resets the SoC); the device node's mode and owner;
the libane checkout's branch, commit, dirty count and `ANEC_HEADER_SIZE`
(must be `0x1000`); `libane/libane.a` and
`bindings/python/dylib/libane_python.so`; and this repository's commit and
dirty count. Any failure exits 2 and names the fix.

`tests/run_h13_linux_hardware.sh` runs this same script and refuses to
compile or submit if it fails, so the gates cannot drift apart.

State on the workstation used to build this kit, for contrast with what jwm1
must show: `~/src/omarchy-ane` is on `omarchy-kmd` at `2ced80b` with
`ANEC_HEADER_SIZE 0x800` and no built library. That checkout would be
rejected. jwm1's checkout must be on `omarchy` (`f8df7bc` has the `0x1000`
header) with both artifacts built.

## Running a rung

```sh
python3 tests/h13_first_run/first_run.py --rung 1              # dry run, no device
python3 tests/h13_first_run/first_run.py --rung 1 --execute    # submit
python3 tests/h13_first_run/first_run.py                       # dry-run all eight
```

A dry run materializes the fixture, compiles the package, asserts the rung's
encoder mix and task-descriptor count, and writes
`/tmp/h13-first-run/<rung>/plan.json`: every binding, every channel, the
per-output pass criterion, and the exact libane call sequence
`tools/h13_run_linux.py` will issue. It needs no ANE host and reaches no
device. Review the plan before `--execute`.

`--execute` hands the fixture to `tests/run_h13_linux_hardware.sh`, which
preflights, recompiles, validates the package with
`research/inspect_anec.py`, writes the plan next to the package, and only
then submits. Every output is compared with `tools/h13_reference.py`; output
files are written only on a pass.

### Fixture layout

`first_run.py` writes each rung to `/tmp/h13-first-run/<rung>/`:

```
model.mil                 copied from tests/h13_first_run/rungs/<rung>/model.mil
models/*.bin              BLOBFILE weights: 128-byte prefix, subheader at 64
inputs/<name>.fp16        dense fp16, one value per element
expected/<name>.fp16      tools/h13_reference.py output, written before compiling
pkg/                      the compiled ANEC package and its manifest
plan.json                 the reviewed dispatch plan
```

### The deterministic generator

`rungs.json` gives every tensor a `scale`, `modulus` and `offset`; element
`i` is `scale * ((i % modulus) - offset)`, or `scale * (i - offset)` when
`modulus` is 0, rounded once to fp16. No RNG, no timestamps: the same bytes
on every host, so a device result can be re-derived years later. The
patterns match the ones already used by the oracle probes and
`tests/hardware/run_alu.mm` — `i*0.125`, `i*0.25`, `0.25*((i%7)-3)`,
`0.125*((i%5)-2)`.

## Which encoder a rung gets, and how to force the legacy one

There is no `--encoder` flag. `build/mil-hwxc` accepts only `--mil`,
`--model-root`, `--output`, `--target` and `--format`
(`tools/mil-hwxc.mm`), and the encoder is a consequence of the operation's
*form*. `lowerOperation` in `plugins/H13/ANEH13Compiler.mm` tries, in order:

1. `broadcastPlan` → `apple-parity-broadcast`
2. `parityPlan` → `h13-oracle-parity` (whole-tensor oracle templates)
3. `normParityPlan` → `apple-parity-norm`
4. the source-qualified fallback → `h13-source-qualified` (the
   allbilly-derived 64-lane `encodeBinary`/`encodeMatvec` path), plus
   `apple-parity-matvec`/`apple-parity-matmul` inside the `matmul` branch

The manifest records the choice per program in `programs[].encoder`, and
`first_run.py` asserts it, so a rung cannot silently change encoder.

To force the legacy encoder for `[1,64,1,1]` `add` — rung 1 — give the
second operand as a **matching-shape `BLOBFILE` tensor constant** rather
than a runtime tensor:

* `parityPlan` refuses it: a constant `y` must be an inline fp16 scalar
  whose bits are `0x3800`, i.e. `0.5` (`ANEH13Compiler.mm`, the
  `scalarBits != 0x3800` guard).
* `broadcastPlan` refuses it too: a `[1,64,1,1]` constant is not a scalar,
  and `supportsBroadcast` has no decoded template for a per-channel
  constant at this geometry.
* So lowering falls through to `encodeBinary`, which is the 64-lane
  descriptor with `tileBytes` surfaces on channels 5, 6 and 4 — the
  descriptor shape `allbilly/libane`'s examples ran on this driver.

Two runtime operands at the same shape take `parityPlan` instead, which is
exactly rung 2: same arithmetic, same expected output, different descriptor.
That pairing is the point — if rung 1 passes and rung 2 fails, the driver is
fine and the Apple-parity descriptor or its constant section is not.

## Pass criteria

`tools/h13_run_linux.py` fails closed before submitting: input byte counts
must equal the manifest's `logicalBytes`, `__ane_src_size`/`__ane_dst_size`
must equal the manifest's `allocationBytes`, and the MIL's returns must be
exactly the manifest's output tensors.

For values, the criterion per output is printed in `plan.json`
(`referenceCriteria`):

* **byte-for-byte equal to the fp16 reference** — rungs 1, 2, 3, 4, 5, 6, 7.
* **`|device - reference| <= 0.02 + 0.02 * |reference|`** — rung 8, and any
  output computed from a chunked reduction. A K=768 reduction is summed in
  512-element chunks with one fp16 rounding per chunk, and that rounding
  propagates through the relu and the residual add, so `y` inherits the
  envelope even though it is an elementwise result.

One caveat that is a reference limitation, not a device fault: Apple
evaluates softmax's exponential and reciprocal through fp16 lookup tables in
the constant section, while `tools/h13_reference.py` evaluates them in
double precision and rounds once. If rung 5 fails by one or two ulps per
element and rungs 1–4 passed, suspect the lookup tables, not the driver;
record the device bytes and compare against the table values in
`research/h13-td-fields.md` before calling it a driver bug.

## The rungs

Every rung below lists the fixture, the command, the criterion, the libane
calls, and what a failure there implicates. Buffer sizes are the surface
allocations the driver must agree with: a `[1,64,1,1]` fp16 tensor occupies
one 16 KiB tile, a `[1,512,1,1]` one 32 KiB pair of tiles.

### Rung 1 — `add [1,64,1,1]`, legacy source-qualified encoder

* **MIL** `tests/h13_first_run/rungs/01-add-legacy/model.mil` — `add(x = a, y = c)`
  where `c` is a `[1,64,1,1]` `BLOBFILE` constant.
* **Weights** `models/c.bin`, 64 fp16, `0.25 * i`.
* **Input** `a`, 64 fp16, `0.125 * i`.
* **Expected** `expected/y.fp16` = `0.375 * i`, from `tools/h13_reference.py`.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 1 --execute`
* **Criterion** `y` byte-for-byte equal to the reference; 128 bytes.
* **libane calls** (one program, no kernel bind — the folded constant is a
  host-packed input surface, and `constantBytes` is 0):

  ```
  pyane_init("program-0.anec", device)
  __ane_src_size(handle, 0) == 16384  # a   (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_src_size(handle, 1) == 16384  # c   (channel 6, constantInputs)
  __ane_send(handle, buffer, 1)
  __ane_dst_size(handle, 0) == 16384  # y   (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here implicates the driver contract, not the encoder.** This is
  the descriptor family `allbilly`'s examples already ran. Read the failure in
  this order: `pyane_init` → the `0x1000` ANEC header offset or the container
  parse; a size mismatch → the 0x4000 tile model or `allocationBytes`;
  `ane_exec` returning nonzero → task submission or the DART mapping; wrong
  values with every call succeeding → channel assignment (4 out, 5 and 6 in)
  or the 64-byte lane layout.

### Rung 2 — `add [1,64,1,1]`, Apple-parity encoder

* **MIL** `rungs/02-add-parity/model.mil` — `add(x = a, y = b)`, both runtime.
* **Weights** none.
* **Inputs** `a` = `0.125 * i`, `b` = `0.25 * i`, 64 fp16 each.
* **Expected** `expected/y.fp16` = `0.375 * i` — identical to rung 1, on purpose.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 2 --execute`
* **Criterion** `y` byte-for-byte equal to the reference; 128 bytes.
* **libane calls**: rung 1's sequence plus, before the sends,
  `ane_bind_kernel(handle, anec[0x1000 + 512], 16384)` — Apple's constant
  section at content offset `0x200`, 16 KiB.
* **A failure here, with rung 1 passing, implicates the Apple-parity
  descriptor or the constant section**: `ane_bind_kernel` rejecting the
  16 KiB block, the kernel-capacity limit, or a task-descriptor word the
  driver interprets differently from Apple. The two rungs compute the same
  numbers from the same inputs, so any difference is descriptor-side.

### Rung 3 — scalar-constant `mul`

* **MIL** `rungs/03-mul-scalar/model.mil` — `mul(x = a, y = fp16(0.5))`.
* **Weights** none; `0.5` is folded into the constant section's scale block.
* **Input** `a`, 64 fp16, `0.125 * i`.
* **Expected** `expected/y.fp16` = `0.0625 * i`.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 3 --execute`
* **Criterion** `y` byte-for-byte equal; 128 bytes.
* **libane calls** — one input surface only:

  ```
  pyane_init("program-0.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 512], 16384)
  __ane_src_size(handle, 0) == 16384  # a   (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_dst_size(handle, 0) == 16384  # y   (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here, with rung 2 passing, implicates the constant section's
  per-channel bias and scale blocks** — the values the engine multiplies by,
  not the descriptor: one input surface instead of two, and the operand comes
  from the bound kernel. Zeros in the output mean the scale block was not
  read; the input echoed back means it was read as `1.0`.

### Rung 4 — Apple-parity matvec K=256 N=512

* **MIL** `rungs/04-matvec-k256-n512/model.mil` — `matmul(x, W)`,
  `transpose_x = false`, `transpose_y = true`, `W` a `[512,256]` `BLOBFILE`.
* **Weights** `models/weights.bin`, 131072 fp16, `0.125 * ((i % 5) - 2)`,
  packed by the compiler into Apple's permutation (256 KiB in the constant
  section).
* **Input** `x`, 256 fp16, `0.25 * ((i % 7) - 3)`.
* **Expected** `expected/y.fp16`, 512 fp16, fp32-accumulated then rounded once.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 4 --execute`
* **Criterion** `y` byte-for-byte equal; 1024 bytes. K=256 is a single
  reduction chunk, so no chunked envelope applies.
* **libane calls**:

  ```
  pyane_init("program-0.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 1152], 262144)
  __ane_src_size(handle, 0) == 16384  # x   (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_dst_size(handle, 0) == 16384  # y   (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here implicates the two-task linked stream and the weight
  DMA.** This is the first rung with two task descriptors — Apple's 126-word
  preparation task then its 157-word compute task, linked by byte offset —
  and the first with a large constant section (256 KiB at content offset
  `0x480`). `ane_bind_kernel` refusing it means kernel capacity or the
  offset; `ane_exec` failing means task linking; structured-wrong values
  (right magnitudes, wrong positions) mean the weight permutation
  (`packMatvecWeights`, `research/h13-td-fields.md`) rather than the driver.

### Rung 5 — Apple-parity softmax `[1,512,1,1]`

* **MIL** `rungs/05-softmax-512/model.mil` — `softmax(x = a, axis = 1)`.
* **Weights** none; the fp16 exp and reciprocal lookup tables are the 256-byte
  constant section.
* **Input** `a`, 512 fp16, `0.25 * ((i % 7) - 3)`.
* **Expected** `expected/y.fp16`, 512 fp16 summing to about 1.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 5 --execute`
* **Criterion** `y` byte-for-byte equal; 1024 bytes — with the lookup-table
  caveat above. A one-to-two-ulp spread is a reference limitation; a wrong
  distribution shape is not.
* **libane calls**: rung 3's shape with 32 KiB surfaces and a 256-byte kernel
  at content offset `0xc00`:

  ```
  pyane_init("program-0.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 3072], 256)
  __ane_src_size(handle, 0) == 32768  # a   (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_dst_size(handle, 0) == 32768  # y   (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here implicates multi-task sequencing inside one program.**
  Five linked descriptors — max, subtract, exponentiate, sum, reciprocal
  multiply — sharing `__bss` scratch. A hang or a partially written output
  means the driver walked the chain wrong or ran only the first task; that is
  the descriptor-linking rule in `docs/ane/task-descriptors.md`, not the
  operation.

### Rung 6 — two-program chain `add` → `mul`

* **MIL** `rungs/06-chain-add-mul/model.mil` — `sum = add(a, b)`,
  `y = mul(sum, b)`.
* **Weights** none.
* **Inputs** `a` = `0.125 * i`, `b` = `0.125 * ((i % 5) - 2)`, 64 fp16 each.
* **Expected** `expected/y.fp16` = `(a + b) * b`.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 6 --execute`
* **Criterion** `y` byte-for-byte equal; 128 bytes.
* **libane calls** — two `pyane_init`/`pyane_free` cycles; `sum` leaves
  program 0 on channel 4 and re-enters program 1 on channel 5:

  ```
  pyane_init("program-0.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 512], 16384)
  __ane_src_size(handle, 0) == 16384  # a     (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_src_size(handle, 1) == 16384  # b     (channel 6)
  __ane_send(handle, buffer, 1)
  __ane_dst_size(handle, 0) == 16384  # sum   (channel 4, intermediate)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  pyane_init("program-1.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 512], 16384)
  __ane_src_size(handle, 0) == 16384  # sum   (channel 5, intermediate)
  __ane_send(handle, buffer, 0)
  __ane_src_size(handle, 1) == 16384  # b     (channel 6)
  __ane_send(handle, buffer, 1)
  __ane_dst_size(handle, 0) == 16384  # y     (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here, with rung 2 passing, implicates the intermediate
  handoff**, not either program: the host reads `sum` back, re-packs it into
  program 1's input layout and re-sends it, so a mismatch means the read-back
  layout, the physical lane conversion, or state leaking across
  `pyane_free`/`pyane_init`. Run rung 2 again afterwards: if it now fails, the
  second `pyane_init` in one boot is the problem, which matches the known
  IOVA leak across module state.

### Rung 7 — runtime-runtime matmul M=K=N=64

* **MIL** `rungs/07-runtime-matmul-64/model.mil` — `matmul(x, w)` with both
  operands runtime, `transpose_x = false`, `transpose_y = false`.
* **Weights** none — `w` is an input, not a constant.
* **Inputs** `x` = `0.25 * ((i % 7) - 3)`, `w` = `0.125 * ((i % 5) - 2)`,
  4096 fp16 each.
* **Expected** `expected/y.fp16`, 4096 fp16.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 7 --execute`
* **Criterion** `y` byte-for-byte equal; 8192 bytes.
* **libane calls** — note the operand order: Apple declares the second
  operand first, so `w` is channel 5 and `x` is channel 6:

  ```
  pyane_init("program-0.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 1152], 16384)
  __ane_src_size(handle, 0) == 16384  # w   (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_src_size(handle, 1) == 16384  # x   (channel 6)
  __ane_send(handle, buffer, 1)
  __ane_dst_size(handle, 0) == 16384  # y   (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here implicates surface ordering and scratch.** Apple lays the
  output surface out first and both operands below it, and the program uses
  `__DATA`/`__bss` scratch rather than a packed weight section. A transposed
  or block-shuffled result means the driver's surface-address assumption
  differs from Apple's ordering; a fault means the scratch allocation.

### Rung 8 — 768 → 1024 → 768 MLP block

* **MIL** `rungs/08-mlp-768-1024-768/model.mil` — `linear(x, w1, b1)`, `relu`,
  `linear(., w2, b2)`, then `add` with `x` as a residual.
* **Weights** `models/mlp-w1.bin` and `models/mlp-w2.bin`, 786432 fp16 each,
  `0.03125 * ((i % 5) - 2)`; `models/mlp-b1.bin` (1024) and
  `models/mlp-b2.bin` (768), `0.125 * ((i % 7) - 3)`.
* **Input** `x`, 768 fp16, `0.25 * ((i % 7) - 3)`.
* **Expected** `expected/y.fp16`, 768 fp16, 1536 bytes.
* **Command** `python3 tests/h13_first_run/first_run.py --rung 8 --execute`
* **Criterion** `|device - reference| <= 0.02 + 0.02 * |reference|` for `y`
  (`plan.json` states it): both reductions are K=768, chunked in 512-element
  pieces, and the rounding propagates through the relu and the residual.
* **libane calls** — 77 programs, 686 calls, in `plan.json`. Program 0 is
  representative:

  ```
  pyane_init("program-0.anec", device)
  ane_bind_kernel(handle, anec[0x1000 + 640], 524288)
  __ane_src_size(handle, 0) == 32768  # x                        (channel 5)
  __ane_send(handle, buffer, 0)
  __ane_dst_size(handle, 0) == 32768  # $h13.hidden.partial0     (channel 4)
  ane_exec(handle)
  __ane_read(handle, buffer, 0)
  pyane_free(handle)
  ```

* **A failure here, with rungs 1–7 passing, implicates sustained dispatch
  rather than any single contract**: 77 `pyane_init`/`pyane_free` cycles in
  one process, 4.02 MiB of constant sections (4.36 MiB of package bytes),
  and 512 KiB kernel binds. Record
  which program index failed — `plan.json` maps index to file, operation and
  bindings. A failure that starts partway through and not at program 0 is a
  resource-exhaustion or leak signature (IOVA mappings, kernel buffers), not
  an encoding bug.

## Teardown

* **Never `rmmod` and reload after a hung submission.** The module's remove
  path leaks IOVA mappings; a second `insmod` in the same boot fails
  `bo_init` with `iommu_map failed at 0x4000` (`EEXIST`). Reboot, then re-run
  the bring-up ladder.
* A hung ANE queue is untrustworthy until reboot. Any result read after a
  hang, including a passing one, is not evidence.
* After a reboot, bring the engine up with
  `~/src/homelab-infra/scripts/jwm1-ane-bringup.sh` (it re-powers the chain
  before every load, runs the three gated stages, and pins runtime PM), then
  re-run `preflight.sh` and restart at the rung that failed — not at the top,
  and not past it.
* Leave the shared DART irq enabled: `ane_disable_dart_irq=1` makes the first
  submission fatal on this host.
* Fixtures live under `/tmp/h13-first-run/`; keep the failing rung's
  directory, it contains the exact package, plan and expected bytes.

## Dry-run evidence

Recorded on the Linux workstation (x86_64, no ANE) on 2026-09-06 at compiler
`d72cb9b`, `feat/h13-m1`:

```
$ python3 tests/h13_first_run/first_run.py
PASS rung 1 01-add-legacy: {'h13-source-qualified': 1}, 1 task descriptors, 1 dispatched programs, 9 libane calls
PASS rung 2 02-add-parity: {'h13-oracle-parity': 1}, 1 task descriptors, 1 dispatched programs, 10 libane calls
PASS rung 3 03-mul-scalar: {'h13-oracle-parity': 1}, 1 task descriptors, 1 dispatched programs, 8 libane calls
PASS rung 4 04-matvec-k256-n512: {'apple-parity-matvec': 1}, 2 task descriptors, 1 dispatched programs, 8 libane calls
PASS rung 5 05-softmax-512: {'apple-parity-norm': 1}, 5 task descriptors, 1 dispatched programs, 8 libane calls
PASS rung 6 06-chain-add-mul: {'h13-oracle-parity': 2}, 2 task descriptors, 2 dispatched programs, 20 libane calls
PASS rung 7 07-runtime-matmul-64: {'apple-parity-matmul': 1}, 2 task descriptors, 1 dispatched programs, 10 libane calls
PASS rung 8 08-mlp-768-1024-768: {'h13-source-qualified': 76, 'h13-oracle-parity': 1}, 77 task descriptors, 77 dispatched programs, 686 libane calls
H13 first-run ladder: 8 rung(s) dry-run OK
```

`preflight.sh` and `--execute` both refuse on that host, as they must:

```
$ python3 tests/h13_first_run/first_run.py --rung 1 --execute
PREFLIGHT FAIL host is x86_64, not aarch64
PREFLIGHT FAIL no /proc/device-tree/soc/ane@* node; boot the ANE device-tree entry
PREFLIGHT FAIL ane module is not loaded; run the jwm1 bring-up ladder first
PREFLIGHT FAIL no character device at /dev/accel/accel0
PREFLIGHT FAIL /home/joshuawarren/src/omarchy-ane is on 'omarchy-kmd', not 'omarchy'
PREFLIGHT FAIL libane reads the ANEC payload at 0x800, not 0x1000
H13 preflight: FAIL; do not submit
FAIL rung 1 hardware run exited 2; stop the ladder here
```

Full receipt: `receipts/2026-09-05-ane-community/h13-first-run-dryrun.log`.
Suites re-run after the kit landed: `make test-h13 test-hwx-inspection
test-h13-reference test-h13-simulation`.
