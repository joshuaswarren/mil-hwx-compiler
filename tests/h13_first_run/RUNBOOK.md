# H13 first-run kit

This kit prepares eight H13 fixtures for device-free checks. Preparation does not authorize hardware submission, and native Linux H13 execution remains unqualified.

## Device-free preparation

`python3 tests/h13_first_run/first_run.py` materializes every fixture, computes dense fp16 reference outputs, compiles each package with the selected compiler, validates its encoder and task-descriptor count, and writes a dry-run plan. `--rung ID` selects one or more rungs. `--work DIR` selects the work directory; it defaults to `/tmp/h13-first-run`.

The plan contains the dispatch order, bindings, allocation sizes, expected-output sizes, numerical criterion, and reviewed libane call sequence. Dry-run loads neither libane nor a device and writes no device output.

Set `ANE_COMPILER_BIN` to the reviewed compiler binary before preparing a package. The first-run tool records its SHA-256 in `pkg/compiler.sha256` as:

`<sha256>  mil-hwxc`

When the wrapper receives `--package DIR`, it requires this marker and compares its first field with the pinned compiler SHA-256. The marker records the selected compiler; reviewed identities and preflight are separate requirements.

## Submission boundary

The Linux hardware wrapper has this argument contract:

`bash tests/run_h13_linux_hardware.sh [--package DIR] MIL MODEL_ROOT NAME=input.fp16 ...`

Without `--package`, the wrapper compiles a fresh package and writes its `compiler.sha256` marker. With `--package`, it checks package structure and the recorded compiler digest before dispatch. The marker is not an attestation of package contents or their correspondence to MIL and model data. The wrapper rejects the fixture-only `PREFLIGHT_ROOT` seam.

Before any submission, `tests/h13_first_run/preflight.sh` requires `ANE_REVIEWED_IDENTITIES`, a reviewed key-value file held outside the repository. It has no approved default values. The file must contain all of:

- `firmware`
- `dt-compatible`
- `module-srcversion`
- `module-ko`
- `module-ko-sha256`
- `libane-python-sha256`
- `libane-archive-sha256`
- `compiler-commit`
- `compiler-sha256`

Preflight compares the reviewed compiler source commit and binary digest, libane library and archive digests, loaded module srcversion, supplied module-file digest, platform binding, device node and runtime state. It records the reviewed firmware label without independently verifying active firmware. A module-file digest is not proof of which bytes were loaded. Preflight only reports and refuses; it does not change host state or establish lifecycle safety.

The runtime loads the selected `--libane-library` and requires this ABI before transfer: `pyane_init`, `pyane_free`, `__ane_src_size`, `__ane_dst_size`, `__ane_send`, `__ane_read`, and `ane_exec`. Each program's returned surface sizes are validated before that program's first transfer. Package structure, dense fp16 input sizes, MIL return names and reference outputs are checked before dispatch; output files are written only after every output meets its numerical criterion.

## Native-only benchmarks

`tools/h13_run_linux.py` offers `--benchmark-json PATH`, optionally `--warmup N` and `--iterations N`, for native execution only. These options are rejected with `--dry-run`; `--warmup` and `--iterations` require `--benchmark-json`. Defaults are three validated warmups and ten measured iterations.

The benchmark evaluates the reference before timing, prepares every program before the first iteration, and validates every warmup and measured output. `setupSeconds` reports initialization, size validation, and buffer allocation separately. Per-program samples cover transfers, submission, and readback. End-to-end samples span the first transfer through the last readback, including Python intermediate composition between programs. Final output unpacking, numerical checks, reference evaluation, and setup are outside measured windows. A dry run never reports native timings.

## Status

Per-op scheduling prefers the source-qualified native encoder for same-shape add/mul/maximum/minimum, inline-scalar multiply, and constant-weight single-row matvec. Fresh compiler packages passed isolated M1 checks for all64 add/multiply outputs and all511/512 matrix-vector outputs. The old Apple-parity add failed on identical inputs. Explicit `--schedule chain` retains its decoded fusion encoders; it is not a Linux-qualified replacement for these native paths.

Each native check used one request after a reboot, with DMA pages, mappings and power retained until the next reboot. These results do not establish safe repeated submission, general-chain correctness, cold-start repeatability or performance. Dry-run plans remain device-free; preflight is not permission to submit.
