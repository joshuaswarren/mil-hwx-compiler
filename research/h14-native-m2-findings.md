# H14 native/cross mint: the real staged programs compile whole; the refusals were spelling artifacts

Date: 2026-09-30. Actor: M2NativeMint. Compiler worktree:
`/home/joshuawarren/src/mil-hwx-h14-integ-wt`, branch `agent/h14-integ`.
Notebook entry:
`~/.local/share/apple-silicon-lab/entries/M2NativeMint/20260930T1213Z-jw14m2-macos-native-mint.md`.

## Question

Apple's cross-compile on the Mac Studio (M1 Ultra, macOS 26.6.2,
`ane-compile-hwx`, tool SHA-256 `2060776c98e385d091ca244a2083be25919ed0c340ba1c8c89840208473b05d5`)
had refused, with `callback_status=1`, the reconstructed family spellings:
`linear` consuming `[1,2048,1,1]`, `reshape` `[1,2048,1,1] -> [1,2048]`,
reshape+linear, the scaled-softmax attention chains at `[16,50]`, standalone
rank-2 `[1,2048] axes=[1]` and `(1,16,128,1)/(1,16,1,128)` axes 2/3
reductions, the head-major broadcasts, and every `concat` spelling. That
blocked all 38 staged Qwen3.8-2B decode programs (0 of 38 as one fused ANEC
on our H14 compiler). Question: does a native Apple compiler accept them?

## Resolution

The M2 macOS session became unnecessary. The Studio h14 cross-target accepts
the **real** staged programs whole: all 38 programs of
`/var/tmp/qwen-real-mil/` (extracted from the pinned ANEForge
`lane/deltanet-split-decode` tip) compiled with `exit=0, callback_status=0`,
each emitting exactly one program descriptor. Every earlier refusal was
established on reconstructed spellings; the real staged MILs avoid those
refusals by construction. **The real staged MILs are the authoritative
spelling source** — the F2/F4/linear/reshape/concat family wire-ups should
copy these MILs verbatim instead of re-deriving spellings.

## Run facts (Studio, load-gated, niced, serial; `TARGETS='h14'`)

| Class | Programs | Tasks | HWX bytes | Real ports (in/out) |
|---|---|---:|---:|---|
| A-first (0, 21) | 2 | 32 | 34,144,256 | 2 / 7 |
| B-state (18) | 18 | 10 | 81,920 | 6 / 2 |
| A-next (11) | 11 | 52 | 118,063,104 | 4 / 8 |
| D-attn (5) | 5 | 120 | 223,117,312 | 11 / 10 |
| C-readout (20) | 1 | 20 | 83,935,232 | 3 / 1 |
| E-final (37) | 1 | 92 | 189,005,824 | 10 / 3 |

Artifacts: HWX at `/var/tmp/qwen-real-hwx-h14/prog_NNN/model.hwx`
(+ `SHA256SUMS`); converted ANEC packages at
`/var/tmp/qwen-real-anec-h14/prog_NNN/program-0.anec` (+ `manifest.json`
with per-program tasks, first-task bytes, constant bytes, named ports and
load flags, + `SHA256SUMS`). Weights verified SHA-256 against
`/var/tmp/qwen-real-mil/weights_index.csv` (2,747,351,488 bytes total).
Notebook artifacts under `~/.local/share/apple-silicon-lab/artifacts/M2NativeMint/`.

## Converter status

Two independent HWX→ANEC converters, both byte-exact against the
hardware-proven `fixtures/h14-anec/add` and `fixtures/h14-anec/relu`
programs (re-minted oracle HWX is structurally identical to the records:
same constant-section SHA, same task descriptors; only volatile HWX header
bytes differ, and none reach the ANEC):

- `tools/hwx_h14_to_anec.py` (lane owner: RealQwenH14, being extended in
  place): strict single-output + task-reference-verified scope; refuses the
  multi-port staged programs by design (all 38).
- Multi-port converter (this lane, scratch at
  `/tmp/m2-native-mint/hwx_to_anec_h14.py`, approach handed to the owner):
  converts 38/38. ANEC header rules per `H14Program.cpp:encodeANEC`
  (`firstTaskBytes` = first task length, `tiles[0] = ceil(content/0x4000)`,
  channel allocations from the HWX `__FVMLIB` surfaces, layouts from the
  tensor descriptors, leading stream frame zeroed).

Open items handed to the owner lane: which surface owns channel 4 per
multi-output program (state_out vs the lane output — name-driven binding
recommended); the gamma constant packing for `garms_chain_c2048_gamma`
re-mints (4-byte-per-channel block; pattern not derivable from the decoded
record).

## Static load flags for the T6021 driver

- 20 of 38 programs exceed the 16 MiB `BO_INIT` constant cap (constants run
  to 223 MB); the 18 B-state programs (16,384-byte constants) fit.
- Every program's `firstTaskBytes` is non-16-byte-aligned (Apple stream
  trait, e.g. 148; the recorded islands show the same trait).
- Runtime input counts span 2..11 and outputs 1..10 — the ANEC channel
  space (32) holds them, but the driver's binding path and the staged
  encoder's 1..3-input envelope do not; multi-output channel assignment is
  flagged per program in the manifest and needs device-side verification.
- Compile/package emit is not device execution; no ANE was run.

## Candidates

All 38 staged programs are now encoder candidates on equal footing: Apple
accepts each real MIL whole at h14, and the emitted streams/constants are
the verbatim templates an H14 encoder can replay per program. The earlier
per-family fix list (F1/F2/F3/F4/F5) should be re-derived from the real
spellings rather than the reconstructed harness MILs in
`artifacts/H14Batch5/run_qwen_matrix_batch5.py`.
