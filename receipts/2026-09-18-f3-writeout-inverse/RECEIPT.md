# F3 write-out inverse — fix landed (jw16 re-gate, 2026-09-18)

Verdict: **the F3 out-proj conv now computes correctly as-emitted on the real
ANE: rel_l2 0.000207 / 0.000208 / 0.000208 (worst 0.000208, ≤ 0.001, rngs
11/33/57) — 9/9 device-gate cases PASS.** The device finding (gate receipt
f631ca8, gate branch `agent/encoder-conv-device-gate` @ `c6f86fb`) is closed
by decoding the measured write-out order into the F3 record set per protocol
and regenerating the templates; byte parity keeps every count (891 cases /
1782 artifacts / 296 convolution) and the E2E placement pins hold EXACTLY.

Refs: mil-hwx-compiler fix commit `e7967f9` on `agent/f3-writeout-inverse`
(off origin/main `504a1e4`; built on jw16, mil-hwxc sha256 `d9194dd4…`);
finding side `f631ca8` + `c6f86fb` (permutation table
`f3-writeout-permutation.json`).

## Mechanism

Dev run k displays ref run R(k) with exact numerics, so the engine computes
output run k from the weights packed at slot R(k) — measured against the
compiler's identity packing with real random weights, the fetch map is fixed:

    R(k) = ((k >> 1) & 7) | (k & 0x10) | ((((~k) >> 5) & 1) << 3) | ((k & 1) << 5)

The compiler parks ref run k at slot R(k): slot p carries ref run R⁻¹(p),
within-run plane order preserved (`writeout_inverse` in mint_conv_probes,
mirrored in `packConvDenseWriteoutInverse`), gated to the one measured
geometry (k1x1 g1 valid stride-1 bias-free, 1024 → 1024, chunks [16]*4,
spatial_width 375). Same engine class the m375 k1024 n1024 linear lowering
compensates (67dfcf1).

Record-level protocol: the F3 record
(`encoder_conv_c1024_n1024_k1x1_s1_g1_bias0_valid.json`) carries the decode
(`device_writeout_decode`: measured table, mechanism, device numbers) written
by `decode_f3_writeout.py`; its captured section is uniform per constant
(`fp16 bits 0x3400 + index`, one value per constant) so the section bytes are
invariant under the run placement — sha256 `b04df14b…` unchanged, byte parity
against the Apple capture keeps holding, and only device execution
distinguishes the orders. `H13ConvTemplates.inc` regenerated via
`mint_conv_probes --emit-templates`: byte-identical (the task stream carries
no weight order); `--check` in test-h13 enforces the record/emitter pair.
C++ and python packers were cross-checked byte-identical on a random payload.

## Compile side (jw16, aarch64, no window needed)

- `make test-h13`: PASS (encoding, anec, 19 CLI suites, mint_conv_probes
  --check).
- `make test-h13-parity`: PASS **891 cases / 1782 artifacts / 296
  convolution** — identical to 504a1e4; the record content changed, the
  count did not.

## Device re-gate (scripts/dev_gate_conv.py, MHWC=fixed build, one flock
window, inode 12, llm-inference stopped→restarted active MainPID 106244)

| form | program sha256 | rel_l2 (11/33/57) | worst | vs baseline |
|---|---|---|---|---|
| F1 in-proj wmaj | `00a1da99…` | 0.000208 ×3 | 0.000208 | sha+rels unchanged |
| F2 depthwise +bias | `baaac08e…` | 0.000208/0.000207/0.000207 | 0.000208 | sha+rels unchanged |
| **F3 out-proj** | **`305456e7…`** (was `26c04852…`) | **0.000207/0.000208/0.000208** | **0.000208** | **fixed** |
| F4 padconv p0010 | `0c3a70c9…` | 0.000208/0.000207/0.000208 | 0.000208 | sha+rels unchanged |
| slice last-dim | `9515166f…` | 0 ×3 | 0 | unchanged |
| transpose r3 ×2 | `211f2bd1…`/`46307619…` | 0 ×3 ×2 | 0 | unchanged |
| transpose r4 ×2 | `7db7d022…`/`04bd10e2…` | 0 ×3 ×2 | 0 | unchanged |

The F3 program sha moved exactly where the fix applies and nowhere else.

## E2E encoder pins after the fix (abc-launch arm, certified runner
`vulkan_encoder_main_o.py` e93500d2, islands bundles, worker 6b63261a,
libane-strict-fill 04a17653)

status match, 72 submissions, matching_prefix 104/104, cpu_tensor_events 0,
transcript `db501a8c…`, hidden `38c73261…`, mel `5b54f4a9…`,
exec_ms 2603.0. All pins identical to f631ca8.

## Discipline

TAKE/RELEASE announced to Main; `flock -w 900 /tmp/m1-gpu.lock` (inode 12,
never stolen/unlinked); llm-inference stopped before the window and
restarted + `is-active=active` confirmed after; ≥3 GiB /tmp free (30 G);
stdout to files. No gate weakened, no generated file hand-patched, no
formatter run.
