# Apple capture: mul-by-0x1p-4 at [1,8,375,375] (bd scale) — CAPTURED & LANDED

Lane: EncoderCompilerCoverage (compiler research branch).
Purpose: single-program ANE lowering for the encoder AC-head bd scale.
Blocks: F-gate ac-head bundle (island-attn-ac-head-L00-18e93b9) currently
diverges median 5,200 ULP because the in-graph bd scale is missing and the
local compiler has no non-chunked lowering for it (see
tests/test_h13_scalar_mul_broadcast.py — failing-first, 17,579 chunk
programs vs the Apple 1-program contract).

## What to run (studio-host, CPU-only tool, no ANE execution)

Tool: ane-compile-hwx (the same wrapper the oracle campaign used,
/tmp/h13-oracle/bin/ane-compile-hwx; last provenance tool_sha256
b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe,
host studio-host.local, macOS 26.6.2).

MIL to capture (exact text):

program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 375]> x) {
    fp16 s = const()[name = string("s"), val = fp16(0x1p-4)];
    tensor<fp16, [1, 8, 375, 375]> y = mul(x = x, y = s)[name = string("y")];
  } -> (y);
}

Command shape (match the campaign):
  ane-compile-hwx CAPTURE_DIR OUTPUT_DIR h13

Record in the case JSON: tool sha256, driver sha256, decoder sha256,
source_commit, hwx_sha256, program_count, task_descriptors,
tensor_descriptors, constant_section, weights — the same fields as
research/oracles/h13/env_bcast_mul_1x768x16x16_scalar.json.

## Acceptance (CPU-side, this repo)

1. Place the decoded case JSON in research/oracles/h13/ as
   env_bcast_mul_1x8x375x375_scalar.json.
2. tests/test_h13_scalar_mul_broadcast.py switches from asserting the
   contract against a bare program count to the parity byte-compare
   against the new oracle (compile locally, byte-compare the task
   stream). It must go GREEN only when the local compiler emits the
   captured 1-program stream for [1, 8, 375, 375].
3. Add the decoded row to kBroadcastTasks (H13EnvelopeTemplates.inc):
   {operation=mul(1), operand=Scalar, x=[1,8,375,375]}, and generalize
   encodeBroadcast/supportsBroadcast scalar-bits handling so a row
   carries its OWN decoded scalar value (current hard rejection of
   scalarBits != 0x3800 stays for rows whose constant section encodes
   0.5; the new row's constant section must be checked against 0x2C00
   semantics during decode).
4. Re-run: tests/test_h13_scalar_mul_broadcast.py (both tests GREEN),
   then the bounded slice of tests/test_h13_parity.py for the
   env_broadcast family (no regression in the 525-row envelope).
5. Full parity suite run before any device window (Main owns hardware).

## Value provenance

bd scale 0x1p-4 = var_371 scalar mul in the pinned encoder MIL
(receipts 2026-09-19-encoder-submit-repair §6, MIL statements 224-225,
applied to matrix_bd). The F-gate fullhead divergence (median 5,200
ULP) is root-caused to this missing scale; the runner-side pre-scaled
arm (window 2026-09-20T142049Z) already proved the composed chain
reaches median 1 ULP vs CPU reference once the scale is applied. The
in-graph row is the permanent fix so the runner never pre-processes.


## OUTCOME (2026-09-20, fulfilled)

Capture executed on studio-host (CPU-only ane-compile-hwx; otool-verified:
links Foundation/ANECompiler/libc++/libSystem/CoreFoundation, no device
framework, no ANE execution):
- model.hwx 49,152 B, ANECCompile=0, sha256 e60db231cf5d8a852ea8b9bc2
  4dc979b66deb5154e82ab99307eb3ec05033f7a; task_count 1 (504-byte task);
  constant section 16,384 B all-zero; scalar 0x1p-4 (fp16 bits 0x2C00)
  baked into task word 109 high half.
- Decoded row landed as kEnvelopeTask258 + kBroadcastTasks row
  {mul, Scalar, [1,8,375,375]} with per-row scalarBits 0x2C00;
  OracleBroadcastTemplate carries scalarBits and encodeBroadcast's
  guard compares the requested scalar against the row (fail-closed);
  broadcastPlan no longer lets the native tensor-shape probe swallow
  Scalar operands before the decoded table is consulted.
- tests/test_h13_scalar_mul_broadcast.py: 4/4 GREEN (one-program,
  task-byte match under the parity binding, wrong-scalar rejection,
  semantic reference). env_broadcast parity slice 8/8 GREEN.
