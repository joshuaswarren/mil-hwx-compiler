#!/usr/bin/env python3
"""Failing-first CPU regression: scalar-mul broadcast lowers as ONE program.

Contract under test (Apple-oracle shape class env_bcast_*_scalar, e.g.
research/oracles/h13/env_bcast_mul_1x768x16x16_scalar.json — Apple
ane-compile-hwx emits a single task for mul-by-fp16-scalar at these
shapes): compiling a rank-4 mul whose second operand is an inline fp16
scalar must produce ONE program at the encoder's real attention scale
[1, 8, 375, 375].

Current local build (mil-hwxc 18e93b9, binary sha 5f447bdc…) fails:
the H13 constant-fold loop (ANEH13Compiler.mm ~L2127-2260,
sliceElements = MIN(64, …)) emits ceil(1,125,000 / 64) = 17,579
per-chunk mul programs. The same regression reproduces the Apple
1-program oracle env_bcast_mul_1x768x16x16_scalar as 3,072 chunks.

Also pins the semantic reference: the CPU fp16 result of scaling a
fixture tensor by 0x1p-4 (the encoder's bd scale) is the expected
output payload for the eventual device decode comparison.

Routing trace (all paths verified in source, no template invented):
1. parityPlan (ANEH13Compiler.mm L1240): scalar operand accepted ONLY
   when scalarBits == 0x3800 (fp16 0.5). bd scale 0x1p-4 = 0x1400 -> NO.
2. broadcastPlan (L1338): Scalar operand reaches the decoded envelope
   table, but kBroadcastTasks (H13EnvelopeTemplates.inc, 525 rows) has
   Scalar-mul rows ONLY for {1,64,8,8}, {1,64,16,16}, {1,768,8,8},
   {1,768,16,16} - and encodeBroadcast hard-rejects scalarBits != 0x3800
   (H13Program.cpp L1927). [1,8,375,375] has no row of any operand kind
   (Runtime/Scalar/Constant) for mul. -> NO.
3. Constant-tensor blob twin (supportsElementwiseConstant): BinaryConstant
   rows exist only for Add{512}, Add{896}, Mul{512} -> NO.
4. Fall-through: the 64-lane fold -> 17,579 chunk programs.

The decoded-envelope approach cannot mint this op without a new Apple
capture (ane-compile-hwx on studio-host, CPU-only tool) of
mul-by-0x1p-4 at [1,8,375,375], then adding the decoded row and
generalizing encodeBroadcast's per-row scalar-bits handling. Capture
request staged at scripts/request_bdscale_capture.md in this branch.
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "build" / "mil-hwxc"
SHAPE = (1, 8, 375, 375)
SCALE = float.fromhex("0x1p-4")  # the encoder bd scale (var_371 class)


MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 375]> x) {
    fp16 s = const()[name = string("s"), val = fp16(0x1p-4)];
    tensor<fp16, [1, 8, 375, 375]> y = mul(x = x, y = s)[name = string("y")];
  } -> (y);
}
"""


def _compile(tmp_path: Path) -> dict:
    mil = tmp_path / "scalar_mul_1x8x375x375.mil"
    mil.write_text(MIL)
    result = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(tmp_path),
         "--target", "H13", "--format", "anec",
         "--output", str(tmp_path / "out")],
        capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads((tmp_path / "out" / "manifest.json").read_text())


def test_scalar_mul_at_attention_scale_is_one_program(tmp_path) -> None:
    manifest = _compile(tmp_path)
    programs = manifest["programs"]
    assert len(programs) == 1, (
        f"scalar mul at {SHAPE} lowered to {len(programs)} chunk programs; "
        "Apple oracle class env_bcast_*_scalar emits 1 — the H13 "
        "constant-fold is chunking instead of using the broadcast binary")
    ops = {p["operation"] for p in programs}
    assert ops == {"mul"}, ops


def test_semantic_reference_fp16_scale(tmp_path) -> None:
    """CPU fp16 semantic reference for the device decode comparison.

    Exactness domain for a power-of-two scale: the RESULT must stay
    normal. x >= 2**-14 is not sufficient — a normal x in
    [2**-14, 2**-10) scales INTO the subnormal range where fp16 rounds,
    so bits are lost even though the input was normal. This test
    documents both regions explicitly.
    """
    elements = math.prod(SHAPE)
    # Deterministic fixture: fp16 bit patterns ramping over the full
    # positive range (zero, subnormals, normals).
    bits = (np.arange(elements, dtype=np.uint32) * 31 % 0x7C00).astype(np.uint16)
    x = bits.view(np.float16).reshape(SHAPE)
    expected = (x.astype(np.float32) * np.float32(SCALE)).astype(np.float16)

    x_f32 = x.astype(np.float32)
    positive = x_f32 > 0
    result_normal = positive & (x_f32 * np.float32(SCALE) >= 2 ** -14)
    result_subnormal = positive & (x_f32 * np.float32(SCALE) < 2 ** -14)

    # Exact (exponent shift only) wherever the result stays normal:
    assert np.array_equal(
        x[result_normal].astype(np.float32) * np.float32(SCALE),
        expected[result_normal].astype(np.float32))
    # Subnormal RESULTS are lossy for at least some inputs — asserted,
    # not assumed (0x1p-4 exactness does not generalize to subnormals).
    if result_subnormal.any():
        some_lossy = np.any(
            x[result_subnormal].astype(np.float32) * np.float32(SCALE)
            != expected[result_subnormal].astype(np.float32))
        assert some_lossy
    (tmp_path / "semantic_reference_f32.npy").write_bytes(
        expected.astype(np.float32).tobytes())