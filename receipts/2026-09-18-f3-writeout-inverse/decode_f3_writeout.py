#!/usr/bin/env python3
"""Decode the measured F3 engine write-out into the F3 oracle record.

The jw16 device gate found the F3 out-projection conv program (k1x1 g1
valid, c1024 -> n1024, [1, 1024, 1, 375]) computing the exact matmul but
emitting its 1024 output planes in a fixed 64-run x 16-plane order: dev
run k displays ref run

    R(k) = ((k >> 1) & 7) | (k & 0x10) | ((((~k) >> 5) & 1) << 3)
           | ((k & 1) << 5)

(receipts/2026-09-18-encoder-conv-device-gate/f3-writeout-permutation.json
on gate branch agent/encoder-conv-device-gate @ c6f86fb; as-emitted
rel_l2 1.413157, 0.000207 after the offline inverse). The measurement ran
against the compiler's identity packing with real random weights, so the
engine's fetch map is fixed: output run k reads packer slot R(k). The
compiler must therefore park ref run k at slot R(k): slot p carries ref
run R^-1(p) with within-run plane order preserved -- the conv spell of
the engine class the m375 k1024 n1024 linear lowering already
compensates (67dfcf1).

This script records that decode in the F3 record: the measured run
table, the mechanism, and the device numbers, verified against the
packer in research/mint_conv_probes.py. The record's captured section
stays byte-identical -- its payload is uniform per constant
(``fp16 bits 0x3400 + index``, one value per constant), so the captured
bytes are invariant under any run placement, and byte parity against the
capture keeps holding with the compensated packer. Only device execution
distinguishes the orders, which is the acceptance gate.

    python3 receipts/2026-09-18-f3-writeout-inverse/decode_f3_writeout.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))

import mint_conv_probes as mc  # noqa: E402

RECORD = ROOT / "research/oracles/h13/encoder_conv_c1024_n1024_k1x1_s1_g1_bias0_valid.json"
TABLE = ROOT / "receipts/2026-09-18-encoder-conv-device-gate/f3-writeout-permutation.json"
REDUCTION = 1024
OUTPUTS = 1024
RUNS = 64
LANES = 16


def r_formula(k: int) -> int:
    return (((k >> 1) & 7) | (k & 0x10) | ((((~k) >> 5) & 1) << 3) |
            ((k & 1) << 5))


def main() -> int:
    record = json.loads(RECORD.read_text())
    table = json.loads(TABLE.read_text())
    measured = table["dev_run_to_ref_run"]
    assert measured == [r_formula(k) for k in range(RUNS)], \
        "packer formula differs from the measured table"
    assert table["dev_plane_to_ref_plane"] == [
        16 * run + plane for run in measured for plane in range(16)], \
        "measured plane table is not the run table expanded"

    weights, bias = mc.case_weights(record)
    assert bias is None
    chunks = mc.lane_chunks(OUTPUTS, mc.lane_cap(375))
    assert chunks == [16] * 4, chunks

    captured = mc.pack_dense(REDUCTION, OUTPUTS, chunks, weights, None,
                             False)
    section = record["constant_section"]
    assert hashlib.sha256(captured).hexdigest() == section["sha256"], \
        "identity packing does not reproduce the captured section"

    compensated = mc.pack_dense_writeout_inverse(REDUCTION, OUTPUTS, chunks,
                                                 weights)
    assert compensated == captured, \
        "the uniform-payload section must be invariant under the run " \
        "placement, or byte parity against the capture would break"
    assert hashlib.sha256(compensated).hexdigest() == section["sha256"]

    record["device_writeout_decode"] = {
        "schema": "f3-writeout-inverse-v1",
        "measured_on": "jw16mbp1-linux (T6001, M1 Max, Asahi)",
        "source": ("receipts/2026-09-18-encoder-conv-device-gate/"
                   "f3-writeout-permutation.json, gate branch "
                   "agent/encoder-conv-device-gate @ c6f86fb"),
        "run_geometry": ("dev run k (16 planes) displays ref run R[k]; the "
                         "engine reads packer slot R(k) for output run k, "
                         "so the compiler parks ref run k at slot R(k): "
                         "slot p carries ref run R^-1(p), within-run plane "
                         "order preserved"),
        "dev_run_to_ref_run": measured,
        "payload_note": ("the captured weights.bin is uniform per constant, "
                         "so the section bytes above are invariant under "
                         "the run placement; the packers (pack_dense_"
                         "writeout_inverse / packConvDenseWriteoutInverse) "
                         "apply the inverse for this geometry and device "
                         "execution is the oracle"),
        "rel_l2_as_emitted_identity": 1.413157,
        "rel_l2_after_inverse_offline": 0.000207,
    }
    RECORD.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    rebuilt = mc.conv_constants(record["parameters"], weights, None, "h13")
    assert hashlib.sha256(rebuilt).hexdigest() == section["sha256"], \
        "covered() path does not reproduce the section"
    print(f"annotated {RECORD.name}: section sha256 "
        f"{section['sha256'][:16]}... unchanged; write-out inverse decoded "
        "into the record")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
