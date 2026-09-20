#!/usr/bin/env python3
"""Turn Main's Apple `ane-compile-hwx` layer_norm captures into oracle-style
records, using the same decoder the minted campaigns use.

Each input is /tmp/ln-factorial-main/<case>: `capture/model.mil` plus
`compiled/model.hwx` (CPU-only Apple tool output). The records land in
`research/oracles/h13-captures/`, which the parity suite does not scan; they
feed `mint_norm_probes.py --emit-templates` and the epsilon CLI test instead.

A capture that Apple refused is recorded with an `error` field, exactly like
`mint_oracles.run_case` records rejections, so the refusal stays auditable.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mint_oracles import parse_hwx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = Path("/tmp/ln-factorial-main")
OUTPUT = ROOT / "research/oracles/h13-captures"

# rank3_exact is the encoder row; the controls isolate one variable each
# against it (epsilon spelling, epsilon value, axes spelling, rank, batch).
CASES = ("rank3_exact", "width_exact", "batch_exact",
         "control_default", "control_exact", "control_fp16",
         "control_unbracketed")

SHAPE = re.compile(r"tensor<fp16, \[([\d, ]+)\]> x")
AXES = re.compile(r"val = tensor<int32, \[1\]>\(\[?(-?\d+)\]?\)")


def shape_of(mil: str) -> list[int]:
    match = SHAPE.search(mil)
    if not match:
        raise ValueError("capture MIL has no fp16 input shape")
    return [int(value) for value in match.group(1).split(",")]


def axes_of(mil: str) -> list[int]:
    match = AXES.search(mil)
    if not match:
        raise ValueError("capture MIL has no recognizable axes constant")
    text = match.group(1)
    return [int(text)]


def record_for(case: str) -> dict:
    capture = CAPTURE_ROOT / case
    mil = (capture / "capture/model.mil").read_text()
    shape = shape_of(mil)
    parameters = {
        "operation": "layer_norm", "shape": shape,
        "axes": axes_of(mil), "affine": "gamma" in mil,
    }
    record = {
        "schema_version": 1,
        "case": case,
        "family": "normalization",
        "parameters": parameters,
        "mil": mil,
        "weights": {},
        "target": "h13",
        "provenance": {
            "capture_dir": str(capture),
            "tool": "ane-compile-hwx (Apple, CPU-only; Main's capture)",
        },
    }
    hwx = capture / "compiled/model.hwx"
    if not hwx.is_file() or hwx.stat().st_size == 0:
        record.update({
            "hwx_sha256": None, "hwx_bytes": None,
            "program_descriptor": None, "tensor_descriptors": [],
            "constant_section": None, "task_descriptors": [],
            "error": "apple-refused: no model.hwx in the capture "
                     "(control_unbracketed pins Apple's rejection of the "
                     "unbracketed tensor<int32, [1]>(N) axes literal)",
        })
        return record
    data = hwx.read_bytes()
    record.update(parse_hwx(data, "h13"))
    record["provenance"]["hwx_sha256"] = hashlib.sha256(data).hexdigest()
    return record


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        record = record_for(case)
        destination = OUTPUT / f"{case}.json"
        destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        status = "decoded" if record.get("error") is None else "refused"
        print(f"{case}: {status} -> {destination.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
