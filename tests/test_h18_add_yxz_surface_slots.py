#!/usr/bin/env python3
"""M6 add y,x->z: surface BAR slots in the v2 HWX must match Apple's oracle."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import mint_oracles  # noqa: E402
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1]).resolve()
ORACLE_PATH = ROOT / "research/oracles/h18/binary_add_names_yx_z.json"
RECORDED = ("program_descriptor", "program_descriptor_words",
            "tensor_descriptors", "constant_section", "kernel_section",
            "task_descriptors")

# The task-stream words at +0x100/+0x118 pick the DART STE contexts for the
# output DMA and are mirrored into the NC records at 0x04d18/0x05110. Apple's
# macOS capture records the macOS-firmware pair (0x23809346/0x24009444); M6
# Linux hardware rejects it (three runs, 0/64, SID 0 write STE invalid) and
# accepts the driver-configured pair (0x22809346/0x23809444; flagswap/addyxz3
# 2026-11-19, 64/64 x20 exact vs CPU fp16, zero faults). Excluded from Apple
# parity here and pinned to the hardware values by
# tests/test_h18_yxz_slot_record_words.py.
STE_RECORD_ADDRESSES = ("0x04d18", "0x05110")
STE_CANONICAL_HEADER = "STE-CONTEXT"


def neutralize_ste_records(task_descriptors):
    for descriptor in task_descriptors:
        for record in descriptor.get("records", []):
            if record.get("address") in STE_RECORD_ADDRESSES:
                record["header"] = STE_CANONICAL_HEADER


def main():
    oracle = json.loads(ORACLE_PATH.read_text())
    assert oracle["case"] == "binary_add_names_yx_z"
    with tempfile.TemporaryDirectory(prefix="h18-yxz-") as directory:
        root = Path(directory)
        path = root / "binary_add_names_yx_z.mil"
        path.write_text(oracle["mil"])
        out = root / "out"
        r = subprocess.run(
            [str(COMPILER), "--mil", str(path), "--model-root", str(root),
             "--target", "H18", "--output", str(out)],
            capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        manifest = json.loads((out / "manifest.json").read_text())
        hwx = out / manifest["programs"][0]["file"]
        parsed = mint_oracles.parse_hwx(hwx.read_bytes(), "h18")
        neutralize_ste_records(parsed["task_descriptors"])
        neutralize_ste_records(oracle["task_descriptors"])
        for key in RECORDED:
            assert parsed[key] == oracle[key], f"{key} differs"
    print("PASS binary_add_names_yx_z surface-slot parity (v2 HWX task_descriptors)")


if __name__ == "__main__":
    main()
