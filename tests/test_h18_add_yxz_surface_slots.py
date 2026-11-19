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
        for key in RECORDED:
            assert parsed[key] == oracle[key], f"{key} differs"
    print("PASS binary_add_names_yx_z surface-slot parity (v2 HWX task_descriptors)")


if __name__ == "__main__":
    main()
