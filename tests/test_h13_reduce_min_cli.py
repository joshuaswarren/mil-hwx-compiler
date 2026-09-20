#!/usr/bin/env python3
"""The captured fp16 reduce_min row (Main-authorized Apple CPU oracle).

The int32 reduce_min the encoder spells stays refused (Apple refuses the
int32 data type); the fp16 form decodes and this test pins its bytes against
the captured record.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve())
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "research"))
import test_h13_parity as parity  # noqa: E402

record = json.loads(
    (ROOT / "research/oracles/h13-captures/capture_reduce_min_f16.json").read_text())
assert record.get("error") is None

with tempfile.TemporaryDirectory(prefix="h13-reduce-min-") as directory:
    root = Path(directory)
    (root / "rmin.mil").write_text(record["mil"])
    out = root / "out"
    run = subprocess_run = __import__("subprocess").run(
        [compiler, "--mil", str(root / "rmin.mil"), "--model-root", str(root),
         "--target", "H13", "--format", "anec", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    parity.check_anec(record, out, manifest)
    assert manifest["programs"][0]["encoder"] == "apple-parity-norm"

print("h13 reduce min cli: PASS")
