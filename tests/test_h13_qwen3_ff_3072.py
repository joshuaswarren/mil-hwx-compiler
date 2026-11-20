#!/usr/bin/env python3
"""Opt-in fusion gate for the 3072-wide layer-0 projection and multiply graph."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from test_h13_qwen3_ff_fusion import oracle_task_hash
from timeouts import scaled_timeout


def main():
    compiler = Path(sys.argv[1]).resolve()
    mil = Path(sys.argv[2]).resolve()
    with tempfile.TemporaryDirectory(prefix="h13-qwen3-ff3072-") as directory:
        out = Path(directory)
        run = subprocess.run(
            [str(compiler), "--target", "H13", "--mil", str(mil),
             "--model-root", str(mil.parent), "--output", str(out)],
            capture_output=True, text=True, timeout=scaled_timeout(120), check=False)
        assert run.returncode == 0, (
            f"wide graph refused: {run.stdout}{run.stderr}")
        manifest = json.loads((out / "manifest.json").read_text())
        plan = manifest["dispatchPlan"]
        files = {path.name for path in out.glob("*.anec")}
        assert len(manifest["programs"]) == 1 and plan == [0] and len(files) == 1, (
            f"expected one program and one dispatch, got {len(files)} files and {len(plan)} dispatches")
        program = manifest["programs"][0]
        assert files == {program["file"]}
        assert program["operation"] == "qwen-ff" and program["taskDescriptors"] == 5, program
        assert program["constantBytes"] == 3 * 1024 * 3072 * 2, "expected three wide FP16 matrices"
        anec = (out / program["file"]).read_bytes()
        assert oracle_task_hash(anec) == (
            "a095ca8f83eb6707236ba58b0f8a9236eeaec16e656b65055f2459bd93e68fbe")
        assert hashlib.sha256(anec[-program["constantBytes"]:]).hexdigest() == (
            "cb05e2008ca59102ba4424490580f07a63c391003c1b478239f066f65176bf05")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] /
                                 "research/inspect_anec.py"), str(out)],
            check=True, stdout=subprocess.DEVNULL, timeout=scaled_timeout(60))
        print("PASS Qwen3 H13 FF 3072 one-dispatch package")


if __name__ == "__main__":
    main()
