#!/usr/bin/env python3
"""Opt-in fusion gate: pass the compiler and the benchmark's ff.mil path."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout


def main():
    compiler, mil = (Path(arg).resolve() for arg in sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="h13-qwen3-ff-") as directory:
        out = Path(directory)
        run = subprocess.run(
            [str(compiler), "--target", "H13", "--mil", str(mil),
             "--model-root", str(mil.parent), "--output", str(out)],
            capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
        assert run.returncode == 0, run.stdout + run.stderr
        manifest = json.loads((out / "manifest.json").read_text())
        plan = manifest["dispatchPlan"]
        files = {path.name for path in out.glob("*.anec")}
        assert len(manifest["programs"]) == 1 and plan == [0] and len(files) == 1, (
            f"expected one H13 program and one dispatch, got {len(files)} files and {len(plan)} dispatches")
        assert files == {manifest["programs"][0]["file"]}
    print("PASS Qwen3 H13 FF one-dispatch package")


if __name__ == "__main__":
    main()
