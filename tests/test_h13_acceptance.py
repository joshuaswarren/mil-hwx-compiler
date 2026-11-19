#!/usr/bin/env python3
"""Behavior test for research/h13_acceptance.py, plus the measured H13 SiLU set."""
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

checker = str(Path(__file__).resolve().parents[1] / "research/h13_acceptance.py")
standalone = Path.home() / "scratch/w72/silu-qualification/H13"


def run(got, ref, bound):
    return subprocess.run([sys.executable, checker, got, ref, str(bound)],
                          capture_output=True, text=True)


def write(path, values):
    with open(path, "wb") as handle:
        handle.write(struct.pack("<%de" % len(values), *values))


def main():
    with tempfile.TemporaryDirectory(prefix="h13-acceptance-") as directory:
        root = Path(directory)
        # Pass: 0.001 error inside a 0.016 bound, exact lanes included.
        ref = [0.0, 1.0, -2.0, 0.5]
        write(root / "ref.fp16", ref)
        write(root / "got.fp16", [0.0, 1.0, -1.999, 0.5])
        result = run(str(root / "got.fp16"), str(root / "ref.fp16"), 0.016)
        assert result.returncode == 0, result.stdout + result.stderr

        # Fail: one lane over the bound.
        write(root / "got.fp16", [0.0, 1.02, -2.0, 0.5])
        result = run(str(root / "got.fp16"), str(root / "ref.fp16"), 0.016)
        assert result.returncode == 1, result.stdout + result.stderr
        assert '"over_bound": 1' in result.stdout, result.stdout

        # Fail: nonfinite output is always a violation.
        inf = float("inf")
        write(root / "got.fp16", [0.0, inf, -2.0, 0.5])
        result = run(str(root / "got.fp16"), str(root / "ref.fp16"), 0.016)
        assert result.returncode == 1, result.stdout + result.stderr
        assert '"nonfinite": 1' in result.stdout, result.stdout

    # Measured evidence: standalone H13 SiLU qualification set, bound 0.016.
    if standalone.exists():
        result = run(str(standalone / "2702.failed.fp16"), str(standalone / "expected.fp16"), 0.016)
        assert result.returncode == 0, result.stdout + result.stderr
        assert '"nonfinite": 0' in result.stdout, result.stdout
    print("PASS h13 acceptance checker")


if __name__ == "__main__":
    main()
