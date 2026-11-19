#!/usr/bin/env python3
"""Backlog docs/lane-backlog.md line 174: next M6 program beyond addyxz.

The 32x32 fp16 matmul must compile for H18 with the compiler alone. Today it
fails with h18.outside-parity-envelope (H18 emits only decoded elementwise,
unary, and scalar-constant templates); this gate goes GREEN when a decoded
matmul template lands in the envelope.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1]).resolve()
MIL = ROOT / "tests/fixtures/h18_matmul_32.mil"


def main():
    out = ROOT / "build/h18-matmul-32"
    r = subprocess.run(
        [str(COMPILER), "--target", "H18", "--mil", str(MIL),
         "--model-root", str(out), "--output", str(out)],
        capture_output=True, text=True, timeout=120, check=False)
    assert r.returncode == 0, f"matmul not yet in the H18 envelope: {r.stdout}{r.stderr}"
    manifest = out / "manifest.json"
    assert manifest.is_file(), "no manifest produced"
    print("PASS h18 matmul 32x32 compiles (envelope extended)")


if __name__ == "__main__":
    main()
