#!/usr/bin/env python3
"""Reproducible M6 H18 add pack: compile the fixture MIL, splice the v2 template,
and require the whole-pack sha the M6 already ran green (receipts 2026-11-19:
flagswap-run-5694a3c0 / D7 run 22, 64/64 x20 exact vs CPU fp16, zero faults)."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = Path(sys.argv[1]).resolve()
MIL = ROOT / "tests/fixtures/h18_addyxz.mil"
TMPL = ROOT / "tests/fixtures/h18_v2_tmpl.bin"
SPLICE = ROOT / "tools/mkapple.py"
WANT = "feedd382a2209c776f1ce5395e6cde8af92217f3733f872462474483941be0bd"


def main():
    with tempfile.TemporaryDirectory(prefix="h18-pack-repro-") as directory:
        root = Path(directory)
        out = root / "out"
        r = subprocess.run(
            [str(COMPILER), "--target", "H18", "--mil", str(MIL),
             "--model-root", str(root), "--output", str(out)],
            capture_output=True, text=True, timeout=120, check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        hwx = out / json.loads((out / "manifest.json").read_text())["programs"][0]["file"]
        pack = root / "pack.bin"
        s = subprocess.run(
            [sys.executable, str(SPLICE), str(TMPL), str(hwx), "-o", str(pack)],
            capture_output=True, text=True, timeout=60, check=False)
        assert s.returncode == 0, s.stdout + s.stderr
        got = hashlib.sha256(pack.read_bytes()).hexdigest()
        assert got == WANT, f"pack sha {got}, want {WANT}"
    print(f"PASS h18 add pack reproducible: {WANT[:16]}")


if __name__ == "__main__":
    main()
