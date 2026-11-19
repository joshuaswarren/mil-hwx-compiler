#!/usr/bin/env python3
"""Opt-in fusion gate: pass the compiler and the benchmark's ff.mil path."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from timeouts import scaled_timeout


def oracle_task_hash(anec):
    task_size, = struct.unpack_from("<Q", anec, 16)
    task = bytearray(anec[0x1000:0x1000 + task_size])
    offset = 0
    for _ in range(5):
        nid, = struct.unpack_from("<I", task, offset)
        next_offset, selectors = struct.unpack_from("<II", task, offset + 28)
        assert (nid >> 16) & 0xff == 0x40
        struct.pack_into("<I", task, offset, nid & ~0x00ff0000)
        for shift in (0, 6, 12):
            if (selectors >> shift) & 31 in (4, 5):
                selectors ^= 1 << shift
        struct.pack_into("<I", task, offset + 32, selectors)
        offset = next_offset
    assert offset == 0
    return hashlib.sha256(task).hexdigest()


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
        program = manifest["programs"][0]
        assert program["taskDescriptors"] == 5
        anec = (out / program["file"]).read_bytes()
        assert oracle_task_hash(anec) == (
            "757faddd04d3c3026ada56dfb4e780fd4f6ee0b1324c2d8d8cb1661c104afe43")
        constants = anec[-program["constantBytes"]:]
        assert len(constants) == 3 * 1024 * 1024 * 2
        assert hashlib.sha256(constants).hexdigest() == (
            "f475e1fcec21145153853147e704cc026be125a8037782e2a0d2c5610e166e2b")
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve().parents[1] /
                                 "research/inspect_anec.py"), str(out)],
            check=True, stdout=subprocess.DEVNULL, timeout=scaled_timeout(60))
        negative = out / "unfused.mil"
        original = mil.read_text()
        changed = original.replace("mul(x = g, y = u)", "add(x = g, y = u)")
        assert changed != original
        negative.write_text(changed)
        other = out / "unfused"
        subprocess.run(
            [str(compiler), "--target", "H13", "--mil", str(negative),
             "--model-root", str(mil.parent), "--output", str(other)],
            check=True, stdout=subprocess.DEVNULL, timeout=scaled_timeout(60))
        unfused = json.loads((other / "manifest.json").read_text())
        assert len(unfused["dispatchPlan"]) > 1
    print("PASS Qwen3 H13 FF one-dispatch package")


if __name__ == "__main__":
    main()
