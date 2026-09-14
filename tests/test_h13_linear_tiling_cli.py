#!/usr/bin/env python3
import json
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def linear_source(rows):
    x = f"tensor<fp16, [1, {rows}, 1024]>"
    w = "tensor<fp16, [128, 1024]>"
    b = "tensor<fp16, [128]>"
    y = f"tensor<fp16, [1, {rows}, 128]>"
    bias = ", ".join("fp16(1.0)" for _ in range(128))
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x} x) {{
    {w} W = const()[name = string("W"), val = {w}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {b} bias = const()[name = string("bias"), val = {b}([{bias}])];
    {y} y = linear(x = x, weight = W, bias = bias)[name = string("projection")];
  }} -> (y);
}}
"""


def blob(payload):
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(payload), 128)
    return bytes(header) + payload


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-linear-tiling-") as directory:
    root = Path(directory)
    (root / "weights.bin").write_bytes(blob(bytes(128 * 1024 * 2)))
    for rows in (1, 3):
        mil = root / f"linear-{rows}.mil"
        out = root / f"linear-{rows}"
        mil.write_text(linear_source(rows))
        run = subprocess.run(
            [compiler, "--mil", str(mil), "--model-root", str(root),
             "--target", "H13", "--output", str(out)],
            capture_output=True, text=True, check=False, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
        manifest = json.loads((out / "manifest.json").read_text())
        ops = [program["operation"] for program in manifest["programs"]]
        assert Counter(ops) == {"matmul": 2 * rows, "add": 3 * rows}, ops
        assert all(program["encoder"] == "h13-source-qualified" or
                   program["operation"] != "matmul"
                   for program in manifest["programs"])
        matmul = [program for program in manifest["programs"]
                  if program["operation"] == "matmul"]
        assert [program["inputs"][0]["slice"]["elementCount"] for program in matmul] == (
            [512, 512] * rows)
print("h13 linear tiling cli: PASS")
