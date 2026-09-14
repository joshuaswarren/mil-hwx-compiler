#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def source(op, shape):
    tensor = f"tensor<fp16, {shape}>"
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({tensor} x) {{
    {tensor} y = {op}(x = x)[name = string("y")];
  }} -> (y);
}}
"""


def compile_source(root, name, text, expected_code=None):
    mil = root / f"{name}.mil"
    out = root / name
    mil.write_text(text)
    run = subprocess.run(
        [compiler, "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=30)
    if expected_code is None:
        assert run.returncode == 0, run.stdout + run.stderr
        return json.loads((out / "manifest.json").read_text())
    assert run.returncode == 65, run.stdout + run.stderr
    assert expected_code in run.stderr, run.stderr
    assert "LUT unaries have no 64-lane split" in run.stderr, run.stderr
    return None


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-silu-envelope-") as directory:
    root = Path(directory)
    for op in ("silu", "sigmoid"):
        for shape in ("[1, 64, 1, 1]", "[1, 512, 1, 1]"):
            manifest = compile_source(root, f"{op}-{shape}", source(op, shape))
            assert len(manifest["programs"]) == 1, manifest["programs"]
            program = manifest["programs"][0]
            assert program["operation"] == op
            assert program["encoder"] == "h13-oracle-parity"
        compile_source(root, f"{op}-glu", source(op, "[1, 1024, 375]"),
                       expected_code="h13.unary-outside-envelope")
    compile_source(root, "silu-128", source("silu", "[1, 128, 1, 1]"),
                   expected_code="h13.unary-outside-envelope")
print("h13 silu envelope cli: PASS")
