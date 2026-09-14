#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())

source = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 128]> x, tensor<fp16, [1, 8, 128, 375]> w) {
    bool tx = const()[name = string("tx"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(false)];
    tensor<fp16, [1, 8, 375, 375]> product = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("product")];
  } -> (product);
}
"""

with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-matmul-envelope-") as directory:
    root = Path(directory)
    mil = root / "attn-r4.mil"
    out = root / "attn-r4"
    mil.write_text(source)
    run = subprocess.run(
        [compiler, "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["programs"]) == 1, [p.get("operation") for p in manifest["programs"]]
    program = manifest["programs"][0]
    assert program["operation"] == "matmul"
    assert program["encoder"] == "apple-parity-batched-matmul"
    assert program["taskDescriptors"] == 208
    assert program["outputs"][0]["shape"] == [1, 8, 375, 375]
    assert program["inputs"][0]["shape"] == [1, 8, 128, 375]
    assert program["inputs"][1]["shape"] == [1, 8, 375, 128]
print("h13 matmul envelope cli: PASS")
