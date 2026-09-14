#!/usr/bin/env python3
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())

source = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 1, 375, 1024]> x, tensor<fp16, [1, 1, 1024, 128]> w) {
    bool tx = const()[name = string("tx"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(false)];
    tensor<fp16, [1, 1, 375, 128]> product = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("product")];
  } -> (product);
}
"""

with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-rank4-linear-") as directory:
    root = Path(directory)
    mil = root / "rank4-linear.mil"
    out = root / "rank4-linear"
    mil.write_text(source)
    run = subprocess.run(
        [compiler, "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=60)
    assert run.returncode == 65, run.stdout + run.stderr
    assert "h13.matmul-outside-envelope" in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
print("h13 rank4 linear cli: PASS")
