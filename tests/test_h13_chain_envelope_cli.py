#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())

island = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 128]> x, tensor<fp16, [1, 8, 128, 375]> w, tensor<fp16, [1, 8, 375, 375]> ninf_rt, tensor<bool, [1, 8, 375, 375]> cond) {
    bool tx = const()[name = string("tx"), val = bool(false)];
    bool ty = const()[name = string("ty"), val = bool(false)];
    tensor<fp16, [1, 8, 375, 375]> product = matmul(transpose_x = tx, transpose_y = ty, x = x, y = w)[name = string("product")];
    tensor<fp16, [1, 8, 375, 375]> y = select(a = ninf_rt, b = product, cond = cond)[name = string("y")];
  } -> (y);
}
"""

add_relu = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1,512,1,1]> a, tensor<fp16, [1,512,1,1]> b) {
    tensor<fp16, [1,512,1,1]> sum = add(x = a, y = b)[name = string("sum")];
    tensor<fp16, [1,512,1,1]> result = relu(x = sum)[name = string("result")];
  } -> (result);
}
"""


def compile_source(root, name, text, schedule=None, expected_code=None):
    mil = root / f"{name}.mil"
    out = root / name
    mil.write_text(text)
    command = [compiler, "--mil", str(mil), "--model-root", str(root),
               "--target", "H13", "--output", str(out)]
    if schedule:
        command += ["--schedule", schedule]
    run = subprocess.run(command, capture_output=True, text=True, check=False,
                         timeout=60)
    if expected_code is None:
        assert run.returncode == 0, run.stdout + run.stderr
        return json.loads((out / "manifest.json").read_text())
    assert run.returncode == 65, run.stdout + run.stderr
    assert expected_code in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
    return None


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-chain-envelope-") as directory:
    root = Path(directory)
    package = compile_source(root, "island-perop", island)
    assert package["schema"] == "mil-hwxc.h13-anec-package.v2"
    assert package["dispatchPlan"] == [0, 1]
    assert package["intermediates"] == ["product"]
    assert [program["operation"] for program in package["programs"]] == [
        "matmul", "select"]
    matmul, select = package["programs"]
    assert matmul["encoder"] == "apple-parity-batched-matmul"
    assert matmul["taskDescriptors"] == 208
    assert select["encoder"] == "apple-parity-boolean"
    assert select["taskDescriptors"] == 5
    compile_source(root, "island-chain", island, schedule="chain",
                   expected_code="h13.chain-outside-envelope")
    fused = compile_source(root, "add-relu-chain", add_relu, schedule="chain")
    assert fused["schedule"] == "chain"
    assert fused["dispatchPlan"] == [0]
    assert len(fused["programs"]) == 1
    assert fused["programs"][0]["encoder"] == "composed-chain"
    assert fused["programs"][0]["operation"] == "chain"
print("h13 chain envelope cli: PASS")
