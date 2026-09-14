#!/usr/bin/env python3
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def blob(payload):
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(payload), 128)
    return bytes(header) + payload


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
    assert not out.exists(), f"failed compilation wrote {out}"
    return None


ln_c1024 = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 1024, 1, 1]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([1])];
    tensor<fp16, [1, 1024, 1, 1]> y = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))[name = string("y")];
  } -> (y);
}
"""

sm_heads = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 128, 128]> x) {
    int32 axis = const()[name = string("axis"), val = int32(-1)];
    tensor<fp16, [1, 8, 128, 128]> y = softmax(x = x, axis = axis)[name = string("y")];
  } -> (y);
}
"""

ln_encoder = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))[name = string("y")];
  } -> (y);
}
"""

ln_encoder_affine = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1024]> gamma = const()[name = string("gamma"), val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/gamma.bin"), offset = uint64(64)))];
    tensor<fp16, [1024]> beta = const()[name = string("beta"), val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/beta.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001), gamma = gamma, beta = beta)[name = string("y")];
  } -> (y);
}
"""

sm_encoder = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 375]> x) {
    int32 axis = const()[name = string("axis"), val = int32(-1)];
    tensor<fp16, [1, 8, 375, 375]> y = softmax(x = x, axis = axis)[name = string("y")];
  } -> (y);
}
"""

with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-norm-envelope-") as directory:
    root = Path(directory)
    (root / "gamma.bin").write_bytes(blob(bytes(1024 * 2)))
    (root / "beta.bin").write_bytes(blob(bytes(1024 * 2)))
    ln = compile_source(root, "ln-c1024", ln_c1024)
    assert len(ln["programs"]) == 1
    assert ln["programs"][0]["operation"] == "layer_norm"
    assert ln["programs"][0]["encoder"] == "apple-parity-norm"
    assert ln["programs"][0]["taskDescriptors"] == 5
    sm = compile_source(root, "sm-8-128", sm_heads)
    assert len(sm["programs"]) == 1
    assert sm["programs"][0]["operation"] == "softmax"
    assert sm["programs"][0]["encoder"] == "apple-parity-norm"
    assert sm["programs"][0]["taskDescriptors"] == 6
    compile_source(root, "ln-enc", ln_encoder, expected_code="h13.norm-outside-envelope")
    compile_source(root, "ln-enc-aff", ln_encoder_affine,
                   expected_code="h13.norm-outside-envelope")
    compile_source(root, "sm-enc", sm_encoder, expected_code="h13.norm-outside-envelope")
print("h13 norm envelope cli: PASS")
