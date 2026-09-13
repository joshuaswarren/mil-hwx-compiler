#!/usr/bin/env python3
import json
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")


def split_source(batch=1):
    input_shape = f"[{batch}, 128, 1, 1]"
    output_shape = f"[{batch}, 64, 1, 1]"
    return f"""program(1) {{
  func main<CoreML8>(tensor<fp16, {input_shape}> input) {{
    tensor<int32, []> count = const()[val = tensor<int32, []>(2)];
    tensor<int32, []> axis = const()[val = tensor<int32, []>(1)];
    tensor<int32, [4]> shape = const()[val = tensor<int32, [4]>([1, 64, 1, 1])];
    (tensor<fp16, {output_shape}> first, tensor<fp16, {output_shape}> second) = split(axis = axis, num_splits = count, x = input);
    tensor<fp16, {output_shape}> second_view = reshape(shape = shape, x = second);
    tensor<fp16, {output_shape}> gate = sigmoid(x = second_view);
    tensor<fp16, {output_shape}> output = mul(x = first, y = gate);
  }} -> (output);
}}
"""


def three_way_split_source():
    return """program(1) {
  func main<CoreML8>(tensor<fp16, [1, 192, 1, 1]> input) {
    tensor<int32, []> count = const()[val = tensor<int32, []>(3)];
    tensor<int32, []> axis = const()[val = tensor<int32, []>(1)];
    (tensor<fp16, [1, 64, 1, 1]> first, tensor<fp16, [1, 64, 1, 1]> second, tensor<fp16, [1, 64, 1, 1]> third) = split(axis = axis, num_splits = count, x = input);
  } -> (first);
}
"""


def compile_source(root, name, source, expected_code=None):
    source_path = root / f"{name}.mil"
    output = root / name
    source_path.write_text(source)
    result = subprocess.run(
        [compiler, "--mil", str(source_path), "--model-root", str(root),
         "--output", str(output), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True, timeout=30, check=False)
    if expected_code is None:
        assert result.returncode == 0, result.stderr
        return output
    assert result.returncode == 65, result.stderr
    assert expected_code in result.stderr, result.stderr
    return None


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    package = compile_source(root, "contiguous-axis-1", split_source())
    manifest = json.loads((package / "manifest.json").read_text())
    assert [program["operation"] for program in manifest["programs"]] == [
        "sigmoid", "mul"]
    sigmoid_input = manifest["programs"][0]["inputs"][0]
    mul_input = next(item for item in manifest["programs"][1]["inputs"]
                     if item["name"] == "input")
    assert sigmoid_input["slice"] == {
        "tensor": "input", "elementOffset": 64, "elementCount": 64}
    assert mul_input["slice"] == {
        "tensor": "input", "elementOffset": 0, "elementCount": 64}
    assert "first" not in manifest["tensors"]
    assert "second" not in manifest["tensors"]
    assert "second_view" not in manifest["tensors"]

    dense = list(range(128))
    axis_reference = [dense[channel] for channel in range(128)]
    first_reference, second_reference = axis_reference[:64], axis_reference[64:]
    first_slice = mul_input["slice"]
    second_slice = sigmoid_input["slice"]
    assert dense[first_slice["elementOffset"]:
                 first_slice["elementOffset"] + first_slice["elementCount"]] == first_reference
    assert dense[second_slice["elementOffset"]:
                 second_slice["elementOffset"] + second_slice["elementCount"]] == second_reference

    inspected = subprocess.run(
        [sys.executable, inspector, str(package)], capture_output=True,
        text=True, timeout=30, check=False)
    assert inspected.returncode == 0, inspected.stderr

    compile_source(root, "noncontiguous-axis-1", split_source(batch=2),
                   "h13.noncontiguous-split-axis")
    compile_source(root, "unsupported-count", three_way_split_source(),
                   "h13.unsupported-split-count")

print("h13 split cli: PASS")
