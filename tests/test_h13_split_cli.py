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
def constant_split_source(reshape=False):
    view = ""
    source = "weights"
    if reshape:
        view = "    tensor<int32, [4]> shape = const()[val = tensor<int32, [4]>([1, 128, 1, 1])];\n    tensor<fp16, [1, 128, 1, 1]> view = reshape(shape = shape, x = weights);\n"
        source = "view"
    return f"""program(1) {{
  func main<CoreML8>() {{
    tensor<fp16, [1, 128, 1, 1]> weights = const()[val = tensor<fp16, [1, 128, 1, 1]>(BLOBFILE(path = string(\"@model_path/missing.bin\"), offset = uint64(0)))];
{view}    tensor<int32, []> count = const()[val = tensor<int32, []>(2)];
    tensor<int32, []> axis = const()[val = tensor<int32, []>(1)];
    (tensor<fp16, [1, 64, 1, 1]> first, tensor<fp16, [1, 64, 1, 1]> second) = split(axis = axis, num_splits = count, x = {source});
    tensor<fp16, [1, 64, 1, 1]> gate = sigmoid(x = second);
    tensor<fp16, [1, 64, 1, 1]> output = mul(x = first, y = gate);
  }} -> (output);
}}
"""


def duplicate_result_source():
    return """program(1) {
  func main<CoreML8>(tensor<fp16, [1, 64, 1, 1]> input) {
    tensor<fp16, [1, 64, 1, 1]> output = sigmoid(x = input);
  } -> (output, output);
}
"""

def reshaped_duplicate_result_source():
    return """program(1) {
  func main<CoreML8>(tensor<fp16, [1, 64, 1, 1]> input) {
    tensor<fp16, [1, 64, 1, 1]> output = sigmoid(x = input);
    tensor<int32, [4]> shape = const()[val = tensor<int32, [4]>([1, 1, 8, 8])];
    tensor<fp16, [1, 1, 8, 8]> view = reshape(shape = shape, x = output);
  } -> (view, view);
}
"""


def sliced_duplicate_result_source():
    return """program(1) {
  func main<CoreML8>(tensor<fp16, [1, 128, 1, 1]> input) {
    tensor<fp16, [1, 128, 1, 1]> output = relu(x = input);
    tensor<int32, []> count = const()[val = tensor<int32, []>(2)];
    tensor<int32, []> axis = const()[val = tensor<int32, []>(1)];
    (tensor<fp16, [1, 64, 1, 1]> first, tensor<fp16, [1, 64, 1, 1]> second) = split(axis = axis, num_splits = count, x = output);
  } -> (second, second);
}
"""

def distinct_split_result_source(reverse=False):
    returned = "wide, tall" if reverse else "tall, wide"
    return f"""program(1) {{
  func main<CoreML8>(tensor<fp16, [1, 128, 1, 1]> input) {{
    tensor<fp16, [1, 128, 1, 1]> output = relu(x = input);
    tensor<int32, []> count = const()[val = tensor<int32, []>(2)];
    tensor<int32, []> axis = const()[val = tensor<int32, []>(1)];
    (tensor<fp16, [1, 64, 1, 1]> first, tensor<fp16, [1, 64, 1, 1]> second) = split(axis = axis, num_splits = count, x = output);
    tensor<int32, [4]> tall_shape = const()[val = tensor<int32, [4]>([1, 8, 8, 1])];
    tensor<int32, [4]> wide_shape = const()[val = tensor<int32, [4]>([1, 1, 8, 8])];
    tensor<fp16, [1, 8, 8, 1]> tall = reshape(shape = tall_shape, x = first);
    tensor<fp16, [1, 1, 8, 8]> wide = reshape(shape = wide_shape, x = second);
  }} -> ({returned});
}}
"""


def independent_result_source(reverse=False):
    returned = "wide, tall" if reverse else "tall, wide"
    return f"""program(1) {{
  func main<CoreML8>(tensor<fp16, [1, 64, 1, 1]> tall_input, tensor<fp16, [1, 64, 1, 1]> wide_input) {{
    tensor<fp16, [1, 64, 1, 1]> tall_storage = relu(x = tall_input);
    tensor<int32, [4]> tall_shape = const()[val = tensor<int32, [4]>([1, 8, 8, 1])];
    tensor<fp16, [1, 8, 8, 1]> tall = reshape(shape = tall_shape, x = tall_storage);
    tensor<fp16, [1, 64, 1, 1]> wide_storage = sigmoid(x = wide_input);
    tensor<int32, [4]> wide_shape = const()[val = tensor<int32, [4]>([1, 1, 8, 8])];
    tensor<fp16, [1, 1, 8, 8]> wide = reshape(shape = wide_shape, x = wide_storage);
  }} -> ({returned});
}}
"""


def mixed_result_source():
    return """program(1) {
  func main<CoreML8>(tensor<fp16, [1, 64, 1, 1]> input) {
    tensor<fp16, [1, 64, 1, 1]> output = sigmoid(x = input);
    tensor<string, []> fp32_dtype = const()[val = tensor<string, []>("fp32")];
    tensor<string, []> int32_dtype = const()[val = tensor<string, []>("int32")];
    tensor<fp32, [1, 64, 1, 1]> fp32_output = cast(dtype = fp32_dtype, x = output);
    tensor<int32, [1, 64, 1, 1]> int32_output = cast(dtype = int32_dtype, x = output);
  } -> (fp32_output, int32_output);
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
        "tensor": "input", "elementOffset": 64, "elementCount": 64,
        "physicalElements": 64}
    assert mul_input["slice"] == {
        "tensor": "input", "elementOffset": 0, "elementCount": 64,
        "physicalElements": 64}
    assert "first" not in manifest["tensors"]
    assert "second" not in manifest["tensors"]
    assert "second_view" not in manifest["tensors"]
    assert all(type(program.get("scratchBytes")) is int
               for program in manifest["programs"])

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


    duplicate = compile_source(root, "duplicate-identity-results",
                               duplicate_result_source())
    duplicate_manifest = json.loads((duplicate / "manifest.json").read_text())
    physical = {"tensor": "output", "dtype": "float16",
                "shape": [1, 64, 1, 1], "logicalBytes": 128}
    mapping = {"name": "output", "dtype": "float16",
               "shape": [1, 64, 1, 1],
               "physical": {"tensor": "output", "elementOffset": 0,
                            "elementCount": 64},
               "conversion": "identity"}
    assert duplicate_manifest["schema"] == "mil-hwxc.h13-anec-package.v2"
    assert duplicate_manifest["physicalOutputs"] == [physical]
    assert duplicate_manifest["logicalResults"] == [mapping, mapping]
    duplicate_inspected = subprocess.run(
        [sys.executable, inspector, str(duplicate)], capture_output=True,
        text=True, timeout=30, check=False)
    assert duplicate_inspected.returncode == 0, duplicate_inspected.stderr

    reshaped = compile_source(root, "reshaped-identity-results",
                              reshaped_duplicate_result_source())
    reshaped_manifest = json.loads((reshaped / "manifest.json").read_text())
    assert reshaped_manifest["physicalOutputs"] == [physical]
    reshaped_mapping = {"name": "view", "dtype": "float16",
                        "shape": [1, 1, 8, 8],
                        "physical": {"tensor": "output", "elementOffset": 0,
                                     "elementCount": 64},
                        "conversion": "identity"}
    assert reshaped_manifest["logicalResults"] == [
        reshaped_mapping, reshaped_mapping]
    reshaped_inspected = subprocess.run(
        [sys.executable, inspector, str(reshaped)], capture_output=True,
        text=True, timeout=30, check=False)
    assert reshaped_inspected.returncode == 0, reshaped_inspected.stderr

    sliced = compile_source(root, "sliced-identity-results",
                            sliced_duplicate_result_source())
    sliced_manifest = json.loads((sliced / "manifest.json").read_text())
    sliced_physical = {"tensor": "output", "dtype": "float16",
                       "shape": [1, 128, 1, 1], "logicalBytes": 256}
    sliced_mapping = {"name": "second", "dtype": "float16",
                      "shape": [1, 64, 1, 1],
                      "physical": {"tensor": "output", "elementOffset": 64,
                                   "elementCount": 64},
                      "conversion": "identity"}
    assert sliced_manifest["physicalOutputs"] == [sliced_physical]
    assert sliced_manifest["logicalResults"] == [sliced_mapping, sliced_mapping]
    sliced_inspected = subprocess.run(
        [sys.executable, inspector, str(sliced)], capture_output=True,
        text=True, timeout=30, check=False)
    assert sliced_inspected.returncode == 0, sliced_inspected.stderr
    tall_mapping = {
        "name": "tall",
        "dtype": "float16",
        "shape": [1, 8, 8, 1],
        "physical": {
            "tensor": "output",
            "elementOffset": 0,
            "elementCount": 64,
        },
        "conversion": "identity",
    }
    wide_mapping = {
        "name": "wide",
        "dtype": "float16",
        "shape": [1, 1, 8, 8],
        "physical": {
            "tensor": "output",
            "elementOffset": 64,
            "elementCount": 64,
        },
        "conversion": "identity",
    }
    for reverse, expected_logical in (
        (False, [tall_mapping, wide_mapping]),
        (True, [wide_mapping, tall_mapping]),
    ):
        distinct_package = compile_source(
            root,
            f"distinct-{'reverse' if reverse else 'forward'}",
            distinct_split_result_source(reverse),
        )
        distinct_manifest = json.loads(
            (distinct_package / "manifest.json").read_text())
        assert distinct_manifest["physicalOutputs"] == [sliced_physical]
        assert distinct_manifest["logicalResults"] == expected_logical
        distinct_inspected = subprocess.run(
            [sys.executable, inspector, str(distinct_package)], capture_output=True,
            text=True, timeout=30, check=False)
        assert distinct_inspected.returncode == 0, distinct_inspected.stderr

    independent_physical = [
        {
            "tensor": "tall_storage",
            "dtype": "float16",
            "shape": [1, 64, 1, 1],
            "logicalBytes": 128,
        },
        {
            "tensor": "wide_storage",
            "dtype": "float16",
            "shape": [1, 64, 1, 1],
            "logicalBytes": 128,
        },
    ]
    independent_tall_mapping = {
        **tall_mapping,
        "physical": {
            "tensor": "tall_storage",
            "elementOffset": 0,
            "elementCount": 64,
        },
    }
    independent_wide_mapping = {
        **wide_mapping,
        "physical": {
            "tensor": "wide_storage",
            "elementOffset": 0,
            "elementCount": 64,
        },
    }
    for reverse, expected_logical in (
        (False, [independent_tall_mapping, independent_wide_mapping]),
        (True, [independent_wide_mapping, independent_tall_mapping]),
    ):
        independent_package = compile_source(
            root,
            f"independent-{'reverse' if reverse else 'forward'}",
            independent_result_source(reverse),
        )
        independent_manifest = json.loads(
            (independent_package / "manifest.json").read_text())
        assert independent_manifest["physicalOutputs"] == independent_physical
        assert independent_manifest["logicalResults"] == expected_logical
        independent_inspected = subprocess.run(
            [sys.executable, inspector, str(independent_package)], capture_output=True,
            text=True, timeout=30, check=False)
        assert independent_inspected.returncode == 0, independent_inspected.stderr

    sliced_manifest["logicalResults"][0]["physical"]["physicalElements"] = 128
    (sliced / "manifest.json").write_text(json.dumps(sliced_manifest))
    invalid_view_fields = subprocess.run(
        [sys.executable, inspector, str(sliced)], capture_output=True,
        text=True, timeout=30, check=False)
    assert invalid_view_fields.returncode != 0
    assert "physical mapping has incorrect fields" in invalid_view_fields.stderr

    duplicate_manifest["logicalResults"][0]["dtype"] = "float32"
    (duplicate / "manifest.json").write_text(json.dumps(duplicate_manifest))
    invalid_identity = subprocess.run(
        [sys.executable, inspector, str(duplicate)], capture_output=True,
        text=True, timeout=30, check=False)
    assert invalid_identity.returncode != 0
    assert "identity result dtype differs" in invalid_identity.stderr

    compile_source(root, "mixed-logical-results", mixed_result_source(),
                   "h13.unsupported-logical-result-conversion")
    compile_source(root, "noncontiguous-axis-1", split_source(batch=2),
                   "h13.noncontiguous-split-axis")
    compile_source(root, "unsupported-count", three_way_split_source(),
                   "h13.unsupported-split-count")
    compile_source(root, "constant-source", constant_split_source(),
                   "h13.unsupported-constant-split-source")
    compile_source(root, "reshaped-constant-source",
                   constant_split_source(reshape=True),
                   "h13.unsupported-constant-split-source")

print("h13 split cli: PASS")
