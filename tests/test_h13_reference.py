#!/usr/bin/env python3
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import h13_reference
import h13_run_linux

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()


def fp16(values):
    return b"".join(struct.pack("<e", value) for value in values)


def values(data):
    return list(struct.unpack(f"<{len(data) // 2}e", data))


def blob(payload):
    result = bytearray(128 + len(payload))
    struct.pack_into("<IIQQ", result, 64, 0xDEADBEEF, 1, len(payload), 128)
    result[128:] = payload
    return result


def test_fp16_rounding_and_ops():
    mil = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [2]> x) {
    tensor<fp16, [2]> c = const()[name = string("c"), val = tensor<fp16, [2]>([fp16(0.00048828125), fp16(-2.0)])];
    tensor<fp16, [2]> a = add(x = x, y = c)[name = string("add")];
    tensor<fp16, [2]> b = relu(x = a)[name = string("relu")];
    tensor<fp16, [2]> y = clip(x = b, alpha = fp32(0.0), beta = fp32(1.0))[name = string("clip")];
  } -> (y);
}
"""
    output = h13_reference.evaluate(mil, Path("."), {"x": fp16([1.0, 1.0])})
    assert output == {"y": fp16([1.0, 0.0])}
    assert h13_reference.fp16(65520.0) == math.inf
    with tempfile.TemporaryDirectory(prefix="h13-reference-cli-") as directory:
        root = Path(directory)
        source, input_path, output_path = root / "model.mil", root / "x.fp16", root / "y.fp16"
        source.write_text(mil)
        input_path.write_bytes(fp16([1.0, 1.0]))
        run = subprocess.run([
            sys.executable, str(ROOT / "tools/h13_reference.py"), str(source),
            "--model-root", str(root), "--input", f"x={input_path}",
            "--output", f"y={output_path}",
        ], capture_output=True, text=True, timeout=15, check=False)
        assert run.returncode == 0, run.stdout + run.stderr
        assert output_path.read_bytes() == fp16([1.0, 0.0])


def test_blob_matmul_transposes():
    mil = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [3, 2]> x) {
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [4, 3]> w = const()[name = string("w"), val = tensor<fp16, [4, 3]>(BLOBFILE(path = string("@model_path/w.bin"), offset = uint64(64)))];
    tensor<fp16, [2, 4]> y = matmul(x = x, y = w, transpose_x = t, transpose_y = t)[name = string("mm")];
  } -> (y);
}
"""
    with tempfile.TemporaryDirectory(prefix="h13-reference-matmul-") as directory:
        root = Path(directory)
        root.joinpath("w.bin").write_bytes(blob(fp16([
            1, 0, 0,
            0, 1, 0,
            0, 0, 1,
            1, 1, 1,
        ])))
        output = h13_reference.evaluate(
            mil, root, {"x": fp16([1, 4, 2, 5, 3, 6])})
    assert values(output["y"]) == [1, 2, 3, 6, 4, 5, 6, 15]


def test_shape_ops_linear_and_binary_ops():
    mil = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 2]> x) {
    tensor<fp16, [2, 2]> w = const()[name = string("w"), val = tensor<fp16, [2, 2]>([fp16(1), fp16(2), fp16(3), fp16(4)])];
    tensor<fp16, [2]> bias = const()[name = string("bias"), val = tensor<fp16, [2]>([fp16(1), fp16(-1)])];
    tensor<fp16, [1, 2]> a = linear(x = x, weight = w, bias = bias)[name = string("linear")];
    tensor<fp16, [1, 2]> b = sub(x = a, y = bias)[name = string("sub")];
    tensor<fp16, [1, 2]> c = real_div(x = b, y = tensor<fp16, [2]>([fp16(2), fp16(4)]))[name = string("div")];
    tensor<fp16, [1, 2]> d = maximum(x = c, y = tensor<fp16, [2]>([fp16(2), fp16(2)]))[name = string("max")];
    tensor<fp16, [1, 2]> e = minimum(x = d, y = tensor<fp16, [2]>([fp16(4), fp16(4)]))[name = string("min")];
    tensor<fp16, [1, 2]> f = mul(x = e, y = tensor<fp16, [2]>([fp16(2), fp16(2)]))[name = string("mul")];
    tensor<int32, [1]> shape = const()[name = string("shape"), val = tensor<int32, [1]>([2])];
    tensor<fp16, [2]> r = reshape(x = f, shape = shape)[name = string("reshape")];
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([0])];
    tensor<fp16, [1, 2]> expanded = expand_dims(x = r, axes = axes)[name = string("expand")];
    tensor<fp16, [2]> y = squeeze(x = expanded)[name = string("squeeze")];
  } -> (y);
}
"""
    output = h13_reference.evaluate(mil, Path("."), {"x": fp16([1, 2])})
    assert values(output["y"]) == [5, 5.5]


def test_chunked_accumulation_difference_bound():
    lhs = [1.0] * 1024
    rhs = [0.001] * 512 + [0.0005] * 512
    single = h13_reference.dot_fp32(lhs, rhs)
    chunked = h13_reference.add_fp16(
        h13_reference.dot_fp32(lhs[:512], rhs[:512]),
        h13_reference.dot_fp32(lhs[512:], rhs[512:]))
    assert single != chunked
    assert h13_run_linux.chunked_close(single, chunked)


def mlp_source():
    return """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 256]> x) {
    tensor<fp16, [512, 256]> w1 = const()[name = string("w1"), val = tensor<fp16, [512, 256]>(BLOBFILE(path = string("@model_path/mlp-w1.bin"), offset = uint64(64)))];
    tensor<fp16, [512]> b1 = const()[name = string("b1"), val = tensor<fp16, [512]>(BLOBFILE(path = string("@model_path/mlp-b1.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 512]> h0 = linear(x = x, weight = w1, bias = b1)[name = string("layer1")];
    tensor<fp16, [1, 512]> h = relu(x = h0)[name = string("activation")];
    tensor<fp16, [64, 512]> w2 = const()[name = string("w2"), val = tensor<fp16, [64, 512]>(BLOBFILE(path = string("@model_path/mlp-w2.bin"), offset = uint64(64)))];
    tensor<fp16, [64]> b2 = const()[name = string("b2"), val = tensor<fp16, [64]>(BLOBFILE(path = string("@model_path/mlp-b2.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 64]> y = linear(x = h, weight = w2, bias = b2)[name = string("layer2")];
  } -> (y);
}
"""

def test_intermediate_physical_buffer_composition():
    tensors = {"h": {"shape": [512], "logicalBytes": 1024, "role": "intermediate"}}
    binding = {
        "name": "h", "dtype": "float16", "shape": [512],
        "logicalBytes": 1024, "allocationBytes": 32768, "index": 5,
        "slice": {"tensor": "h", "elementOffset": 0, "elementCount": 512},
    }
    regions = []
    for chunk in range(8):
        data = bytearray(16384)
        for element in range(64):
            struct.pack_into("<H", data, element * 64, chunk * 64 + element)
        regions.append((chunk * 64, 64, bytes(data)))
    result = h13_run_linux._intermediate_buffer(binding, tensors, regions)
    assert len(result) == 32768
    assert [struct.unpack_from("<H", result, index * 64)[0] for index in range(512)] == list(range(512))


def test_linux_runner_dry_run():
    with tempfile.TemporaryDirectory(prefix="h13-linux-runner-") as directory:
        root = Path(directory)
        mil = root / "h13-mlp.mil"
        package = root / "package"
        mil.write_text(mlp_source())
        for name, count in (("mlp-w1.bin", 512 * 256),
                            ("mlp-b1.bin", 512),
                            ("mlp-w2.bin", 64 * 512),
                            ("mlp-b2.bin", 64)):
            root.joinpath(name).write_bytes(blob(bytes(count * 2)))
        compiled = subprocess.run([
            str(COMPILER), "--mil", str(mil), "--model-root", str(root),
            "--target", "H13", "--output", str(package),
        ], capture_output=True, text=True, timeout=30, check=False)
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        input_path = root / "input.fp16"
        output_path = root / "output.fp16"
        input_path.write_bytes(bytes(256 * 2))
        command = [
            sys.executable, str(ROOT / "tools/h13_run_linux.py"), str(package),
            "--mil", str(mil), "--model-root", str(root),
            "--input", f"x={input_path}", "--output", f"y={output_path}",
            "--dry-run",
        ]
        run = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        assert run.returncode == 0, run.stdout + run.stderr
        plan = json.loads(run.stdout)
        manifest = json.loads(package.joinpath("manifest.json").read_text())
        assert plan["schema"] == "mil-hwxc.h13-linux-plan.v1"
        assert plan["deviceCalls"] is False
        assert [program["index"] for program in plan["programs"]] == manifest["dispatchPlan"]
        assert plan["referenceOutputs"] == {"y": 128}
        assert not output_path.exists()
        manifest["dispatchPlan"] = [0] * len(manifest["programs"])
        package.joinpath("manifest.json").write_text(json.dumps(manifest))
        rejected = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
        assert rejected.returncode == 1
        assert "dispatchPlan must contain every program index exactly once" in rejected.stderr
        assert not output_path.exists()


def main():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS {len(tests)} H13 reference tests")


if __name__ == "__main__":
    main()
