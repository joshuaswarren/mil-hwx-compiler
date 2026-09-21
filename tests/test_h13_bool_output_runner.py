#!/usr/bin/env python3
"""Bool output handling in the H13 reference runner.

The runner executed every output as fp16: `_unpack_outputs` copied
2-byte elements, `convert_tensor`/`dense_slice` assumed 2-byte lanes, and
`_check_outputs` decoded every comparison through `compare_fp16`. The two
decoded boolean families with bool results (cast fp16->bool, the bool
tail-swap transpose) therefore fail the runner before any device byte is
compared. These tests pin the minimal dtype-aware, exact-byte handling:

  - reference `cast` gains the fp16 -> bool direction (nonzero -> 1,
    zero -> 0, exact; no tolerance anywhere),
  - reference `transpose` evaluates the tail-swap permutation for bool
    and fp16 tensors,
  - `inspect_anec.convert_tensor`/`dense_slice` size elements from the
    binding/tensor dtype (1 byte bool, 2 bytes fp16),
  - `_unpack_outputs` assembles bool outputs per element,
  - `_check_outputs` compares bool outputs as exact bytes and never
    routes them through the fp16 decoder.

Label: software runner parity, not a device result.
"""
import json
import importlib.util
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve())
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import h13_run_linux  # noqa: E402
import h13_reference  # noqa: E402
from research import inspect_anec  # noqa: E402


CAST_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 1, 8]> x) {
    tensor<string, []> dt = const()[name = string("dt"), val = tensor<string, []>("bool")];
    tensor<bool, [1, 1, 8]> out = cast(dtype = dt, x = x)[name = string("out")];
  } -> (out);
}
"""

TRANSPOSE_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 3, 4]> x) {
    tensor<int32, [3]> perm = const()[name = string("perm"), val = tensor<int32, [3]>([0, 2, 1])];
    tensor<bool, [1, 4, 3]> out = transpose(perm = perm, x = x)[name = string("out")];
  } -> (out);
}
"""


def run_compile(root, name, text):
    model_root = root / f"{name}-root"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / f"{name}.mil").write_text(text)
    out = model_root / name
    run = subprocess.run(
        [compiler, "--mil", str(model_root / f"{name}.mil"),
         "--model-root", str(model_root), "--target", "H13",
         "--format", "anec", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=120)
    assert run.returncode == 0, f"{name}: {run.stdout}{run.stderr}"
    return out, json.loads((out / "manifest.json").read_text())


# 1. Reference cast fp16 -> bool: the exact zero/nonzero mapping.
cast_input = b"".join(struct.pack("<e", value) for value in
                      (0.0, -0.0, 1.0, -2.5, float("inf"), float("-inf"),
                       float("nan"), 6.0e-8))
cast_out = h13_reference.evaluate(CAST_MIL, ".", {"x": cast_input})["out"]
assert cast_out == bytes([0, 0, 1, 1, 1, 1, 1, 1]), cast_out
print("PASS reference cast fp16->bool exact zero/nonzero")

# 2. Reference transpose: bool tail-swap permutes elements, keeps dtype.
transpose_input = bytes((i * 7 + 1) % 3 == 0 for i in range(12))
transpose_out = h13_reference.evaluate(
    TRANSPOSE_MIL, ".", {"x": transpose_input})["out"]
rows = [transpose_input[0:4], transpose_input[4:8], transpose_input[8:12]]
expected = bytes(rows[j][i] for i in range(4) for j in range(3))
assert transpose_out == expected, (transpose_out, expected)
print("PASS reference transpose bool tail-swap")

# 3. convert_tensor round-trip on a bool binding (1-byte lanes, 64-byte
# row pitch): pack dense bool into the physical surface and read it back.
pack_binding = {
    "name": "lanes", "dtype": "bool", "index": 4,
    "logicalBytes": 8, "allocationBytes": 512,
    "shape": [8],
    "nchw": [1, 1, 1, 8, 64, 64],
}
dense = bytes([1, 0, 1, 1, 0, 0, 1, 0])
packed = inspect_anec.convert_tensor(pack_binding, dense, True)
assert len(packed) == pack_binding["allocationBytes"]
assert packed[0:8] == dense
assert inspect_anec.convert_tensor(pack_binding, packed, False) == dense
print("PASS convert_tensor bool 1-byte lanes")

# 4. dense_slice on a bool tensor slices per element, not per 2 bytes.
tensors = {"mask": {"dtype": "bool", "logicalBytes": 6,
                    "role": "input", "shape": [6]}}
sliced = inspect_anec.dense_slice(bytes([1, 0, 0, 1, 1, 0]),
                                  {"name": "mask", "logicalBytes": 6,
                                   "shape": [6]}, tensors)
assert sliced == bytes([1, 0, 0, 1, 1, 0])
print("PASS dense_slice bool element count")

# 5. _unpack_outputs assembles a whole bool output per element.
manifest = {
    "tensors": {"out": {"dtype": "bool", "logicalBytes": 4,
                        "role": "output", "shape": [4]}},
    "programs": [], "dispatchPlan": [],
}
out_binding = {
    "name": "out", "dtype": "bool", "index": 4,
    "logicalBytes": 4, "allocationBytes": 256,
    "shape": [4],
    "nchw": [1, 1, 1, 4, 64, 64],
}
regions = {"out": [(out_binding, bytes([1, 0, 0, 1]) + bytes(252))]}
actual = h13_run_linux._unpack_outputs(manifest, regions, ["out"])
assert actual == {"out": bytes([1, 0, 0, 1])}, actual
print("PASS _unpack_outputs bool elements")

# 6. _check_outputs compares bool outputs as exact bytes; a one-bit
# difference raises, and the fp16 decoder never sees the bytes. The odd
# 375-element output length (the decoded cast geometry) is exercised, not
# just the even toy length above.
odd_manifest = {
    "tensors": {"out": {"dtype": "bool", "logicalBytes": 375,
                        "role": "output", "shape": [1, 1, 375]}},
    "programs": [], "dispatchPlan": [],
}
h13_run_linux._check_outputs(
    odd_manifest, odd_manifest["tensors"], ["out"],
    {"out": bytes(i % 3 == 0 for i in range(375))},
    {"out": bytes(i % 3 == 0 for i in range(375))})
try:
    h13_run_linux._check_outputs(
        odd_manifest, odd_manifest["tensors"], ["out"],
        {"out": bytes(i % 3 == 0 for i in range(375))[:-1] + bytes([1])},
        {"out": bytes(i % 3 == 0 for i in range(375))})
except ValueError:
    pass
else:
    raise AssertionError("odd-length bool output mismatch accepted")
try:
    h13_run_linux._check_outputs(
        odd_manifest, odd_manifest["tensors"], ["out"],
        {"out": bytes(374)},
        {"out": bytes(375)})
except ValueError:
    pass
else:
    raise AssertionError("short bool output accepted")
print("PASS _check_outputs bool exact-byte compare")

# 6b. _unpack_outputs refuses a produced surface of the wrong byte count
# and assembles the odd 375-element output per element.
odd_binding = {
    "name": "out", "dtype": "bool", "index": 4,
    "logicalBytes": 375, "allocationBytes": 16384,
    "shape": [1, 1, 375],
    "nchw": [1, 1, 1, 375, 384, 384],
}
odd_regions = {"out": [(odd_binding, bytes(16383))]}
try:
    h13_run_linux._unpack_outputs(odd_manifest, odd_regions, ["out"])
except ValueError:
    pass
else:
    raise AssertionError("short physical bool surface accepted")
odd_surface = bytearray(16384)
for i in range(375):
    odd_surface[i] = 1 if i % 3 == 0 else 0
odd_regions = {"out": [(odd_binding, bytes(odd_surface))]}
actual = h13_run_linux._unpack_outputs(odd_manifest, odd_regions, ["out"])
assert actual == {"out": bytes(i % 3 == 0 for i in range(375))}, actual
print("PASS _unpack_outputs odd 375-element bool output")

# 7. End to end through the compiled packages at the captured geometries
# (the compiler envelope is fail-closed to the decoded rows): the runner
# dry-run plan must build (reference evaluation included) for the cast
# and transpose fixtures, with the bool outputs declared in the plan.
E2E_CAST_MIL = CAST_MIL.replace("[1, 1, 8]", "[1, 1, 375]")
E2E_TRANSPOSE_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 375, 375]> x) {
    tensor<int32, [3]> perm = const()[name = string("perm"), val = tensor<int32, [3]>([0, 2, 1])];
    tensor<bool, [1, 375, 375]> out = transpose(perm = perm, x = x)[name = string("out")];
  } -> (out);
}
"""
with tempfile.TemporaryDirectory(prefix="h13-bool-runner-") as directory:
    root = Path(directory)
    cast_pkg, cast_manifest = run_compile(root, "cast", E2E_CAST_MIL)
    cast_input = b"".join(struct.pack("<e", 0.0 if i % 4 == 0 else float(i + 1))
                          for i in range(375))
    cast_input_path = root / "x.fp16"
    cast_input_path.write_bytes(cast_input)
    _, _, plan = h13_run_linux.run_package(
        cast_pkg, cast_pkg.parent / "cast.mil", str(cast_pkg),
        {"x": cast_input_path}, {"out": None}, None)
    assert plan["referenceCriteria"]["out"] == "exact bool bytes", plan
    assert plan["referenceOutputs"]["out"] == 375
    assert cast_manifest["tensors"]["out"]["dtype"] == "bool"

    transpose_pkg, transpose_manifest = run_compile(
        root, "transpose", E2E_TRANSPOSE_MIL)
    # Asymmetric bounded pattern: coefficients 7 and 11 differ, so the
    # transpose of the mask differs from the mask (a symmetric mask
    # cannot detect a transpose that never permutes).
    transpose_input = bytes((i * 7 + j * 11) % 5 == 0
                            for j in range(375) for i in range(375))
    assert bytes(transpose_input[i * 375 + j]
                 for j in range(375) for i in range(375)) != transpose_input
    transpose_input_path = root / "x.bool"
    transpose_input_path.write_bytes(transpose_input)
    _, _, plan = h13_run_linux.run_package(
        transpose_pkg,
        transpose_pkg.parent / "transpose.mil",
        str(transpose_pkg), {"x": transpose_input_path}, {"out": None}, None)
    assert plan["referenceCriteria"]["out"] == "exact bool bytes", plan
    assert plan["referenceOutputs"]["out"] == 375 * 375
    assert transpose_manifest["tensors"]["out"]["dtype"] == "bool"
print("PASS runner dry-run plans for bool cast and bool transpose")

# 8. The D fixture's recorded mask: the generator's pattern must be
# asymmetric under tail-swap (the prior (i + j) % 4 == 0 pattern equaled
# its own transpose and could not detect a no-op transpose), and the
# reference transpose must reorder it exactly like an independent
# row/column swap.
sys.path.insert(0, str(ROOT / "tests"))
spec = importlib.util.spec_from_file_location(
    "h13_rank3_m1-test-host_fixtures",
    ROOT / "tests/h13_rank3_m1-test-host_fixtures.py")
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
fixture_mask = fixtures.mask_bytes(375, 375)
swapped = bytes(fixture_mask[i * 375 + j]
                for j in range(375) for i in range(375))
assert fixture_mask != swapped, \
    "fixture D mask must be asymmetric under tail-swap"
permutation_mil = E2E_TRANSPOSE_MIL
permuted = h13_reference.evaluate(
    permutation_mil, ".", {"x": fixture_mask})["out"]
assert permuted == swapped, \
    "reference transpose must equal the independent row/column swap"
print("PASS fixture D mask asymmetric and transpose-exact")

print("h13 bool output runner: PASS")
