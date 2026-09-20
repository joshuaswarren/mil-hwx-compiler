#!/usr/bin/env python3
"""The exact host boundary casts: fp16 -> fp32 and bool -> int32 returns.

The real encoder returns encoder_hidden (fp32, cast from fp16) and
encoder_mask (int32, cast from bool). Both widen losslessly — fp16 is a
subsumed IEEE format and bool 0/1 maps to int32 0/1 — so the compiler binds
the returned logical output to the source storage, the manifest declares
hostConvert on the output record, and the host performs the widening after
readback. No ANE conversion program is emitted and no other direction is
accepted. Label: host boundary conversion, not an ANE-executed conversion.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve())
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "tools"))
import h13_run_linux  # noqa: E402
import h13_reference  # noqa: E402

ARANGE = ", ".join(str(i) for i in range(375))

BOUNDARY = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x,
                   tensor<fp16, [1, 1]> lengths) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1, 375, 1024]> hidden_fp16 = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))[name = string("hidden_fp16")];
    tensor<string, []> dtype_fp32 = const()[name = string("dtype_fp32"), val = tensor<string, []>("fp32")];
    tensor<fp32, [1, 375, 1024]> encoder_hidden = cast(dtype = dtype_fp32, x = hidden_fp16)[name = string("encoder_hidden")];
    tensor<fp16, [375]> arange = const()[name = string("arange"), val = tensor<fp16, [375]>([%s])];
    tensor<bool, [375]> mask = less(x = arange, y = lengths)[name = string("mask")];
    tensor<string, []> dtype_i32 = const()[name = string("dtype_i32"), val = tensor<string, []>("int32")];
    tensor<int32, [375]> encoder_mask = cast(dtype = dtype_i32, x = mask)[name = string("encoder_mask")];
  } -> (encoder_hidden, encoder_mask);
}
""" % ARANGE

# The identical graph without the two boundary casts: the fp16/bool sources
# the conversions widen, and what the reference compares against.
SOURCES = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x,
                   tensor<fp16, [1, 1]> lengths) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1, 375, 1024]> hidden_fp16 = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))[name = string("hidden_fp16")];
    tensor<fp16, [375]> arange = const()[name = string("arange"), val = tensor<fp16, [375]>([%s])];
  } -> (hidden_fp16);
}
""" % ARANGE


def input_x():
    return b"".join(struct.pack("<e", (i % 2048) / 1024.0 - 1.0)
                    for i in range(375 * 1024))


def lengths_input():
    return struct.pack("<e", 37.5)


def run_compile(root, name, text, expected_code=None):
    model_root = root / f"{name}-root"
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / f"{name}.mil").write_text(text)
    out = model_root / name
    run = subprocess.run(
        [compiler, "--mil", str(model_root / f"{name}.mil"),
         "--model-root", str(model_root), "--target", "H13",
         "--format", "anec", "--output", str(out)],
        input=None, capture_output=True, text=True, check=False, timeout=120)
    if expected_code is not None:
        assert run.returncode == 65, f"{name}: {run.stdout}{run.stderr}"
        assert expected_code in run.stderr, run.stderr
        return None
    assert run.returncode == 0, f"{name}: {run.stdout}{run.stderr}"
    return out, json.loads((out / "manifest.json").read_text())


with tempfile.TemporaryDirectory(prefix="h13-boundary-cast-") as directory:
    root = Path(directory)
    inputs = {"x": input_x(), "lengths": lengths_input()}

    # The source surfaces the two conversions widen, computed by the CPU
    # reference: hidden_fp16 (fp16 bytes) and mask (bool 0/1 bytes).
    source_outputs = h13_reference.evaluate(
        SOURCES, str(root),
        {"x": inputs["x"], "lengths": inputs["lengths"]})

    out, manifest = run_compile(root, "boundary", BOUNDARY)
    operations = [program["operation"] for program in manifest["programs"]]
    assert "cast" not in operations, \
        "a boundary cast must never become an ANE program"
    tensors = manifest["tensors"]
    assert tensors["encoder_hidden"]["hostConvert"] == "fp32"
    assert tensors["encoder_hidden"]["aliasOf"] == "hidden_fp16"
    assert tensors["encoder_hidden"]["logicalBytes"] == 375 * 1024 * 4
    assert tensors["encoder_mask"]["hostConvert"] == "int32"
    assert tensors["encoder_mask"]["aliasOf"] == "mask"

    # The runner's widening of the read-back surfaces must equal the
    # reference's converted outputs byte for byte.
    assert h13_run_linux._host_boundary_convert(
        "fp32", source_outputs["hidden_fp16"]) == \
        h13_reference.evaluate(BOUNDARY, str(root), inputs)["encoder_hidden"]
    # The mask bool surface is recomputed from the compare semantics (the
    # bool result never surfaces as bytes from the fp16-only reference run):
    lengths = struct.unpack("<e", inputs["lengths"])[0]
    bool_surface = bytes(1 if i < lengths else 0
                         for i in range(375))
    assert h13_run_linux._host_boundary_convert(
        "int32", bool_surface) == \
        h13_reference.evaluate(BOUNDARY, str(root), inputs)["encoder_mask"]

    # The int32 semantics, computed independently of the reference: the mask
    # is arange[i] < lengths[0] with lengths = 37.5 floored to 37.
    mask_i32 = struct.unpack("<375i", h13_run_linux._host_boundary_convert(
        "int32", bool_surface))
    assert mask_i32 == tuple(1 if i < lengths else 0 for i in range(375))

    # The fp32 widening must be the identity on values: widen the fp16
    # source bytes and compare floats, not bytes, so endianness bugs surface.
    widened = struct.unpack("<375002f", h13_run_linux._host_boundary_convert(
        "fp32", source_outputs["hidden_fp16"])) if False else \
        struct.unpack(f"<{375 * 1024}f", h13_run_linux._host_boundary_convert(
            "fp32", source_outputs["hidden_fp16"]))
    source_fp16 = struct.unpack(f"<{375 * 1024}e",
                                source_outputs["hidden_fp16"])
    assert widened == tuple(float(v) for v in source_fp16)

    # Fail-closed: a non-widening direction (fp16 -> fp16 through the same
    # cast spelling) is not a boundary conversion and stays refused.
    same_dtype = BOUNDARY.replace('"fp32"', '"fp16"').replace(
        "tensor<fp32, [1, 375, 1024]> encoder_hidden",
        "tensor<fp16, [1, 375, 1024]> encoder_hidden")
    narrow_mil = root / "narrow.mil"
    narrow_mil.write_text(same_dtype)
    refusal = subprocess.run(
        [compiler, "--mil", str(narrow_mil), "--model-root", str(root),
         "--target", "H13", "--format", "anec", "--output",
         str(root / "narrow")],
        capture_output=True, text=True, check=False)
    assert refusal.returncode == 65, refusal.stdout + refusal.stderr

print("h13 boundary cast cli: PASS")

# Pre-declared comparison classes: LN/reduce statistics outputs carry the
# fp16-envelope class through every downstream elementwise op; elementwise
# chains outside a statistics stage stay exact. (Main-directed: the LN
# numeric envelope is declared, never retrofitted onto a failed exact run.)
ones = ", ".join(["fp16(0x1p+0)"] * 1024)
envelope_source = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1, 1, 1024]> gamma = const()[name = string("gamma"), val = tensor<fp16, [1, 1, 1024]>([%s])];
    tensor<fp16, [1, 1, 1024]> beta = const()[name = string("beta"), val = tensor<fp16, [1, 1, 1024]>([%s])];
    tensor<fp16, [1, 375, 1024]> normalized = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))[name = string("normalized")];
    tensor<fp16, [1, 375, 1024]> scaled = mul(x = normalized, y = gamma)[name = string("scaled")];
    tensor<fp16, [1, 375, 1024]> encoder_hidden = add(x = scaled, y = beta)[name = string("encoder_hidden")];
  } -> (encoder_hidden);
}
""" % (ones, ones)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    out, manifest = run_compile(root, "envelope-classes", envelope_source)
    tensors = manifest["tensors"]
    assert tensors["normalized"].get("comparison") == "fp16-envelope", tensors["normalized"]
    assert tensors["scaled"].get("comparison") == "fp16-envelope", tensors["scaled"]
    assert tensors["encoder_hidden"].get("comparison") == "fp16-envelope"
print("h13 envelope comparison classes: PASS")
