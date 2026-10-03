#!/usr/bin/env python3
"""Compares emitted H17 or H18 HWX word-for-word with decoded Apple oracles.

Usage: test_h17_parity.py COMPILER H17|H18

Every decoded elementwise, unary and scalar-constant oracle of the target is
compiled from its own MIL and parsed with the parser that recorded the oracle.
Every task word, every nonzero program-descriptor word, the tensor
descriptors and the constant-section hashes must equal the oracle. Shapes,
constants and families off the decoded points must be refused by name.
H18 shares H17's task framing and descriptor layout, so one test serves both.
Byte parity says nothing about execution on an H17 or H18 device.
"""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import inspect_hwx  # noqa: E402
import mint_oracles  # noqa: E402
from generate_h14_templates import selected_oracles  # noqa: E402
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1]).resolve()
TARGET = sys.argv[2]
PREFIX = TARGET.lower()
ISA = {"H17": 19, "H18": 20}[TARGET]
RECORDED = ("program_descriptor", "program_descriptor_words",
            "tensor_descriptors", "constant_section", "kernel_section",
            "task_descriptors")


def compile_mil(mil, root, name, extra=()):
    path = root / f"{name}.mil"
    path.write_text(mil)
    output = root / name
    result = subprocess.run(
        [str(COMPILER), "--mil", str(path), "--model-root", str(root),
         "--target", TARGET, "--output", str(output), *extra],
        capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
    return result, output


def check_oracle(oracle, root):
    assert oracle["weights"]["storage"] != "BLOBFILE", oracle["case"]
    result, output = compile_mil(oracle["mil"], root, oracle["case"])
    assert result.returncode == 0, \
        f"{oracle['case']}: {result.stdout}{result.stderr}"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["target"] == TARGET and manifest["artifactFormat"] == "hwx"
    assert len(manifest["programs"]) == 1, oracle["case"]
    record = manifest["programs"][0]
    assert record["encoder"] == f"{PREFIX}-oracle-parity", oracle["case"]
    path = output / record["file"]
    parsed = mint_oracles.parse_hwx(path.read_bytes(), PREFIX)
    assert parsed["program_count"] == 1, oracle["case"]
    for key in RECORDED:
        assert parsed[key] == oracle[key], f"{oracle['case']}: {key} differs"
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        inspect_hwx.main(str(path))
    text = report.getvalue()
    assert f"name={TARGET} isa={ISA}" in text, oracle["case"]
    assert f"{PREFIX}_tasks count={len(oracle['task_descriptors'])}" in text, \
        oracle["case"]


def unary(operation, channels):
    kind = f"tensor<fp16, [1, {channels}, 1, 1]>"
    return mint_oracles.program(
        f"{kind} x", [f'{kind} y = {operation}(x = x)[name = string("y")];'], "y")


def scalar(operation, channels, value):
    kind = f"tensor<fp16, [1, {channels}, 1, 1]>"
    return mint_oracles.program(f"{kind} x", [
        f'fp16 z = const()[name = string("z"), val = fp16({value})];',
        f'{kind} y = {operation}(x = x, y = z)[name = string("y")];'], "y")


def check_refusals(root):
    """Inputs one step off a decoded point are refused, never interpolated."""
    outside = f"[{PREFIX}.outside-parity-envelope]"
    decoded = {oracle["case"] for oracle in selected_oracles(PREFIX)}
    # The decoded points these neighbours sit next to must exist.
    assert {"binary_add_1x64x1x1", "unary_sigmoid_c64",
            "binary_mul_c64_constant_scalar"} <= decoded
    cases = {
        "binary_add_1x65x1x1": mint_oracles.binary_runtime(
            "add", (1, 65, 1, 1))["mil"],
        "unary_sigmoid_c128": unary("sigmoid", 128),
        "binary_mul_c64_scalar_quarter": scalar("mul", 64, "0x1p-2"),
        "binary_pow_1x64x1x1": mint_oracles.binary_runtime(
            "pow", (1, 64, 1, 1))["mil"],
        "softmax_1x512x1x1": mint_oracles.normalization(
            "softmax", (1, 512, 1, 1))["mil"],
        "conv_k1_c256_o512": mint_oracles.convolution(
            1, 256, 512, 1, False)["mil"],
    }
    for name, mil in cases.items():
        result, output = compile_mil(mil, root, f"refuse-{name}")
        assert result.returncode != 0, f"{name} compiled outside the envelope"
        assert outside in result.stderr, f"{name}: {result.stderr}"
        assert not output.exists(), name
    result, _ = compile_mil(selected_oracles(PREFIX)[0]["mil"], root,
                            "refuse-anec", ("--format", "anec"))
    assert result.returncode != 0 and \
        f"[{PREFIX}.unsupported-format]" in result.stderr, result.stderr
    return len(cases) + 1


def main():
    oracles = selected_oracles(PREFIX)
    assert len(oracles) == 165, \
        f"expected 165 decoded {TARGET} elementwise oracles, found {len(oracles)}"
    families = {oracle["family"] for oracle in oracles}
    with tempfile.TemporaryDirectory(prefix=f"{PREFIX}-parity-") as directory:
        root = Path(directory)
        for oracle in oracles:
            check_oracle(oracle, root)
        refused = check_refusals(root)
    print(f"{TARGET} oracle parity: PASS ({len(oracles)} cases over "
          f"{', '.join(sorted(families))}; {refused} off-point inputs refused)")


if __name__ == "__main__":
    main()
