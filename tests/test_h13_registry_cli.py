#!/usr/bin/env python3
"""Registry ops of the boolean/mask path: less, floor, select, floor_div.

Ground truth is the stream-3 decision doc (mlx-omarchy
docs/2026-09-13-h13-boolean-mask-fp16.md): no finite +,-,*,/ expression
produces a comparison, floor has no arithmetic identity, and the attention
bias selects fill -inf where the blend identity NaNs — so all four lower
only through decoded encoders the corpus does not yet hold. The decoded
template tables carry no comparison, floor, or lane-pick op (surveyed), so
these contracts reject with the exact reason and the mint dependency
instead of inventing task words. The divide half of floor_div is real
today: a constant power-of-two divisor lowers through the decoded
reciprocal multiply, pinned here with the doc's own divisor of 2.0.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")


def header(shape):
    return f"tensor<fp16, [{', '.join(map(str, shape))}]>"


def source(body, inputs, result):
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({inputs}) {{
{body}  }} -> ({result});
}}
"""


def const(name, literal):
    return f"    {literal.split('(')[0]} {name} = const()" \
           f"[name = string(\"{name}\"), val = {literal}];\n"


def tensor_const(name, dtype, values):
    return const(name, f"tensor<{dtype}, [{len(values)}]>([{', '.join(map(str, values))}])")


def fp16_const(name, shape, values):
    body = ", ".join(f"fp16({v})" for v in values)
    dims = ", ".join(map(str, shape))
    return const(name, f"tensor<fp16, [{dims}]>([{body}])")


def compile_source(root, name, text, expected_code=None, expected_message=None,
                   format="anec"):
    path = root / f"{name}.mil"
    path.write_text(text)
    output = root / name
    result = subprocess.run(
        [compiler, "--mil", str(path), "--model-root", str(root),
         "--output", str(output), "--target", "H13", "--format", format],
        capture_output=True, text=True, timeout=120, check=False)
    if expected_code is None:
        assert result.returncode == 0, result.stderr
        return output
    assert result.returncode == 65, result.stderr
    assert expected_code in result.stderr, result.stderr
    if expected_message:
        assert expected_message in result.stderr, result.stderr
    return None


def validate(root, package):
    result = subprocess.run([sys.executable, inspector, str(package)],
                            capture_output=True, text=True, timeout=60,
                            check=False)
    assert result.returncode == 0, result.stderr


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)

    body = const("dt", "string(\"fp16\")")
    body += fp16_const("arange", [1, 375], [i * 0.03125 for i in range(375)])
    body += ("    tensor<fp16, [1, 375]> m = less(x = arange, y = len)"
             "[name = string(\"m\")];\n")
    body += ("    tensor<fp16, [1, 375]> y = cast(dtype = dt, x = m)"
             "[name = string(\"y\")];\n")
    compile_source(root, "registry-less", source(
        body, header([1]) + " len", "y"),
        expected_code="h13.less-needs-decoded-encoder",
        expected_message="no finite +,-,*,/ expression over fp16 produces (x < y) as 0/1 exactly")

    # floor — the encoder's [1]-shaped length scalars.
    body = ("    tensor<fp16, [1]> y = floor(x = a)"
            "[name = string(\"y\")];\n")
    compile_source(root, "registry-floor", source(
        body, header([1]) + " a", "y"),
        expected_code="h13.floor-needs-decoded-encoder",
        expected_message="total and exact over fp16")

    # floor_div — the encoder's length halving; blocked on the floor encoder
    # while the divide itself is already exact.
    body = fp16_const("two", [1], [2.0])
    body += ("    tensor<fp16, [1]> y = floor_div(x = a, y = two)"
             "[name = string(\"y\")];\n")
    compile_source(root, "registry-floor-div", source(
        body, header([1]) + " a", "y"),
        expected_code="h13.floor-div-needs-decoded-encoder",
        expected_message="3199/32 divides-and-rounds to exactly 100.0 in fp16")

    # select — the encoder's attention-bias family: -inf fill against an
    # fp16 0/1 mask (the decision doc's boundary conversion means no bool
    # dtype reaches the device).
    payload = struct.pack("<e", float("-inf"))
    header_bytes = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(payload), 64 + 24)
    (root / "weights.bin").write_bytes(b"\0" * 64 + header_bytes + payload)
    body = const("neg_inf", "tensor<fp16, []>"
                             "(BLOBFILE(path = string(\"@model_path/weights.bin\"),"
                             " offset = uint64(64)))")
    body += ("    tensor<fp16, [1, 8, 375, 375]> y = select("
             "a = neg_inf, b = b, cond = m)[name = string(\"y\")];\n")
    compile_source(root, "registry-select-neg-inf", source(
        body, header([1, 8, 375, 375]) + " b, " + header([1, 8, 375, 375]) + " m",
        "y"),
        expected_code="h13.select-needs-decoded-encoder",
        expected_message="0 * -inf = NaN destroys every unmasked lane")

    # The +0.0-fill family rejects the same way here: it belongs to the
    # frontend mul rewrite, not to a compiler encoder.
    body = const("zero", "fp16(0.0)")
    body += ("    tensor<fp16, [1, 1024, 375]> y = select("
             "a = zero, b = b, cond = m)[name = string(\"y\")];\n")
    compile_source(root, "registry-select-zero-fill", source(
        body, header([1, 1024, 375]) + " b, " + header([1, 1024, 375]) + " m",
        "y"),
        expected_code="h13.select-needs-decoded-encoder",
        expected_message="the +0.0-fill family belongs to the frontend mul rewrite")

    # Positive control: the divide half of floor_div is real today — a
    # constant power-of-two divisor lowers through the decoded reciprocal
    # multiply, byte-identical to the equivalent mul.
    def constant_source(op, values):
        elements = ", ".join(f"fp16({v})" for v in values)
        body = const("c", f"tensor<fp16, [64]>([{elements}])")
        body += (f"    tensor<fp16, [64]> y = {op}(x = a, y = c)"
                 f"[name = string(\"y\")];\n")
        return source(body, "tensor<fp16, [64]> a", "y")

    divide = compile_source(root, "floor-div-divide-half",
                            constant_source("real_div", [2.0] * 64))
    multiply = compile_source(root, "floor-div-divide-reference",
                              constant_source("mul", [0.5] * 64))
    assert (divide / "program-0.anec").read_bytes() == \
        (multiply / "program-0.anec").read_bytes()
    divide_manifest = json.loads((divide / "manifest.json").read_text())
    assert divide_manifest["operation"] == "mul"
    assert bytes.fromhex(
        divide_manifest["constantInputs"]["c"]) == struct.pack("<e", 0.5) * 64
    validate(root, divide)

    print("h13 registry cli: PASS")
