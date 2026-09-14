#!/usr/bin/env python3
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def compile_source(root, name, text, expected_code=None, expected_message=None):
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
    if expected_message:
        assert expected_message in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
    return None


def runtime(shape):
    t = f"tensor<fp16, {shape}>"
    m = f"tensor<bool, {shape}>"
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({t} a, {t} b, {m} cond) {{
    {t} y = select(a = a, b = b, cond = cond)[name = string("y")];
  }} -> (y);
}}
"""


def const_a(shape, a_type):
    t = f"tensor<fp16, {shape}>"
    m = f"tensor<bool, {shape}>"
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({t} b, {m} cond) {{
    {a_type} a = const()[name = string("a"), val = {a_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {t} y = select(a = a, b = b, cond = cond)[name = string("y")];
  }} -> (y);
}}
"""


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-select-envelope-") as directory:
    root = Path(directory)
    payload = struct.pack("<e", float("-inf"))
    header = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(payload), 64 + 24)
    (root / "weights.bin").write_bytes(b"\0" * 64 + header + payload)
    for name, shape in (("enc-rrb", "[1, 8, 375, 375]"),
                        ("vec-rrb", "[1, 64, 1, 1]")):
        hit = compile_source(root, name, runtime(shape))
        assert len(hit["programs"]) == 1, hit["programs"]
        program = hit["programs"][0]
        assert program["operation"] == "select"
        assert program["encoder"] == "apple-parity-boolean"
        assert program["taskDescriptors"] == 5
    compile_source(root, "enc-ninf-scalar",
                   const_a("[1, 8, 375, 375]", "tensor<fp16, []>"),
                   expected_code="h13.select-needs-decoded-encoder",
                   expected_message="materializes the fill as a runtime constant input")
    compile_source(root, "enc-ninf-tensor",
                   const_a("[1, 8, 375, 375]", "tensor<fp16, [1, 8, 375, 375]>"),
                   expected_code="h13.select-needs-decoded-encoder",
                   expected_message="materializes the fill as a runtime constant input")
    compile_source(root, "glu-rrb", runtime("[1, 1024, 375]"),
                   expected_code="h13.boolean-outside-envelope")
print("h13 select envelope cli: PASS")
