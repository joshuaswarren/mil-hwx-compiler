#!/usr/bin/env python3
"""The decoded encoder transposes lower as one 1-task program; every other
transpose form still refuses by name."""
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


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
        return run
    assert run.returncode == 65, run.stdout + run.stderr
    assert expected_code in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
    return None


def transpose(name, input_shape, output_shape, perm):
    rank = len(eval(input_shape))
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, {input_shape}> x) {{
    tensor<int32, [{rank}]> perm = const()[name = string("perm"), val = tensor<int32, [{rank}]>({perm})];
    tensor<fp16, {output_shape}> y = transpose(perm = perm, x = x)[name = string("y")];
  }} -> (y);
}}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="h13-transpose-") as directory:
        root = Path(directory)
        covered = [
            ("r3_1x375x1024", "[1, 375, 1024]", "[1, 1024, 375]",
             "[0, 2, 1]"),
            ("r3_1x1024x375", "[1, 1024, 375]", "[1, 375, 1024]",
             "[0, 2, 1]"),
        ]
        for name, inp, out, perm in covered:
            compile_source(root, name, transpose(name, inp, out, perm))
        r4 = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 128]> x) {
    tensor<int32, [4]> perm = const()[name = string("perm"), val = tensor<int32, [4]>([0, 2, 1, 3])];
    tensor<fp16, [1, 375, 8, 128]> y = transpose(perm = perm, x = x)[name = string("y")];
  } -> (y);
}
"""
        compile_source(root, "r4_1x8x375x128", r4)
        r4p = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 256, 16]> x) {
    tensor<int32, [4]> perm = const()[name = string("perm"), val = tensor<int32, [4]>([0, 2, 1, 3])];
    tensor<fp16, [1, 256, 375, 16]> y = transpose(perm = perm, x = x)[name = string("y")];
  } -> (y);
}
"""
        compile_source(root, "r4_1x375x256x16", r4p)
        # An identity permutation over a returned function input stays a
        # layout-preserving alias, which the function boundary refuses.
        compile_source(root, "identity_perm", transpose(
            "identity_perm", "[1, 375, 1024]", "[1, 375, 1024]", "[0, 1, 2]"),
            "h13.returned-input-alias")
        refusals = [
            ("uncovered_pair", "[1, 256, 375, 16]", "[1, 375, 256, 16]",
             "[0, 2, 1, 3]", "h13.nonfoldable-transpose"),
            ("uncovered_shape", "[1, 375, 512]", "[1, 512, 375]", "[0, 2, 1]",
             "h13.nonfoldable-transpose"),
        ]
        for name, inp, out, perm, code in refusals:
            compile_source(root, name, transpose(name, inp, out, perm), code)
    print("H13 transpose CLI: PASS (4 covered, %d refusals)" % (len(refusals) + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
