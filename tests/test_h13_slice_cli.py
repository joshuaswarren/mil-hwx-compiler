#!/usr/bin/env python3
"""The decoded encoder slice_by_index last-dim window lowers as one 1-task
program; every other slice form still refuses by name."""
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


def slice_source(input_shape, output_shape, begin, end,
                 const_extra="", call_extra=""):
    rank = len(eval(input_shape))
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, {input_shape}> x) {{
    tensor<int32, [{rank}]> eb = const()[name = string("eb"), val = tensor<int32, [{rank}]>({begin})];
    tensor<int32, [{rank}]> ee = const()[name = string("ee"), val = tensor<int32, [{rank}]>({end})];{const_extra}
    tensor<fp16, {output_shape}> y = slice_by_index(x = x, begin = eb, end = ee{call_extra})[name = string("y")];
  }} -> (y);
}}
"""


STRIDE_CONST = ('\n    tensor<int32, [4]> st = const()[name = string("st"), '
                'val = tensor<int32, [4]>([1, 1, 1, 1])];')

ENDMASK_CONST = ('\n    tensor<bool, [4]> em = const()[name = string("em"), '
                 'val = tensor<bool, [4]>([true, true, true, false])];')


def main():
    with tempfile.TemporaryDirectory(prefix="h13-slice-") as directory:
        root = Path(directory)
        # The decoded rel-pos windowing spell, both spellings.
        compile_source(root, "lastdim_window", slice_source(
            "[1, 8, 375, 749]", "[1, 8, 375, 375]",
            "[0, 0, 0, 0]", "[1, 8, 375, 375]"))
        compile_source(root, "lastdim_window_endmask", slice_source(
            "[1, 8, 375, 749]", "[1, 8, 375, 375]", "[0, 0, 0, 0]",
            "[0, 0, 0, 375]", const_extra=ENDMASK_CONST,
            call_extra=", end_mask = em"))
        refusals = [
            # The same shape pair opened mid-row has no decoded task stream.
            ("window_offset", "[1, 8, 375, 749]", "[1, 8, 375, 375]",
             "[0, 0, 0, 374]", "[1, 8, 375, 749]", "", "",
             "h13.noncontiguous-slice"),
            # An uncovered input surface pair stays refused.
            ("uncovered_shape", "[1, 4, 375, 749]", "[1, 4, 375, 375]",
             "[0, 0, 0, 0]", "[1, 4, 375, 375]", "", "",
             "h13.noncontiguous-slice"),
            # The encoder's other slice family narrows a non-last axis
            # (mask slicing [1,8,750,375] -> [1,8,749,375]); no oracle.
            ("mid_dim_narrow", "[1, 8, 750, 375]", "[1, 8, 749, 375]",
             "[0, 0, 0, 0]", "[1, 8, 749, 375]", "", "",
             "h13.noncontiguous-slice"),
            # A strided window is not one contiguous binding slice.
            ("strided_window", "[1, 8, 375, 749]", "[1, 8, 375, 375]",
             "[0, 0, 0, 0]", "[1, 8, 375, 375]", STRIDE_CONST,
             ", stride = st", "h13.noncontiguous-slice"),
        ]
        for name, inp, out, begin, end, const_extra, call_extra, code \
                in refusals:
            compile_source(root, name, slice_source(
                inp, out, begin, end, const_extra, call_extra), code)
    print("H13 slice CLI: PASS (2 covered, %d refusals)" % len(refusals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
