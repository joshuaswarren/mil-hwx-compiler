#!/usr/bin/env python3
"""SUP-89, Parakeet lowering class 8: casts, quant/dequant, dtype promotion.

Failing-first on main, per the SUP-82 method: every class-8 op the Parakeet
encoder MIL uses must reach an ANE program. Probed on main (ef87854):

  cast (all directions) -> RED. H16G: h16g.legalize.unsupported-operation
    'cast' (ANEOperationKindUnsupported); H13: h13.unsupported-program /
    h13.unsupported-logical-result-conversion. Parakeet's mask prelude
    carries cast(fp16->int32) on the length chain and cast back to fp16;
    bool mask materialization needs cast(fp16->bool); the fp32 boundary
    needs both promotion directions.
  quantize/dequantize standalone -> RED. Folded into numeric boundaries by
    h16g.legalize (markFoldedIntoNumericBoundary), so a boundary pair as the
    package's own input/output compiles rc=0 with ZERO artifacts -- the int8
    surface never materializes. Inside a conv chain they are GREEN (below).
  constexpr_affine_dequantize -> GREEN inside the W8A8 conv boundary only;
    the weight-side forms constexpr_lut_to_dense / sparse_to_dense /
    blockwise_shift_scale have no importer/parser handling at all.

Green list (what already lowers, asserted as the passing control):
  - the W8A8 conv boundary chain: quantize(fp16->int8) and
    dequantize(int8->fp16) folded into conv numeric modes, with
    constexpr_affine_dequantize int8 weights -- tests/fixtures/
    w8a8_conv_chain.mil compiles to a real program-0.hwx on H16G
    (covered by test_cli.sh's w8a8 case; restated here as the class
    control).

Run: python3 tests/test_parakeet_class8_cast_quant_lowering.py [mil-hwxc]
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
REPO = Path(__file__).resolve().parent.parent


def compile_source(root, name, text, target="H16G", model_root=None):
    """Compile one class-8 program. Returns the parsed manifest."""
    mil = root / f"{name}.mil"
    out = root / name
    mil.write_text(text)
    cmd = [compiler, "--mil", str(mil), "--model-root",
           str(model_root or root), "--target", target, "--output", str(out)]
    run = subprocess.run(cmd, capture_output=True, text=True,
                         check=False, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    return json.loads((out / "manifest.json").read_text())


def assert_real_program(manifest):
    """The compile must emit at least one artifact holding a real program."""
    assert manifest["artifacts"], "compiled package has no artifact"
    assert manifest["artifacts"][0]["file"].endswith(".hwx")
    assert manifest["artifacts"][0]["bytes"] > 0


def cast_mil(src_type, dst_type):
    shape = "[1, 64, 64]" if dst_type != "bool" else "[1, 375, 375]"
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<{src_type}, {shape}> x) {{
    tensor<{dst_type}, {shape}> y = cast(x = x, dtype = string("{dst_type}"))[name = string("y")];
  }} -> (y);
}}
"""


# Red 1-5: cast directions. Parakeet's mask prelude is cast(fp16->int32) on
# the length chain with cast(int32->fp16) back; bool masks materialize via
# cast(fp16->bool); the fp32 boundary needs both promotion directions.
# Decoded on h16 (research/oracles/h16/boundary_cast_*): fp16->bool,
# fp16->fp32 and fp32->fp16 compile to real boundary programs; Apple rejects
# every int32-surface form, so the int32 directions assert the documented
# refusal instead of an artifact.
CASTS = [
    ("fp16", "int32"),
    ("int32", "fp16"),
    ("fp16", "bool"),
    ("fp16", "fp32"),
    ("fp32", "fp16"),
]

INT32_CASTS = {("fp16", "int32"), ("int32", "fp16")}

# Red 6: a standalone quantize package output must materialize the int8
# surface as a real boundary program; today the fold drops it and the
# package compiles to zero artifacts.
quantize_output_mil = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 64, 64, 64]> x) {
    string qd = const()[name = string("qd"), val = string("int8")];
    fp16 qs = const()[name = string("qs"), val = fp16(0x1p-3)];
    tensor<int8, [1, 64, 64, 64]> y = quantize(input = x, output_dtype = qd, scale = qs)[name = string("y")];
  } -> (y);
}
"""

# Red 7: a standalone dequantize package input must materialize the fp16
# surface the same way.
dequantize_input_mil = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<int8, [1, 64, 64, 64]> x) {
    fp16 dqs = const()[name = string("dqs"), val = fp16(0x1p-3)];
    tensor<fp16, [1, 64, 64, 64]> y = dequantize(input = x, scale = dqs)[name = string("y")];
  } -> (y);
}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="mil-hwx-parakeet-class8-") as directory:
        root = Path(directory)

        # Green: the W8A8 boundary chain lowers today (control).
        chain = (REPO / "tests" / "fixtures" / "w8a8_conv_chain.mil").read_text()
        mil = root / "w8a8_chain.mil"
        mil.write_text(chain)
        run = subprocess.run(
            [compiler, "--mil", str(mil), "--model-root",
             str(REPO / "tests" / "models" / "w8a8"), "--target", "H16G",
             "--output", str(root / "w8a8_chain")],
            capture_output=True, text=True, check=False, timeout=30)
        assert run.returncode == 0, run.stdout + run.stderr
        manifest = json.loads(
            (root / "w8a8_chain" / "manifest.json").read_text())
        assert_real_program(manifest)

        # Decoded cast directions emit a real boundary program; the int32
        # directions hit the documented refusal (no int32 surface on h16).
        for src_type, dst_type in CASTS:
            name = f"cast_{src_type}_to_{dst_type}"
            if (src_type, dst_type) in INT32_CASTS:
                mil = root / f"{name}.mil"
                mil.write_text(cast_mil(src_type, dst_type))
                run = subprocess.run(
                    [compiler, "--mil", str(mil), "--model-root", str(root),
                     "--target", "H16G", "--output", str(root / name)],
                    capture_output=True, text=True, check=False, timeout=30)
                assert run.returncode != 0
                assert "h16g.legalize.unsupported-operation" in run.stderr
                assert "int32" in run.stderr
                continue
            manifest = compile_source(root, name,
                                      cast_mil(src_type, dst_type))
            assert_real_program(manifest)

        # Red until quantize/dequantize materialize standalone boundary
        # surfaces instead of folding into an empty package.
        manifest = compile_source(root, "quantize_output",
                                  quantize_output_mil)
        assert_real_program(manifest)
        manifest = compile_source(root, "dequantize_input",
                                  dequantize_input_mil)
        assert_real_program(manifest)

    print("parakeet class-8 cast/quant lowering: PASS")


if __name__ == "__main__":
    main()
