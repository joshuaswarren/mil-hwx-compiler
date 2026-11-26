#!/usr/bin/env python3
"""SUP-85, Parakeet lowering class 4: normalization ops (layer/rms/group).

Failing-first on main, per the SUP-82 method: every class-4 op the Parakeet
encoder uses must reach an ANE norm program. Today only the no-affine
LayerNorm form lowers (apple-parity-norm, 35 H13 / 34 H14 oracle templates);
the affine LayerNorm form the encoder actually carries is envelope-refused
(Apple refuses it too), and RMSNorm / GroupNorm have no lowering anywhere
(no NormOperation entry, no templates, no decompose pass handling) — so
their cases assert the desired outcome and stay red until the lowerings
land. Green list of what already lowers lives in the class docstring.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def blob(payload):
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(payload), 128)
    return bytes(header) + payload


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
        return json.loads((out / "manifest.json").read_text())
    assert run.returncode == 65, run.stdout + run.stderr
    assert expected_code in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
    return None


# Green list: what already lowers today (keep covered by the envelope test's
# oracle geometry; restated here with the Parakeet encoder shape).
ln_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001))[name = string("y")];
  } -> (y);
}
"""

# Red 1: Parakeet carries the affine form (gamma/beta) — envelope-refused
# today; the class-4 plan is an affine LayerNorm lowering at this geometry.
ln_parakeet_affine = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([-1])];
    tensor<fp16, [1024]> gamma = const()[name = string("gamma"), val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/gamma.bin"), offset = uint64(64)))];
    tensor<fp16, [1024]> beta = const()[name = string("beta"), val = tensor<fp16, [1024]>(BLOBFILE(path = string("@model_path/beta.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001), gamma = gamma, beta = beta)[name = string("y")];
  } -> (y);
}
"""

# Red 2: RMSNorm has no MIL parsing and no lowering anywhere in the tree.
rms_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<fp16, [1, 375, 1024]> y = rms_norm(x = x, epsilon = fp32(0.00001))[name = string("y")];
  } -> (y);
}
"""

# Red 3: GroupNorm (the conv-subsampling norm family) has no lowering.
gn_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 64, 375, 80]> x) {
    tensor<fp16, [64]> gamma = const()[name = string("gamma"), val = tensor<fp16, [64]>(BLOBFILE(path = string("@model_path/gn_gamma.bin"), offset = uint64(64)))];
    tensor<fp16, [64]> beta = const()[name = string("beta"), val = tensor<fp16, [64]>(BLOBFILE(path = string("@model_path/gn_beta.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 64, 375, 80]> y = group_norm(x = x, gamma = gamma, beta = beta, groups = 32, epsilon = fp32(0.00001))[name = string("y")];
  } -> (y);
}
"""


def main():
    with tempfile.TemporaryDirectory(prefix="mil-hwx-parakeet-class4-") as directory:
        root = Path(directory)
        (root / "gamma.bin").write_bytes(blob(bytes(1024 * 2)))
        (root / "beta.bin").write_bytes(blob(bytes(1024 * 2)))
        (root / "gn_gamma.bin").write_bytes(blob(bytes(64 * 2)))
        (root / "gn_beta.bin").write_bytes(blob(bytes(64 * 2)))

        # Green: the no-affine encoder geometry lowers today.
        ln = compile_source(root, "ln-parakeet", ln_parakeet)
        assert len(ln["programs"]) == 1
        assert ln["programs"][0]["operation"] == "layer_norm"
        assert ln["programs"][0]["encoder"] == "apple-parity-norm"

        # Red until the affine LayerNorm envelope opens: Parakeet's encoder
        # norm carries gamma/beta and must reach the same parity program.
        ln_aff = compile_source(root, "ln-parakeet-affine", ln_parakeet_affine)
        assert len(ln_aff["programs"]) == 1
        assert ln_aff["programs"][0]["operation"] == "layer_norm"
        assert ln_aff["programs"][0]["encoder"] == "apple-parity-norm"

        # Red until RMSNorm lowers: the attention/FF blocks use it.
        rms = compile_source(root, "rms-parakeet", rms_parakeet)
        assert len(rms["programs"]) == 1
        assert rms["programs"][0]["encoder"] == "apple-parity-norm"

        # Red until GroupNorm lowers: the conv-subsampling norm family.
        gn = compile_source(root, "gn-parakeet", gn_parakeet)
        assert len(gn["programs"]) == 1
        assert gn["programs"][0]["encoder"] == "apple-parity-norm"

    print("parakeet class-4 norm lowering: PASS")


if __name__ == "__main__":
    main()
