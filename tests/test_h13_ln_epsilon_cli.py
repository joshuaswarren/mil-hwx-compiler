#!/usr/bin/env python3
"""The captured Apple encoder layer_norm row: exact epsilon, exact bytes.

Main's CPU-only `ane-compile-hwx` captures (research/oracles/h13-captures/)
prove the [1, 375, 1024] axes=[-1] non-affine layer_norm against Apple's own
task bytes, and the control captures prove the epsilon is baked at fp16
granularity: fp32(1e-5), fp32(0x1.5p-17) and fp16(0x1.5p-17) all compile to
byte-identical programs (fp16 0x00a8). The compiler therefore claims a norm
row by the captured fp16 epsilon halves and refuses any epsilon that rounds
elsewhere, exactly as the broadcast planner claims a scalar by its baked
bits. The real model spells the epsilon as a rank-0 fp16 constant (BLOBFILE
record, payload fp16 0x00a8) and the axes as an unbracketed one-element
tensor literal; both spellings must reach the same Apple bytes.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve())

# test_h13_parity resolves its own compiler from argv; scrub before import.
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "research"))
import test_h13_parity as parity  # noqa: E402

CAPTURES = ROOT / "research/oracles/h13-captures"


def capture(case):
    record = json.loads((CAPTURES / f"{case}.json").read_text())
    assert record.get("error") is None, case
    return record


def mil_encoder(axes_spelling, epsilon_spelling, shape="1, 375, 1024"):
    return """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [%s]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = %s];
    tensor<fp16, [%s]> y = layer_norm(x = x, axes = axes, %s)[name = string("y")];
  } -> (y);
}
""" % (shape, axes_spelling, shape, {
        "fp32_default": "epsilon = fp32(0.00001)",
        "fp32_exact": "epsilon = fp32(0x1.5p-17)",
        "fp16_exact": "epsilon = fp16(0x1.5p-17)",
        "fp32_1e6": "epsilon = fp32(0.000001)",
        "fp32_2e5": "epsilon = fp32(0.00002)",
    }[epsilon_spelling])


def mil_const_epsilon(payload):
    # The real model's spelling: epsilon is a rank-0 fp16 constant, either an
    # fp16 literal or the BLOBFILE record the real weights carry.
    val = ("BLOBFILE(path = string(\"@model_path/weights/epsilon.bin\"), "
           "offset = uint64(64))" if payload == "blob" else "fp16(0x1.5p-17)")
    return """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>(-1)];
    tensor<fp16, []> epsilon_bits = const()[name = string("epsilon_bits"), val = tensor<fp16, []>(%s)];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, axes = axes, epsilon = epsilon_bits)[name = string("y")];
  } -> (y);
}
""" % val


def mil_affine():
    gamma = ", ".join(["fp16(0x1p+0)"] * 1024)
    beta = ", ".join(["fp16(0x0p+0)"] * 1024)
    return """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>(-1)];
    tensor<fp16, [1024]> gamma = const()[name = string("gamma"), val = tensor<fp16, [1024]>([%s])];
    tensor<fp16, [1024]> beta = const()[name = string("beta"), val = tensor<fp16, [1024]>([%s])];
    tensor<fp16, []> epsilon_bits = const()[name = string("epsilon_bits"), val = tensor<fp16, []>(fp16(0x1.5p-17))];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, axes = axes, epsilon = epsilon_bits, gamma = gamma, beta = beta)[name = string("y")];
  } -> (y);
}
""" % (gamma, beta)


def epsilon_blob():
    # The real model's BLOBFILE layout: the offset names a 24-byte record
    # (magic, count, payload length, payload offset); the payload is the
    # captured fp16 0x00a8 = 0x1.5p-17 exactly.
    blob = bytearray(64 + 24 + 2)
    struct.pack_into("<IIQQ", blob, 64, 0xDEADBEEF, 1, 2, 88)
    struct.pack_into("<H", blob, 88, 0x00A8)
    return bytes(blob)


def run_compile(root, name, text, artifact_format, expected_code=None):
    out = root / f"{name}-{artifact_format}"
    model_root = root / f"{name}-{artifact_format}-root"
    (model_root / "weights").mkdir(parents=True, exist_ok=True)
    (model_root / "weights/epsilon.bin").write_bytes(epsilon_blob())
    (model_root / f"{name}.mil").write_text(text)
    run = subprocess.run(
        [compiler, "--mil", str(model_root / f"{name}.mil"),
         "--model-root", str(model_root), "--target", "H13",
         "--format", artifact_format, "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=60)
    if expected_code is not None:
        assert run.returncode == 65, f"{name}: {run.stdout}{run.stderr}"
        assert expected_code in run.stderr, run.stderr
        assert not out.exists(), f"{name}: failed compilation wrote output"
        return None
    assert run.returncode == 0, f"{name}: {run.stdout}{run.stderr}"
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["programs"]) == 1, name
    assert manifest["programs"][0]["encoder"] == "apple-parity-norm", name
    # The same acceptance the 896-case parity suite applies, against the
    # captured Apple bytes instead of a minted oracle.
    if artifact_format == "anec":
        parity.check_anec(record_for_assert, out, manifest)
    else:
        parity.check_hwx(record_for_assert, out, manifest)


def assert_matches(record, root, name, text):
    global record_for_assert
    record_for_assert = record
    for artifact_format in ("anec", "hwx"):
        run_compile(root, name, text, artifact_format)


rank3 = capture("rank3_exact")
width = capture("width_exact")
control_default = capture("control_default")
control_exact = capture("control_exact")
control_fp16 = capture("control_fp16")

# The controls pin the captured epsilon semantics: Apple's bytes for
# fp32(1e-5), fp32(0x1.5p-17), and fp16(0x1.5p-17) at [1, 1024, 1, 1] are
# identical, so the row's baked fp16 epsilon (0x00a8) is the claim key.
assert control_default["task_descriptors"] == control_exact["task_descriptors"]
assert control_default["task_descriptors"] == control_fp16["task_descriptors"]

with tempfile.TemporaryDirectory(prefix="h13-ln-epsilon-") as directory:
    root = Path(directory)

    # The captured encoder row: bracketed axes and the exact fp32 epsilon,
    # byte-compared with Apple's capture in both artifact formats.
    assert_matches(rank3, root, "rank3-capture-mil", rank3["mil"])

    # The real model's spellings reach the same Apple bytes.
    assert_matches(rank3, root, "rank3-unbracketed-axes",
                   mil_encoder("tensor<int32, [1]>(-1)", "fp32_exact"))
    assert_matches(rank3, root, "rank3-fp16-attribute",
                   mil_encoder("tensor<int32, [1]>(-1)", "fp16_exact"))
    assert_matches(rank3, root, "rank3-const-fp16-epsilon",
                   mil_const_epsilon("fp16"))
    assert_matches(rank3, root, "rank3-const-blob-epsilon",
                   mil_const_epsilon("blob"))

    # The rank-4 spelling of the encoder geometry captures its own Apple
    # row: [1, 375, 1, 1024] axes=[3] binds the transposed surface.
    assert_matches(width, root, "width-capture-mil", width["mil"])

    # The controls byte-match their own captures at [1, 1024, 1, 1].
    assert_matches(control_default, root, "control-default-mil",
                   control_default["mil"])
    assert_matches(control_exact, root, "control-exact-mil",
                   control_exact["mil"])
    assert_matches(control_fp16, root, "control-fp16-mil",
                   control_fp16["mil"])

    # Fail-closed: an epsilon Apple has no captured row for refuses by name.
    run_compile(root, "epsilon-1e6",
                mil_encoder("tensor<int32, [1]>([-1])", "fp32_1e6"),
                "anec", expected_code="h13.norm-outside-envelope")
    run_compile(root, "epsilon-2e5",
                mil_encoder("tensor<int32, [1]>([-1])", "fp32_2e5"),
                "anec", expected_code="h13.norm-outside-envelope")

    # Fail-closed: a geometry Apple has no captured row for refuses.
    run_compile(root, "geometry-376",
                mil_encoder("tensor<int32, [1]>([-1])", "fp32_exact",
                            shape="1, 376, 1024"),
                "anec", expected_code="h13.norm-outside-envelope")

    # Fail-closed: the affine form stays refused (no decoded affine row).
    run_compile(root, "affine-exact-epsilon", mil_affine(),
                "anec", expected_code="h13.norm-outside-envelope")

print("h13 ln epsilon cli: PASS")
