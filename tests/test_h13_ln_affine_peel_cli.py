#!/usr/bin/env python3
"""The affine layer_norm peel: three decoded Apple programs per statement.

Apple's own tool refuses the affine form at the encoder geometry (capture
triage 2026-09-20: affine arms refused with the exact epsilon in both fp32
spellings; the blob record layout and the non-affine row decoded). The
compiler therefore composes the peel itself: the non-affine layer_norm row,
then the decoded per-channel constant mul (gamma) and constant add (beta)
broadcast rows. Every program must be byte-exact against its decoded row —
this suite compares all three, including constant sections — and the
manifest keeps the three programs distinct. A peel stage outside its decoded
envelope (gamma without beta) leaves the affine refusal in force.
"""
import hashlib
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
import test_h13_parity as parity
import mint_oracles
from h13_td import decode_task  # noqa: E402


def oracle(name):
    return json.loads((ROOT / "research/oracles/h13" / f"{name}.json").read_text())


rank3 = json.loads(
    (ROOT / "research/oracles/h13-captures/rank3_exact.json").read_text())
mul_row = oracle("env_bcast_mul_1x375x1024_blob_1x1x1024")
add_row = oracle("env_bcast_add_1x375x1024_blob_1x1x1024")

# The payload the blob-broadcast oracles compiled against: fp16 0.5 repeated,
# which pins the emitted constant sections to the recorded hashes.
PAYLOAD = b"\x00\x38" * 1024


def blob():
    # The campaign's own blob fixture, byte-for-byte: both constants point
    # at its single record @64, so the compiler resolves the same window the
    # oracle rows decoded and the sections compare byte-exactly.
    return mint_oracles.blob(PAYLOAD)


def affine_mil(gamma=True, beta=True):
    lines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({})]",
        "{",
        "  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {",
        '    tensor<int32, [1]> axes = const()[name = string("axes"), '
        'val = tensor<int32, [1]>([-1])];',
    ]
    if gamma:
        lines.append('    tensor<fp16, [1024]> gamma = const()'
                     '[name = string("gamma"), val = tensor<fp16, [1024]>'
                     '(BLOBFILE(path = string("@model_path/weights.bin"), '
                     'offset = uint64(64)))];')
    if beta:
        lines.append('    tensor<fp16, [1024]> beta = const()'
                     '[name = string("beta"), val = tensor<fp16, [1024]>'
                     '(BLOBFILE(path = string("@model_path/weights.bin"), '
                     'offset = uint64(64)))];')
    lines.append('    tensor<fp16, []> epsilon = const()'
                 '[name = string("epsilon"), val = tensor<fp16, []>'
                 '(fp16(0x1.5p-17))];')
    operands = "epsilon = epsilon"
    if gamma:
        operands += ", gamma = gamma"
    if beta:
        operands += ", beta = beta"
    lines.append('    tensor<fp16, [1, 375, 1024]> y = layer_norm(x = x, '
                 f'axes = axes, {operands})[name = string("y")];')
    lines += ["  } -> (y);", "}", ""]
    return "\n".join(lines)


def run_compile(root, name, text, weights, expected_code=None):
    model_root = root / f"{name}-root"
    (model_root / "weights").mkdir(parents=True, exist_ok=True)
    (model_root / "weights.bin").write_bytes(weights)
    (model_root / f"{name}.mil").write_text(text)
    out = root / name
    run = subprocess.run(
        [compiler, "--mil", str(model_root / f"{name}.mil"),
         "--model-root", str(model_root), "--target", "H13",
         "--format", "anec", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=60)
    if expected_code is not None:
        assert run.returncode == 65, f"{name}: {run.stdout}{run.stderr}"
        assert expected_code in run.stderr, run.stderr
        assert not out.exists(), f"{name}: failed compilation wrote output"
        return None
    assert run.returncode == 0, f"{name}: {run.stdout}{run.stderr}"
    return out, json.loads((out / "manifest.json").read_text())


def assert_program(record, out, program):
    """Task stream + constant section byte-exact against the decoded row,
    under the parity suite's own channel-binding convention."""
    payload = (out / program["file"]).read_bytes()
    tasks, constants, constant_offset = parity.anec_contents(payload)
    binding = parity.channel_binding(record)
    expected = [parity.bound_task_descriptor(task, binding)
                for task in record["task_descriptors"]]
    assert len(tasks) == len(expected), program["file"]
    for index, task in enumerate(tasks):
        assert decode_task(task, "h13") == expected[index], \
            f"{program['file']} task {index} differs from the decoded row"
    section = record["constant_section"]
    assert len(constants) == section["size"], program["file"]
    assert hashlib.sha256(constants).hexdigest() == section["sha256"], \
        f"{program['file']} constant section differs from the decoded row"
    assert program["constantOffset"] == constant_offset, program["file"]


with tempfile.TemporaryDirectory(prefix="h13-ln-affine-peel-") as directory:
    root = Path(directory)

    out, manifest = run_compile(root, "affine-peel", affine_mil(), blob())
    assert [program["operation"] for program in manifest["programs"]] == \
        ["layer_norm", "mul", "add"], manifest["programs"]
    assert [program["encoder"] for program in manifest["programs"]] == \
        ["apple-parity-norm", "apple-parity-broadcast",
         "apple-parity-broadcast"], manifest["programs"]
    assert [program["taskDescriptors"] for program in manifest["programs"]] == \
        [5, 2, 2], manifest["programs"]
    assert_program(rank3, out, manifest["programs"][0])
    assert_program(mul_row, out, manifest["programs"][1])
    assert_program(add_row, out, manifest["programs"][2])

    # Fail-closed: the peel gate needs BOTH constants. Gamma without beta
    # stays on the affine refusal path, exactly as before the peel landed.
    run_compile(root, "gamma-only", affine_mil(beta=False), blob(),
                expected_code="h13.norm-outside-envelope")
    run_compile(root, "beta-only", affine_mil(gamma=False), blob(),
                expected_code="h13.norm-outside-envelope")

print("h13 ln affine peel cli: PASS")
