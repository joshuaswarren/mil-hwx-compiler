#!/usr/bin/env python3
"""Generic relu beyond the decoded c64/c512 oracle shapes (NE-05).

The parity corpus decodes relu at exactly two {c,1,1} geometries, 64 and
512 channels. The c64 and c512 task streams differ only at seven words:
two channel fields (c), four DMA word-count fields (c * 16 words, flag
bits preserved), and the stream's descriptor word, which the
seven-channel abs corpus fixes as floor(c * 9 / 2048) on H14/H17 and
floor(c * 7 / 2048) on H18. The generic encoder rewrites those words for
any channel count divisible by 16 and keeps everything else byte-identical.

For every covered shape this test compiles the generic program AND the
c512 oracle program with the same compiler, then diffs the emitted task
streams word-for-word: every differing word must move from its c512
value to the derived value, and nothing else may differ. Refusal legs
pin the envelope boundary: channels not divisible by 16, and non-{c,1,1}
shapes, still fail with outside-parity-envelope.

Usage: test_generic_relu.py [path-to-mil-hwxc]
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import mint_oracles  # noqa: E402
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()

CHANNEL_WORD = 512
DMA_WORD = 512 * 16
# floor(c * K / 2048): the abs corpus (c64..c4096, every generation) pins K.
DESCRIPTOR_DIVISOR = {("H14", "H17"): (9, 2048), ("H18",): (7, 2048)}
ANEC_HEADER_BYTES = 0x1000
GENERIC_CHANNELS = (128, 384, 1024)

MIL = """program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {c}, 1, 1]> x) {{
    tensor<fp16, [1, {c}, 1, 1]> y = relu(x = x)[name = string("y")];
  }} -> (y);
}}
"""

MIL_HW = """program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {c}, {h}, {w}]> x) {{
    tensor<fp16, [1, {c}, {h}, {w}]> y = relu(x = x)[name = string("y")];
  }} -> (y);
}}
"""


def descriptor_word(target, channels):
    for targets, (numerator, denominator) in DESCRIPTOR_DIVISOR.items():
        if target in targets:
            return channels * numerator // denominator
    raise AssertionError(target)


def compile_relu(root, target, mil, name):
    path = root / f"{name}.mil"
    path.write_text(mil)
    output = root / name
    result = subprocess.run(
        [str(COMPILER), "--mil", str(path), "--model-root", str(root),
         "--target", target, "--output", str(output)],
        capture_output=True, text=True, timeout=scaled_timeout(120), check=False)
    return result, output


def artifact_bytes(output, suffix):
    programs = sorted(output.glob(f"program-*.{suffix}"))
    assert len(programs) == 1, f"expected one .{suffix} program, got {len(programs)}"
    return programs[0].read_bytes()


def anec_stream(blob, words):
    return list(struct.unpack_from(f"<{words}I", blob, ANEC_HEADER_BYTES))


def stream_word_count(target):
    return {"H14": 51, "H17": 65, "H18": 67}[target]


def check_word_diffs(target, reference, generic, channels):
    """Every changed word moves 512 -> c, 512*16 -> c*16 (flags kept), or
    the descriptor word -> floor(c * K / 2048). Nothing else may differ."""
    problems = []
    desc_ref = descriptor_word(target, CHANNEL_WORD)
    desc_new = descriptor_word(target, channels)
    for index, (old, new) in enumerate(zip(reference, generic)):
        if old == new:
            continue
        if old == CHANNEL_WORD and new == channels:
            continue
        if old == DMA_WORD and (new & ~0x3F) == channels * 16:
            continue
        if old == desc_ref and new == desc_new:
            continue
        problems.append(
            f"word[{index}]: {old:#010x} -> {new:#010x} is not a shape field")
    if len(reference) != len(generic):
        problems.append(
            f"stream length {len(generic)} != reference {len(reference)}")
    return problems


def hwx_word_map(parsed):
    words = {}
    for descriptor in parsed["task_descriptors"]:
        for block in descriptor["blocks"].values():
            for offset, word in block["words"].items():
                words[offset] = int(word, 16)
    return words


def check_h14(target, root):
    problems = []
    ref_result, ref_out = compile_relu(root, target, MIL.format(c=CHANNEL_WORD), f"{target}_relu_c512")
    assert ref_result.returncode == 0, ref_result.stderr
    ref_blob = artifact_bytes(ref_out, "anec")
    words = stream_word_count(target)
    reference = anec_stream(ref_blob, words)
    for channels in GENERIC_CHANNELS:
        result, output = compile_relu(root, target, MIL.format(c=channels),
                                      f"{target}_relu_c{channels}")
        if result.returncode != 0:
            problems.append(f"{target} c{channels}: refused: {result.stderr.strip()}")
            continue
        blob = artifact_bytes(output, "anec")
        generic = anec_stream(blob, words)
        problems += [f"{target} c{channels}: {p}" for p in
                     check_word_diffs(target, reference, generic, channels)]
        header = struct.unpack_from("<II", blob, 32)
        if header != (1, 1):
            problems.append(f"{target} c{channels}: io counts {header}")
        for channel in (4, 5):
            nchw = struct.unpack_from("<6Q", blob, 168 + 48 * channel)
            if nchw != (1, channels, 1, 1, 64, 64):
                problems.append(f"{target} c{channels}: ch{channel} nchw {nchw}")
    return problems


def check_hwx(target, root):
    problems = []
    ref_result, ref_out = compile_relu(root, target, MIL.format(c=CHANNEL_WORD), f"{target}_relu_c512")
    assert ref_result.returncode == 0, ref_result.stderr
    reference = hwx_word_map(mint_oracles.parse_hwx(
        artifact_bytes(ref_out, "hwx"), target.lower()))
    for channels in GENERIC_CHANNELS:
        result, output = compile_relu(root, target, MIL.format(c=channels),
                                      f"{target}_relu_c{channels}")
        if result.returncode != 0:
            problems.append(f"{target} c{channels}: refused: {result.stderr.strip()}")
            continue
        parsed = mint_oracles.parse_hwx(artifact_bytes(output, "hwx"),
                                        target.lower())
        generic = hwx_word_map(parsed)
        offsets = sorted(set(reference) | set(generic))
        reference_list = [reference.get(offset, 0) for offset in offsets]
        generic_list = [generic.get(offset, 0) for offset in offsets]
        diff_problems = check_word_diffs(target, reference_list, generic_list,
                                       channels)
        problems += [f"{target} c{channels}: {p}" for p in diff_problems]
        tensors = parsed["tensor_descriptors"]
        channels_seen = [t.get("channels") for t in tensors]
        if channels_seen and any(value not in (channels, None) for value in channels_seen):
            problems.append(f"{target} c{channels}: tensor channels {channels_seen}")
    return problems


def check_refusals(root):
    problems = []
    cases = [
        ("H14", MIL.format(c=100), "c100 not divisible by 16"),
        ("H18", MIL.format(c=100), "c100 not divisible by 16"),
        # {37,2,1} flattens to {74,1,1}: still not divisible by 16.
        ("H14", MIL_HW.format(c=37, h=2, w=1), "{37,2,1} flattens to 74"),
        ("H18", MIL_HW.format(c=37, h=2, w=1), "{37,2,1} flattens to 74"),
    ]
    for index, (target, mil, label) in enumerate(cases):
        result, _ = compile_relu(root, target, mil, f"{target}_refused_{index}")
        if result.returncode == 0:
            problems.append(f"{target}: {label} compiled unexpectedly")
        elif "outside-parity-envelope" not in result.stderr:
            problems.append(f"{target}: {label} refused with the wrong error: "
                            f"{result.stderr.strip()}")
    # Non-{c,1,1} shapes compile through the flattened {c*h*w,1,1} candidate;
    # {64,2,1} is 128 elementwise elements and emits the generic c128 program.
    for target in ("H14", "H17", "H18"):
        result, output = compile_relu(root, target,
                                      MIL_HW.format(c=64, h=2, w=1),
                                      f"{target}_flattened")
        if result.returncode != 0:
            problems.append(f"{target}: {{64,2,1}} flattened form refused: "
                            f"{result.stderr.strip()}")
    return problems


def main():
    bad = 0
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for problems, label in (
            (check_h14("H14", root), "H14 anec"),
            (check_hwx("H17", root), "H17 hwx"),
            (check_hwx("H18", root), "H18 hwx"),
            (check_refusals(root), "envelope boundary"),
        ):
            if problems:
                bad += 1
                for problem in problems:
                    print(f"FAIL {label}: {problem}")
            else:
                print(f"PASS {label}")
    print("generic relu: " + ("PASS" if not bad else "FAIL"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
