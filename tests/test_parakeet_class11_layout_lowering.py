#!/usr/bin/env python3
"""Parakeet lowering class 11: layout / data-movement ops.

Failing-first on main, per the SUP-82 method: every class-11 op the
Parakeet encoder uses must reach an ANE data-movement program. Today
transpose lowers (apple-parity-transpose) and four runtime-operand tile
captures lower (apple-parity-tile), but the view ops Apple mints real
programs for - reshape, squeeze - collapse to input aliases here and are
refused when the alias is the returned value
(h13.returned-input-alias). Envelope probing on h13 (2026-11-28,
research/oracles/h13/layout_*.json):

- reshape [1,375,1024] -> [1,375,64,16]: decoded 2-task program.
- squeeze [1,375,1] -> [1,375]: decoded 1-task program.
- tile [1,64,1] reps [1,1,64] -> [1,64,64]: decoded 1-task program;
  the tile envelope widens past the four captured forms.
- split (axis 0 and axis 1) and pad refuse on Apple too
  (callback_status=1): frontend elimination jobs, pinned here as
  documented refusals.

The red rows assert an apple-parity-layout / apple-parity-tile program
whose task stream is byte-exact against the decoded oracle.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
ROOT = Path(__file__).resolve().parents[1]


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
        return json.loads((out / "manifest.json").read_text()), out
    assert run.returncode == 65, run.stdout + run.stderr
    assert expected_code in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
    return None, None


def reconstruct_task(td):
    words = [int(w, 16) for w in td["header_words"]]
    flat = {}
    for block in td["blocks"].values():
        for a, v in block["words"].items():
            flat[int(a, 16)] = int(v, 16)
    for rec in td["records"]:
        words.append(int(rec["header"], 16))
        base = int(rec["address"], 16)
        for index in range(rec["count"]):
            words.append(flat[base + index * 4])
    return words


SURFACE_REMAP = {5: 4, 4: 5}


def capture_stream(record):
    stream = bytearray()
    offset = 0
    for td in record["task_descriptors"]:
        words = reconstruct_task(td)
        blob = struct.pack(f"<{len(words)}I", *words)
        if len(stream) < offset + len(blob):
            stream.extend(b"\0" * (offset + len(blob) - len(stream)))
        linked = list(words)
        linked[0] = (linked[0] & ~0x00FF0000) | 0x00400000
        for shift in (0, 6, 12):
            channel = (linked[8] >> shift) & 0x1F
            if channel in SURFACE_REMAP:
                linked[8] = (linked[8] & ~(0x1F << shift)) | \
                    (SURFACE_REMAP[channel] << shift)
        stream[offset:offset + len(blob)] = struct.pack(
            f"<{len(linked)}I", *linked)
        offset = words[7]
    return bytes(stream)


def anec_task_stream(path):
    data = path.read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    return data[0x1000:0x1000 + size]


# Control: transpose already lowers through apple-parity-transpose.
transpose_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [3]> perm = const()[name = string("perm"), val = tensor<int32, [3]>([0, 2, 1])];
    tensor<fp16, [1, 1024, 375]> y = transpose(x = x, perm = perm)[name = string("y")];
  } -> (y);
}
"""

# Red 1: reshape is a pure view today - the returned alias is refused.
# Apple mints a real 2-task data-movement program for it.
reshape_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [4]> shape = const()[name = string("shape"), val = tensor<int32, [4]>([1, 375, 64, 16])];
    tensor<fp16, [1, 375, 64, 16]> y = reshape(x = x, shape = shape)[name = string("y")];
  } -> (y);
}
"""

# Red 2: squeeze - same alias refusal; Apple mints a 1-task program.
squeeze_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1]> x) {
    tensor<fp16, [1, 375]> y = squeeze(x = x)[name = string("y")];
  } -> (y);
}
"""

# Red 3: tile outside the four captured forms - the decoded envelope
# widens (this capture mints on h13) but our table only covers the
# original four.
tile_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 64, 1]> x) {
    tensor<int32, [3]> reps = const()[name = string("reps"), val = tensor<int32, [3]>([1, 1, 64])];
    tensor<fp16, [1, 64, 64]> y = tile(x = x, reps = reps)[name = string("y")];
  } -> (y);
}
"""

# Documented refusal 1: split has no decoded primitive anywhere - Apple
# refuses both axis positions, so it is a frontend elimination job.
split_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [2, 375, 1024]> x) {
    tensor<int32, []> axis = const()[name = string("axis"), val = tensor<int32, []>(0)];
    tensor<int32, []> count = const()[name = string("count"), val = tensor<int32, []>(2)];
    (tensor<fp16, [1, 375, 1024]> a, tensor<fp16, [1, 375, 1024]> b) = split(axis = axis, num_splits = count, x = x)[name = string("s")];
  } -> (a);
}
"""

# Documented refusal 2: pad refuses on Apple in every form including the
# identity view (6/6 callback_status=1) - frontend elimination too.
pad_parakeet = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [6]> pads = const()[name = string("pads"), val = tensor<int32, [6]>([0, 0, 0, 1, 0, 0])];
    tensor<fp16, [1, 376, 1024]> y = pad(x = x, pad = pads, mode = string("constant"))[name = string("y")];
  } -> (y);
}
"""


def parity_program(manifest, name, encoder, tasks):
    program = manifest["programs"][0]
    assert program["operation"] == name, (name, program["operation"])
    assert program["encoder"] == encoder, (name, program["encoder"])
    assert program["taskDescriptors"] == tasks, (name, program)
    return program


with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)

    manifest, _ = compile_source(root, "transpose", transpose_parakeet)
    parity_program(manifest, "transpose", "apple-parity-transpose", 1)

    manifest, out = compile_source(root, "reshape", reshape_parakeet)
    parity_program(manifest, "reshape", "apple-parity-layout", 2)
    oracle = json.loads(
        (ROOT / "research/oracles/h13"
         "/layout_reshape_1x375x1024_to_1x375x64x16.json").read_text())
    assert anec_task_stream(out / "program-0.anec") == \
        capture_stream(oracle), "reshape task stream != oracle"

    manifest, out = compile_source(root, "squeeze", squeeze_parakeet)
    parity_program(manifest, "squeeze", "apple-parity-layout", 1)
    oracle = json.loads(
        (ROOT / "research/oracles/h13"
         "/layout_squeeze_1x375x1_to_1x375.json").read_text())
    assert anec_task_stream(out / "program-0.anec") == \
        capture_stream(oracle), "squeeze task stream != oracle"

    manifest, out = compile_source(root, "tile", tile_parakeet)
    parity_program(manifest, "tile", "apple-parity-tile", 1)
    oracle = json.loads(
        (ROOT / "research/oracles/h13"
         "/layout/tile_r_1x64x1_reps_1164.json").read_text())
    assert anec_task_stream(out / "program-0.anec") == \
        capture_stream(oracle), "tile task stream != oracle"

    compile_source(root, "split", split_parakeet,
                   expected_code="h13.returned-input-alias")
    compile_source(root, "pad", pad_parakeet,
                   expected_code="h13.unsupported-pad")

    print("parakeet class-11 layout lowering: PASS")
