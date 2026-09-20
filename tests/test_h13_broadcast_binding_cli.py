#!/usr/bin/env python3
"""Broadcast manifest bindings must name the channels the replayed task
stream addresses.

The 2026-09-19 ac-head mint caught the runtime-runtime broadcast rows
declaring x=out-of-order/y=output channels that their captured streams do
not address - a swap byte-parity cannot see (it compares task words only)
and that would have misrouted every runtime broadcast at run time. This
regression decodes the selector fields of each emitted task and requires
the manifest input/output channels to match the decoded per-role channels:
the same-shape attention-add row binds x=4, y=5, result=6; broadcasting
runtime rows stay refused until their positional fill is derived from the
runtime bundle layout.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def compile_package(root, name, mil):
    (root / f"{name}.mil").write_text(mil)
    out = root / name
    run = subprocess.run(
        [compiler, "--mil", str(root / f"{name}.mil"), "--model-root", str(root),
         "--output", str(out), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    return out


def decoded_channels(anec_path):
    data = anec_path.read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    count = struct.unpack_from("<I", data, 12)[0]
    first = struct.unpack_from("<I", data, 8)[0]
    section = data[0x1000:0x1000 + size]
    words = struct.unpack(f"<{len(section) // 4}I", section)
    selectors = words[8]
    registers = {}
    cursor = 10 + (1 if words[9] & 3 == 3 else 0)
    while cursor < len(words):
        header = words[cursor]
        run_count = (header >> 26) + 1
        base = header & 0x03FFFFFF
        for step in range(run_count):
            registers[base + step * 4] = words[cursor + 1 + step]
        cursor += 1 + run_count
    src, dst = set(), set()
    for shift, role in ((0, "src"), (6, "src"), (12, "dst")):
        channel = (selectors >> shift) & 31
        if channel < 4:
            continue
        if role == "dst":
            if registers.get(0x17800, 0) & 0xFF:
                dst.add(channel)
        elif registers.get(0x13800, 0) & 0xFF:
            src.add(channel)
    # The selectors decoded here belong to the first task in the stream,
    # which carries the broadcast row's own convention regardless of how
    # many tasks the program totals.
    return sorted(src), sorted(dst), first


SAME_ADD = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 375]> x, tensor<fp16, [1, 8, 375, 375]> y) {
    tensor<fp16, [1, 8, 375, 375]> z = add(x = x, y = y)[name = string("z")];
  } -> (z);
}
"""

BCAST_ADD = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 256, 8, 8]> x, tensor<fp16, [1, 256, 1, 1]> y) {
    tensor<fp16, [1, 256, 8, 8]> z = add(x = x, y = y)[name = string("z")];
  } -> (z);
}
"""

with tempfile.TemporaryDirectory(prefix="h13-bcast-binding-") as directory:
    root = Path(directory)

    # The proven same-shape attention-add row: x=4, y=5, result=6.
    package = compile_package(root, "bcast-same-add", SAME_ADD)
    manifest = json.loads((package / "manifest.json").read_text())
    program = manifest["programs"][0]
    assert program["encoder"] == "apple-parity-broadcast"
    src_channels, dst_channels, _ = decoded_channels(package / program["file"])
    assert src_channels == [4, 5] and dst_channels == [6]
    assert sorted(b["index"] for b in program["inputs"]) == [4, 5]
    assert [b["index"] for b in program["outputs"]] == [6]

    # Broadcasting runtime rows are refused: their captured src2 slot
    # names no surface channel (the engine's L2 path), and bundle
    # schema 4 cannot deliver a runtime operand through a channel the
    # decoded stream never reads (proven by the role-consistency CLI).
    # The mint materializes the narrow operand and resubmits the
    # same-shape form.
    (root / "bcast-refused.mil").write_text(BCAST_ADD)
    run = subprocess.run(
        [compiler, "--mil", str(root / "bcast-refused.mil"),
         "--model-root", str(root), "--output", str(root / "bcast-refused"),
         "--target", "H13", "--format", "anec"],
        capture_output=True, text=True, timeout=300, check=False)
    assert run.returncode == 65, run.stdout + run.stderr
    assert "h13.runtime-broadcast-needs-materialized-operand" in run.stderr, \
        run.stderr

print("h13 broadcast binding cli: PASS")
