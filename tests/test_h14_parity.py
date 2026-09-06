#!/usr/bin/env python3
"""Compares emitted H14 task streams word-for-word with decoded Apple oracles."""
import contextlib
import hashlib
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_td import decode_task, split_h14_tasks  # noqa: E402
import inspect_hwx  # noqa: E402
from generate_h14_templates import selected_oracles  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
TILE_BYTES = 0x4000


def compile_oracle(oracle, output, artifact_format):
    mil = output.parent / f"{oracle['case']}-{artifact_format}.mil"
    mil.write_text(oracle["mil"])
    result = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(output.parent),
         "--target", "H14", "--format", artifact_format, "--output", str(output)],
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, \
        f"{oracle['case']} {artifact_format}: {result.stdout}{result.stderr}"
    return json.loads((output / "manifest.json").read_text())


def unpack_commands(hwx):
    """Decode the H14 fields research/mint_oracles.py records from Apple HWX."""
    magic, _, subtype, _, count, command_bytes, _, _ = struct.unpack_from("<8I", hwx)
    assert magic == 0xBEEFFACE and subtype == 5, (hex(magic), subtype)
    cursor = 32
    end = cursor + command_bytes
    sections = {}
    program = None
    tensors = []
    for _ in range(count):
        command, size = struct.unpack_from("<2I", hwx, cursor)
        kind = struct.unpack_from("<I", hwx, cursor + 8)[0] if size >= 12 else None
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", hwx, cursor)
            segment = fields[2].split(b"\0", 1)[0].decode()
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                values = struct.unpack_from("<16s16s2Q8I", hwx, section_cursor)
                section = values[0].split(b"\0", 1)[0].decode()
                sections[(segment, section)] = {
                    "address": values[2], "size": values[3], "offset": values[4]}
                section_cursor += 80
        elif command == 4 and kind == 4 and size >= 0x838:
            program = {
                "kind": kind, "command_size": size,
                "text_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x10)[0]),
                "constant_address": hex(
                    struct.unpack_from("<Q", hwx, cursor + 0x20)[0]),
                "resource_addresses": [
                    hex(value) for value in
                    struct.unpack_from("<5Q", hwx, cursor + 0x30)],
                "text_words": struct.unpack_from("<I", hwx, cursor + 0x824)[0],
                "task_count": struct.unpack_from("<I", hwx, cursor + 0x830)[0],
                "trailing_words": [
                    f"0x{value:08x}" for value in struct.unpack_from(
                        f"<{(size - 0x810) // 4}I", hwx, cursor + 0x810)],
            }
        elif command == 4 and kind == 3 and size >= 0x78:
            tensors.append({
                "binding": struct.unpack_from("<I", hwx, cursor + 0x14)[0],
                "element_code": struct.unpack_from("<I", hwx, cursor + 0x24)[0],
                "shape": list(struct.unpack_from("<4I", hwx, cursor + 0x28)),
                "strides": list(struct.unpack_from("<4Q", hwx, cursor + 0x50)),
                "total_bytes": struct.unpack_from("<Q", hwx, cursor + 0x70)[0],
            })
        cursor += size
    assert cursor == end
    return sections, program, tensors


def anec_contents(payload):
    content_bytes, first_task_bytes, task_count, stream_bytes, constant_bytes, \
        inputs, outputs = struct.unpack_from("<QIIQQII", payload)
    stream = payload[0x1000:0x1000 + stream_bytes]
    tasks = split_h14_tasks(stream)
    constant_offset = (stream_bytes + 0x3F) & ~0x3F
    constants = payload[0x1000 + constant_offset:
                        0x1000 + constant_offset + constant_bytes]
    header = {"first_task_bytes": first_task_bytes, "task_count": task_count,
              "stream_bytes": stream_bytes, "inputs": inputs, "outputs": outputs,
              "content_bytes": content_bytes}
    return tasks, constants, constant_offset, header


def assert_tasks(actual, oracle):
    expected = oracle["task_descriptors"]
    assert len(actual) == len(expected), \
        f"{oracle['case']}: {len(actual)} tasks, oracle has {len(expected)}"
    for index, task in enumerate(actual):
        assert decode_task(task, "h14") == expected[index], \
            f"{oracle['case']} task {index} words differ from the oracle"


def assert_constants(constants, oracle):
    expected = oracle["constant_section"]
    assert len(constants) == expected["size"], oracle["case"]
    assert hashlib.sha256(constants).hexdigest() == expected["sha256"], \
        f"{oracle['case']} constant section differs from the oracle"


def check_anec(oracle, output, manifest):
    record = manifest["programs"][0]
    assert record["encoder"] == "h14-oracle-parity", oracle["case"]
    payload = (output / record["file"]).read_bytes()
    tasks, constants, constant_offset, header = anec_contents(payload)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)
    text_bytes = oracle["program_descriptor"]["text_words"] * 4
    assert header["stream_bytes"] == text_bytes, oracle["case"]
    assert header["task_count"] == oracle["program_descriptor"]["task_count"], \
        oracle["case"]
    assert header["first_task_bytes"] == oracle["task_descriptors"][0]["size_bytes"], \
        oracle["case"]
    assert header["inputs"] == len(oracle["tensor_descriptors"]) - 1, oracle["case"]
    assert header["outputs"] == 1, oracle["case"]
    assert record["constantOffset"] == constant_offset == \
        (text_bytes + 0x3F) & ~0x3F, oracle["case"]
    assert header["content_bytes"] == constant_offset + len(constants), \
        oracle["case"]


def check_hwx(oracle, output, manifest):
    record = manifest["programs"][0]
    assert record["encoder"] == "h14-oracle-parity", oracle["case"]
    path = output / record["file"]
    payload = path.read_bytes()
    sections, program, tensors = unpack_commands(payload)
    assert program == oracle["program_descriptor"], oracle["case"]
    assert tensors == oracle["tensor_descriptors"], oracle["case"]
    text = sections[("__TEXT", "__text")]
    constants = sections[("__TEXT", "__const")]
    assert text["size"] == program["text_words"] * 4, oracle["case"]
    assert constants["size"] == oracle["constant_section"]["size"], oracle["case"]
    assert constants["address"] - text["address"] == record["constantOffset"], \
        oracle["case"]
    # The surface allocations Apple's text address implies: 16 KiB tiles per
    # tensor, starting at the scratch base the descriptor records in slot 4.
    allocated = sum((descriptor["total_bytes"] + TILE_BYTES - 1) // TILE_BYTES
                    * TILE_BYTES for descriptor in tensors)
    assert text["address"] == int(program["resource_addresses"][4], 16) + allocated, \
        oracle["case"]
    assert_tasks(split_h14_tasks(payload[text["offset"]:
                                        text["offset"] + text["size"]]), oracle)
    assert_constants(payload[constants["offset"]:
                             constants["offset"] + constants["size"]], oracle)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        inspect_hwx.main(str(path))
    text_report = report.getvalue()
    assert "name=H14 isa=11" in text_report, oracle["case"]
    assert f"h14_tasks count={len(oracle['task_descriptors'])}" in text_report, \
        oracle["case"]


def main():
    oracles = selected_oracles()
    assert len(oracles) == 87, f"expected 87 decoded parity oracles, found {len(oracles)}"
    with tempfile.TemporaryDirectory(prefix="h14-parity-") as directory:
        root = Path(directory)
        for oracle in oracles:
            for artifact_format, check in (("anec", check_anec), ("hwx", check_hwx)):
                output = root / f"{oracle['case']}-{artifact_format}"
                manifest = compile_oracle(oracle, output, artifact_format)
                assert len(manifest["programs"]) == 1, \
                    f"{oracle['case']}: {len(manifest['programs'])} programs"
                check(oracle, output, manifest)
    print(f"H14 oracle parity: PASS ({len(oracles)} cases, "
          f"{len(oracles) * 2} artifacts)")


if __name__ == "__main__":
    main()
