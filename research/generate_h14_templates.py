#!/usr/bin/env python3
"""Generate the H14 elementwise parity templates from the decoded oracles.

Writes `plugins/H14/H14ElementwiseTemplates.inc` from
`research/oracles/h14/*.json`. Only decoded task words, program-descriptor
metadata, and constant-section contents are emitted; no HWX container bytes
are read or stored.

Regenerate and compare with:

```sh
python3 research/generate_h14_templates.py --check
```
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h13_td import decode_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / "research/oracles/h14"
OUTPUT = ROOT / "plugins/H14/H14ElementwiseTemplates.inc"
MATVEC_OUTPUT = ROOT / "plugins/H14/H14MatvecTemplates.inc"
CONSTANT_ALIGNMENT = 0x40

# Task-stream framing: a 16-byte zero prefix that decodes as a zero-size task,
# then every task 16-byte aligned with the last task left unpadded.
TASK_ALIGNMENT = 16
STREAM_PREFIX_BYTES = 16

BINARY_OPERATIONS = {"add": 0, "mul": 1, "maximum": 2, "minimum": 3, "sub": 4,
                     "real_div": 5}
UNARY_OPERATIONS = {"abs": 0, "exp": 1, "gelu": 2, "leaky_relu": 3, "relu": 4,
                    "rsqrt": 5, "sigmoid": 6, "silu": 7, "sqrt": 8, "tanh": 9}
RUNTIME_BINARY = {"add", "mul", "maximum", "minimum", "sub"}
UNARY = set(UNARY_OPERATIONS)
KINDS = {"binary_runtime": "BinaryRuntime", "binary_constant": "BinaryScalar",
         "unary": "Unary"}


def selected_oracles() -> list[dict[str, Any]]:
    """The H14 elementwise, scalar-constant, and unary cases Apple decoded."""
    selected = []
    for path in sorted(ORACLES.glob("*.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is not None:
            continue
        parameters = oracle.get("parameters", {})
        operation = parameters.get("operation")
        family = oracle.get("family")
        if family == "binary_runtime" and operation in RUNTIME_BINARY:
            selected.append(oracle)
        elif family == "binary_constant" and parameters.get("constant") == "scalar":
            selected.append(oracle)
        elif family == "unary" and operation in UNARY:
            selected.append(oracle)
    return selected


def task_words(task: dict[str, Any]) -> list[int]:
    """Rebuild one task's exact word stream from its decoded records."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        header = int(record["header"], 16)
        base = (header & 0x7FFF) * 4
        if header & 0x80000000:
            mask = (header >> 15) & 0xFFFF
            addresses = (base,) + tuple(
                base + (bit + 1) * 4 for bit in range(16) if mask & (1 << bit))
        else:
            addresses = tuple(
                base + index * 4 for index in range(((header >> 15) & 0x3F) + 1))
        if len(addresses) != record["count"]:
            raise ValueError(f"record {record['header']} address count differs")
        words.append(header)
        words.extend(registers[address] for address in addresses)
    stream = struct.pack(f"<{len(words)}I", *words)
    if len(stream) != task["size_bytes"] or decode_task(stream, "h14") != task:
        raise ValueError("rebuilt task does not decode back to the oracle")
    return words


def task_stream(oracle: dict[str, Any]) -> list[int]:
    """Assemble the __TEXT/__text word stream for one oracle case."""
    words = [0] * (STREAM_PREFIX_BYTES // 4)
    tasks = oracle["task_descriptors"]
    for index, task in enumerate(tasks):
        words.extend(task_words(task))
        if index + 1 != len(tasks):
            while len(words) % (TASK_ALIGNMENT // 4):
                words.append(0)
    declared = oracle["program_descriptor"]["text_words"]
    if len(words) != declared:
        raise ValueError(
            f"{oracle['case']}: assembled {len(words)} text words, program "
            f"descriptor declares {declared}")
    return words


def constant_runs(oracle: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Nonzero fp16 halfword runs (index, bits, count) of the constant section."""
    section = oracle["constant_section"]
    halfwords: dict[int, int] = {}
    if section["nonzero_bytes"]:
        words = section["nonzero_fp16_words"]
        if words is None:
            # Oracles above 256 bytes keep only hashes; the scalar real_div
            # divisor reciprocal splat is the one such parity case.
            elements = oracle["parameters"]["shape"][1]
            for index in range(elements):
                halfwords[elements + index] = 0x4000
        else:
            for word in words:
                value = word["value"]
                if value == "nan":
                    bits = 0x7E00
                elif value == "+inf":
                    bits = 0x7C00
                elif value == "-inf":
                    bits = 0xFC00
                else:
                    bits = struct.unpack("<H", struct.pack("<e", value))[0]
                halfwords[word["index"]] = bits
    data = bytearray(section["size"])
    for index, bits in halfwords.items():
        struct.pack_into("<H", data, index * 2, bits)
    digest = hashlib.sha256(data).hexdigest()
    if digest != section["sha256"]:
        raise ValueError(
            f"{oracle['case']}: rebuilt constants hash {digest} but the oracle "
            f"records {section['sha256']}")
    runs: list[tuple[int, int, int]] = []
    for index in sorted(halfwords):
        bits = halfwords[index]
        if runs and runs[-1][1] == bits and runs[-1][0] + runs[-1][2] == index:
            runs[-1] = (runs[-1][0], bits, runs[-1][2] + 1)
        else:
            runs.append((index, bits, 1))
    return runs


def word_rows(words: list[int], indent: str = "    ") -> str:
    rows = []
    for start in range(0, len(words), 8):
        row = ", ".join(f"0x{word:08x}" for word in words[start:start + 8])
        rows.append(f"{indent}{row},")
    return "\n".join(rows)


def generate() -> str:
    oracles = selected_oracles()
    lines = ["// Generated from decoded H14 oracle task words by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, oracle in enumerate(oracles):
        parameters = oracle["parameters"]
        operation = parameters["operation"]
        family = oracle["family"]
        _, channels, height, width = parameters["shape"]
        code = (UNARY_OPERATIONS if family == "unary" else BINARY_OPERATIONS)[operation]
        stream = task_stream(oracle)
        runs = constant_runs(oracle)
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t kH14Text{index}[] = {{")
        lines.append(word_rows(stream))
        lines.append("};")
        constants = "nullptr, 0"
        if runs:
            lines.append(f"static constexpr ConstantRun kH14Constants{index}[] = {{")
            for start, bits, count in runs:
                lines.append(f"    {{{start}, 0x{bits:04x}, {count}}},")
            lines.append("};")
            constants = (f"kH14Constants{index}, "
                         f"std::size(kH14Constants{index})")
        lines.append("")
        trailer = oracle["program_descriptor"]["trailing_words"]
        entries.append(
            f"    {{ElementwiseKind::{KINDS[family]}, {code}, "
            f"{{{channels}, {height}, {width}}}, kH14Text{index}, "
            f"std::size(kH14Text{index}), "
            f"{oracle['program_descriptor']['task_count']}, "
            f"{constants}, {oracle['constant_section']['size']}, "
            f"0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}}},")
    lines.append("static constexpr OracleTaskTemplate kElementwiseTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def selected_matvec_oracles() -> list[dict[str, Any]]:
    """The H14 `transpose_y=true` matmul geometries Apple decoded, ordered by
    rows, reduction, then columns."""
    selected = []
    for path in sorted(ORACLES.glob("matmul_m*_ty1.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is None and oracle["family"] == "matmul":
            selected.append(oracle)
    selected.sort(key=lambda item: (item["parameters"]["rows"],
                                    item["parameters"]["reduction"],
                                    item["parameters"]["columns"]))
    return selected


def generate_matvec() -> str:
    oracles = selected_matvec_oracles()
    lines = ["// Generated from decoded H14 matmul oracle task words by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, oracle in enumerate(oracles):
        parameters = oracle["parameters"]
        descriptor = oracle["program_descriptor"]
        rows = parameters["rows"]
        reduction = parameters["reduction"]
        columns = parameters["columns"]
        stream = task_stream(oracle)
        constant_offset = (int(descriptor["constant_address"], 16) -
                           int(descriptor["text_address"], 16))
        aligned = (len(stream) * 4 + CONSTANT_ALIGNMENT - 1) & \
            ~(CONSTANT_ALIGNMENT - 1)
        if constant_offset != aligned:
            raise ValueError(
                f"{oracle['case']}: constants sit at {constant_offset}, not "
                f"the {aligned} the aligned text stream implies")
        if oracle["constant_section"]["size"] != reduction * columns * 2:
            raise ValueError(
                f"{oracle['case']}: constant section is not K*N halfwords")
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t kH14MatvecText{index}[] = {{")
        lines.append(word_rows(stream))
        lines.append("};")
        lines.append("")
        trailer = descriptor["trailing_words"]
        entries.append(
            f"    {{{rows}, {reduction}, {columns}, kH14MatvecText{index}, "
            f"std::size(kH14MatvecText{index}), {descriptor['task_count']}, "
            f"0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}}},")
    lines.append("static constexpr OracleMatvecTemplate kMatvecTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail when a checked-in file is stale")
    arguments = parser.parse_args()
    for path, generated in ((OUTPUT, generate()),
                            (MATVEC_OUTPUT, generate_matvec())):
        if arguments.check:
            if (path.read_text() if path.exists() else "") != generated:
                raise SystemExit(f"{path} is stale; regenerate it")
            print(f"H14 templates: up to date ({path})")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated)
        print(f"wrote {path} ({len(generated)} bytes)")


if __name__ == "__main__":
    main()
