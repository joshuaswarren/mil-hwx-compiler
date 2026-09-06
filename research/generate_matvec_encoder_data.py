#!/usr/bin/env python3
"""Emit plugins/H13/H13MatvecTemplates.inc from the decoded H13 matmul oracles.

Each record contributes the exact task stream Apple emitted: the ten-word task
header, every register record in order, and the zero padding the linked task
array requires. No HWX container bytes are read; the oracle JSON already holds
the decoded words.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from h13_td import decode_task  # noqa: E402

HEADER_WORDS = 10
CONSTANT_ALIGNMENT = 0x40
SCRATCH_ALIGNMENT = 0x4000


def task_words(task: dict) -> list[int]:
    """Rebuild one task descriptor's words from its decoded form."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        base = int(record["address"], 16)
        words.append(int(record["header"], 16))
        words.extend(registers[base + index * 4] for index in range(record["count"]))
    if len(words) * 4 != task["size_bytes"]:
        raise ValueError("rebuilt task size differs from the decoded size")
    return words


def stream_words(record: dict) -> list[int]:
    """Rebuild the whole linked task array, including alignment padding."""
    tasks = record["task_descriptors"]
    words: list[int] = []
    for index, task in enumerate(tasks):
        rebuilt = task_words(task)
        if decode_task(struct.pack(f"<{len(rebuilt)}I", *rebuilt), "h13") != task:
            raise ValueError(f"{record['case']} task {index} does not round-trip")
        words.extend(rebuilt)
        next_offset = int(task["header_words"][7], 16)
        if index + 1 == len(tasks):
            if next_offset:
                raise ValueError(f"{record['case']} final task links onward")
            continue
        if next_offset % 4 or next_offset < len(words) * 4:
            raise ValueError(f"{record['case']} task {index} link is invalid")
        words.extend([0] * (next_offset // 4 - len(words)))
    return words


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def scratch_bytes(record: dict) -> int:
    """Apple's __DATA/__bss allocation, which shifts every surface address."""
    text = int(record["program_descriptor"]["text_address"], 16)
    inputs, outputs = record["tensor_descriptors"][:2]
    return (text - 0x30000000 - align(inputs["total_bytes"], SCRATCH_ALIGNMENT)
            - align(outputs["total_bytes"], SCRATCH_ALIGNMENT))


def selected(target: str) -> list[dict]:
    records = []
    for path in sorted((ROOT / "research/oracles" / target).glob("matmul_m*_ty1.json")):
        record = json.loads(path.read_text())
        if record.get("error") is None and record["family"] == "matmul":
            records.append(record)
    records.sort(key=lambda item: (item["parameters"]["rows"],
                                   item["parameters"]["reduction"],
                                   item["parameters"]["columns"]))
    return records


def emit(records: list[dict], out) -> None:
    print("// Generated from decoded H13 matmul oracle task words by "
          "research/generate_matvec_encoder_data.py.", file=out)
    print("// No HWX container bytes; regenerate after re-minting the oracles.",
          file=out)
    rows = []
    for index, record in enumerate(records):
        words = stream_words(record)
        parameters = record["parameters"]
        descriptor = record["program_descriptor"]
        first_task = (descriptor["task_words_minus_one"] + 1) * 4
        constant = int(descriptor["constant_address"], 16) - \
            int(descriptor["text_address"], 16)
        if constant != align(len(words) * 4, CONSTANT_ALIGNMENT):
            raise ValueError(f"{record['case']} constant offset is not stream-aligned")
        if record["constant_section"]["size"] != \
                parameters["reduction"] * parameters["columns"] * 2:
            raise ValueError(f"{record['case']} constant section is not K*N halfwords")
        symbol = f"kMatvecTask{index}"
        print(f"static constexpr std::uint32_t {symbol}[] = {{", file=out)
        for start in range(0, len(words), 8):
            row = ", ".join(f"0x{word:08x}" for word in words[start:start + 8])
            print(f"    {row},", file=out)
        print("};", file=out)
        rows.append(
            f"    {{{parameters['rows']}, {parameters['reduction']}, "
            f"{parameters['columns']}, {symbol}, std::size({symbol}), "
            f"{first_task}, {descriptor['task_count']}, {constant}, "
            f"{scratch_bytes(record)}}},")
    print("static constexpr OracleMatvecTemplate kMatvecTasks[] = {", file=out)
    for row in rows:
        print(row, file=out)
    print("};", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="h13")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "plugins/H13/H13MatvecTemplates.inc")
    arguments = parser.parse_args()
    records = selected(arguments.target)
    if not records:
        raise SystemExit("no decoded matmul oracles found")
    with arguments.output.open("w") as out:
        emit(records, out)
    print(f"{arguments.output}: {len(records)} matvec templates")


if __name__ == "__main__":
    main()
