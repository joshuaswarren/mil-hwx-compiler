#!/usr/bin/env python3
"""Sweeps the decoded H13 matmul oracles and reports which words scale with M/K/N."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(target: str, pattern: str = "matmul_m*_ty1.json") -> list[dict]:
    records = []
    for path in sorted((ROOT / "research/oracles" / target).glob(pattern)):
        record = json.loads(path.read_text())
        if record.get("error") is None:
            records.append(record)
    return records


def key(record: dict) -> tuple[int, int, int]:
    parameters = record["parameters"]
    return parameters["rows"], parameters["reduction"], parameters["columns"]


def task_words(record: dict, index: int) -> dict[str, int]:
    task = record["task_descriptors"][index]
    words = {f"header[{position}]": int(value, 16)
             for position, value in enumerate(task["header_words"])}
    for block in task["blocks"].values():
        for address, value in block["words"].items():
            words[address] = int(value, 16)
    words["size_bytes"] = task["size_bytes"]
    return words


def report(records: list[dict], task_index: int) -> None:
    table = {key(record): task_words(record, task_index) for record in records
             if len(record["task_descriptors"]) > task_index}
    addresses = sorted({address for words in table.values() for address in words},
                       key=lambda text: (not text.startswith("header"), text))
    print(f"### task {task_index}: {len(table)} cases")
    invariant = []
    for address in addresses:
        values = {case: words.get(address) for case, words in table.items()}
        distinct = set(values.values())
        if len(distinct) == 1:
            invariant.append((address, distinct.pop()))
            continue
        depends = []
        for position, name in enumerate("MKN"):
            groups: dict[tuple, set] = {}
            for case, value in values.items():
                rest = tuple(v for i, v in enumerate(case) if i != position)
                groups.setdefault(rest, set()).add(value)
            if any(len(group) > 1 for group in groups.values()):
                depends.append(name)
        print(f"{address:>12} varies with {''.join(depends) or '?'}: " +
              " ".join(f"m{c[0]}k{c[1]}n{c[2]}=0x{v:08x}"
                       for c, v in sorted(values.items())))
    print(f"invariant words: {len(invariant)}")
    for address, value in invariant:
        print(f"{address:>12} = 0x{value:08x}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="h13")
    parser.add_argument("--pattern", default="matmul_m*_ty1.json")
    parser.add_argument("--task", type=int, action="append")
    arguments = parser.parse_args()
    records = load(arguments.target, arguments.pattern)
    print(f"{len(records)} decoded records; task counts: "
          + str(sorted({len(r['task_descriptors']) for r in records})))
    for task_index in arguments.task or (0, 1):
        report(records, task_index)


if __name__ == "__main__":
    main()
