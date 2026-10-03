#!/usr/bin/env python3
"""Decode H13 through H18 ANE task-descriptor register streams."""
from __future__ import annotations

import struct
from typing import Any, Iterable

# Fixed header words before the first register record. H14 and H15 carry
# eight; H16, H17 and H18 carry nine (the reference parser's
# `i = 8 if subtype in (5, 6) else 9`), which every decoded h16-h18 oracle
# confirms by decoding with no unknown register address.
HEADER_WORDS = {"h13": 10, "h14": 8, "h15": 8, "h16": 9, "h17": 9, "h18": 9}

# H15 and later move every block above 0x4000 except `common`. Bases and
# counts for H16, H17 and H18 are the reference parser's
# (freedomtan/coreml_to_ane_hwx ce54664e hwx_dump/ane_hwx_regs.h, BSD-3-Clause).
# For H15 that parser reuses the H14 counts, but Apple's h15 objects write
# tile_dma_dst word 14 (0x5138) and kernel_dma_src words 70-71 (0x5618,
# 0x561c), so those two H15 counts are widened to the words observed.
BLOCKS = {
    "h13": (
        ("common", 0x00000, 16),
        ("l2", 0x04800, 18),
        ("pe", 0x08800, 4),
        ("ne", 0x0C800, 5),
        ("tile_dma_src", 0x13800, 28),
        ("tile_dma_dst", 0x17800, 7),
        ("kernel_dma_src", 0x1F800, 62),
    ),
    "h14": (
        ("common", 0x0000, 19),
        ("l2", 0x0500, 25),
        ("pe", 0x0900, 5),
        ("ne", 0x0D00, 5),
        ("tile_dma_src", 0x1100, 53),
        ("tile_dma_dst", 0x1500, 10),
        ("kernel_dma_src", 0x1900, 70),
    ),
    "h15": (
        ("common", 0x0000, 19),
        ("l2", 0x4100, 25),
        ("pe", 0x4500, 15),
        ("ne", 0x4900, 5),
        ("tile_dma_src", 0x4D00, 53),
        ("tile_dma_dst", 0x5100, 15),
        ("kernel_dma_src", 0x5500, 72),
        ("cache_dma", 0x5900, 12),
    ),
    "h16": (
        ("common", 0x0000, 23),
        ("l2", 0x4100, 41),
        ("pe_index", 0x44D0, 1),
        ("pe", 0x4500, 15),
        ("ne", 0x4900, 12),
        ("tile_dma_src", 0x4D00, 81),
        ("tile_dma_dst", 0x5100, 21),
        ("kernel_dma_src", 0x5500, 72),
        ("cache_dma", 0x5900, 12),
    ),
    "h17": (
        ("common", 0x0000, 23),
        ("l2", 0x4100, 42),
        ("pe", 0x4500, 16),
        ("ne", 0x4900, 13),
        ("tile_dma_src", 0x4D00, 83),
        ("tile_dma_dst", 0x5100, 23),
        ("kernel_dma_src", 0x5500, 74),
        ("cache_dma", 0x5900, 14),
    ),
    "h18": (
        ("common", 0x0000, 23),
        ("l2", 0x4100, 43),
        ("pe", 0x4500, 16),
        ("ne", 0x4900, 13),
        ("tile_dma_src", 0x4D00, 81),
        ("tile_dma_dst", 0x5100, 27),
        ("kernel_dma_src", 0x5500, 83),
        ("cache_dma", 0x5900, 14),
    ),
}


def _words(data: bytes, label: str) -> tuple[int, ...]:
    if len(data) % 4:
        raise ValueError(f"{label} size 0x{len(data):x} is not word-aligned")
    return struct.unpack(f"<{len(data) // 4}I", data)


def _block(target: str, address: int) -> tuple[str, int] | None:
    for name, start, count in BLOCKS[target]:
        if start <= address < start + count * 4:
            return name, start
    return None


def _extra_header_words(words: tuple[int, ...], header_count: int) -> int:
    """Low bits 0b11 of the last fixed header word announce one extra word before the records.

    Corpus values: H13 header[9] in {0x0, 0x21, 0x23, 0x26}; H14 header[7] in
    {0x1, 0x10001, 0x30001, 0x40001, 0x50001, 0x50003}. Only 0x23 and 0x50003
    carry the extra word (observed 0x0 or 0x7), so the predicate is both low
    bits set, not bit 1 alone. The h15-h18 corpus never sets both bits: its
    last header word is one of 0x1, 0x9, 0x30001, 0x30009, 0x40001, 0x40009,
    0x50001, 0x50009.
    """
    return 1 if len(words) >= header_count and words[header_count - 1] & 0x3 == 0x3 else 0

def _h13_records(words: tuple[int, ...]) -> list[dict[str, Any]]:
    index = HEADER_WORDS["h13"] + _extra_header_words(words, HEADER_WORDS["h13"])
    records = []
    while index < len(words):
        header = words[index]
        count = (header >> 26) + 1
        address = header & 0x03FFFFFF
        if address % 4:
            raise ValueError(
                f"H13 record[{len(records)}] has unaligned address 0x{address:x}")
        if count > len(words) - index - 1:
            raise ValueError(f"H13 record[{len(records)}] values are truncated")
        addresses = tuple(address + offset * 4 for offset in range(count))
        values = words[index + 1:index + 1 + count]
        records.append({"header": header, "addresses": addresses, "words": values})
        index += count + 1
    return records


def _packed_records(words: tuple[int, ...], target: str) -> list[dict[str, Any]]:
    """H14-format records, shared by H14 through H18."""
    header_count = HEADER_WORDS[target]
    index = header_count + _extra_header_words(words, header_count)
    records = []
    while index < len(words):
        header = words[index]
        base = (header & 0x7FFF) * 4
        if header & 0x80000000:
            mask = (header >> 15) & 0xFFFF
            addresses = (base,) + tuple(
                base + (bit + 1) * 4 for bit in range(16) if mask & (1 << bit))
        else:
            count = ((header >> 15) & 0x3F) + 1
            addresses = tuple(base + offset * 4 for offset in range(count))
        if len(addresses) > len(words) - index - 1:
            raise ValueError(
                f"{target.upper()} record[{len(records)}] values are truncated")
        values = words[index + 1:index + 1 + len(addresses)]
        records.append({"header": header, "addresses": addresses, "words": values})
        index += len(addresses) + 1
    return records


def decode_task(data: bytes, target: str) -> dict[str, Any]:
    """Decode one task descriptor into header, records, and register blocks."""
    target = target.lower()
    if target not in BLOCKS:
        raise ValueError(f"unsupported task descriptor target {target!r}")
    words = _words(data, f"{target.upper()} task descriptor")
    header_count = HEADER_WORDS[target]
    if len(words) < header_count:
        raise ValueError(f"{target.upper()} task descriptor header is truncated")
    records = _h13_records(words) if target == "h13" else _packed_records(words, target)
    blocks: dict[str, dict[str, Any]] = {}
    unknown = 0
    for record in records:
        for address, value in zip(record["addresses"], record["words"]):
            known = _block(target, address)
            if known is None:
                key = f"0x{address:05x}"
                name = f"unknown_{unknown}"
                unknown += 1
            else:
                name, start = known
                key = f"0x{start:05x}"
            block = blocks.setdefault(key, {"name": name, "words": {}})
            block["words"][f"0x{address:05x}"] = f"0x{value:08x}"
    return {
        "size_bytes": len(data),
        "header_words": [f"0x{word:08x}" for word in
                         words[:header_count + _extra_header_words(words, header_count)]],
        "records": [
            {
                "header": f"0x{record['header']:08x}",
                "address": f"0x{record['addresses'][0]:05x}",
                "count": len(record["words"]),
            }
            for record in records
        ],
        "blocks": blocks,
    }


def split_h13_tasks(
        section: bytes, task_words_minus_one: int, task_count: int
) -> list[bytes]:
    """Follow an H13 linked task array using its program and task headers."""
    if task_count <= 0:
        raise ValueError("H13 program descriptor has no tasks")
    tasks = []
    offset = 0
    task_words = task_words_minus_one + 1
    for task_index in range(task_count):
        task_bytes = task_words * 4
        if task_bytes < HEADER_WORDS["h13"] * 4:
            raise ValueError(f"H13 task[{task_index}] size 0x{task_bytes:x} is too small")
        if task_bytes > len(section) - offset:
            raise ValueError(f"H13 task[{task_index}] extends beyond __TEXT/__text")
        task = section[offset:offset + task_bytes]
        tasks.append(task)
        words = _words(task, f"H13 task[{task_index}]")
        next_offset = words[7]
        if task_index + 1 == task_count:
            if next_offset:
                raise ValueError(f"H13 final task points to 0x{next_offset:x}")
            if offset + task_bytes != len(section):
                raise ValueError(
                    f"H13 tasks consume 0x{offset + task_bytes:x} bytes but "
                    f"__TEXT/__text is 0x{len(section):x} bytes")
            continue
        if next_offset < offset + task_bytes or next_offset > len(section):
            raise ValueError(
                f"H13 task[{task_index}] has invalid next pointer 0x{next_offset:x}")
        if any(section[offset + task_bytes:next_offset]):
            raise ValueError(f"H13 task[{task_index}] has nonzero alignment padding")
        task_words = ((words[1] >> 16) & 0x1FF) + 1
        offset = next_offset
    return tasks


def split_h14_tasks(section: bytes) -> list[bytes]:
    """Split an H14-format (H14 through H18) aligned task array, skipping
    zero-size 16-byte headers."""
    tasks = []
    offset = 0
    while offset < len(section):
        if len(section) - offset < 4:
            if any(section[offset:]):
                raise ValueError("H14 trailing alignment bytes are nonzero")
            break
        task_words = struct.unpack_from("<H", section, offset + 2)[0] & 0x7FF
        if task_words == 0:
            offset = min(offset + 16, len(section))
            continue
        task_bytes = task_words * 4
        if task_words < HEADER_WORDS["h14"]:
            raise ValueError(f"H14 task[{len(tasks)}] has invalid size {task_words} words")
        if task_bytes > len(section) - offset:
            raise ValueError(
                f"H14 task[{len(tasks)}] declares {task_words} words beyond "
                "__TEXT/__text")
        tasks.append(section[offset:offset + task_bytes])
        next_offset = min((offset + task_bytes + 15) & ~15, len(section))
        if any(section[offset + task_bytes:next_offset]):
            raise ValueError(f"H14 task[{len(tasks) - 1}] has nonzero alignment padding")
        offset = next_offset
    return tasks


def register_words(task: dict[str, Any]) -> dict[int, int]:
    """Flatten decoded blocks to integer register addresses and values."""
    return {
        int(address, 16): int(value, 16)
        for block in task["blocks"].values()
        for address, value in block["words"].items()
    }


def diff_tasks(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exact header and register differences between decoded tasks."""
    differences = []
    left_header = left["header_words"]
    right_header = right["header_words"]
    for index in range(max(len(left_header), len(right_header))):
        a = left_header[index] if index < len(left_header) else None
        b = right_header[index] if index < len(right_header) else None
        if a != b:
            differences.append({"kind": "header", "index": index, "left": a, "right": b})
    left_words = register_words(left)
    right_words = register_words(right)
    for address in sorted(set(left_words) | set(right_words)):
        a = left_words.get(address)
        b = right_words.get(address)
        if a != b:
            differences.append({
                "kind": "register", "address": f"0x{address:05x}",
                "left": None if a is None else f"0x{a:08x}",
                "right": None if b is None else f"0x{b:08x}",
            })
    return differences


def varying_words(tasks: Iterable[dict[str, Any]]) -> dict[str, list[str | None]]:
    """Return header/register locations whose values differ across tasks."""
    tasks = list(tasks)
    if not tasks:
        return {}
    values: dict[str, list[str | None]] = {}
    header_count = max(len(task["header_words"]) for task in tasks)
    for index in range(header_count):
        key = f"header[{index}]"
        values[key] = [task["header_words"][index]
                       if index < len(task["header_words"]) else None
                       for task in tasks]
    addresses = sorted(set().union(*(register_words(task) for task in tasks)))
    flattened = [register_words(task) for task in tasks]
    for address in addresses:
        values[f"0x{address:05x}"] = [
            None if address not in words else f"0x{words[address]:08x}"
            for words in flattened]
    return {key: observed for key, observed in values.items()
            if len(set(observed)) > 1}
