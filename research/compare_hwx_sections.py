#!/usr/bin/env python3
"""Compare named Mach-O sections in HWX reference objects by 32-bit word."""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def sections(data: bytes) -> dict[tuple[str, str], bytes]:
    result: dict[tuple[str, str], bytes] = {}
    command_count = struct.unpack_from("<I", data, 0x10)[0]
    cursor = 0x20
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<II", data, cursor)
        if command == 0x19:
            segment = cstring(data[cursor + 8 : cursor + 24])
            section_count = struct.unpack_from("<I", data, cursor + 0x40)[0]
            section_cursor = cursor + 0x48
            for _ in range(section_count):
                section = cstring(data[section_cursor : section_cursor + 16])
                size = struct.unpack_from("<Q", data, section_cursor + 0x28)[0]
                offset = struct.unpack_from("<I", data, section_cursor + 0x30)[0]
                result[(segment, section)] = data[offset : offset + size]
                section_cursor += 0x50
        cursor += command_size
    return result


def words(data: bytes) -> list[int]:
    return list(struct.unpack(f"<{len(data) // 4}I", data))


def main(paths: list[str]) -> None:
    if len(paths) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} SEGMENT SECTION FILE...")
    segment, section, *files = paths
    rows: list[tuple[Path, bytes]] = []
    for raw_path in files:
        path = Path(raw_path)
        data = sections(path.read_bytes()).get((segment, section))
        if data is None:
            print(f"{path.name}: missing {segment}/{section}")
            continue
        print(
            f"{path.name}: bytes=0x{len(data):x} words={len(data)//4} "
            f"sha256={hashlib.sha256(data).hexdigest()}"
        )
        rows.append((path, data))
    if not rows:
        return
    base_path, base_data = rows[0]
    base_words = words(base_data)
    for path, data in rows[1:]:
        current_words = words(data)
        print(f"\n{path.name} relative to {base_path.name}:")
        for index in range(max(len(base_words), len(current_words))):
            left = base_words[index] if index < len(base_words) else None
            right = current_words[index] if index < len(current_words) else None
            if left != right:
                left_text = "--------" if left is None else f"{left:08x}"
                right_text = "--------" if right is None else f"{right:08x}"
                print(f"  +0x{index * 4:03x}: {left_text} -> {right_text}")


if __name__ == "__main__":
    main(sys.argv[1:])
