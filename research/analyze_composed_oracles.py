#!/usr/bin/env python3
"""Report H16G descriptor words that vary across composed primitive oracles."""

from __future__ import annotations

import pathlib
import hashlib
import struct
import sys


def sections(data: bytes) -> list[tuple[str, str, bytes, int, int, int]]:
    _, _, _, _, command_count, _, _, _ = struct.unpack_from("<8I", data, 0)
    cursor = 32
    result = []
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            section_count = struct.unpack_from("<I", data, cursor + 64)[0]
            section_cursor = cursor + 72
            for _ in range(section_count):
                fields = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
                name = fields[0].split(b"\0", 1)[0]
                segment = fields[1].split(b"\0", 1)[0]
                contents = data[fields[4]:fields[4] + fields[3]]
                result.append((name.decode(), segment.decode(), contents,
                               fields[6], fields[7], fields[2]))
                section_cursor += 80
        cursor += command_size
    return result


def program_metadata(data: bytes) -> list[tuple[int, int, int, int, list[int]]]:
    _, _, _, _, command_count, _, _, _ = struct.unpack_from("<8I", data, 0)
    cursor = 32
    result = []
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        if command == 4 and command_size >= 0x894:
            pointers = [struct.unpack_from("<Q", data, cursor + offset)[0]
                        for offset in range(0x10, 0xb8, 0x10)]
            result.append((command_size,
                           struct.unpack_from("<I", data, cursor + 0x830)[0],
                           struct.unpack_from("<I", data, cursor + 0x860)[0],
                           struct.unpack_from("<I", data, cursor + 0x890)[0],
                           pointers))
        cursor += command_size
    return result


def text_section(data: bytes) -> bytes:
    for name, segment, contents, _, _, _ in sections(data):
        if name == "__text" and segment == "__TEXT":
            return contents
    raise ValueError("object has no __TEXT/__text section")


def words(path: pathlib.Path) -> tuple[int, ...]:
    text = text_section(path.read_bytes())
    return struct.unpack(f"<{len(text) // 4}I", text)


def main(root: pathlib.Path, dump: bool = False) -> None:
    paths = sorted(root.glob("*.hwx"))
    for path in paths:
        values = words(path)
        nonzero = [(index * 4, value) for index, value in enumerate(values) if value]
        print(f"{path.stem}: bytes={len(values) * 4} nonzero={len(nonzero)}")
        if dump:
            for command_size, task_count, record_count, format_code, pointers in program_metadata(path.read_bytes()):
                print(f"  program bytes={command_size} task_count={task_count} "
                      f"record_count={record_count} format={format_code}")
                if task_count:
                    print(f"    pointers={[hex(value) for value in pointers]}")
            for name, segment, contents, relocation_offset, relocation_count, address in sections(path.read_bytes()):
                digest = hashlib.sha256(contents).hexdigest()
                print(f"  section {segment}/{name}: bytes={len(contents)} "
                      f"address=0x{address:x} "
                      f"sha256={digest} reloff={relocation_offset} "
                      f"nreloc={relocation_count}")
                if relocation_count:
                    addresses = [struct.unpack_from("<i", path.read_bytes(),
                                 relocation_offset + index * 8)[0]
                                 for index in range(relocation_count)]
                    print(f"    relocation_addresses={addresses}")
                if name == "__const" and segment == "__TEXT" and len(contents) >= 0x894:
                    task_count = struct.unpack_from("<I", contents, 0x830)[0]
                    record_count = struct.unpack_from("<I", contents, 0x860)[0]
                    format_code = struct.unpack_from("<I", contents, 0x890)[0]
                    print(f"    program task_count={task_count} "
                          f"record_count={record_count} format={format_code}")
            for offset, value in nonzero:
                print(f"  0x{offset:04x}: 0x{value:08x}")
    stems = {path.stem: path for path in paths}
    for stem, path in sorted(stems.items()):
        if not stem.endswith("_n128"):
            continue
        other = stems.get(stem.removesuffix("128") + "256")
        if not other:
            continue
        left, right = words(path), words(other)
        changes = []
        for index in range(max(len(left), len(right))):
            a = left[index] if index < len(left) else None
            b = right[index] if index < len(right) else None
            if a != b:
                changes.append((index * 4, a, b))
        print(f"DIFF {stem} -> {other.stem}: {len(changes)} words")
        for offset, a, b in changes:
            print(f"  0x{offset:04x}: {a!s:>10} -> {b!s}")
    operation_pairs = [
        ("add_row_row", "mul_row_row"),
        ("add_row_row", "maximum_row_row"),
        ("sub_matrix_row", "mul_matrix_row"),
        ("mul_matrix_row", "real_div_matrix_row"),
    ]
    for size in (128, 256):
        for left_name, right_name in operation_pairs:
            left_path = stems.get(f"{left_name}_n{size}")
            right_path = stems.get(f"{right_name}_n{size}")
            if not left_path or not right_path:
                continue
            left, right = words(left_path), words(right_path)
            changes = [
                (index * 4, a, b)
                for index, (a, b) in enumerate(zip(left, right)) if a != b
            ]
            if len(left) != len(right):
                changes.append((-1, len(left), len(right)))
            print(
                f"OPDIFF {left_path.stem} -> {right_path.stem}: "
                f"{len(changes)} changes"
            )
            for offset, a, b in changes:
                print(f"  0x{offset:04x}: 0x{a:08x} -> 0x{b:08x}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit(f"usage: {sys.argv[0]} ORACLE_DIRECTORY [--dump]")
    main(pathlib.Path(sys.argv[1]), "--dump" in sys.argv[2:])
