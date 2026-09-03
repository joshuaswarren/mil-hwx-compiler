#!/usr/bin/env python3
"""Compare Apple-compiled chain oracles against their single-operation parts.

For every HWX in the directory the script prints the program descriptor
summary (task count, record count, format, scratch size), the tensor
descriptors, and the non-zero descriptor words. For every `chain:a+b` pair
given on the command line it prints a word-level alignment of the chain's
task stream against the concatenation of the two single-operation streams so
the bridge fields stand out.
"""

from __future__ import annotations

import pathlib
import struct
import sys


def commands(data: bytes):
    _, _, _, _, command_count, _, _, _ = struct.unpack_from("<8I", data, 0)
    cursor = 32
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        yield command, command_size, cursor
        cursor += command_size


def sections(data: bytes):
    result = []
    for command, command_size, cursor in commands(data):
        if command != 0x19:
            continue
        section_count = struct.unpack_from("<I", data, cursor + 64)[0]
        section_cursor = cursor + 72
        for _ in range(section_count):
            fields = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
            name = fields[0].split(b"\0", 1)[0].decode()
            segment = fields[1].split(b"\0", 1)[0].decode()
            contents = data[fields[4]:fields[4] + fields[3]]
            result.append((segment, name, fields[2], fields[3], contents))
            section_cursor += 80
    return result


def text_words(data: bytes):
    for segment, name, _, _, contents in sections(data):
        if segment == "__TEXT" and name == "__text":
            return list(struct.unpack(f"<{len(contents) // 4}I", contents))
    raise ValueError("no __TEXT/__text")


def program_info(data: bytes):
    for command, command_size, cursor in commands(data):
        kind = struct.unpack_from("<I", data, cursor + 8)[0]
        if command == 4 and kind == 4 and command_size >= 0x898:
            task_count = struct.unpack_from("<I", data, cursor + 0x830)[0]
            records = struct.unpack_from("<I", data, cursor + 0x860)[0]
            scratch = struct.unpack_from("<Q", data, cursor + 0x40)[0]
            layout = struct.unpack_from("<I", data, cursor + 0xc)[0]
            pointers = [struct.unpack_from("<Q", data, cursor + off)[0]
                        for off in range(0x70, 0xc0, 0x10)]
            return dict(size=command_size, tasks=task_count, records=records,
                        scratch=scratch, layout=layout, pointers=pointers)
    return None


def tensor_descriptors(data: bytes):
    result = []
    for command, command_size, cursor in commands(data):
        kind = struct.unpack_from("<I", data, cursor + 8)[0]
        if command == 4 and kind == 3 and command_size >= 0x80:
            binding = struct.unpack_from("<I", data, cursor + 0x14)[0]
            shape = struct.unpack_from("<4I", data, cursor + 0x28)
            strides = struct.unpack_from("<4Q", data, cursor + 0x50)
            symbol_offset = struct.unpack_from("<I", data, cursor + 0x20)[0]
            symbol = data[cursor + symbol_offset:cursor + symbol_offset + 64]
            symbol = symbol.split(b"\0", 1)[0].decode(errors="replace")
            result.append((binding, symbol, shape, strides))
    return result


def scratch_segment(data: bytes):
    for segment, name, address, size, _ in sections(data):
        if segment == "__DATA" and name == "__bss":
            return address, size
    return None


def describe(path: pathlib.Path, dump: bool):
    data = path.read_bytes()
    info = program_info(data)
    words = text_words(data)
    print(f"ORACLE {path.stem}: text_words={len(words)} "
          f"tasks={info['tasks'] if info else '?'} "
          f"records={info['records'] if info else '?'} "
          f"descriptor_bytes={info['size'] if info else '?'} "
          f"layout=0x{info['layout']:x} " if info else "",
          end="")
    scratch = scratch_segment(data)
    print(f"scratch={'0x%x' % scratch[1] if scratch else 'none'}")
    for binding, symbol, shape, strides in tensor_descriptors(data):
        print(f"  tensor binding={binding} symbol={symbol} shape={shape} "
              f"strides={strides}")
    if dump:
        for index, value in enumerate(words):
            if value:
                print(f"  0x{index * 4:04x}: 0x{value:08x}")


def task_blocks(words):
    """Split a task stream into per-task blocks using the header words.

    A task block starts with (packet_words << 16 | task_index) at offset 0x10
    of the first block; subsequent blocks begin at the position after the
    previous block's packet words. This is heuristic and reported as such.
    """
    blocks = []
    cursor = 4  # the 0x10-byte program start header
    while cursor < len(words):
        header = words[cursor]
        packet_words = header >> 16
        if packet_words == 0:
            break
        blocks.append((cursor, packet_words, header & 0xffff))
        cursor += packet_words + 1
    return blocks


def diff_words(label, left, right):
    changes = []
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a != b:
            changes.append((index, a, b))
    print(f"{label}: left_words={len(left)} right_words={len(right)} "
          f"changes={len(changes)}")
    for index, a, b in changes:
        left_text = f"0x{a:08x}" if a is not None else "   (none)"
        right_text = f"0x{b:08x}" if b is not None else "   (none)"
        print(f"  0x{index * 4:04x}: {left_text} -> {right_text}")


def align_words(label, left, right, boundaries):
    import difflib
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    opcodes = matcher.get_opcodes()
    changed = sum(1 for op in opcodes if op[0] != "equal")
    print(f"{label}: left_words={len(left)} right_words={len(right)} "
          f"edit_groups={changed}")
    for offset, name in boundaries:
        print(f"    part {name} starts at left 0x{offset * 4:04x}")
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            print(f"    equal left 0x{i1 * 4:04x}..0x{i2 * 4:04x} "
                  f"right 0x{j1 * 4:04x}..0x{j2 * 4:04x} ({i2 - i1} words)")
            continue
        print(f"    {tag} left 0x{i1 * 4:04x}..0x{i2 * 4:04x} "
              f"right 0x{j1 * 4:04x}..0x{j2 * 4:04x}")
        width = max(i2 - i1, j2 - j1)
        for k in range(width):
            a = left[i1 + k] if i1 + k < i2 else None
            b = right[j1 + k] if j1 + k < j2 else None
            left_text = f"0x{a:08x}" if a is not None else "   (none)"
            right_text = f"0x{b:08x}" if b is not None else "   (none)"
            left_offset = f"0x{(i1 + k) * 4:04x}" if a is not None else "      "
            right_offset = f"0x{(j1 + k) * 4:04x}" if b is not None else "      "
            print(f"      {left_offset} {left_text} -> {right_offset} {right_text}")


def main(argv):
    root = pathlib.Path(argv[1])
    dump = "--dump" in argv
    pairs = [arg for arg in argv[2:] if ":" in arg]
    for path in sorted(root.glob("*.hwx")):
        describe(path, dump)
    for spec in pairs:
        chain, parts = spec.split(":", 1)
        part_names = parts.split("+")
        chain_path = root / f"{chain}.hwx"
        part_paths = [root / f"{name}.hwx" for name in part_names]
        if not chain_path.exists() or not all(p.exists() for p in part_paths):
            print(f"PAIR {spec}: missing oracle")
            continue
        chain_words = text_words(chain_path.read_bytes())
        print(f"PAIR {spec}: chain blocks={task_blocks(chain_words)}")
        for part_path in part_paths:
            part_words = text_words(part_path.read_bytes())
            print(f"  part {part_path.stem}: blocks={task_blocks(part_words)}")
        # Sequence-align the chain stream against the concatenated part
        # streams (the parts' 4-word program headers dropped after the first)
        # so insertions, deletions, and replacements are reported in place.
        concatenated = []
        boundaries = []
        for index, part_path in enumerate(part_paths):
            part_words = text_words(part_path.read_bytes())
            start = 0 if index == 0 else 4
            boundaries.append((len(concatenated), part_path.stem))
            concatenated.extend(part_words[start:])
        align_words(f"  align {'+'.join(part_names)} -> {chain}",
                    concatenated, chain_words, boundaries)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} ORACLE_DIR [--dump] "
                         "[chain:partA+partB ...]")
    main(sys.argv)
