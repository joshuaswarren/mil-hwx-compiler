#!/usr/bin/env python3
"""Extract decoded matmul TD packet words from a reference HWX object."""

import struct
import sys


def text_words(path: str) -> list[int]:
    data = open(path, "rb").read()
    command_count = struct.unpack_from("<I", data, 16)[0]
    cursor = 32
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19 and data[cursor + 8:cursor + 24].rstrip(b"\0") == b"__TEXT":
            section = cursor + 72
            file_offset = struct.unpack_from("<I", data, section + 48)[0]
            byte_length = struct.unpack_from("<Q", data, section + 40)[0]
            return list(struct.unpack_from(
                f"<{byte_length // 4}I", data, file_offset))
        cursor += command_size
    raise ValueError("HWX object has no __TEXT.__text section")


def blocks(words: list[int]) -> list[list[int]]:
    ends = [i for i, word in enumerate(words)
            if word in (0x22001340, 0x22001440)]
    block_ends = ends[1::2]
    starts = [4] + [end + 2 for end in block_ends[:-1]]
    stops = [end + 2 for end in block_ends]
    return [words[start:stop] for start, stop in zip(starts, stops)]


def print_array(name: str, words: list[int]) -> None:
    print(f"static const uint32_t {name}[] = {{")
    for offset in range(0, len(words), 6):
        chunk = words[offset:offset + 6]
        print("    " + ", ".join(f"0x{word:08x}" for word in chunk) + ",")
    print("};")
    print(f"// {len(words)} words / 0x{len(words) * 4:x} bytes")


def main(path: str, mode: str) -> None:
    words = text_words(path)
    if mode == "single":
        print_array("kSingleTileWords", words)
        return
    split = blocks(words)
    if len(split) < 4:
        raise ValueError("reference does not contain a tiled matmul stream")
    print_array("kTiledHeader", words[:4])
    print_array("kTiledPrologue", split[0])
    print_array("kTiledFirst", split[1])
    print_array("kTiledMiddle", split[2])
    print_array("kTiledLast", split[-1])
    print(f"// decoded block count in reference: {len(split)}")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] not in ("single", "tiled"):
        raise SystemExit(f"usage: {sys.argv[0]} FILE.hwx single|tiled")
    main(sys.argv[1], sys.argv[2])
