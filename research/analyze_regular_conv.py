#!/usr/bin/env python3
"""Compare regular-convolution TD families and print field deltas."""

import hashlib
import os
import struct
import sys


def text(path: str) -> bytes:
    data = open(path, "rb").read()
    command_count = struct.unpack_from("<I", data, 16)[0]
    cursor = 32
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            section_count = struct.unpack_from("<I", data, cursor + 64)[0]
            for section_index in range(section_count):
                section = cursor + 72 + section_index * 80
                name = data[section : section + 16].split(b"\0", 1)[0]
                if name == b"__text":
                    size = struct.unpack_from("<Q", data, section + 40)[0]
                    offset = struct.unpack_from("<I", data, section + 48)[0]
                    return data[offset : offset + size]
        cursor += command_size
    raise ValueError(f"no __text in {path}")


def words(data: bytes):
    return struct.unpack(f"<{len(data) // 4}I", data)


def compare(label: str, left: bytes, right: bytes) -> None:
    a, b = words(left), words(right)
    print(label, f"lengths=0x{len(left):x}/0x{len(right):x}")
    for index in range(min(len(a), len(b))):
        if a[index] != b[index]:
            print(f"  +0x{index*4:03x}: 0x{a[index]:08x} -> 0x{b[index]:08x}")


def main(root: str) -> None:
    names = [
        "cw_3x3_c64_s32.hwx", "cw_3x3_c64_s64.hwx",
        "cw_5x5_c64_s64.hwx", "cw_3x3_c128_s32.hwx",
        "cw_3x3_c128_s64.hwx", "cw_5x5_c128_s64.hwx",
    ]
    streams = {name: text(os.path.join(root, name)) for name in names}
    for name, stream in streams.items():
        print(name, f"length=0x{len(stream):x}",
              f"sha256={hashlib.sha256(stream).hexdigest()}")
    compare("C64 S32 -> S64",streams[names[0]],streams[names[1]])
    compare("C64 3x3 -> 5x5",streams[names[1]],streams[names[2]])
    compare("C128 S32 -> S64",streams[names[3]],streams[names[4]])
    compare("C128 3x3 -> 5x5",streams[names[4]],streams[names[5]])
    for name in (names[1], names[4]):
        print("ARRAY",name)
        values = words(streams[name])
        for index in range(0,len(values),8):
            print("  " + ",".join(f"0x{value:08x}" for value in values[index:index+8]) + ",")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} REF_DIRECTORY")
    main(sys.argv[1])
