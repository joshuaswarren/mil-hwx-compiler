#!/usr/bin/env python3
"""Print TD, KERN and relocation metadata for convolution references."""

import glob
import hashlib
import os
import struct
import sys


def metadata(path: str):
    data = open(path, "rb").read()
    command_count = struct.unpack_from("<I", data, 16)[0]
    cursor = 32
    text_size = kernel_size = relocation_offset = relocation_count = 0
    text_offset = kernel_offset = 0
    for _ in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            section_count = struct.unpack_from("<I", data, cursor + 64)[0]
            for section_index in range(section_count):
                section = cursor + 72 + section_index * 80
                name = data[section : section + 16].split(b"\0", 1)[0]
                size = struct.unpack_from("<Q", data, section + 40)[0]
                if name == b"__text":
                    text_size = size
                    text_offset = struct.unpack_from("<I", data, section + 48)[0]
                    relocation_offset, relocation_count = struct.unpack_from(
                        "<2I", data, section + 56)
                elif name == b"__kern_0":
                    kernel_size = size
                    kernel_offset = struct.unpack_from("<I", data, section + 48)[0]
        cursor += command_size
    relocations = [
        struct.unpack_from("<I", data, relocation_offset + index * 8)[0]
        for index in range(relocation_count)
    ]
    return (text_size, kernel_size, relocations,
            hashlib.sha256(data[text_offset:text_offset+text_size]).hexdigest(),
            hashlib.sha256(data[kernel_offset:kernel_offset+kernel_size]).hexdigest())


def main(root: str) -> None:
    patterns = ["cw_3x3_c*_s*.hwx", "cw_5x5_c*_s*.hwx"]
    paths = sorted(path for pattern in patterns
                   for path in glob.glob(os.path.join(root, pattern)))
    for path in paths:
        text_size, kernel_size, relocations, text_hash, kernel_hash = metadata(path)
        print(
            os.path.basename(path), f"text=0x{text_size:x}",
            f"kern=0x{kernel_size:x}",
            "relocs=" + ",".join(f"0x{value:x}" for value in relocations),
            f"text_sha256={text_hash}", f"kern_sha256={kernel_hash}"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} REF_DIRECTORY")
    main(sys.argv[1])
