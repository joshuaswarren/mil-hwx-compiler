#!/usr/bin/env python3
"""Recover channel and tap ordering from focused depthwise oracle HWX files."""

import struct
import sys


def main(path: str, channels: int, pattern: str) -> None:
    image = open(path, "rb").read()
    packet_bytes = channels + 64
    kernel = image[0xC000 : 0xC000 + 16 * packet_bytes]
    for packet in range(16):
        values = struct.unpack_from(
            f"<{packet_bytes // 2}e", kernel, packet * packet_bytes)
        if pattern == "channel":
            nonzero = [int(value) - 1 for value in values if value != 0]
            decoded = [nonzero[index] for index in range(0, len(nonzero), 9)]
        else:
            decoded = [int(value) for value in values if value != 0]
        print(f"packet={packet:02d} {decoded}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} ORACLE.hwx CHANNELS channel|tap")
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
