#!/usr/bin/env python3
"""Print the leading packed coordinates from dense-convolution oracles."""

import struct
import sys


def main(path: str, count: int = 192) -> None:
    data = open(path, "rb").read()
    values = struct.unpack_from(f"<{count}e", data, 0xC000)
    print([int(value) - 1 for value in values])


if __name__ == "__main__":
    if len(sys.argv) not in (2,3):
        raise SystemExit(f"usage: {sys.argv[0]} ORACLE.hwx [COUNT]")
    main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)==3 else 192)
