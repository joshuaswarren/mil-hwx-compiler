#!/usr/bin/env python3
"""Emit decoded H16G reduction TD arrays from independently minted HWX refs."""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

from compare_hwx_sections import sections


NAME = re.compile(
    r"reduce_(sum|mean|max)_c(\d+)_h(\d+)_w(\d+)_a(\d+)\.hwx"
)


def symbol(operation: str, channels: str, height: str,
           width: str, axis: str) -> str:
    return (
        f"{operation.capitalize()}C{channels}H{height}W{width}A{axis}TD"
    )


def emit_array(name: str, data: bytes, width: int = 6) -> None:
    words = struct.unpack(f"<{len(data) // 4}I", data)
    print(f"static const uint32_t k{name}Words[] = {{")
    for start in range(0, len(words), width):
        row = ", ".join(
            f"0x{value:08x}" for value in words[start:start + width]
        )
        print(f"    {row},")
    print("};")
    print()


def main(root_text: str) -> None:
    root = Path(root_text)
    paths = sorted(root.glob("reduce_*.hwx"))
    if len(paths) != 24:
        raise SystemExit(f"expected 24 reduction references, found {len(paths)}")
    for path in paths:
        match = NAME.fullmatch(path.name)
        if not match:
            raise SystemExit(f"unexpected reference name: {path.name}")
        td = sections(path.read_bytes())[("__TEXT", "__text")]
        emit_array(symbol(*match.groups()), td)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} REDUCTION_ORACLES")
    main(sys.argv[1])
