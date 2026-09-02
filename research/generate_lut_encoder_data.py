#!/usr/bin/env python3
"""Emit decoded H16G unary TD/table arrays from independently minted HWX refs."""

from __future__ import annotations

import struct
import sys
from pathlib import Path

from compare_hwx_sections import sections


def symbol(text: str) -> str:
    return "".join(part.capitalize() for part in text.replace("-", "_").split("_"))


def emit_array(name: str, data: bytes, width: int = 6) -> None:
    words = struct.unpack(f"<{len(data) // 4}I", data)
    print(f"static const uint32_t k{name}Words[] = {{")
    for start in range(0, len(words), width):
        row = ", ".join(f"0x{value:08x}" for value in words[start:start + width])
        print(f"    {row},")
    print("};")
    print()


def section(path: Path, segment: str, name: str) -> bytes:
    return sections(path.read_bytes())[(segment, name)]


def main(v2_root: str, legacy_root: str) -> None:
    v2 = Path(v2_root)
    legacy = Path(legacy_root)
    for family in ("sigmoid", "relu", "rsqrt"):
        for size in (128, 256, 512, 1024, 2048):
            path = v2 / f"{family}_n{size}.hwx"
            emit_array(f"{symbol(family)}{size}TD", section(path, "__TEXT", "__text"))
    emit_array("Log256TD", section(legacy / "act_log.hwx", "__TEXT", "__text"))
    for operation in (
        "sigmoid", "tanh", "gelu", "silu", "exp", "log", "sqrt", "rsqrt",
        "recip",
    ):
        path = legacy / f"act_{operation}.hwx"
        emit_array(f"{symbol(operation)}KERN", section(path, "__KERN_0", "__kern_0"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {sys.argv[0]} V2_ORACLES LEGACY_REFS")
    main(sys.argv[1], sys.argv[2])
