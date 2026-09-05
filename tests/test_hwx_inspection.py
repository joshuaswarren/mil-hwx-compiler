#!/usr/bin/env python3
"""Device-free checks for source-backed H14 and existing H16G inspection."""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

INSPECTOR = Path(__file__).resolve().parents[1] / "research" / "inspect_hwx.py"
MAGIC = 0xBEEFFACE
H14_BLOCKS = (
    ("common", 0x0000, 19),
    ("l2", 0x0500, 25),
    ("pe", 0x0900, 5),
    ("ne", 0x0D00, 5),
    ("tile_dma_src", 0x1100, 53),
    ("tile_dma_dst", 0x1500, 9),
    ("kernel_dma_src", 0x1900, 70),
)


def dense_record(start: int, count: int, seed: int) -> list[int]:
    records = []
    written = 0
    while written < count:
        span = min(64, count - written)
        records.append(((span - 1) << 15) | (start // 4 + written))
        records.extend(seed + written + index for index in range(span))
        written += span
    return records


def h14_task() -> bytes:
    words = [0] * 8
    for index, (_, start, count) in enumerate(H14_BLOCKS, 1):
        words.extend(dense_record(start, count, index << 24))
    words[0] = 1 | (len(words) << 16)
    words[1] = 1
    payload = struct.pack(f"<{len(words)}I", *words)
    return payload + bytes((-len(payload)) % 16)


def hwx(subtype: int, text: bytes) -> bytes:
    command_size = 72 + 80
    file_offset = 32 + command_size
    header = struct.pack("<8I", MAGIC, 0x80, subtype, 1, 1, command_size, 0, 0)
    segment = struct.pack(
        "<2I16s4Q4I", 0x19, command_size, b"__TEXT", 0, len(text),
        file_offset, len(text), 7, 5, 1, 0)
    section = struct.pack(
        "<16s16s2Q8I", b"__text", b"__TEXT", 0, len(text), file_offset,
        2, 0, 0, 0x28, 0, 0, 0)
    return header + segment + section + text


def inspect(path: Path, success: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(INSPECTOR), str(path)], capture_output=True,
        text=True, timeout=10, check=False)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    return result


with tempfile.TemporaryDirectory(prefix="mil-hwx-inspection-") as directory:
    root = Path(directory)
    source_derived = root / "source-derived-h14.hwx"
    source_derived.write_bytes(hwx(5, h14_task()))
    result = inspect(source_derived)
    assert "architecture subtype=0x0005 name=H14 isa=11" in result.stdout
    for name, start, count in H14_BLOCKS:
        end = start + count * 4
        expected = (
            f"h14_block name={name} range=0x{start:04x}-0x{end:04x} "
            f"source_words={count} observed_words={count}")
        assert expected in result.stdout, result.stdout
    assert "h14_tasks count=1" in result.stdout

    mixed = bytearray(hwx(5, h14_task()))
    text_offset = 32 + 72 + 80
    struct.pack_into("<I", mixed, text_offset + 8 * 4, 0x4100 // 4)
    mixed_path = root / "mixed-generation-h14.hwx"
    mixed_path.write_bytes(mixed)
    result = inspect(mixed_path, success=False)
    assert "outside source-backed H14 blocks" in result.stderr, result.stderr

    truncated = hwx(5, h14_task()[:-16])
    truncated_path = root / "truncated-h14.hwx"
    truncated_path.write_bytes(truncated)
    result = inspect(truncated_path, success=False)
    assert "declares" in result.stderr and "beyond __TEXT/__text" in result.stderr

    h16g_path = root / "synthetic-h16g-envelope.hwx"
    h16g_path.write_bytes(hwx(7, b""))
    result = inspect(h16g_path)
    assert "architecture subtype=0x0007 name=H16G isa=17" in result.stdout
    assert "__TEXT/__text" in result.stdout

print("HWX inspection: PASS (synthetic source-derived H14 fixture; no real H14 artifact)")
