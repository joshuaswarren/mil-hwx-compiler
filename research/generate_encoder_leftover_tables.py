#!/usr/bin/env python3
"""Emit plugins/H13/H13LinearTemplates.inc from the decoded encoder leftover
oracles (rank-3 linear and the d1024 s375 FFN chain).

Each record contributes the exact task stream Apple emitted plus the fields
the encoder needs. The constant-section models are verified byte-for-byte
against the recorded ``__TEXT,__const`` sidecars before anything is emitted:

- ``none``/``uniform`` bias modes carry the existing ``packMatvecWeights``
  interleave and nothing else; a uniform bias folds into a scalar register,
  so its block is absent from the section.
- The ``block`` mode lays the section out as column tiles of
  ``[bias group][group columns x reduction, reduction-outer][zero pad]``;
  the per-geometry tile groups are part of the verified model.
- The FFN chain interleaves two such tile regions behind a 128-byte kernel
  header that repeats at the head of every first-stage tile except the last.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from generate_matvec_encoder_data import (  # noqa: E402
    CONSTANT_ALIGNMENT, SCRATCH_ALIGNMENT, align, scratch_bytes, stream_words)

ORACLES = ROOT / "receipts/2026-09-16-compiler-leftover/oracles-round1"

# case -> (reduction, columns, bias mode, tile groups (group, count, stride
# halfwords)). The bias0 record needs no tiles; block modes carry the verified
# column-tile layout. Uniform-bias rows reuse the plain captures' streams.
LINEAR_ROWS = [
    ("encoder_linear_m375_k1024_n1024_bias0_idx", 1024, 1024, "none", []),
    ("encoder_linear_m375_k1024_n1024_bias1", 1024, 1024, "uniform", []),
    ("encoder_linear_m375_k1024_n128_bias1", 1024, 128, "uniform", []),
    ("encoder_linear_m375_k1024_n4096_bias1", 1024, 4096, "uniform", []),
    ("encoder_linear_m375_k1024_n640_bias1", 1024, 640, "uniform", []),
    ("encoder_linear_m375_k4096_n1024_bias1", 4096, 1024, "uniform", []),
    ("encoder_linear_m375_k1024_n1024_bias1_idx", 1024, 1024, "block",
     [(16, 64, 16416)]),
    ("encoder_linear_m375_k1024_n128_bias1_idx", 1024, 128, "block",
     [(8, 16, 8224)]),
    ("encoder_linear_m375_k1024_n640_bias1_idx", 1024, 640, "block",
     [(16, 32, 16416), (8, 16, 8224)]),
    ("encoder_linear_m375_k1024_n4096_bias1_idx", 1024, 4096, "block",
     [(16, 256, 16416)]),
    ("encoder_linear_m375_k4096_n1024_bias1_idx", 4096, 1024, "block",
     [(6, 160, 24608), (4, 16, 16416)]),
]

CHAIN_CASE = "chain_ffn_d1024_s375_matmul_silu"
CHAIN_TILES = {
    # stage -> (reduction, columns, groups (group, count, stride halfwords),
    # headered). The first stage repeats the kernel header before every tile
    # but the last; the second stage has no headers.
    "mm1": (1024, 4096, [(16, 256, 16480)], True),
    "mm2": (4096, 1024, [(6, 160, 24608), (4, 16, 16416)], False),
}


def record(name: str) -> dict:
    return json.loads((ORACLES / f"{name}.json").read_text())


def const_halves(name: str) -> list[int]:
    data = (ORACLES / f"{name}.const.bin").read_bytes()
    return list(struct.unpack(f"<{len(data) // 2}H", data))


def verify_linear_tiles(halves: list[int], reduction: int, columns: int,
                        groups: list[tuple[int, int, int]]) -> None:
    """The block-mode section: per tile [bias group][weights][zero pad]."""
    bias_base = reduction * columns
    pos = 0
    column = 0
    for group, count, stride in groups:
        for _ in range(count):
            for j in range(group):
                expected = bias_base + column + j + 1
                if halves[pos + j] != expected & 0xFFFF:
                    raise SystemExit(
                        f"bias strip mismatch at half {pos + j}: "
                        f"{halves[pos + j]} != {expected & 0xFFFF}")
            for red in range(reduction):
                for c in range(group):
                    expected = ((column + c) * reduction + red + 1) & 0xFFFF
                    at = pos + group + red * group + c
                    if halves[at] != expected:
                        raise SystemExit(
                            f"weight mismatch at half {at}: {halves[at]} "
                            f"!= {expected}")
            for at in range(pos + group + reduction * group, pos + stride):
                if halves[at] != 0:
                    raise SystemExit(f"tile pad not zero at half {at}")
            pos += stride
            column += group
    if pos != len(halves) or column != columns:
        raise SystemExit(
            f"tile model covers {pos} of {len(halves)} halves, "
            f"{column} of {columns} columns")


def verify_chain(halves: list[int], header: list[int]) -> None:
    """[128-byte header][mm1 tiles][mm2 tiles], verified against the record."""
    pos = 0
    if halves[:len(header)] != header:
        raise SystemExit("chain kernel header mismatch")
    pos = 0
    for stage in ("mm1", "mm2"):
        reduction, columns, groups, headered = CHAIN_TILES[stage]
        weight_bits = 0x3400 if stage == "mm1" else 0x3402
        bias_bits = 0x3401 if stage == "mm1" else 0x3403
        column = 0
        for group, count, stride in groups:
            for _ in range(count):
                start = pos
                if headered:
                    if halves[pos:pos + len(header)] != header:
                        raise SystemExit(f"chain {stage} tile header mismatch")
                    pos += len(header)
                if halves[pos:pos + group] != [bias_bits] * group:
                    raise SystemExit(f"chain {stage} bias strip mismatch")
                span = group * reduction
                if any(halves[pos + group + i] != weight_bits
                       for i in range(span)):
                    raise SystemExit(f"chain {stage} weight mismatch")
                for at in range(pos + group + span, start + stride):
                    if halves[at] != 0:
                        raise SystemExit(f"chain {stage} pad not zero at {at}")
                pos = start + stride
                column += group
        if column != columns:
            raise SystemExit(f"chain {stage} covers {column} of {columns}")
    if pos != len(halves):
        raise SystemExit(f"chain model covers {pos} of {len(halves)} halves")


def emit_words(out, name: str, words: list[int]) -> None:
    print(f"static constexpr std::uint32_t {name}[] = {{", file=out)
    for start in range(0, len(words), 8):
        row = ", ".join(f"0x{word:08x}" for word in words[start:start + 8])
        print(f"    {row},", file=out)
    print("};", file=out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "plugins/H13/H13LinearTemplates.inc")
    arguments = parser.parse_args()

    rows = []
    seen = set()
    with arguments.output.open("w") as out:
        print("// Generated from the decoded encoder-leftover oracles by "
              "research/generate_encoder_leftover_tables.py.", file=out)
        print("// No HWX container bytes; regenerate after re-minting the "
              "oracles.", file=out)
        for name, reduction, columns, mode, groups in LINEAR_ROWS:
            if name in seen:
                continue
            seen.add(name)
            rec = record(name)
            words = stream_words(rec)
            descriptor = rec["program_descriptor"]
            first_task = (descriptor["task_words_minus_one"] + 1) * 4
            constant = int(descriptor["constant_address"], 16) - \
                int(descriptor["text_address"], 16)
            if constant != align(len(words) * 4, CONSTANT_ALIGNMENT):
                raise SystemExit(f"{name} constant offset is not aligned")
            constant_bytes = rec["constant_section"]["size"]
            if mode == "block":
                verify_linear_tiles(const_halves(name), reduction, columns,
                                    groups)
                covered = sum(count * stride for _, count, stride in groups)
                if covered * 2 != constant_bytes:
                    raise SystemExit(f"{name} tile model size mismatch")
                tiles = ", ".join(f"{{{group}, {count}, {stride * 2}}}"
                                  for group, count, stride in groups)
            else:
                if constant_bytes != reduction * columns * 2:
                    raise SystemExit(f"{name} unexpected section size")
                tiles = ""
            symbol = f"kLinearTask{len(rows)}"
            emit_words(out, symbol, words)
            tile_symbol = ""
            if tiles:
                tile_symbol = f"kLinearTiles{len(rows)}"
                print(f"static constexpr H13LinearTileGroup {tile_symbol}[] = "
                      f"{{{tiles}}};", file=out)
            rows.append(
                f"    {{{rec['parameters']['rows']}, {reduction}, {columns}, "
                f"LinearBiasMode::{mode.capitalize()}, {symbol}, "
                f"std::size({symbol}), {first_task}, "
                f"{descriptor['task_count']}, {constant}, {constant_bytes}, "
                f"{scratch_bytes(rec)}, "
                f"{tile_symbol or 'nullptr'}, "
                f"{len(groups) if groups else 0}}},")

        chain = record(CHAIN_CASE)
        words = stream_words(chain)
        descriptor = chain["program_descriptor"]
        first_task = (descriptor["task_words_minus_one"] + 1) * 4
        constant = int(descriptor["constant_address"], 16) - \
            int(descriptor["text_address"], 16)
        if constant != align(len(words) * 4, CONSTANT_ALIGNMENT):
            raise SystemExit("chain constant offset is not aligned")
        chain_halves = const_halves(CHAIN_CASE)
        header = chain_halves[:64]
        verify_chain(chain_halves, header)
        emit_words(out, "kFFNChainTask", words)
        header_bytes = struct.pack(f"<{len(header)}H", *header)
        header_words = struct.unpack("<32I", header_bytes)
        data = ", ".join(f"0x{word:08x}" for word in header_words)
        print(f"static constexpr std::uint32_t kFFNChainKernelHeader[32] = "
              f"{{{data}}};", file=out)
        for stage in ("mm1", "mm2"):
            _, _, groups, _ = CHAIN_TILES[stage]
            tiles = ", ".join(f"{{{group}, {count}, {stride * 2}}}"
                              for group, count, stride in groups)
            print(f"static constexpr H13LinearTileGroup kFFNChain"
                  f"{stage.upper()}Tiles[] = {{{tiles}}};", file=out)
        rows.append(
            f"    {{kFFNChainTask, std::size(kFFNChainTask), {first_task}, "
            f"{descriptor['task_count']}, {constant}, "
            f"{chain['constant_section']['size']}, "
            f"{scratch_bytes(chain)}}},")

        print("static constexpr OracleLinearTemplate kLinearTasks[] = {",
              file=out)
        for row in rows[:-1]:
            print(row, file=out)
        print("};", file=out)
        print("static constexpr OracleFFNChainTemplate kFFNChainTasks[] = {",
              file=out)
        print(rows[-1], file=out)
        print("};", file=out)
    print(f"{arguments.output}: {len(rows)} templates")


if __name__ == "__main__":
    main()
