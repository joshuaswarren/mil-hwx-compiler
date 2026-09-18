#!/usr/bin/env python3
"""Emits plugins/H13/H13TransposeTemplates.inc from the decoded transpose
oracles in receipts/2026-09-16-compiler-leftover/oracles-round1/.

Every row's task words are the recorded Apple task stream verbatim; the
shapes are the Apple-normalized elementwise triples of the MIL input and
result (rank 3 [1, A, B] -> (1, A, B), rank 4 [1, C, H, W] -> (C, H, W)).
Regenerate after re-minting the oracles.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "receipts/2026-09-16-compiler-leftover/oracles-round1"
OUT = ROOT / "plugins/H13/H13TransposeTemplates.inc"

CASES = [
    "encoder_transpose_r3_021",
    "encoder_transpose_r3_021_1x1024x375",
    "encoder_transpose_r4_0213",
    "encoder_transpose_r4_0213_1x375x256x16",
]


def normalized(shape):
    if len(shape) == 3:
        assert shape[0] == 1, shape
        return (1, shape[1], shape[2])
    assert len(shape) == 4 and shape[0] == 1, shape
    return (shape[1], shape[2], shape[3])


def raw_words(task):
    """Inverts h13_td.decode_task: the stream is the header words followed by
    each record's header word and its values (addresses are contiguous)."""
    words = [int(word, 16) for word in task["header_words"]]
    amap = {}
    for block in task["blocks"].values():
        for address, word in block["words"].items():
            amap[int(address, 16)] = int(word, 16)
    for record in task["records"]:
        address = int(record["address"], 16)
        words.append(int(record["header"], 16))
        words.extend(amap[address + offset * 4] for offset in range(record["count"]))
    return words


def main():
    rows = []
    for name in CASES:
        oracle = json.loads((ORACLES / f"{name}.json").read_text())
        assert oracle["error"] is None, name
        tasks = oracle["task_descriptors"]
        assert len(tasks) == 1, name
        words = raw_words(tasks[0])
        # The recorded task stream is 504 bytes; the decoded view must agree.
        expected = tasks[0]["size_bytes"]
        assert len(words) * 4 == expected, (name, len(words) * 4, expected)
        input_shape = normalized(oracle["tensor_descriptors"][0]["shape"])
        output_shape = normalized(oracle["tensor_descriptors"][1]["shape"])
        constant = oracle["constant_section"]
        assert constant["nonzero_bytes"] == 0, name
        first_task = oracle["program_descriptor"]["task_section"]
        assert first_task == {"offset": 0, "size": expected}, name
        constant_offset = (expected + 0x3F) & ~0x3F
        rows.append((name, input_shape, output_shape, words, expected,
                     constant["size"], constant_offset))

    lines = [
        "// Generated from decoded H13 encoder-transpose oracle task words by",
        "// receipts/2026-09-17-encoder-transpose-lowering/emit_transposes.py.",
        "// No HWX container bytes; regenerate after re-minting the oracles.",
        "",
    ]
    for index, (name, _, _, words, _, _, _) in enumerate(rows):
        lines.append(f"static constexpr std::uint32_t kTransposeTask{index}[] = {{")
        for start in range(0, len(words), 8):
            chunk = ", ".join(f"0x{word:08x}" for word in words[start:start + 8])
            lines.append(f"    {chunk},")
        lines.append("};")
        lines.append("")
    lines.append("static constexpr OracleTransposeTemplate kTransposeTasks[] = {")
    for index, (name, inp, out, words, first_task, constant_bytes,
                constant_offset) in enumerate(rows):
        lines.append(
            f"    {{ {{ {inp[0]}, {inp[1]}, {inp[2]} }}, "
            f"{{ {out[0]}, {out[1]}, {out[2]} }}, "
            f"kTransposeTask{index}, std::size(kTransposeTask{index}), "
            f"{first_task}, 1, {constant_offset}, {constant_bytes}, 0 }},"
            f"  // {name}")
    lines.append("};")
    lines.append("")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
