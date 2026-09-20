#!/usr/bin/env python3
"""Real-encoder probe for the H13 boundary at the LN-peel tip.

The real encoder MIL returns two converted values (fp32 `encoder_hidden`,
int32 `encoder_mask`), which the H13 result gate refuses before any op
lowers. This probe strips exactly those two result casts and returns the
fp16/bool producers instead — a census probe, not a model claim — and
synthesizes the referenced weight files (records at the MIL's exact
BLOBFILE offsets, zero payloads of the declared element counts) so covered
plans can resolve constants and the lowering reaches the first genuinely
uncovered operation.
"""
import re
import struct
import sys
from pathlib import Path

MIL = Path("/tmp/mil-hwxc-cov/encoder-pinned.mil")
ROOT = Path("/tmp/enc-peel-probe")

CONST = re.compile(
    r"tensor<(?P<elem>fp16|fp32|int32|bool), (?P<dims>\[[^\]]*\])> "
    r"(?P<name>[A-Za-z_0-9]+) = const\(\)\[name = tensor<string, \[\]>\("
    r'"[^"]*"\), val = tensor<(?P=elem), (?P=dims)>\(BLOBFILE\('
    r'path = string\("@model_path/(?P<file>[^"]+)"\), '
    r"offset = uint64\((?P<offset>\d+)\)\)\)\];")

ELEMENT_SIZES = {"fp16": 2, "fp32": 4, "int32": 4, "bool": 1}


def dims_of(text: str) -> int:
    inner = text[1:-1].strip()
    if not inner:
        return 1
    count = 1
    for part in inner.split(","):
        count *= int(part.strip())
    return count


def main() -> int:
    text = MIL.read_text()
    (ROOT / "weights").mkdir(parents=True, exist_ok=True)
    files = {}
    for match in CONST.finditer(text):
        entry = match.groupdict()
        entry["count"] = dims_of(entry["dims"])
        files.setdefault(entry["file"], []).append(entry)
    for name, entries in files.items():
        regions = []
        for entry in entries:
            offset = int(entry["offset"])
            payload = offset + 64  # the real model's record->payload stride
            regions.append((offset, 24, None))
            regions.append((payload, entry["count"] * ELEMENT_SIZES[entry["elem"]],
                            entry["name"]))
        regions.sort()
        for (_, end, _), (start, _, _) in zip(regions, regions[1:]):
            if start < end:
                raise SystemExit(f"{name}: regions overlap at {start} < {end}")
        size = regions[-1][0] + regions[-1][1]
        with open(ROOT / name, "wb") as stream:
            stream.truncate(size)
            for entry in entries:
                offset = int(entry["offset"])
                payload = offset + 64
                stream.seek(offset)
                stream.write(struct.pack("<IIQQ", 0xDEADBEEF, 1,
                    entry["count"] * ELEMENT_SIZES[entry["elem"]], payload))
                # Known real payloads (Main, 2026-09-20): the length
                # divisors var_23* are fp16 1.0; the add operands
                # var_133/var_177 are fp16 1.0; the select -inf fills
                # var_8/var_13 are fp16 -inf. The epsilon record is the
                # receipt-verified fp16 0x00a8 = 0x1.5p-17. Everything
                # else stays zero-filled: values never feed a compile
                # gate except through the named constants above.
                known = {
                    "var_5_to_fp16": b"\xa8\x00",
                    "var_23_promoted_to_fp16": b"\x00\x3c",
                    "var_23_promoted_1_to_fp16": b"\x00\x3c",
                    "var_133_promoted_to_fp16": b"\x00\x3c",
                    "var_177_promoted_to_fp16": b"\x00\x3c",
                    "var_8_to_fp16": b"\x00\xfc",
                    "var_13_to_fp16": b"\x00\xfc",
                }
                if entry["name"] in known:
                    if entry["elem"] != "fp16" or entry["count"] != 1:
                        raise SystemExit(
                            f"{name}: known payload {entry['name']} is not "
                            "one fp16 lane")
                    stream.seek(payload)
                    stream.write(known[entry["name"]])
        print(f"{name}: {len(entries)} constants, sparse file {size} bytes")
    probe = text.replace(
        '    tensor<string, []> linear_217_cast_fp16_to_fp32_dtype_0 = '
        'const()[name = tensor<string, []>("linear_217_cast_fp16_to_fp32_'
        'dtype_0"), val = tensor<string, []>("fp32")];\n', "")
    probe = probe.replace(
        '    tensor<string, []> var_3903_dtype_0 = const()[name = tensor<'
        'string, []>("op_3903_dtype_0"), val = tensor<string, []>("int32")];\n',
        "")
    probe = probe.replace(
        '    tensor<fp32, [1, 375, 640]> encoder_hidden = cast(dtype = '
        'linear_217_cast_fp16_to_fp32_dtype_0, x = linear_217_cast_fp16)'
        '[name = tensor<string, []>("cast_0")];\n', "")
    probe = probe.replace(
        '    tensor<int32, [1, 375]> encoder_mask = cast(dtype = '
        'var_3903_dtype_0, x = output_mask)[name = tensor<string, []>'
        '("cast_1")];\n', "")
    probe = probe.replace("} -> (encoder_hidden, encoder_mask);",
                          "} -> (linear_217_cast_fp16, output_mask);")
    assert "-> (linear_217_cast_fp16, output_mask);" in probe
    assert "encoder_hidden" not in probe.split("} ->")[0].split("func main")[1]
    (ROOT / "encoder_probe.mil").write_text(probe)
    print(f"probe MIL: {ROOT / 'encoder_probe.mil'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
