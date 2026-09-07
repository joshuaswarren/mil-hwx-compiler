#!/usr/bin/env python3
"""Emit the chain-schedule template tables for H13 and H14 from decoded oracles.

``plugins/H13/H13ChainTemplates.inc`` and ``plugins/H14/H14ChainTemplates.inc``
carry, per decoded ``chain_*`` oracle: the whole task stream Apple emitted, the
surface layouts it declared, the scratch it allocated below them, the ordered
constant-section plan that rebuilds its ``__TEXT/__const`` byte-for-byte, and a
label for every task naming the MIL operation it encodes and the fusion it
carries. The compilers walk a MIL function in dataflow order, derive the same
schedule key, and instantiate the matching template as one program.

The constant plan is fitted, not guessed: the emitter rebuilds each candidate
layout from the packed weights and the decoded lookup tables and accepts the
arrangement whose SHA-256 equals the recorded section hash. An arrangement the
fitter cannot reproduce refuses the case loudly, so the parity set never
silently shrinks; ``--report`` lists the refusals.

    python3 research/mint_sched_probes.py --emit-templates
    python3 research/mint_sched_probes.py --check
    python3 research/mint_sched_probes.py --report

The ``sched_`` campaign mints the sweep points the fitted plans still lack --
repeated identical constants (Apple deduplicates them), the scratch fraction of
tiled chains, and the fused per-channel bias block -- on the oracle host:

    python3 research/mint_sched_probes.py --host macstudio --targets h13 h14
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import pathlib
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

import mint_chain_probes as mc  # noqa: E402  (Chain builder, task labeler)
import mint_oracles as om  # noqa: E402

OUTPUTS = {"h13": ROOT / "plugins/H13/H13ChainTemplates.inc",
           "h14": ROOT / "plugins/H14/H14ChainTemplates.inc"}
RECORD_BASE, RECORD_BYTES, RECORD_CAPACITY = 64, 24, 4096
VALUE_BASE, VALUE_SPAN = 0x3400, 0x0800
LUT_UNARY = {"gelu": "Gelu", "silu": "Silu", "exp": "Exponential",
             "sigmoid": "Sigmoid", "tanh": "Tanh", "sqrt": "SquareRoot"}
# Table identifiers shared with the ChainTableKind enum in the encoders.
TABLE_IDS = {"Exponential": 1, "ExponentialReciprocal": 2, "Gelu": 3,
             "Silu": 4, "Sigmoid": 5, "Tanh": 6, "SquareRoot": 7}
SHORT_TABLE = {"Exponential": "Exp", "Gelu": "Gelu", "Silu": "Silu",
               "Sigmoid": "Sigmoid", "Tanh": "Tanh", "SquareRoot": "Sqrt"}
YRE = re.compile(r"(?<![\w])y = (\w+)")
XRE = re.compile(r"(?<![\w])x = (\w+)")
TABLE_SIZES = (128, 256, 512, 1024, 2048)
FLOOR_BYTES = 16384
LABEL_KINDS = {
    "matmul_const": ("matmul", None), "matmul_rt": ("matmul", None),
    "softmax": ("softmax", None), "layer_norm": ("layer_norm", None),
    "gelu": ("unary", "gelu"), "silu": ("unary", "silu"),
    "exp": ("unary", "exp"), "sigmoid": ("unary", "sigmoid"),
    "tanh": ("unary", "tanh"), "sqrt": ("unary", "sqrt"),
    "relu": ("unary", "relu"), "leaky_relu": ("unary", "leaky_relu"),
    "add": ("binary", "add"), "mul": ("binary", "mul"),
    "add_bcast": ("binary", "add"), "mul_bcast": ("binary", "mul"),
    "add_const": ("binary", "add"), "mul_const": ("binary", "mul"),
    "maximum": ("binary", "maximum"), "minimum": ("binary", "minimum"),
}


# --------------------------------------------------------------------------
# Constant-section reconstruction
# --------------------------------------------------------------------------

def record_offset(index: int) -> int:
    return RECORD_BASE + index * RECORD_BYTES


def weight_blob(shapes: list[tuple[int, ...]],
                payloads: list[bytes] | None = None) -> bytes:
    """Build the campaign BLOBFILE with one payload region per constant."""
    if payloads is None:
        payloads = [struct.pack("<H", VALUE_BASE + index % VALUE_SPAN)
                    * math.prod(shape) for index, shape in enumerate(shapes)]
    expected = [math.prod(shape) * 2 for shape in shapes]
    if len(payloads) != len(shapes) or [len(payload) for payload in payloads] != expected:
        raise ValueError("constant payload sizes do not match their shapes")
    data_start = (record_offset(RECORD_CAPACITY) + 0x3F) & ~0x3F
    blob = bytearray(data_start + sum(len(payload) for payload in payloads))
    struct.pack_into("<II", blob, 0, len(shapes), 2)
    offset = data_start
    for index, payload in enumerate(payloads):
        struct.pack_into("<IIQQ", blob, record_offset(index),
                         0xDEADBEEF, 1, len(payload), offset)
        blob[offset:offset + len(payload)] = payload
        offset += len(payload)
    return bytes(blob)


def blob_regions(shapes: list[tuple[int, ...]]) -> list[bytes]:
    blob = weight_blob(shapes)
    cursor = (record_offset(RECORD_CAPACITY) + 0x3F) & ~0x3F
    regions = []
    for shape in shapes:
        size = math.prod(shape) * 2
        regions.append(blob[cursor:cursor + size])
        cursor += size
    return regions


def _kern_tables() -> tuple[dict[str, bytes], bytes]:
    text = (ROOT / "plugins/H13/H13ElementwiseConstants.inc").read_text()
    tables = {}
    for match in re.finditer(r"k(\w+)KERNWords\[\]\s*=\s*\{(.*?)\};", text, re.S):
        words = [int(word, 16)
                 for word in re.findall(r"0x[0-9a-fA-F]+", match.group(2))]
        tables[match.group(1)] = b"".join(struct.pack("<I", word)
                                          for word in words)
    recip_match = re.search(r"kRecipKERNWords\[\]\s*=\s*\{(.*?)\};", text, re.S)
    recip = b"".join(struct.pack("<I", int(word, 16)) for word in re.findall(
        r"0x[0-9a-fA-F]+", recip_match.group(1)))
    return tables, recip


_TABLE_SOURCE = ROOT / "plugins/H13/H13ElementwiseConstants.inc"
TABLES, RECIP = _kern_tables() if _TABLE_SOURCE.exists() else ({}, b"")


def pack_h13(rows: int, reduction: int, columns: int, weights: bytes) -> bytes:
    group = min(16, columns // 16)
    if rows > 128:
        group = min(group, 32768 // reduction)
    group = max(1, group)
    groups = columns // group
    packed = bytearray(len(weights))
    for column in range(columns):
        plane = column // group
        destination_plane = (plane % 16) * (groups // 16) + plane // 16
        destination = (destination_plane * reduction * group + column % group) * 2
        source = column * reduction * 2
        for _ in range(reduction):
            packed[destination:destination + 2] = weights[source:source + 2]
            destination += group * 2
            source += 2
    return bytes(packed)


def pack_h14(reduction: int, columns: int, weights: bytes) -> bytes:
    group = min(16, columns // 16)
    planes = columns // group
    section = max(1024, len(weights))
    packed = bytearray(section)
    stride = section // 2 // planes
    for column in range(columns):
        plane = column // group
        destination_plane = (plane % 16) * (planes // 16) + plane // 16
        destination = (destination_plane * stride + column % group) * 2
        source = column * reduction * 2
        for _ in range(reduction):
            packed[destination:destination + 2] = weights[source:source + 2]
            destination += group * 2
            source += 2
    return bytes(packed)


def lane_interleave(packed: bytes, prefix: bytes,
                    partitions: list[int]) -> bytes:
    lane_bytes = len(packed) // 16
    if sum(partitions) != lane_bytes:
        raise ValueError("fused LUT partitions do not cover a packed lane")
    out = bytearray()
    cursor = [lane * lane_bytes for lane in range(16)]
    for partition in partitions:
        for lane in range(16):
            out += prefix
            out += packed[cursor[lane]:cursor[lane] + partition]
            cursor[lane] += partition
    return bytes(out)


def table_variants(kind: str, table: str | None) -> list[tuple[str, int, bytes]]:
    """Candidate table payloads: decoded 128-byte table, zero padded."""
    if kind == "softmax":
        contents = [("Exponential", TABLES["Exp"]),
                    ("ExponentialReciprocal",
                     (TABLES["Exp"] + RECIP).ljust(256, b"\0")[:256])]
    else:
        contents = [(table, TABLES[SHORT_TABLE[table]])]
    out = []
    for name, content in contents:
        for size in TABLE_SIZES:
            if len(content) <= size:
                out.append((name, size, content.ljust(size, b"\0")))
    return out


# --------------------------------------------------------------------------
# Chain model
# --------------------------------------------------------------------------

def parse_ops(mil: str) -> list[dict]:
    ops = []
    for match in re.finditer(
            r"tensor<fp16, \[([^\]]+)\]> (\w+) = (\w+)\((.*?)\)\[name[^;]*;",
            mil, re.S):
        shape = tuple(int(value) for value in
                      match.group(1).replace(" ", "").split(","))
        ops.append({"name": match.group(3), "shape": shape,
                    "args": match.group(4), "result": match.group(2),
                    "text": match.group(0)})
    return ops


def const_of(ops: list[dict], name: str) -> dict | None:
    return next((op for op in ops
                 if op["result"] == name and op["name"] == "const"), None)


def chw(shape: tuple[int, ...]) -> tuple[int, int, int]:
    """The CHW surface a logical MIL shape lays out as (leading 1s collapse)."""
    dimensions = list(shape)
    while len(dimensions) > 3 and dimensions[0] == 1:
        dimensions.pop(0)
    while len(dimensions) < 3:
        dimensions.insert(0, 1)
    return tuple(dimensions)


def fp16_scalar(args: str, ops: list[dict]) -> int | None:
    match = re.search(r"y = fp16\(0x1p([+-]\d+)\)", args)
    if not match:
        name = YRE.search(args)
        constant = const_of(ops, name.group(1)) if name else None
        match = re.search(r"val = fp16\(0x1p([+-]\d+)\)",
                          constant["text"] if constant else "")
    if not match:
        return None
    value = 2.0 ** int(match.group(1))
    return struct.unpack("<H", struct.pack("<e", value))[0]


def op_segments(record: dict) -> list[dict]:
    """Scheduled operations in dataflow order, fusion folded into producers."""
    mil = record["mil"]
    shapes = [tuple(shape) for shape in
              record.get("parameters", {}).get("constant_matrices", [])]
    ops = parse_ops(mil)
    constants = iter(list(zip(shapes, blob_regions(shapes))))
    compute = [op for op in ops if op["name"] != "const"]
    segments: list[dict] = []
    index = 0
    while index < len(compute):
        op = compute[index]
        name = op["name"]
        c, h, w = chw(op["shape"])
        if name == "matmul":
            y_const = const_of(ops, YRE.search(op["args"]).group(1))
            weight = None
            if y_const and "BLOBFILE" in y_const["text"]:
                (columns, reduction), region = next(constants)
                rows = op["shape"][1] if len(op["shape"]) == 3 else op["shape"][0]
                weight = {"rows": rows, "reduction": reduction,
                          "columns": columns, "region": region,
                          "transpose_x": "transpose_x = t" in op["args"],
                          "transpose_y": "transpose_y = f" not in op["args"]}
            fusion = None
            if index + 1 < len(compute):
                nxt = compute[index + 1]
                if nxt["name"] in LUT_UNARY and nxt["shape"] == op["shape"]:
                    fusion = ("lut", LUT_UNARY[nxt["name"]])
                    index += 1
                elif nxt["name"] == "relu" and nxt["shape"] == op["shape"]:
                    fusion = ("relu", None)
                    index += 1
                elif nxt["name"] == "mul" and nxt["shape"] == op["shape"] and \
                        fp16_scalar(nxt["args"], ops) is not None:
                    fusion = ("scale", fp16_scalar(nxt["args"], ops))
                    index += 1
                elif nxt["name"] == "add":
                    ny = const_of(ops, YRE.search(nxt["args"]).group(1))
                    if ny and "BLOBFILE" in ny["text"] and \
                            len([d for d in ny["shape"] if d != 1]) == 1:
                        bias = next(constants)
                        fusion = ("bias", bias)
                        index += 1
                        if index + 1 < len(compute):
                            follow = compute[index + 1]
                            if follow["name"] in LUT_UNARY and \
                                    follow["shape"] == op["shape"]:
                                fusion = ("bias-lut", bias,
                                          LUT_UNARY[follow["name"]])
                                index += 1
            if weight:
                key = (f"mm{weight['rows']}x{weight['reduction']}"
                       f"x{weight['columns']}t{int(weight['transpose_x'])}"
                       f"{int(weight['transpose_y'])}")
            else:
                key = (f"mmrt{op['shape'][-2]}x{op['shape'][-1]}"
                       f"t{int('transpose_x = t' in op['args'])}"
                       f"{int('transpose_y = f' not in op['args'])}")
            if fusion:
                tag = fusion[0]
                if tag == "lut":
                    detail = f":{fusion[1]}"
                elif tag == "scale":
                    detail = f":{fusion[1]:04x}"
                elif tag == "bias-lut":
                    detail = f":{fusion[2]}"
                else:
                    detail = ""
                key += f"+{tag}{detail}"
            segments.append({"kind": "matmul", "op": op, "weight": weight,
                                         "fusion": fusion, "key": key,
                                         "source": XRE.search(op["args"]).group(1)})
        elif name == "softmax":
            segments.append({"kind": "softmax", "op": op,
                             "key": f"sm:{c}x{h}x{w}"})
        elif name == "layer_norm":
            segments.append({"kind": "layer_norm", "op": op,
                             "key": f"ln:{c}x{h}x{w}"})
        elif name in ("add", "mul", "maximum", "minimum", "sub"):
            y_const = const_of(ops, YRE.search(op["args"]).group(1))
            if y_const and "BLOBFILE" in y_const["text"]:
                (cshape, region) = next(constants)
                per_channel = len([d for d in cshape if d != 1]) == 1
                segments.append({"kind": "binary", "op": op, "operand": region,
                                 "per_channel": per_channel,
                                 "key": f"{name}:{c}x{h}x{w}+"
                                        f"{'pc' if per_channel else 'full'}"})
            elif fp16_scalar(op["args"], ops) is not None:
                segments.append({"kind": "binary", "op": op, "operand": None,
                                 "key": f"{name}:{c}x{h}x{w}"
                                        f"s{fp16_scalar(op['args'], ops):04x}"})
            else:
                segments.append({"kind": "binary", "op": op, "operand": None,
                                 "key": f"{name}:{c}x{h}x{w}"})
        elif name in LUT_UNARY or name in ("relu", "leaky_relu"):
            segments.append({"kind": "unary", "op": op,
                             "table": LUT_UNARY.get(name),
                             "key": f"{name}:{c}x{h}x{w}"})
        else:
            raise ValueError(f"unscheduled operation {name}")
        index += 1
    return segments


def schedule_key(segments: list[dict]) -> str:
    return ";".join(segment["key"] for segment in segments)


def assign_operands(segments: list[dict]) -> None:
    """Index every constant operand in MIL declaration order."""
    operand = 0
    for segment in segments:
        if segment["kind"] == "matmul" and segment["weight"]:
            segment["operand_index"] = operand
            operand += 1
            if segment.get("fusion") and \
                    segment["fusion"][0] in ("bias", "bias-lut"):
                segment["fusion"] += (operand,)
                operand += 1
        elif segment["kind"] == "binary" and segment["operand"] is not None:
            segment["operand_index"] = operand
            operand += 1


# --------------------------------------------------------------------------
# Constant-plan fitting
# --------------------------------------------------------------------------

def packed_weight(target: str, weight: dict) -> bytes:
    if target == "h13":
        return pack_h13(weight["rows"], weight["reduction"],
                        weight["columns"], weight["region"])
    return pack_h14(weight["reduction"], weight["columns"], weight["region"])


def _post_op_values(record: dict, target: str) -> list[int]:
    address = "0x0c804" if target == "h13" else "0x00d04"
    values = []
    for task in record["task_descriptors"]:
        value = next((int(block["words"][address], 16)
                      for block in task["blocks"].values()
                      if address in block["words"]), 0)
        values.append(value)
    return values


def _fused_lut_partitions(record: dict, target: str) -> list[list[int]]:
    size_address = 0x1F888 if target == "h13" else 0x1998
    partitions = []
    active = None
    for task, value in zip(record["task_descriptors"],
                           _post_op_values(record, target)):
        fused = bool(value & 0x20000 and value & 0xFF in (0, 0x10))
        if fused and active is None:
            active = []
            partitions.append(active)
        elif not fused:
            active = None
        if fused:
            words = {int(address, 16): int(word, 16)
                     for block in task["blocks"].values()
                     for address, word in block["words"].items()}
            active.append(words[size_address] - 128)
    return partitions


def _standalone_table_sizes(record: dict, target: str) -> list[int]:
    if target == "h14":
        return [128 for value in _post_op_values(record, target)
                if value & 0x20000 and value & 0xFF]
    sizes = []
    for task, post_op in zip(record["task_descriptors"],
                             _post_op_values(record, target)):
        if not (post_op & 0x20000 and post_op & 0xFF):
            continue
        words = {address: int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
        active_lanes = sum(words.get(f"0x{address:05x}", 0) & 1
                           for address in range(0x1F808, 0x1F848, 4))
        if not active_lanes:
            raise ValueError("standalone LUT task has no active kernel-DMA lane")
        sizes.append(active_lanes * 128)
    return sizes


def fused_bias_weight(target: str, weight: dict, bias: bytes) -> bytes:
    """Insert one bias row before each packed output plane."""
    packed = packed_weight(target, weight)
    columns = weight["columns"]
    group = max(1, min(16, columns // 16))
    planes = columns // group
    bias_packed = pack_h13(weight["rows"], 1, columns, bias)
    weight_plane = len(packed) // planes
    bias_plane = len(bias_packed) // planes
    planes_per_lane = planes // 16
    out = bytearray()
    for lane in range(16):
        lane_bytes = bytearray()
        for relative in range(planes_per_lane):
            plane = lane * planes_per_lane + relative
            lane_bytes += bias_packed[plane * bias_plane:(plane + 1) * bias_plane]
            lane_bytes += packed[plane * weight_plane:(plane + 1) * weight_plane]
        out += lane_bytes
        out += b"\0" * (-len(lane_bytes) % 64)
    return bytes(out)

def _constant_order(segments: list[dict]) -> list[dict]:
    """Apply Apple's two-branch and QK-table constant scheduling."""
    ordered = list(segments)
    index = 0
    while index < len(ordered):
        segment = ordered[index]
        if segment["kind"] != "matmul" or not segment["weight"]:
            index += 1
            continue
        end = index + 1
        while end < len(ordered) and ordered[end]["kind"] == "matmul" and \
                ordered[end]["weight"] and \
                ordered[end]["source"] == segment["source"]:
            end += 1
        if end - index >= 3:
            table_index = next((position for position in range(end, len(ordered))
                                if ordered[position]["kind"] == "softmax"), None)
            if table_index is not None:
                table = ordered.pop(table_index)
                ordered.insert(index + 2, table)
                index = end + 1
                continue
        if end - index >= 2:
            ordered[index], ordered[index + 1] = ordered[index + 1], ordered[index]
        index = end
    return ordered

class ScheduleRefusal(Exception):
    """A constant-section rule outside the measured envelope, named."""


def fit_constant_plan(record: dict, target: str,
                      segments: list[dict]) -> tuple[list[dict], bool] | None:
    """Derive and byte-verify the chain constant-section plan."""
    table_sizes = iter(_standalone_table_sizes(record, target))
    fused_partitions = iter(_fused_lut_partitions(record, target))
    entries = []
    for segment in _constant_order(segments):
        if segment["kind"] == "matmul" and segment["weight"]:
            fusion = segment["fusion"]
            if fusion and fusion[0] == "lut":
                partitions = next(fused_partitions, None)
                if partitions is None:
                    return None
                if len(set(partitions)) != 1:
                    raise ScheduleRefusal(
                        "h13.chain-unequal-fused-lut-slices")
                payload = lane_interleave(
                    packed_weight(target, segment["weight"]),
                    TABLES[SHORT_TABLE[fusion[1]]], partitions)
                entry = {"kind": "fused-weight",
                         "operand": segment["operand_index"],
                         "table": fusion[1], "slices": len(partitions)}
            elif fusion and fusion[0] in ("bias", "bias-lut"):
                _, (_, bias), *detail = fusion
                bias_operand = detail[-1]
                payload = fused_bias_weight(target, segment["weight"], bias)
                entry = {"kind": "fused-bias-weight",
                         "operand": segment["operand_index"],
                         "bias_operand": bias_operand}
                if fusion[0] == "bias-lut":
                    partitions = next(fused_partitions, None)
                    if partitions is None:
                        return None
                    if len(set(partitions)) != 1:
                        raise ScheduleRefusal(
                            "h13.chain-unequal-fused-lut-slices")
                    payload = lane_interleave(
                        payload, TABLES[SHORT_TABLE[fusion[2]]], partitions)
                    entry.update(kind="fused-bias-lut-weight",
                                 table=fusion[2], slices=len(partitions))
            else:
                payload = packed_weight(target, segment["weight"])
                entry = {"kind": "weight", "operand": segment["operand_index"]}
            entries.append(dict(entry, payload=payload))
        elif segment["kind"] == "softmax" or \
                (segment["kind"] == "unary" and segment["table"]):
            size = next(table_sizes, None)
            if size is None:
                return None
            table = "Exponential" if segment["kind"] == "softmax" \
                else segment["table"]
            payload = TABLES[SHORT_TABLE[table]].ljust(size, b"\0")
            entries.append({"kind": "table", "table": table,
                            "bytes": size, "payload": payload})
        elif segment["kind"] == "binary" and segment["operand"] is not None:
            entries.append({"kind": "per-channel" if segment["per_channel"]
                            else "full-constant",
                            "operand": segment["operand_index"],
                            "payload": segment["operand"]})
    if not entries:
        entries = [{"kind": "floor", "payload": b"\0" * FLOOR_BYTES}]

    target_size = record["constant_section"]["size"]
    target_sha = record["constant_section"]["sha256"]
    for dedup in (True, False):
        seen = set()
        payload = bytearray()
        for entry in entries:
            digest = hashlib.sha256(entry["payload"]).digest()
            if dedup and digest in seen:
                continue
            seen.add(digest)
            payload += entry["payload"]
        if len(payload) == target_size and \
                hashlib.sha256(payload).hexdigest() == target_sha:
            return [{key: value for key, value in entry.items()
                     if key != "payload"} for entry in entries], dedup
    return None

# --------------------------------------------------------------------------
# Task stream reconstruction
# --------------------------------------------------------------------------

def task_words(record: dict) -> list[list[int]]:
    tasks = []
    target = record["target"]
    for task in record["task_descriptors"]:
        values = {int(address, 16): int(value, 16)
                  for block in task["blocks"].values()
                  for address, value in block["words"].items()}
        words = [int(value, 16) for value in task["header_words"]]
        for encoded in task["records"]:
            header = int(encoded["header"], 16)
            base = int(encoded["address"], 16)
            words.append(header)
            if target == "h14" and header & 0x80000000:
                mask = (header >> 15) & 0xFFFF
                addresses = [base] + [base + (bit + 1) * 4
                                      for bit in range(16) if mask & (1 << bit)]
            else:
                addresses = [base + offset * 4
                             for offset in range(encoded["count"])]
            words.extend(values[address] for address in addresses)
        if len(words) * 4 != task["size_bytes"]:
            raise ValueError("decoded words do not cover the task")
        tasks.append(words)
    return tasks

def text_bytes(record: dict) -> int:
    section = record["program_descriptor"].get("task_section")
    if section:
        return int(section["size"])
    if "text_words" in record["program_descriptor"]:
        return int(record["program_descriptor"]["text_words"]) * 4
    raise ValueError("program descriptor records no text size")


def build_stream(tasks: list[list[int]], total: int,
                 target: str) -> list[int]:
    if target == "h13":
        words = [word for task in tasks for word in task]
        if len(words) * 4 != total:
            raise ValueError(f"H13 stream {len(words) * 4} vs {total}")
        return words
    body = sum((len(task) * 4 + 15) & ~15 for task in tasks)
    prefix = total - body
    if prefix not in (0, 16):
        raise ValueError(f"H14 prefix frame is {prefix} bytes")
    words = [0] * (prefix // 4)
    for task in tasks:
        words.extend(task)
        padding = ((len(task) * 4 + 15) & ~15) - len(task) * 4
        words.extend([0] * (padding // 4))
    if len(words) * 4 != total:
        raise ValueError("H14 stream does not fill __TEXT/__text")
    return words


# --------------------------------------------------------------------------
# Surfaces, descriptors, labels
# --------------------------------------------------------------------------

def surface_layouts(record: dict) -> tuple[list[dict], dict, int]:
    descriptors = record["tensor_descriptors"]
    addresses = [int(value, 16) for value in
                 record["program_descriptor"]["resource_addresses"]
                 if isinstance(value, str)]
    live = [address for address in addresses if address]
    scratch = (min(live) - 0x30000000) if live and target_is_h13(record) else 0
    if live and not target_is_h13(record):
        scratch = 0

    def layout(descriptor: dict) -> dict:
        n, c, h, w = descriptor["shape"]
        _, plane, row = descriptor["strides"][:3]
        batch = descriptor["strides"][0]
        allocation = (max(batch, descriptor["total_bytes"]) + 0x3FFF) & ~0x3FFF
        return {"index": descriptor["binding"],
                "nchw": [n, c, h, w, plane, row],
                "allocation": allocation}

    inputs = [layout(d) for d in descriptors if d["binding"] == 1]
    outputs = [layout(d) for d in descriptors if d["binding"] == 2]
    if len(outputs) != 1:
        raise ValueError("chain declares more than one result surface")
    return inputs, outputs[0], max(0, scratch)


def target_is_h13(record: dict) -> bool:
    return record["target"] == "h13"


def h14_descriptor_words(record: dict) -> tuple[int, int, int]:
    trailing = [int(word, 16) for word in
                record["program_descriptor"]["trailing_words"]]
    record_count = trailing[(0x860 - 0x810) // 4]
    scratch_word = trailing[(0x858 - 0x810) // 4]
    unresolved = trailing[(0x880 - 0x810) // 4]
    return record_count, scratch_word, unresolved


def label_tasks(records: list[dict], target: str) -> dict[str, list[dict]]:
    reference = mc.build_reference(
        mc.load_records(target, ROOT / "research/oracles"))
    return {record["case"]: [mc.label_task(task, reference)
                             for task in record["task_descriptors"]]
            for record in records}


def task_labels(signature_labels: list[dict], segments: list[dict]) -> list[dict]:
    """One {operation, fusion} per task, assigned in schedule order.

    The signature labeler names each task's closest single operation; walking
    the labels in emission order and consuming the matching scheduled segment
    attributes every task, and a matmul task names the consumer folded into it.
    """
    cursors = collections.Counter()
    labels = []
    for signature in signature_labels:
        names = signature["label"].split("|")
        kind = None
        for name in names:
            if name in LABEL_KINDS:
                kind = LABEL_KINDS[name]
                break
        if kind is None:
            labels.append({"operation": "staging", "fusion": None})
            continue
        matches = [s for s in segments if
                   (s["kind"] == kind[0] and
                    (kind[0] != "unary" or s["op"]["name"] == kind[1]))]
        position = cursors[kind] if kind[0] != "unary" else cursors[("unary", kind[1])]
        cursors[kind] += 1
        if kind[0] == "unary":
            cursors[("unary", kind[1])] += 1
        segment = matches[position] if position < len(matches) else None
        if segment is None:
            labels.append({"operation": kind[1] or kind[0], "fusion": None})
            continue
        operation = segment["op"]["name"]
        fusion = segment.get("fusion")
        fusion_name = fusion[0] if fusion else None
        if fusion_name == "lut":
            fusion_name = {"gelu": "gelu", "silu": "silu", "exp": "exp",
                           "sigmoid": "sigmoid", "tanh": "tanh",
                           "sqrt": "sqrt"}[{
                               "Gelu": "gelu", "Silu": "silu",
                               "Exponential": "exp", "Sigmoid": "sigmoid",
                               "Tanh": "tanh", "SquareRoot": "sqrt"}[fusion[1]]]
        labels.append({"operation": operation, "fusion": fusion_name})
    return labels


# --------------------------------------------------------------------------
# Template emission
# --------------------------------------------------------------------------

def selected(target: str) -> list[dict]:
    records = []
    for path in sorted((ROOT / f"research/oracles/{target}").glob("chain_*.json")):
        record = json.loads(path.read_text())
        if record.get("error") is not None:
            continue
        if not record.get("task_descriptors"):
            continue
        if record.get("program_count", 1) != 1:
            continue
        if record.get("task_decode_errors"):
            continue
        records.append(record)
    return records


def build_entries(target: str, labels: dict[str, list[dict]]) -> tuple[list[dict], list[tuple[str, str]]]:
    accepted, refused = [], []
    for record in selected(target):
        case = record["case"]
        try:
            segments = op_segments(record)
            assign_operands(segments)
            fitted = fit_constant_plan(record, target, segments)
            if fitted is None:
                refused.append((case, "constant plan did not fit"))
                continue
            entries, dedup = fitted
            tasks = task_words(record)
            words = build_stream(tasks, text_bytes(record), target)
            inputs, output, scratch = surface_layouts(record)
            labels_for_case = task_labels(labels[case], segments)
        except ScheduleRefusal as error:
            refused.append((case, str(error)))
            continue
        except ValueError as error:
            refused.append((case, str(error)))
            continue
        accepted.append({
            "case": case, "key": schedule_key(segments), "words": words,
            "first_task": len(tasks[0]) * 4, "task_count": len(tasks),
            "scratch": scratch, "inputs": inputs, "output": output,
            "labels": labels_for_case, "entries": entries, "dedup": dedup,
            "record": record,
        })
    return accepted, refused


def c_string(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit_target(target: str, labels: dict[str, list[dict]],
                stream_out) -> tuple[int, list[tuple[str, str]]]:
    accepted, refused = build_entries(target, labels)
    out = stream_out
    out.write(f"// Generated from decoded {target.upper()} chain oracle task "
              "words and constant sections by\n"
              "// research/mint_sched_probes.py --emit-templates. No HWX "
              "container bytes; regenerate after\n"
              "// re-minting the oracles.\n\n")
    for index, entry in enumerate(accepted):
        words = entry["words"]
        out.write("static constexpr std::uint32_t kChainStream%d[] = {\n" % index)
        for start in range(0, len(words), 8):
            out.write("    " + " ".join(f"0x{word:08x},"
                                        for word in words[start:start + 8]) + "\n")
        out.write("};\n")
        out.write("static constexpr ChainConstantSegment kChainConstants"
                  f"{index}[] = {{\n")
        segment_kinds = {"weight": "Weight", "fused-weight": "FusedLUTWeight",
                         "fused-bias-weight": "FusedBiasWeight",
                         "fused-bias-lut-weight": "FusedBiasLUTWeight"}
        for segment in entry["entries"]:
            kind = segment["kind"]
            if kind in segment_kinds:
                table = TABLE_IDS.get(segment.get("table", ""), 0)
                bias = segment.get("bias_operand", 0)
                out.write(f"    {{ChainSegment::{segment_kinds[kind]}, "
                          f"{segment['operand']}, {bias}, {table}, "
                          f"{segment.get('slices', 1)}}},\n")
            elif kind == "table":
                table = TABLE_IDS.get(segment["table"], 0)
                out.write(f"    {{ChainSegment::Table, 0, {table}, "
                          f"{segment['bytes']}, 1}},\n")
            elif kind == "per-channel":
                out.write(f"    {{ChainSegment::PerChannelAdd, "
                          f"{segment['operand']}, 0, 0, 1}},\n")
            elif kind == "full-constant":
                out.write(f"    {{ChainSegment::FullConstant, "
                          f"{segment['operand']}, 0, 0, 1}},\n")
            elif kind == "floor":
                out.write(f"    {{ChainSegment::ZeroFloor, 0, 0, "
                          f"{FLOOR_BYTES}, 1}},\n")
        out.write("};\n")
        out.write("static constexpr ChainTaskLabel kChainTaskLabels"
                  f"{index}[] = {{\n")
        for label in entry["labels"]:
            fusion = label["fusion"]
            out.write(f"    {{{c_string(label['operation'])}, "
                      f"{c_string(fusion) if fusion else 'nullptr'}}},\n")
        out.write("};\n")
    out.write("static constexpr ChainTemplate kChainTemplates[] = {\n")
    for index, entry in enumerate(accepted):
        layouts = list(entry["inputs"])
        while len(layouts) < 3:
            layouts.append({"index": 0, "nchw": [0] * 6, "allocation": 0})
        inputs = ", ".join(
            f"{{{layout['index']}, "
            f"{{{', '.join(str(v) for v in layout['nchw'])}}}, "
            f"{layout['allocation']}}}" for layout in layouts)
        output = entry["output"]
        h14 = ""
        if target == "h14":
            record_count, scratch_word, unresolved = h14_descriptor_words(
                entry["record"])
            h14 = f", {record_count}, {scratch_word}, {unresolved}"
        out.write(
            f"    {{{c_string(entry['case'])}, {c_string(entry['key'])},\n"
            f"        kChainStream{index}, std::size(kChainStream{index}), "
            f"{entry['first_task']}, {entry['task_count']}, "
            f"{entry['scratch']},\n"
            f"        {{{inputs}}}, {len(entry['inputs'])},\n"
            f"        {{{output['index']}, "
            f"{{{', '.join(str(v) for v in output['nchw'])}}}, "
            f"{output['allocation']}}},\n"
            f"        kChainConstants{index}, "
            f"std::size(kChainConstants{index}), {int(entry['dedup'])}, "
            f"kChainTaskLabels{index}{h14}}},  // {entry['case']}\n")
    out.write("};\n")
    return len(accepted), refused


def emit() -> dict[str, tuple[int, list[tuple[str, str]]]]:
    summary = {}
    labels = {target: label_tasks(selected(target), target)
              for target in ("h13", "h14")}
    for target in ("h13", "h14"):
        import io
        buffer = io.StringIO()
        accepted, refused = emit_target(target, labels[target], buffer)
        OUTPUTS[target].write_text(buffer.getvalue())
        summary[target] = (accepted, refused)
        print(f"{target}: {accepted} chain templates, {len(refused)} refused")
        for case, why in refused:
            print(f"  refused {case}: {why}")
    return summary


def check() -> bool:
    import io
    labels = {target: label_tasks(selected(target), target)
              for target in ("h13", "h14")}
    for target in ("h13", "h14"):
        buffer = io.StringIO()
        accepted, _ = emit_target(target, labels[target], buffer)
        if buffer.getvalue() != OUTPUTS[target].read_text():
            print(f"{OUTPUTS[target].name} is stale; regenerate it with "
                  f"research/mint_sched_probes.py --emit-templates "
                  f"({accepted} templates)")
            return False
    return True


# --------------------------------------------------------------------------
# The sched_ probe campaign
# --------------------------------------------------------------------------

def _constant_bytes(data: bytes) -> bytes:
    """Return __TEXT/__const without retaining the HWX image."""
    _, _, _, _, count, _, _, _ = struct.unpack_from("<8I", data)
    cursor = 32
    for _ in range(count):
        command, size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = om.cstring(fields[2])
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
                if (segment, om.cstring(entry[0])) == ("__TEXT", "__const"):
                    return data[entry[4]:entry[4] + entry[3]]
                section_cursor += 80
        cursor += size
    raise ValueError("HWX has no __TEXT/__const section")


def _compile_constant(item: dict, target: str, tool: Path, weights: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="mil-hwx-sched-layout-") as temporary:
        capture = Path(temporary) / "capture"
        compiled = Path(temporary) / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        (capture / "weights.bin").write_bytes(weights)
        result = subprocess.run([str(tool), str(capture), str(compiled), target],
                                capture_output=True, text=True,
                                timeout=om.COMPILE_TIMEOUT_SECONDS, check=False)
        image = compiled / "model.hwx"
        if result.returncode or not image.is_file():
            raise RuntimeError((result.stderr or result.stdout).strip() or
                               f"compiler exited {result.returncode}")
        return _constant_bytes(image.read_bytes())


def _channel_payload(channels: int, plane: int | None) -> bytes:
    low, high = 0x3801, 0x3C82
    if plane is None:
        values = [low] * channels
    elif plane < 0:
        values = [high] * channels
    else:
        values = [high if channel >> plane & 1 else low
                  for channel in range(channels)]
    return struct.pack(f"<{channels}H", *values)


def mint_bias_layout(item: dict, target: str, output: Path, tool: Path,
                     source_commit: str) -> str:
    """Mint a bit-plane map that names every stored fused-bias byte."""
    shapes = [tuple(shape) for shape in item["parameters"]["constant_matrices"]]
    channels = shapes[-1][-1]
    weight = struct.pack("<H", VALUE_BASE) * math.prod(shapes[0])

    def section(plane: int | None) -> bytes:
        return _compile_constant(
            item, target, tool,
            weight_blob(shapes, [weight, _channel_payload(channels, plane)]))

    baseline = section(None)
    saturated = section(-1)
    planes = [section(plane) for plane in range(max(1, (channels - 1).bit_length()))]
    if len({len(baseline), len(saturated), *(len(value) for value in planes)}) != 1:
        raise RuntimeError("bias layout probe constant sizes differ")
    low = struct.pack("<H", 0x3801)
    high = struct.pack("<H", 0x3C82)
    sources = []
    for offset, (before, after) in enumerate(zip(baseline, saturated)):
        if before == after:
            continue
        channel = sum(1 << plane for plane, image in enumerate(planes)
                      if image[offset] != before)
        half = low.index(before) if before in low else None
        if half is None or after != high[half]:
            raise RuntimeError(f"unclassified fused-bias byte at {offset}")
        sources.append([offset, channel, half])
    if len(sources) != channels * 2 or len({entry[1] for entry in sources}) != channels:
        raise RuntimeError("fused-bias map does not name every channel byte exactly once")
    record = {
        "schema_version": 1, "case": item["name"],
        "family": "sched_bias_layout", "parameters": item["parameters"],
        "target": target, "source_commit": source_commit,
        "compiler": {"tool": str(tool),
                     "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
                     "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "host": platform.node(), "platform": platform.platform()},
        "constant_section": {"size": len(baseline),
                             "baseline_sha256": hashlib.sha256(baseline).hexdigest(),
                             "saturated_sha256": hashlib.sha256(saturated).hexdigest(),
                             "plane_sha256": [hashlib.sha256(value).hexdigest()
                                              for value in planes]},
        "bias_byte_sources": sources, "error": None,
    }
    destination = output / target / f"{item['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return "decoded"
def dedup_probes() -> list[dict]:
    """Chains whose constants repeat, to measure Apple's deduplication.

    Every projection reads the same blob record, so the declared weight bytes
    are four times the stored ones if Apple keeps one copy per distinct
    payload -- which is what the encoder must reproduce.
    """
    probes = []
    for width, sequence in ((256, 64), (512, 128)):
        shape = (1, sequence, width)
        chain = mc.Chain({"x": shape})
        shared = chain.constant((width, width))
        projected = [chain.matmul("x", shape, shared) for _ in range(4)]
        result = projected[0]
        for value in projected[1:]:
            result = chain.binary("add", result, value, shape)
        probes.append(mc.chain_case(
            f"sched_dedup4_d{width}_s{sequence}", "sched_dedup",
            {"width": width, "sequence": sequence, "copies": 4,
             "shared": True}, chain, result))
    return probes


def scratch_probes() -> list[dict]:
    """Tiled feed-forwards, to sweep the scratch fraction rule."""
    probes = []
    for width, sequence in ((256, 128), (256, 192), (512, 128), (512, 256),
                            (1024, 128)):
        shape = (1, sequence, width)
        chain = mc.Chain({"x": shape})
        result = chain.feed_forward("x", shape, 4 * width)
        probes.append(mc.chain_case(
            f"sched_scratch_ffn_d{width}_s{sequence}", "sched_scratch",
            {"width": width, "sequence": sequence, "hidden": 4 * width},
            chain, result))
    return probes


def bias_probes() -> list[dict]:
    """Fused per-channel bias sizes plus one byte-level layout probe."""
    probes = []
    for width, columns in ((256, 64), (256, 128), (256, 256),
                           (256, 512), (256, 1024), (1024, 256)):
        shape = (1, 64, width)
        chain = mc.Chain({"x": shape})
        projected = chain.project("x", shape, columns)
        result = chain.bias(projected, shape[:-1] + (columns,), "bcast")
        probes.append(mc.chain_case(
            f"sched_bias_c{columns}_d{width}_s64", "sched_bias",
            {"columns": columns, "width": width, "sequence": 64},
            chain, result))
    for width, columns in ((256, 1024), (1024, 256)):
        shape = (1, 64, width)
        output_shape = (1, 64, columns)
        chain = mc.Chain({"x": shape})
        projected = chain.project("x", shape, columns)
        biased = chain.bias(projected, output_shape, "bcast")
        result = chain.unary("gelu", biased, output_shape)
        probes.append(mc.chain_case(
            f"sched_bias_lut_c{columns}_d{width}_s64", "sched_bias_lut",
            {"columns": columns, "width": width, "sequence": 64},
            chain, result))
    layout = next(item for item in probes
                  if item["parameters"]["columns"] == 256 and
                  item["parameters"]["width"] == 256)
    layout = dict(layout, name="sched_bias_layout_c256_d256_s64",
                  family="sched_bias_layout", layout_probe=True)
    return probes + [layout]


def softmax_probes() -> list[dict]:
    probes = []
    for heads in (1, 8, 12):
        shape = (heads, 128, 128)
        chain = mc.Chain({"x": shape})
        result = chain.softmax("x", shape)
        probes.append(mc.chain_case(
            f"sched_softmax_surface_h{heads}_s128", "sched_softmax",
            {"heads": heads, "sequence": 128, "source": "surface"},
            chain, result))
    for heads in (1, 8, 12):
        shape = (heads, 128, 64)
        chain = mc.Chain({"q": shape, "k": shape, "v": shape})
        result = chain.attention("q", "k", "v", shape)
        probes.append(mc.chain_case(
            f"sched_softmax_scratch_h{heads}_s128", "sched_softmax",
            {"heads": heads, "sequence": 128, "source": "scratch"},
            chain, result))
    return probes


def fused_lut_probes() -> list[dict]:
    probes = []
    for sequence in (128, 256, 512):
        shape = (1, sequence, 512)
        chain = mc.Chain({"x": shape})
        result = chain.feed_forward("x", shape, 2048)
        probes.append(mc.chain_case(
            f"sched_fused_lut_d512_s{sequence}", "sched_fused_lut",
            {"width": 512, "sequence": sequence, "hidden": 2048},
            chain, result))
    return probes


def branch_order_probes() -> list[dict]:
    probes = []
    shape = (1, 64, 256)
    for count in (2, 3):
        chain = mc.Chain({"x": shape})
        projected = [chain.project("x", shape, 256) for _ in range(count)]
        result = projected[0]
        for value in projected[1:]:
            result = chain.binary("add", result, value, shape)
        probes.append(mc.chain_case(
            f"sched_branch{count}_d256_s64", "sched_branch_order",
            {"branches": count, "shared_input": True}, chain, result))
    chain = mc.Chain({"x": shape})
    result = "x"
    for _ in range(3):
        result = chain.project(result, shape, 256)
    probes.append(mc.chain_case(
        "sched_serial3_d256_s64", "sched_branch_order",
        {"branches": 3, "shared_input": False}, chain, result))
    return probes


def campaign() -> list[dict]:
    return (dedup_probes() + scratch_probes() + bias_probes() +
            softmax_probes() + fused_lut_probes() + branch_order_probes())


def selected_cases(pattern: str | None) -> list[dict]:
    return [case for case in campaign() if not pattern or pattern in case["name"]]


def local_run(args: argparse.Namespace) -> int:
    selected = selected_cases(args.case)
    if not selected:
        print("no sched_ cases selected", file=sys.stderr)
        return 2
    tool = Path(args.oracle_tool)
    output = Path(args.output)
    decoded = rejected = 0
    for target in args.targets:
        for item in selected:
            destination = output / target / f"{item['name']}.json"
            if destination.exists() and not args.force:
                existing = json.loads(destination.read_text())
                decoded += existing.get("error") is None
                rejected += existing.get("error") is not None
                continue
            if item.get("layout_probe"):
                status = mint_bias_layout(item, target, output, tool,
                                          args.source_commit)
            else:
                shapes = [tuple(shape) for shape
                          in item["parameters"]["constant_matrices"]]
                item["weights"] = weight_blob(shapes) if shapes else None
                try:
                    status = om.run_case(item, target, output, tool,
                                         args.source_commit)
                finally:
                    item["weights"] = None
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} "
          f"decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-sched-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-sched-probes."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", str(script),
                        str(script.with_name("mint_chain_probes.py")),
                        str(script.with_name("mint_oracles.py")),
                        str(script.with_name("h13_td.py")),
                        f"{args.host}:{root}/"], check=True)
        command = ["python3", f"{root}/{script.name}", "--local",
                   "--oracle-tool", args.oracle_tool,
                   "--output", f"{root}/oracles",
                   "--source-commit", args.source_commit,
                   "--targets", *args.targets]
        if args.case:
            command.extend(["--case", args.case])
        if args.force:
            command.append("--force")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
        with tempfile.TemporaryDirectory(prefix="mil-hwx-sched-json-") as staging:
            subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                            staging], check=True)
            shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


def report() -> int:
    for target in ("h13", "h14"):
        records = selected(target)
        families = collections.Counter(r.get("family") for r in records)
        print(f"{target}: {len(records)} decoded chain oracles {dict(families)}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--emit-templates", action="store_true",
                        help="regenerate the chain template tables")
    parser.add_argument("--check", action="store_true",
                        help="verify the checked-in tables match the oracles")
    parser.add_argument("--report", action="store_true",
                        help="print chain oracle coverage")
    parser.add_argument("--local", action="store_true",
                        help="mint sched_ probes with a local oracle tool")
    parser.add_argument("--host",
                        help="mint sched_ probes on this SSH host")
    parser.add_argument("--oracle-tool",
                        default="/tmp/h13-oracle/bin/ane-compile-hwx",
                        help="Apple compiler the local mint invokes")
    parser.add_argument("--output", type=Path, default=ROOT / "research/oracles",
                        help="where decoded records land")
    parser.add_argument("--source-commit", default="HEAD",
                        help="commit the records tag themselves with")
    parser.add_argument("--targets", nargs="+", default=["h13", "h14"],
                        choices=["h13", "h14"])
    parser.add_argument("--case", help="restrict the mint to one case name")
    parser.add_argument("--force", action="store_true",
                        help="re-mint records that already exist")
    args = parser.parse_args(argv)
    if args.emit_templates:
        emit()
        return 0
    if args.check:
        return 0 if check() else 1
    if args.report:
        return report()
    if args.local:
        return local_run(args)
    if args.host:
        return remote_run(args)
    parser.error("choose --emit-templates, --check, --report, --local or --host")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
