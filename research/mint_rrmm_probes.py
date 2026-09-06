#!/usr/bin/env python3
"""Mint the runtime-runtime matmul, extended matvec, and broadcast probes, and
emit the H13 encoder templates for all three families.

The envelope campaign in ``mint_oracles.py`` established that Apple accepts
matmul with both operands runtime in all four transpose combinations, keeps
every accepted form in one program, and accepts per-channel and spatial
broadcasts. It swept those forms coarsely: the runtime-runtime grid jumps from
M=16 to M=128, the constant-weight grid skips M=64, and nothing separates a
rank-2 case from a rank-3 case at the same M, K and N. These probes fill those
gaps, so the emitted templates cover what an attention block and a
convolution bias actually ask for, and they answer three questions the
envelope left open:

* does a rank-3 runtime-runtime matmul emit the same words as the rank-2 case
  at the same M, K and N?
* where between M=128 and M=256 does the constant-weight packing switch its
  row-group size (the `32768/K` term this script's `weight_group` records)?
* does Apple accept a `transpose_y=false` BLOBFILE weight, which the first
  campaign refused and the envelope campaign never retested?

``--emit-templates`` reads the decoded JSON back and writes
``plugins/H13/H13EnvelopeTemplates.inc``: the exact task stream Apple emitted
for each covered geometry, with no HWX container bytes. Every emitted row is
checked against the encoder's own surface, scratch, binding-order and
constant-section formulas first, so a template is never written for a case the
encoder would reproduce differently.

    python3 research/mint_rrmm_probes.py --host macstudio --targets h13
    python3 research/mint_rrmm_probes.py --emit-templates
    python3 research/mint_rrmm_probes.py --check
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402
from h13_td import decode_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "plugins/H13/H13EnvelopeTemplates.inc"
CONSTANT_ALIGNMENT = 0x40
SURFACE_ALIGNMENT = 0x4000
SURFACE_BASE = 0x30000000
ROW_FLOOR = 64

BINARY_OPERATIONS = {"add": 0, "mul": 1, "maximum": 2, "minimum": 3, "sub": 4,
                     "real_div": 5}
OPERAND_ENUM = {"runtime": "BroadcastOperand::Runtime",
                "scalar": "BroadcastOperand::Scalar",
                "blob": "BroadcastOperand::Constant"}
# fp16 1.0, the scale Apple stores beside a per-channel bias.
UNIT_HALF = b"\x00\x3c"


# --------------------------------------------------------------------------
# Probe campaign
# --------------------------------------------------------------------------

def runtime_matmul(rows: int, reduction: int, columns: int,
                   batch: int | None = None, transpose_x: bool = False,
                   transpose_y: bool = True) -> dict[str, Any]:
    """One matmul with both operands runtime, the attention form."""
    item = om.env_matmul(rows, reduction, columns, batch=batch,
                         transpose_x=transpose_x, transpose_y=transpose_y,
                         x_storage="runtime", w_storage="runtime")
    item["name"] = "rrmm_" + item["name"][len("env_mm_"):]
    item["family"] = "rrmm_matmul"
    return item


def constant_matmul(rows: int, reduction: int, columns: int,
                    batch: int | None = None, transpose_x: bool = False,
                    transpose_y: bool = True) -> dict[str, Any]:
    """One matmul whose weight is a BLOBFILE constant."""
    item = om.env_matmul(rows, reduction, columns, batch=batch,
                         transpose_x=transpose_x, transpose_y=transpose_y,
                         x_storage="runtime", w_storage="blob")
    item["name"] = "rrmm_" + item["name"][len("env_mm_"):]
    item["family"] = "rrmm_matvec"
    return item


def broadcast(operation: str, x_shape: tuple[int, ...],
              y_shape: tuple[int, ...], mode: str = "runtime") -> dict[str, Any]:
    item = om.env_broadcast(operation, x_shape, y_shape, mode=mode)
    item["name"] = "rrmm_bcast_" + item["name"][len("env_bcast_"):]
    item["family"] = "rrmm_broadcast"
    return item


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 1. Rank-2 against rank-3 at the same geometry. The envelope campaign ran
    # rank 3 only at small shapes and rank 2 only at large ones, so nothing in
    # the corpus proves the leading batch of 1 is free for runtime-runtime.
    for rows, reduction, columns in ((64, 64, 64), (128, 64, 128)):
        for transpose_y in (False, True):
            cases.append(runtime_matmul(rows, reduction, columns,
                                        transpose_y=transpose_y))
            cases.append(runtime_matmul(rows, reduction, columns, batch=1,
                                        transpose_y=transpose_y))

    # 2. How M drives the task count and the scaling words, at a geometry
    # small enough to mint quickly. 48 and 96 are not multiples of 64, so a
    # word that tracks a tile count separates from one that tracks M.
    for rows in (16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
        for transpose_y in (False, True):
            cases.append(runtime_matmul(rows, 64, 64, transpose_y=transpose_y))

    # 3. K and N sweeps at fixed M, which separate the reduction-driven words
    # from the column-driven ones.
    for reduction in (32, 64, 128, 256, 512, 1024, 2048):
        cases.append(runtime_matmul(64, reduction, 64))
    for columns in (32, 64, 128, 256, 512, 1024, 2048):
        cases.append(runtime_matmul(64, 64, columns))

    # 4. All four transpose combinations at one asymmetric geometry, plus the
    # attention pair (scores then context) at a size the envelope skipped.
    for transpose_x in (False, True):
        for transpose_y in (False, True):
            cases.append(runtime_matmul(64, 128, 256, transpose_x=transpose_x,
                                        transpose_y=transpose_y))
    for sequence in (32, 96):
        cases.append(runtime_matmul(sequence, 64, sequence, batch=1,
                                    transpose_y=True))
        cases.append(runtime_matmul(sequence, sequence, 64, batch=1,
                                    transpose_y=False))

    # 5. Constant-weight fill-ins. M=64 is missing from the envelope grid, and
    # M=144 through M=224 bracket the row count at which the packing's
    # row-group size starts tracking 32768/K: M=128 still packs 16 rows per
    # group at K=4096 and M=192 packs 8.
    for rows in (64, 144, 160, 176, 192, 224):
        for reduction, columns in ((2048, 2048), (4096, 2048)):
            cases.append(constant_matmul(rows, reduction, columns))
    cases.append(constant_matmul(192, 8192, 2048))
    for rows in (16, 128, 512):
        cases.append(constant_matmul(rows, 2048, 2048, transpose_x=True))
    # The transposed-x form at the geometries the shipped compiler already
    # lowers, where the corpus so far only proves the untransposed program.
    for reduction in (256, 512):
        cases.append(constant_matmul(1, reduction, 512, transpose_x=True))
    # Small N, where the row-group size is N/16 rather than 16.
    for columns in (64, 128, 256):
        cases.append(constant_matmul(1, 256, columns))
        cases.append(constant_matmul(32, 256, columns))

    # 6. Does Apple accept a transpose_y=false BLOBFILE weight? The first
    # campaign refused every such case with an inline constant and the
    # envelope campaign only retested it with both operands runtime.
    for rows, reduction, columns in ((1, 256, 256), (32, 2048, 2048),
                                     (128, 2048, 2048)):
        cases.append(constant_matmul(rows, reduction, columns,
                                     transpose_y=False))

    # 7. Broadcast fill-ins: the channel counts a transformer and a
    # convolution bias use, as a runtime operand and as a BLOBFILE constant.
    for operation in ("add", "mul"):
        for channels in (256, 1024, 3072):
            cases.append(broadcast(operation, (1, channels, 8, 8),
                                   (1, channels, 1, 1)))
            cases.append(broadcast(operation, (1, channels, 8, 8),
                                   (1, channels, 1, 1), mode="blob"))
        cases.append(broadcast(operation, (1, 256, 16, 16), (1, 1, 16, 16)))
        cases.append(broadcast(operation, (1, 256, 32, 32), (1, 256, 1, 1)))
        cases.append(broadcast(operation, (1, 256, 32, 32),
                               (1, 256, 1, 1), mode="blob"))
        cases.append(broadcast(operation, (4, 256, 8, 8), (1, 256, 1, 1)))

    # The sweeps overlap at their shared corner geometries; one record per
    # case name is enough, and `--force` re-mints it once.
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
        if not item["name"].startswith("rrmm_"):
            raise AssertionError(f"probe {item['name']} lacks the rrmm_ prefix")
    return list(unique.values())


def local_run(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    tool = Path(args.oracle_tool)
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
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
            status = om.run_case(item, target, output, tool, args.source_commit)
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} "
          f"decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-rrmm-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-rrmm-probes."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", str(script),
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
        if args.list:
            command.append("--list")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
        if not args.list:
            with tempfile.TemporaryDirectory(prefix="mil-hwx-rrmm-json-") as staging:
                subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


# --------------------------------------------------------------------------
# Task-stream rebuild
# --------------------------------------------------------------------------

def task_words(task: dict[str, Any]) -> list[int]:
    """Rebuild one task descriptor's words from its decoded form."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        base = int(record["address"], 16)
        words.append(int(record["header"], 16))
        words.extend(registers[base + index * 4]
                     for index in range(record["count"]))
    if len(words) * 4 != task["size_bytes"]:
        raise ValueError("rebuilt task size differs from the decoded size")
    return words


def stream_words(record: dict[str, Any]) -> list[int]:
    """Rebuild the whole linked task array, including alignment padding."""
    tasks = record["task_descriptors"]
    words: list[int] = []
    for index, task in enumerate(tasks):
        rebuilt = task_words(task)
        if decode_task(struct.pack(f"<{len(rebuilt)}I", *rebuilt), "h13") != task:
            raise ValueError(f"{record['case']} task {index} does not round-trip")
        words.extend(rebuilt)
        next_offset = int(task["header_words"][7], 16)
        if index + 1 == len(tasks):
            if next_offset:
                raise ValueError(f"{record['case']} final task links onward")
            continue
        if next_offset % 4 or next_offset < len(words) * 4:
            raise ValueError(f"{record['case']} task {index} link is invalid")
        words.extend([0] * (next_offset // 4 - len(words)))
    return words


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


# --------------------------------------------------------------------------
# Surfaces, exactly as plugins/H13/H13Program.cpp builds them
# --------------------------------------------------------------------------

def elementwise_surface(shape: tuple[int, ...]) -> dict[str, Any]:
    batch, channels, height, width = shape
    row = max(ROW_FLOOR, width * 2)
    plane = row * height
    element = plane * channels
    return {"shape": [batch, channels, height, width],
            "strides": [element, plane, row, 2], "total": element,
            "allocation": align(batch * element, SURFACE_ALIGNMENT)}


def matmul_surface(rows: int, width: int) -> dict[str, Any]:
    row = max(ROW_FLOOR, width * 2)
    plane = row * rows
    return {"shape": [1, 1, rows, width], "strides": [plane, plane, row, 2],
            "total": plane, "allocation": align(plane, SURFACE_ALIGNMENT)}


def check_surfaces(record: dict[str, Any], surfaces: list[dict[str, Any]],
                   binding_order: list[int]) -> int:
    """Compares the encoder's surfaces and VM layout with the oracle's, and
    returns the __DATA/__bss scratch allocation Apple placed below them.

    `surfaces` is in descriptor order (inputs, then the output);
    `binding_order` indexes it in the order Apple assigns addresses.
    """
    case = record["case"]
    decoded = record["tensor_descriptors"]
    if len(decoded) != len(surfaces):
        raise SystemExit(f"{case} has {len(decoded)} surfaces, encoder builds "
                         f"{len(surfaces)}")
    for index, (tensor, surface) in enumerate(zip(decoded, surfaces)):
        expected = {"element_code": 5, "shape": surface["shape"],
                    "strides": surface["strides"],
                    "total_bytes": surface["total"]}
        actual = {key: value for key, value in tensor.items() if key != "binding"}
        if actual != expected:
            raise SystemExit(f"{case} surface {index} is {actual}, but the "
                             f"encoder formula gives {expected}")
        role = 2 if index + 1 == len(surfaces) else 1
        if tensor["binding"] != role:
            raise SystemExit(f"{case} surface {index} has binding "
                             f"{tensor['binding']}, expected {role}")
    text = int(record["program_descriptor"]["text_address"], 16)
    scratch = text - SURFACE_BASE - sum(item["allocation"] for item in surfaces)
    if scratch < 0:
        raise SystemExit(f"{case} surfaces do not fit below its text address")
    cursor = SURFACE_BASE + scratch
    expected_addresses = []
    for index in binding_order:
        expected_addresses.append(cursor)
        cursor += surfaces[index]["allocation"]
    addresses = [int(value, 16) for value in
                 record["program_descriptor"]["resource_addresses"][:len(surfaces)]]
    if addresses != expected_addresses:
        raise SystemExit(
            f"{case} surface addresses are "
            f"{[hex(value) for value in addresses]}, but the encoder's binding "
            f"order gives {[hex(value) for value in expected_addresses]}")
    return scratch


def constant_offset(record: dict[str, Any], words: list[int]) -> int:
    descriptor = record["program_descriptor"]
    offset = int(descriptor["constant_address"], 16) - \
        int(descriptor["text_address"], 16)
    if offset != align(len(words) * 4, CONSTANT_ALIGNMENT):
        raise SystemExit(f"{record['case']} constant offset {offset} is not the "
                         "aligned end of its task stream")
    return offset


# --------------------------------------------------------------------------
# Constant sections
# --------------------------------------------------------------------------
def weight_group(rows: int, reduction: int, columns: int) -> int:
    """Apple's row-group size for a packed [columns, reduction] fp16 weight.

    Sixteen weight rows interleave at halfword granularity, except that a
    column count below 256 interleaves `columns / 16` rows, and above 128 x
    rows -- the row count at which Apple starts partitioning the program into
    `1 + K * N / 2^19` tasks -- the group also caps at `32768 / reduction`
    halfwords. Both terms are measured: this rule reproduces the constant
    section hash of every decoded constant-weight oracle, M=128 with K=4096
    still packs 16 rows per group and M=144 packs 8, and no other rule in the
    corpus reproduces the partitioned sections. Where between 129 and 144 the
    switch happens is not resolved; every probed row count is a multiple of 16.
    """
    group = min(16, columns // 16)
    if rows > 128:
        group = min(group, 32768 // reduction)
    return max(1, group)


def pack_weights(rows: int, reduction: int, columns: int,
                 weights: bytes) -> bytes:
    """The constant section for a [columns, reduction] row-major fp16 weight."""
    group = weight_group(rows, reduction, columns)
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


def per_channel_constants(operation: str, channels: int,
                          constant: bytes) -> bytes:
    """Apple's per-channel section: a bias block then a scale block, each one
    fp16 value per channel. `add` puts the constant in the bias block and 1.0
    in the scale block; `mul` leaves the bias zero and scales by the constant.
    """
    if len(constant) != channels * 2:
        raise ValueError("per-channel constant is not one fp16 per channel")
    if operation == "add":
        return constant + UNIT_HALF * channels
    if operation == "mul":
        return bytes(channels * 2) + constant
    raise ValueError(f"no decoded per-channel section for {operation}")


def uniform_blob(elements: int) -> bytes:
    """The bytes the campaign's `weights.bin` resolves to for `elements` fp16
    values. `mint_oracles.blob` writes its chunk header with unpadded fields,
    so Apple's loader and `ANEBlobResolver` both read the payload from file
    offset 0; the recorded sections show the header bytes, and reproducing them
    is what proves the section layout rather than the payload value.
    """
    return om.blob(om.half_payload(elements))[:elements * 2]


def check_constants(record: dict[str, Any], content: bytes) -> int:
    section = record["constant_section"]
    if len(content) != section["size"]:
        raise SystemExit(f"{record['case']} constant section is {section['size']} "
                         f"bytes, the encoder builds {len(content)}")
    digest = hashlib.sha256(content).hexdigest()
    if digest != section["sha256"]:
        raise SystemExit(f"{record['case']} constant section hashes {digest}, "
                         f"the oracle records {section['sha256']}")
    return len(content)


# --------------------------------------------------------------------------
# Template selection
# --------------------------------------------------------------------------

def matmul_key(record: dict[str, Any]) -> tuple:
    parameters = record["parameters"]
    return (parameters["rows"], parameters["reduction"], parameters["columns"],
            parameters["transpose_x"], parameters["transpose_y"],
            parameters["w_storage"] == "runtime")


def broadcast_key(record: dict[str, Any]) -> tuple:
    parameters = record["parameters"]
    operand = parameters["operand"]
    y_shape = (0, 0, 0, 0) if operand == "scalar" \
        else tuple(parameters["operand_shape"])
    return (parameters["operation"], operand, tuple(parameters["shape"]),
            y_shape)


def matmul_records(records: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if record["family"] not in ("env_matmul", "rrmm_matmul", "rrmm_matvec"):
            continue
        if record.get("error") is not None:
            continue
        parameters = record["parameters"]
        if parameters["x_storage"] != "runtime":
            # A constant `x` is a different program: the runtime operand is
            # `y`, and no encoder lowers that form.
            continue
        agree(selected, matmul_key(record), record)
    return selected


def broadcast_records(records: list[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    selected: dict[tuple, dict[str, Any]] = {}
    for record in records:
        if record["family"] not in ("env_broadcast", "rrmm_broadcast"):
            continue
        if record.get("error") is not None:
            continue
        agree(selected, broadcast_key(record), record)
    return selected


def agree(selected: dict[tuple, dict[str, Any]], key: tuple,
          record: dict[str, Any]) -> None:
    """Keeps one record per template key, and fails when two records that
    share a key disagree on anything the encoder reproduces."""
    previous = selected.setdefault(key, record)
    if previous is record:
        return
    for field in ("task_descriptors", "constant_section", "program_descriptor",
                  "tensor_descriptors"):
        if previous[field] != record[field]:
            raise SystemExit(f"{record['case']} and {previous['case']} share a "
                             f"template key but differ in {field}")


def load_records(target: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text())
            for path in sorted((ROOT / "research/oracles" / target).glob("*.json"))]


# --------------------------------------------------------------------------
# Emission
# --------------------------------------------------------------------------

def word_rows(words: list[int]) -> list[str]:
    return [f"    {', '.join(f'0x{word:08x}' for word in words[start:start + 8])},"
            for start in range(0, len(words), 8)]


def matmul_row(record: dict[str, Any], symbol: str,
               words: list[int]) -> str:
    parameters = record["parameters"]
    descriptor = record["program_descriptor"]
    rows = parameters["rows"]
    reduction = parameters["reduction"]
    columns = parameters["columns"]
    transpose_x = parameters["transpose_x"]
    transpose_y = parameters["transpose_y"]
    runtime_weight = parameters["w_storage"] == "runtime"
    x = matmul_surface(reduction, rows) if transpose_x \
        else matmul_surface(rows, reduction)
    output = matmul_surface(rows, columns)
    if runtime_weight:
        weight = matmul_surface(columns, reduction) if transpose_y \
            else matmul_surface(reduction, columns)
        # Apple lays the output out first, then the second operand, then `x`,
        # and declares the operands in the opposite order.
        scratch = check_surfaces(record, [weight, x, output], [2, 0, 1])
        constants = check_constants(record, bytes(record["constant_section"]["size"]))
    else:
        scratch = check_surfaces(record, [x, output], [1, 0])
        if not transpose_y:
            raise SystemExit(
                f"{record['case']} decoded a transpose_y=false constant weight; "
                "the packing for that form is not derived")
        constants = check_constants(record, pack_weights(
            rows, reduction, columns,
            uniform_blob(reduction * columns)))
        if constants != reduction * columns * 2:
            raise SystemExit(f"{record['case']} constant section is not K*N "
                             "halfwords")
    return (f"    {{{rows}, {reduction}, {columns}, "
            f"{'true' if transpose_x else 'false'}, "
            f"{'true' if transpose_y else 'false'}, "
            f"{'true' if runtime_weight else 'false'}, {symbol}, "
            f"std::size({symbol}), "
            f"{(descriptor['task_words_minus_one'] + 1) * 4}, "
            f"{descriptor['task_count']}, {constant_offset(record, words)}, "
            f"{constants}, {scratch}}},  // {record['case']}")


def broadcast_row(record: dict[str, Any], symbol: str,
                  words: list[int]) -> str:
    parameters = record["parameters"]
    descriptor = record["program_descriptor"]
    operation = parameters["operation"]
    operand = parameters["operand"]
    x_shape = tuple(parameters["shape"])
    output_shape = tuple(parameters["output_shape"])
    x = elementwise_surface(x_shape)
    output = elementwise_surface(output_shape)
    if operand == "runtime":
        y_shape = tuple(parameters["operand_shape"])
        y = elementwise_surface(y_shape)
        # Identical operands are laid out in declaration order; any broadcast
        # puts the output between them.
        order = [0, 1, 2] if y_shape == x_shape else [0, 2, 1]
        scratch = check_surfaces(record, [x, y, output], order)
        content = bytes(record["constant_section"]["size"])
    else:
        y_shape = (0, 0, 0, 0) if operand == "scalar" \
            else tuple(parameters["operand_shape"])
        scratch = check_surfaces(record, [x, output], [0, 1])
        content = bytes(record["constant_section"]["size"]) if operand == "scalar" \
            else per_channel_constants(operation, x_shape[1],
                                       uniform_blob(x_shape[1]))
    constants = check_constants(record, content)
    if output_shape != tuple(max(left, right)
                             for left, right in zip(x_shape, y_shape or x_shape)):
        raise SystemExit(f"{record['case']} output shape is not the broadcast "
                         "of its operands")
    return (f"    {{{BINARY_OPERATIONS[operation]}, {OPERAND_ENUM[operand]}, "
            f"{{{', '.join(map(str, x_shape))}}}, "
            f"{{{', '.join(map(str, y_shape))}}}, {symbol}, "
            f"std::size({symbol}), "
            f"{(descriptor['task_words_minus_one'] + 1) * 4}, "
            f"{descriptor['task_count']}, {constant_offset(record, words)}, "
            f"{constants}, {scratch}}},  // {record['case']}")


def emit(records: list[dict[str, Any]]) -> str:
    lines = [
        "// Generated from decoded H13 runtime-runtime matmul, extended matvec,",
        "// and broadcast oracle task words by",
        "// research/mint_rrmm_probes.py --emit-templates. No HWX container",
        "// bytes; regenerate after re-minting the oracles.",
        "",
    ]
    symbols: dict[tuple[int, ...], str] = {}
    body: list[str] = []

    def stream_symbol(record: dict[str, Any], prefix: str) -> tuple[str, list[int]]:
        words = stream_words(record)
        symbol = symbols.get(tuple(words))
        if symbol is None:
            symbol = f"k{prefix}{len(symbols)}"
            symbols[tuple(words)] = symbol
            body.append(f"static constexpr std::uint32_t {symbol}[] = {{")
            body.extend(word_rows(words))
            body.append("};")
        return symbol, words

    matmul_rows = []
    for key, record in sorted(matmul_records(records).items()):
        symbol, words = stream_symbol(record, "EnvelopeTask")
        matmul_rows.append(matmul_row(record, symbol, words))
    broadcast_rows = []
    for key, record in sorted(broadcast_records(records).items(), key=repr):
        symbol, words = stream_symbol(record, "EnvelopeTask")
        broadcast_rows.append(broadcast_row(record, symbol, words))
    if not matmul_rows or not broadcast_rows:
        raise SystemExit("no decoded matmul or broadcast oracles found")
    lines.extend(body)
    lines.append("static constexpr OracleMatmulTemplate kMatmulEnvelopeTasks[] = {")
    lines.extend(matmul_rows)
    lines.append("};")
    lines.append("static constexpr OracleBroadcastTemplate kBroadcastTasks[] = {")
    lines.extend(broadcast_rows)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def emit_templates(args: argparse.Namespace) -> int:
    generated = emit(load_records(args.targets[0]))
    path = Path(args.template_output)
    if args.check:
        if (path.read_text() if path.exists() else "") != generated:
            raise SystemExit(f"{path} is stale; regenerate it with "
                             "research/mint_rrmm_probes.py --emit-templates")
        print(f"H13 envelope templates: up to date ({path})")
        return 0
    path.write_text(generated)
    print(f"wrote {path} ({len(generated)} bytes)")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    mode.add_argument("--emit-templates", action="store_true")
    mode.add_argument("--check", action="store_true",
                      help="fail when the checked-in templates are stale")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=["h13"])
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--template-output", default=str(OUTPUT))
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.emit_templates or arguments.check:
        raise SystemExit(emit_templates(arguments))
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
