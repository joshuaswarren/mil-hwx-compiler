#!/usr/bin/env python3
"""Mint H13 softmax, layer_norm, and reduction oracles across the transformer
shape envelope, and emit the encoder template table from what decoded.

``mint_oracles.py`` covers two normalization shapes and one reduction shape,
which cannot separate a shape-driven task word from an axis-driven one. These
probes sweep the channel counts a transformer block actually uses
([1, C, 1, 1] with C in 64..4096), the last-axis softmax a attention score
matrix needs, layer_norm with and without gamma/beta, and reduction over the
channel axis versus the spatial axes. Every probe is weightless: gamma and beta
go in as inline fp16 tensor constants, so no case touches the BLOBFILE path.

No Apple HWX bytes are retained; this reuses ``mint_oracles.run_case``, which
records the same decoded words, descriptors, and section hashes as the shipped
campaign.

The ``--emit-templates`` mode reads the decoded JSON back and writes
``plugins/H13/H13NormTemplates.inc``: the exact task stream Apple emitted for
each covered (operation, shape, axes), with no HWX container bytes.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import platform
import re
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
HEADER_WORDS = 10
CONSTANT_ALIGNMENT = 0x40
SURFACE_ALIGNMENT = 0x4000

# The fp16 gamma and beta a layer_norm probe uses when it carries them: gamma
# 1.0 and beta 0.0 keep the numerics an identity scale, so a task-word
# difference against the gamma-free case is attributable to the operands alone.
GAMMA_BITS = "fp16(0x1p+0)"
BETA_BITS = "fp16(0x0p+0)"


def shape_name(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def axes_name(axes: tuple[int, ...]) -> str:
    return "_".join(str(axis).replace("-", "n") for axis in axes)


def softmax(shape: tuple[int, ...], axis: int) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    body = [
        f'int32 axis = const()[name = string("axis"), val = int32({axis})];',
        f'{kind} y = softmax(x = x, axis = axis)[name = string("y")];',
    ]
    name = f"norm_softmax_a{str(axis).replace('-', 'n')}_{shape_name(shape)}"
    return om.case(name, "normalization",
                   {"operation": "softmax", "shape": shape, "axis": axis},
                   om.program(f"{kind} x", body, "y"))


def layer_norm(shape: tuple[int, ...], axes: tuple[int, ...],
               affine: bool) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    axis_kind = f"tensor<int32, [{len(axes)}]>"
    normalized = tuple(shape[axis] for axis in axes)
    body = [
        f'{axis_kind} axes = const()[name = string("axes"), '
        f'val = {axis_kind}([{", ".join(map(str, axes))}])];',
    ]
    operands = ""
    if affine:
        affine_kind = om.tensor_type(normalized)
        elements = 1
        for dimension in normalized:
            elements *= dimension
        body.extend([
            f'{affine_kind} gamma = const()[name = string("gamma"), '
            f'val = {affine_kind}([{", ".join([GAMMA_BITS] * elements)}])];',
            f'{affine_kind} beta = const()[name = string("beta"), '
            f'val = {affine_kind}([{", ".join([BETA_BITS] * elements)}])];',
        ])
        operands = ", beta = beta, gamma = gamma"
    body.append(
        f'{kind} y = layer_norm(x = x, axes = axes{operands}, '
        'epsilon = fp32(0.00001))[name = string("y")];')
    suffix = "_affine" if affine else ""
    name = (f"norm_layer_norm_ax{axes_name(axes)}_{shape_name(shape)}{suffix}")
    return om.case(name, "normalization", {
        "operation": "layer_norm", "shape": shape, "axes": axes,
        "affine": affine,
    }, om.program(f"{kind} x", body, "y"))


def reduce(operation: str, shape: tuple[int, ...], axes: tuple[int, ...],
           keep_dims: bool = True) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    if keep_dims:
        output = tuple(1 if index in axes else value
                       for index, value in enumerate(shape))
    else:
        output = tuple(value for index, value in enumerate(shape)
                       if index not in axes)
    axis_kind = f"tensor<int32, [{len(axes)}]>"
    flag = "true" if keep_dims else "false"
    body = [
        f'{axis_kind} axes = const()[name = string("axes"), '
        f'val = {axis_kind}([{", ".join(map(str, axes))}])];',
        f'{om.tensor_type(output)} y = {operation}(x = x, axes = axes, '
        f'keep_dims = bool({flag}))[name = string("y")];',
    ]
    keep = "" if keep_dims else "_flat"
    name = (f"reduce_probe_{operation[len('reduce_'):]}_ax{axes_name(axes)}_"
            f"{shape_name(shape)}{keep}")
    return om.case(name, "reduction", {
        "operation": operation, "shape": shape, "axes": axes,
        "keep_dims": keep_dims,
    }, om.program(f"{kind} x", body, "y"))


# The channel counts a transformer block uses, as flat [1, C, 1, 1] tensors.
CHANNELS = (64, 128, 256, 512, 1024, 2048, 4096)
# Score-matrix and sequence geometries: softmax over the last axis.
SEQUENCE = ((1, 1, 1, 64), (1, 1, 1, 128), (1, 1, 1, 512), (1, 8, 1, 64),
            (1, 8, 1, 128), (1, 64, 1, 64), (1, 32, 1, 32), (1, 64, 1, 128))
SPATIAL = ((1, 64, 8, 8), (1, 128, 16, 16), (1, 32, 4, 4))
# Attention score matrices, the [1, heads, sequence, sequence] form a decoder
# block softmaxes over its last axis.
SCORE = ((1, 1, 64, 64), (1, 8, 64, 64), (1, 12, 64, 64), (1, 8, 128, 128),
         (1, 1, 256, 256))


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Softmax over the channel axis, which is what a flat [1, C, 1, 1] logit
    # vector needs, and over the last axis, which is the shipped campaign's
    # form. Both are swept over the same channel counts so a task word that
    # tracks C separates from one that tracks the reduced extent.
    for channels in CHANNELS:
        for axis in (1, -1, 3):
            cases.append(softmax((1, channels, 1, 1), axis))
    for shape in SEQUENCE:
        for axis in (-1, 3, 1):
            cases.append(softmax(shape, axis))
    for shape in SPATIAL:
        for axis in (1, 2, 3, -1):
            cases.append(softmax(shape, axis))
    # Attention score matrices: softmax over the last axis of [1, H, S, S].
    for shape in SCORE:
        for axis in (-1, 3):
            cases.append(softmax(shape, axis))

    # layer_norm over the channel axis alone (the transformer form), over every
    # non-batch axis (the shipped campaign's form), and with gamma/beta.
    for channels in CHANNELS:
        cases.append(layer_norm((1, channels, 1, 1), (1,), False))
        cases.append(layer_norm((1, channels, 1, 1), (1,), True))
        cases.append(layer_norm((1, channels, 1, 1), (1, 2, 3), False))
        cases.append(layer_norm((1, channels, 1, 1), (1, 2, 3), True))
    for shape in SPATIAL:
        for axes in ((1,), (3,), (2, 3), (1, 2, 3)):
            cases.append(layer_norm(shape, axes, False))
        cases.append(layer_norm(shape, (1, 2, 3), True))
    for shape in SEQUENCE[:4]:
        cases.append(layer_norm(shape, (3,), False))
        cases.append(layer_norm(shape, (1, 2, 3), False))

    # Reductions over the channel axis versus the spatial axes, plus the flat
    # channel sweep and the last-axis form softmax builds on.
    for operation in ("reduce_sum", "reduce_max", "reduce_mean"):
        for channels in CHANNELS:
            cases.append(reduce(operation, (1, channels, 1, 1), (1,)))
        for shape in SPATIAL:
            for axes in ((1,), (2,), (3,), (2, 3), (1, 2, 3)):
                cases.append(reduce(operation, shape, axes))
        for shape in SEQUENCE[:6]:
            cases.append(reduce(operation, shape, (3,)))
            cases.append(reduce(operation, shape, (1,)))
        cases.append(reduce(operation, (1, 64, 8, 8), (2, 3), keep_dims=False))
        cases.append(reduce(operation, (1, 512, 1, 1), (1,), keep_dims=False))

    names = [item["name"] for item in cases]
    if len(names) != len(set(names)):
        raise AssertionError("norm probe campaign has duplicate case names")
    for name in names:
        if not name.startswith(("norm_", "reduce_")):
            raise AssertionError(f"probe {name} lacks the norm_/reduce_ prefix")
    return cases


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
         "mktemp -d /tmp/mil-hwx-norm-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-norm-probes."):
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
            with tempfile.TemporaryDirectory(prefix="mil-hwx-norm-json-") as staging:
                subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


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


def scratch_bytes(record: dict[str, Any]) -> int:
    """Apple's __DATA/__bss allocation, which shifts every surface address."""
    text = int(record["program_descriptor"]["text_address"], 16)
    surfaces = sum(align(tensor["total_bytes"], SURFACE_ALIGNMENT)
                   for tensor in record["tensor_descriptors"])
    return text - 0x30000000 - surfaces


SOFTMAX_AXIS = re.compile(r"val = int32\((-?\d+)\)")
REDUCED_AXES = re.compile(r"val = tensor<int32, \[\d+\]>\(\[([-\d, ]*)\]\)")


def normalized_axes(record: dict[str, Any]) -> tuple[int, ...]:
    """The reduced axes, read from the oracle's own MIL and resolved against
    the rank. ``parameters`` cannot be trusted for this: the shipped campaign
    records no axes for layer_norm, whose MIL reduces every non-batch axis."""
    parameters = record["parameters"]
    rank = len(parameters["shape"])
    mil = record["mil"]
    if parameters["operation"] == "softmax":
        match = SOFTMAX_AXIS.search(mil)
        if not match:
            raise ValueError(f"{record['case']} has no softmax axis constant")
        axes = [int(match.group(1))]
    else:
        match = REDUCED_AXES.search(mil)
        if not match:
            raise ValueError(f"{record['case']} has no axes constant")
        axes = [int(value) for value in match.group(1).split(",")]
    declared = parameters.get("axes")
    if declared is not None and list(declared) != axes:
        raise ValueError(f"{record['case']} axes parameter contradicts its MIL")
    return tuple(sorted(axis % rank for axis in axes))


def canonical_shape(shape: list[int] | tuple[int, ...]) -> tuple[int, int, int]:
    """The physical CHW an H13 surface uses for a logical MIL shape: leading
    unit dimensions collapse into the batch, and a rank below three pads on the
    left. This is what Apple's decoded surfaces show for every rank the oracles
    cover, including the rank-reduced ``keep_dims = false`` outputs."""
    dimensions = list(shape)
    while len(dimensions) > 3 and dimensions[0] == 1:
        dimensions.pop(0)
    while len(dimensions) < 3:
        dimensions.insert(0, 1)
    if len(dimensions) != 3:
        raise ValueError(f"shape {shape} has no H13 surface form")
    return tuple(dimensions)


RESULT_TYPE = re.compile(r"tensor<fp16, \[([\d, ]+)\]> y =")
KEEP_DIMS = re.compile(r"keep_dims = bool\((true|false)\)")


def result_shape(record: dict[str, Any]) -> tuple[int, ...]:
    match = RESULT_TYPE.search(record["mil"])
    if not match:
        raise ValueError(f"{record['case']} has no fp16 result type")
    return tuple(int(value) for value in match.group(1).split(","))


def keep_dims(record: dict[str, Any]) -> bool:
    match = KEEP_DIMS.search(record["mil"])
    return match.group(1) == "true" if match else True


def surface(chw: tuple[int, int, int]) -> dict[str, Any]:
    """The tensor descriptor H13Program's surface formula produces for a CHW."""
    channels, height, width = chw
    row = max(64, width * 2)
    plane = row * height
    return {"element_code": 5, "shape": [1, channels, height, width],
            "strides": [plane * channels, plane, row, 2],
            "total_bytes": plane * channels}


def covered(record: dict[str, Any]) -> bool:
    """True when the record is a decoded single-input normalization or
    reduction whose two surfaces the encoder's own formula reproduces."""
    if record.get("error") is not None:
        return False
    if record["family"] not in ("normalization", "reduction"):
        return False
    if len(record["tensor_descriptors"]) != 2:
        return False
    expected = (surface(canonical_shape(record["parameters"]["shape"])),
                surface(canonical_shape(result_shape(record))))
    for index, tensor in enumerate(record["tensor_descriptors"]):
        decoded = {key: value for key, value in tensor.items()
                   if key != "binding"}
        if decoded != expected[index]:
            raise SystemExit(
                f"{record['case']} surface {index} is {decoded}, but the "
                f"encoder formula gives {expected[index]}")
    return True


def template_key(record: dict[str, Any]) -> tuple:
    """What the compiler knows before it picks a program: the operation, the
    input surface, which axes reduce, and whether the rank is kept."""
    return (record["parameters"]["operation"],
            canonical_shape(record["parameters"]["shape"]),
            normalized_axes(record), keep_dims(record))


LUT_WORDS = re.compile(r"k(\w+)KERNWords\[\] = \{([^}]*)\}")


def kernel_tables() -> dict[str, bytes]:
    """The fp16 LUT blocks the H13 encoder already carries, read back from the
    generated constants header so the derived kind is checked, not assumed."""
    source = (ROOT / "plugins/H13/H13ElementwiseConstants.inc").read_text()
    tables = {}
    for name, body in LUT_WORDS.findall(source):
        words = [int(value, 16)
                 for value in re.findall(r"0x([0-9a-fA-F]{8})", body)]
        tables[name] = struct.pack(f"<{len(words)}I", *words)
    return tables


def constant_kind(record: dict[str, Any], tables: dict[str, bytes]) -> str:
    """Which constant section the encoder must build, proven against the
    recorded SHA-256 rather than inferred from the operation."""
    section = record["constant_section"]
    size = section["size"]
    exponential = tables["Exp"]
    reciprocal = tables["Recip"]
    candidates = {"Zero": bytes(size)}
    if size >= len(exponential):
        candidates["Exponential"] = \
            exponential + bytes(size - len(exponential))
    if size >= len(exponential) + len(reciprocal):
        packed = bytearray(size)
        packed[:len(exponential)] = exponential
        packed[size - len(reciprocal):] = reciprocal
        candidates["ExponentialReciprocal"] = bytes(packed)
    for kind, content in candidates.items():
        if hashlib.sha256(content).hexdigest() == section["sha256"]:
            return kind
    raise SystemExit(
        f"{record['case']} constant section ({size} bytes, "
        f"{section['nonzero_bytes']} nonzero) matches no known H13 LUT layout")


def selected(target: str) -> list[dict[str, Any]]:
    records: dict[tuple, dict[str, Any]] = {}
    for path in sorted((ROOT / "research/oracles" / target).glob("*.json")):
        record = json.loads(path.read_text())
        if not covered(record):
            continue
        key = template_key(record)
        previous = records.setdefault(key, record)
        if previous is not record:
            for field in ("task_descriptors", "constant_section",
                          "program_descriptor", "tensor_descriptors"):
                if previous[field] != record[field]:
                    raise SystemExit(
                        f"{record['case']} and {previous['case']} share a "
                        f"template key but differ in {field}")
    return [records[key] for key in sorted(records)]


OPERATION_ENUM = {
    "softmax": "NormOperation::Softmax",
    "layer_norm": "NormOperation::LayerNorm",
    "reduce_sum": "NormOperation::ReduceSum",
    "reduce_max": "NormOperation::ReduceMax",
    "reduce_mean": "NormOperation::ReduceMean",
}


def axis_mask(axes: tuple[int, ...]) -> int:
    mask = 0
    for axis in axes:
        mask |= 1 << axis
    return mask


def emit(records: list[dict[str, Any]], out) -> None:
    print("// Generated from decoded H13 softmax, layer_norm, and reduction "
          "oracle task words", file=out)
    print("// by research/mint_norm_probes.py --emit-templates. No HWX "
          "container bytes;", file=out)
    print("// regenerate after re-minting the oracles.", file=out)
    tables = kernel_tables()
    symbols: dict[tuple[int, ...], str] = {}
    rows = []
    for record in records:
        words = tuple(stream_words(record))
        descriptor = record["program_descriptor"]
        constant = int(descriptor["constant_address"], 16) - \
            int(descriptor["text_address"], 16)
        if constant != align(len(words) * 4, CONSTANT_ALIGNMENT):
            raise SystemExit(
                f"{record['case']} constant offset is not stream-aligned")
        symbol = symbols.get(words)
        if symbol is None:
            symbol = f"kNormTask{len(symbols)}"
            symbols[words] = symbol
            print(f"static constexpr std::uint32_t {symbol}[] = {{", file=out)
            for start in range(0, len(words), 8):
                row = ", ".join(f"0x{word:08x}"
                                for word in words[start:start + 8])
                print(f"    {row},", file=out)
            print("};", file=out)
        operation, input_shape, axes, kept = template_key(record)
        output_shape = canonical_shape(result_shape(record))
        rows.append(
            f"    {{{OPERATION_ENUM[operation]}, "
            f"{{{input_shape[0]}, {input_shape[1]}, {input_shape[2]}}}, "
            f"{{{output_shape[0]}, {output_shape[1]}, {output_shape[2]}}}, "
            f"0x{axis_mask(axes):02x}, {'true' if kept else 'false'}, "
            f"NormConstants::{constant_kind(record, tables)}, "
            f"{symbol}, std::size({symbol}), "
            f"{(descriptor['task_words_minus_one'] + 1) * 4}, "
            f"{descriptor['task_count']}, {constant}, "
            f"{record['constant_section']['size']}, {scratch_bytes(record)}}},"
            f"  // {record['case']}")
    print("static constexpr OracleNormTemplate kNormTasks[] = {", file=out)
    for row in rows:
        print(row, file=out)
    print("};", file=out)


def emit_templates(args: argparse.Namespace) -> int:
    records = selected(args.targets[0])
    if not records:
        raise SystemExit("no decoded normalization or reduction oracles found")
    with Path(args.template_output).open("w") as out:
        emit(records, out)
    print(f"{args.template_output}: {len(records)} norm/reduce templates")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    mode.add_argument("--emit-templates", action="store_true")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=["h13"])
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--template-output",
                        default=str(ROOT / "plugins/H13/H13NormTemplates.inc"))
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.emit_templates:
        raise SystemExit(emit_templates(arguments))
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
