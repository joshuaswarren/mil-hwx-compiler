#!/usr/bin/env python3
"""Mint and decode H13/H14 ANE compiler oracles without retaining HWX bytes."""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from typing import Any

from h13_td import decode_task, split_h13_tasks, split_h14_tasks

MAGIC = 0xBEEFFACE
SUBTYPES = {"h13": 4, "h14": 5}
DEFAULT_TOOL = "/tmp/h13-oracle/bin/ane-compile-hwx"


def decode_constant_half_words(data: bytes) -> list[dict[str, Any]] | None:
    if len(data) > 256 or len(data) % 2:
        return None
    words = []
    for index, (value,) in enumerate(struct.iter_unpack("<e", data)):
        if value == 0.0:
            continue
        if value != value:
            decoded: float | str = "nan"
        elif value == float("inf"):
            decoded = "+inf"
        elif value == float("-inf"):
            decoded = "-inf"
        else:
            decoded = value
        words.append({"index": index, "value": decoded})
    return words


def shape_text(shape: tuple[int, ...]) -> str:
    return "[" + ", ".join(map(str, shape)) + "]"


def tensor_type(shape: tuple[int, ...]) -> str:
    return f"tensor<fp16, {shape_text(shape)}>"


def program(arguments: str, body: list[str], result: str) -> str:
    lines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({})]",
        "{",
        f"  func main<ios18>({arguments}) {{",
        *(f"    {line}" for line in body),
        f"  }} -> ({result});",
        "}",
    ]
    return "\n".join(lines) + "\n"


def blob(payload: bytes) -> bytes:
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IQQ", result, 64, 0xDEADBEEF, len(payload), 128)
    result[128:] = payload
    return bytes(result)


def half_payload(elements: int) -> bytes:
    return b"\x00\x38" * elements


def case(name: str, family: str, parameters: dict[str, Any], mil: str,
         weights: bytes | None = None, weights_description: dict[str, Any] | None = None
         ) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "parameters": parameters,
        "mil": mil,
        "weights": weights,
        "weights_description": weights_description or {"storage": "none"},
    }


def binary_runtime(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    kind = tensor_type(shape)
    name = f"binary_{operation}_{'x'.join(map(str, shape))}"
    mil = program(
        f"{kind} x, {kind} z",
        [f'{kind} y = {operation}(x = x, y = z)[name = string("y")];'], "y")
    return case(name, "binary_runtime", {"operation": operation, "shape": shape}, mil)


def binary_constant(operation: str, channels: int, storage: str) -> dict[str, Any]:
    shape = (1, channels, 1, 1)
    kind = tensor_type(shape)
    name = f"binary_{operation}_c{channels}_constant_{storage}"
    if storage == "scalar":
        constant = 'fp16 z = const()[name = string("z"), val = fp16(0x1p-1)];'
        weights = None
        description = {"storage": "inline_scalar", "value": "fp16(0x1p-1)"}
    elif storage == "inline":
        values = ", ".join(["fp16(0x1p-1)"] * channels)
        constant = (
            f'{kind} z = const()[name = string("z"), '
            f'val = {kind}([{values}])];')
        weights = None
        description = {"storage": "inline_tensor", "shape": shape,
                       "value": "fp16(0x1p-1)"}
    else:
        payload = half_payload(channels)
        constant = (
            f'{kind} z = const()[name = string("z"), val = {kind}'
            '(BLOBFILE(path = string("@model_path/weights.bin"), '
            'offset = uint64(64)))];')
        weights = blob(payload)
        description = {"storage": "BLOBFILE", "shape": shape,
                       "payload_bytes": len(payload), "value": "fp16(0x1p-1)"}
    mil = program(
        f"{kind} x", [constant,
        f'{kind} y = {operation}(x = x, y = z)[name = string("y")];'], "y")
    return case(name, "binary_constant",
                {"operation": operation, "shape": shape, "constant": storage},
                mil, weights, description)


def unary(operation: str, channels: int) -> dict[str, Any]:
    shape = (1, channels, 1, 1)
    kind = tensor_type(shape)
    attributes = {
        "gelu": ', mode = string("EXACT")',
        "rsqrt": ", epsilon = fp32(0.000001)",
        "clip": ", alpha = fp32(-1.0), beta = fp32(1.0)",
        "leaky_relu": ", alpha = fp32(0.125)",
    }.get(operation, "")
    mil = program(
        f"{kind} x",
        [f'{kind} y = {operation}(x = x{attributes})[name = string("y")];'], "y")
    return case(f"unary_{operation}_c{channels}", "unary",
                {"operation": operation, "shape": shape}, mil)


def reduction(operation: str, axes: tuple[int, ...]) -> dict[str, Any]:
    shape = (1, 64, 8, 8)
    output = tuple(1 if index in axes else value for index, value in enumerate(shape))
    axis_values = ", ".join(map(str, axes))
    kind = tensor_type(shape)
    out_kind = tensor_type(output)
    axis_kind = f"tensor<int32, [{len(axes)}]>"
    mil = program(f"{kind} x", [
        f'{axis_kind} axes = const()[name = string("axes"), '
        f'val = {axis_kind}([{axis_values}])];',
        f'{out_kind} y = {operation}(x = x, axes = axes, '
        'keep_dims = bool(true))[name = string("y")];',
    ], "y")
    suffix = "_".join(map(str, axes))
    return case(f"{operation}_axes_{suffix}", "reduction",
                {"operation": operation, "shape": shape, "axes": axes}, mil)


def normalization(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    kind = tensor_type(shape)
    body = []
    if operation == "softmax":
        body.extend([
            'int32 axis = const()[name = string("axis"), val = int32(-1)];',
            f'{kind} y = softmax(x = x, axis = axis)[name = string("y")];',
        ])
    else:
        axes = tuple(range(1, len(shape)))
        axis_kind = f"tensor<int32, [{len(axes)}]>"
        body.extend([
            f'{axis_kind} axes = const()[name = string("axes"), '
            f'val = {axis_kind}([{", ".join(map(str, axes))}])];',
            f'{kind} y = layer_norm(x = x, axes = axes, '
            'epsilon = fp32(0.00001))[name = string("y")];',
        ])
    name = f"{operation}_{'x'.join(map(str, shape))}"
    return case(name, "normalization", {"operation": operation, "shape": shape},
                program(f"{kind} x", body, "y"))


def weight_constant(name: str, shape: tuple[int, ...]) -> tuple[str, bytes, dict[str, Any]]:
    elements = 1
    for dimension in shape:
        elements *= dimension
    payload = half_payload(elements)
    kind = tensor_type(shape)
    source = (
        f'{kind} {name} = const()[name = string("{name}"), val = {kind}'
        '(BLOBFILE(path = string("@model_path/weights.bin"), '
        'offset = uint64(64)))];')
    return source, blob(payload), {
        "storage": "BLOBFILE", "shape": shape, "payload_bytes": len(payload),
        "value": "fp16(0x1p-1)",
    }


def convolution(kernel: int, inputs: int, outputs: int, spatial: int,
                bias: bool, chain_relu: bool = False) -> dict[str, Any]:
    x_shape = (1, inputs, spatial, spatial)
    y_shape = (1, outputs, spatial, spatial)
    w_shape = (outputs, inputs, kernel, kernel)
    x_kind, y_kind = tensor_type(x_shape), tensor_type(y_shape)
    weight, weights, description = weight_constant("w", w_shape)
    body = [
        'string pt = const()[name = string("pt"), val = string("same")];',
        'tensor<int32, [2]> st = const()[name = string("st"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'tensor<int32, [4]> pd = const()[name = string("pd"), '
        'val = tensor<int32, [4]>([0, 0, 0, 0])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        'int32 gp = const()[name = string("gp"), val = int32(1)];', weight,
    ]
    bias_arg = ""
    if bias:
        values = ", ".join(["fp16(0x0p+0)"] * outputs)
        bias_kind = tensor_type((outputs,))
        body.append(f'{bias_kind} b = const()[name = string("b"), '
                    f'val = {bias_kind}([{values}])];')
        bias_arg = ", bias = b"
    result = "c" if chain_relu else "y"
    body.append(
        f'{y_kind} {result} = conv(dilations = dl, groups = gp, pad = pd, '
        f'pad_type = pt, strides = st, weight = w, x = x{bias_arg})'
        f'[name = string("{result}")];')
    if chain_relu:
        body.append(f'{y_kind} y = relu(x = c)[name = string("y")];')
    name = f"conv_k{kernel}_c{inputs}_n{outputs}_s{spatial}_bias{int(bias)}"
    if chain_relu:
        name += "_relu"
    return case(name, "chain" if chain_relu else "convolution", {
        "kernel": kernel, "input_channels": inputs, "output_channels": outputs,
        "spatial": spatial, "bias": bias, "relu": chain_relu,
    }, program(f"{x_kind} x", body, "y"), weights, description)


def matmul(reduction: int, columns: int, rows: int, transpose: bool,
           chain_bias_relu: bool = False) -> dict[str, Any]:
    x_shape = (rows, reduction)
    w_shape = (columns, reduction) if transpose else (reduction, columns)
    y_shape = (rows, columns)
    x_kind, y_kind = tensor_type(x_shape), tensor_type(y_shape)
    weight, weights, description = weight_constant("w", w_shape)
    flag = "true" if transpose else "false"
    body = [
        'bool f = const()[name = string("f"), val = bool(false)];',
        f'bool ty = const()[name = string("ty"), val = bool({flag})];', weight,
        f'{y_kind} product = matmul(transpose_x = f, transpose_y = ty, '
        'x = x, y = w)[name = string("product")];',
    ]
    result = "product"
    if chain_bias_relu:
        bias_kind = tensor_type((columns,))
        values = ", ".join(["fp16(0x0p+0)"] * columns)
        body.extend([
            f'{bias_kind} b = const()[name = string("b"), '
            f'val = {bias_kind}([{values}])];',
            f'{y_kind} biased = add(x = product, y = b)[name = string("biased")];',
            f'{y_kind} y = relu(x = biased)[name = string("y")];',
        ])
        result = "y"
    name = f"matmul_m{rows}_k{reduction}_n{columns}_ty{int(transpose)}"
    if chain_bias_relu:
        name += "_bias_relu"
    return case(name, "chain" if chain_bias_relu else "matmul", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "transpose_y": transpose, "bias": chain_bias_relu, "relu": chain_bias_relu,
    }, program(f"{x_kind} x", body, result), weights, description)


def linear(reduction: int, columns: int) -> dict[str, Any]:
    x_shape, w_shape, y_shape = (1, reduction), (columns, reduction), (1, columns)
    x_kind, y_kind = tensor_type(x_shape), tensor_type(y_shape)
    weight, weights, description = weight_constant("w", w_shape)
    bias_kind = tensor_type((columns,))
    values = ", ".join(["fp16(0x0p+0)"] * columns)
    body = [weight,
        f'{bias_kind} b = const()[name = string("b"), val = {bias_kind}([{values}])];',
        f'{y_kind} y = linear(x = x, weight = w, bias = b)[name = string("y")];']
    return case(f"linear_k{reduction}_n{columns}", "linear",
                {"reduction": reduction, "columns": columns},
                program(f"{x_kind} x", body, "y"), weights, description)


def add_relu_chain() -> dict[str, Any]:
    shape = (1, 512, 1, 1)
    kind = tensor_type(shape)
    body = [
        f'{kind} added = add(x = x, y = z)[name = string("added")];',
        f'{kind} y = relu(x = added)[name = string("y")];',
    ]
    return case("chain_add_relu_c512", "chain", {"operations": ["add", "relu"],
                "shape": shape}, program(f"{kind} x, {kind} z", body, "y"))


def campaign() -> list[dict[str, Any]]:
    cases = []
    shapes = [(1, channels, 1, 1) for channels in
              (64, 128, 256, 512, 1024, 2048, 4096)]
    shapes += [(1, 64, 8, 8), (1, 128, 16, 16), (1, 3, 224, 224)]
    for operation in ("add", "mul", "maximum", "minimum", "sub", "real_div", "pow"):
        cases.extend(binary_runtime(operation, shape) for shape in shapes)
    cases.extend(unary("abs", shape[1]) for shape in shapes[:7])
    for operation in ("add", "mul", "maximum", "minimum", "sub", "real_div"):
        for channels in (64, 512):
            cases.extend(binary_constant(operation, channels, storage)
                         for storage in ("inline", "blob", "scalar"))
    for operation in ("relu", "sigmoid", "tanh", "gelu", "silu", "exp", "log",
                      "sqrt", "rsqrt", "clip", "leaky_relu"):
        cases.extend(unary(operation, channels) for channels in (64, 512))
    for operation in ("reduce_sum", "reduce_max", "reduce_mean"):
        cases.extend(reduction(operation, axes) for axes in ((1,), (2, 3)))
    for operation in ("softmax", "layer_norm"):
        cases.extend(normalization(operation, shape)
                     for shape in ((1, 512, 1, 1), (1, 64, 8, 8)))
    for inputs in (256, 512, 1024):
        for outputs in (512, 1024):
            cases.extend(convolution(1, inputs, outputs, 1, bias)
                         for bias in (False, True))
    cases.append(convolution(3, 64, 64, 8, False))
    for rows in (1, 2, 8, 64):
        for inner in (256, 512, 1024):
            for columns in (256, 512, 1024):
                cases.extend(matmul(inner, columns, rows, transpose)
                             for transpose in (False, True))
    cases.extend(linear(inner, columns)
                 for inner, columns in ((256, 512), (512, 1024)))
    cases.extend([
        add_relu_chain(),
        matmul(512, 512, 1, False, True),
        convolution(1, 64, 64, 8, False, True),
    ])
    names = [item["name"] for item in cases]
    if len(names) != len(set(names)):
        raise AssertionError("campaign contains duplicate case names")
    return cases


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def parse_hwx(data: bytes, target: str) -> dict[str, Any]:
    if len(data) < 32:
        raise ValueError("HWX Mach-O header is truncated")
    magic, _, subtype, _, command_count, command_bytes, _, _ = struct.unpack_from(
        "<8I", data)
    if magic != MAGIC:
        raise ValueError(f"invalid HWX magic 0x{magic:08x}")
    if subtype != SUBTYPES[target]:
        raise ValueError(f"compiler returned subtype {subtype} for {target}")
    sections: dict[tuple[str, str], dict[str, int]] = {}
    tensors = []
    program_descriptor: dict[str, Any] | None = None
    cursor = 32
    command_end = cursor + command_bytes
    if command_end > len(data):
        raise ValueError("HWX load command table is truncated")
    for command_index in range(command_count):
        if cursor + 8 > command_end:
            raise ValueError(f"load command[{command_index}] is truncated")
        command, size = struct.unpack_from("<2I", data, cursor)
        if size < 8 or cursor + size > command_end:
            raise ValueError(f"load command[{command_index}] has invalid size")
        kind = struct.unpack_from("<I", data, cursor + 8)[0] if size >= 12 else None
        if command == 0x19:
            if size < 72:
                raise ValueError("LC_SEGMENT_64 is truncated")
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = cstring(fields[2])
            section_cursor = cursor + 72
            if fields[-2] * 80 > size - 72:
                raise ValueError("LC_SEGMENT_64 section table is truncated")
            for _ in range(fields[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
                section = cstring(entry[0])
                sections[(segment, section)] = {
                    "address": entry[2], "size": entry[3], "offset": entry[4],
                }
                section_cursor += 80
        elif command == 4 and kind == 3 and size >= 0x78:
            tensors.append({
                "binding": struct.unpack_from("<I", data, cursor + 0x14)[0],
                "element_code": struct.unpack_from("<I", data, cursor + 0x24)[0],
                "shape": list(struct.unpack_from("<4I", data, cursor + 0x28)),
                "strides": list(struct.unpack_from("<4Q", data, cursor + 0x50)),
                "total_bytes": struct.unpack_from("<Q", data, cursor + 0x70)[0],
            })
        elif command == 4 and kind == 1 and size >= 0x820:
            program_descriptor = {
                "kind": kind, "command_size": size,
                "code": struct.unpack_from("<I", data, cursor + 0x0C)[0],
                "text_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x10)[0]:x}",
                "constant_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x18)[0]:x}",
                "resource_addresses": [
                    f"0x{value:x}" for value in struct.unpack_from("<5Q", data, cursor + 0x30)
                ],
                "task_words_minus_one": struct.unpack_from("<I", data, cursor + 0x818)[0],
                "task_count": struct.unpack_from("<I", data, cursor + 0x81C)[0],
            }
        elif command == 4 and kind == 4 and size >= 0x838:
            program_descriptor = {
                "kind": kind, "command_size": size,
                "text_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x10)[0]:x}",
                "constant_address": f"0x{struct.unpack_from('<Q', data, cursor + 0x20)[0]:x}",
                "resource_addresses": [
                    f"0x{value:x}" for value in struct.unpack_from("<5Q", data, cursor + 0x30)
                ],
                "text_words": struct.unpack_from("<I", data, cursor + 0x824)[0],
                "task_count": struct.unpack_from("<I", data, cursor + 0x830)[0],
                "trailing_words": [
                    f"0x{value:08x}" for value in struct.unpack_from(
                        f"<{(size - 0x810) // 4}I", data, cursor + 0x810)
                ],
            }
        cursor += size
    if cursor != command_end:
        raise ValueError("HWX load command sizes do not match the header")
    text = sections.get(("__TEXT", "__text")) or sections.get(("__TEXT", "__TEXT"))
    constants = sections.get(("__TEXT", "__const"))
    if not text or not constants or not program_descriptor:
        raise ValueError("HWX lacks text, constant, or program descriptor metadata")
    text_end = text["offset"] + text["size"]
    constant_end = constants["offset"] + constants["size"]
    if text_end > len(data):
        raise ValueError("HWX task section is truncated")
    if constant_end > len(data):
        raise ValueError("HWX constant section is truncated")
    text_data = data[text["offset"]:text_end]
    constant_data = data[constants["offset"]:constant_end]
    prefix = constant_data[:128]
    chunk_bytes = 0x800
    chunks = [
        {
            "offset": offset,
            "size": len(chunk),
            "sha256": hashlib.sha256(chunk).hexdigest(),
            "nonzero_bytes": sum(value != 0 for value in chunk),
        }
        for offset in range(0, len(constant_data), chunk_bytes)
        for chunk in (constant_data[offset:offset + chunk_bytes],)
    ] if len(constant_data) <= 0x10000 else []
    if target == "h13":
        raw_tasks = split_h13_tasks(
            text_data, program_descriptor["task_words_minus_one"],
            program_descriptor["task_count"])
    else:
        raw_tasks = split_h14_tasks(text_data)
        if len(raw_tasks) != program_descriptor["task_count"]:
            raise ValueError(
                f"H14 program declares {program_descriptor['task_count']} tasks but "
                f"decoded {len(raw_tasks)}")
    return {
        "hwx_sha256": hashlib.sha256(data).hexdigest(),
        "hwx_bytes": len(data),
        "program_descriptor": program_descriptor,
        "tensor_descriptors": tensors,
        "constant_section": {
            "size": len(constant_data),
            "sha256": hashlib.sha256(constant_data).hexdigest(),
            "nonzero_bytes": sum(value != 0 for value in constant_data),
            "prefix_128_sha256": hashlib.sha256(prefix).hexdigest(),
            "prefix_128_nonzero_bytes": sum(value != 0 for value in prefix),
            "tail_after_128_nonzero_bytes": sum(
                value != 0 for value in constant_data[128:]),
            "nonzero_fp16_words": decode_constant_half_words(constant_data),
            "chunk_bytes": chunk_bytes,
            "chunk_count": (len(constant_data) + chunk_bytes - 1) // chunk_bytes,
            "chunks": chunks,
        },
        "task_descriptors": [decode_task(task, target) for task in raw_tasks],
    }


def run_case(item: dict[str, Any], target: str, output: Path,
             oracle_tool: Path, source_commit: str) -> str:
    destination = output / target / f"{item['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "case": item["name"],
        "family": item["family"],
        "parameters": item["parameters"],
        "mil": item["mil"],
        "weights": item["weights_description"],
        "target": target,
        "source_commit": source_commit,
        "compiler": {
            "tool": str(oracle_tool),
            "tool_sha256": hashlib.sha256(oracle_tool.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256(
                Path(__file__).with_name("h13_td.py").read_bytes()).hexdigest(),
            "host": platform.node(),
            "platform": platform.platform(),
        },
    }
    with tempfile.TemporaryDirectory(prefix=f"mil-hwx-{target}-{item['name']}-") as root:
        root_path = Path(root)
        capture = root_path / "capture"
        compiled = root_path / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        weights = item["weights"]
        if weights is None:
            (capture / "weights.bin").write_bytes(b"")
        else:
            (capture / "weights.bin").write_bytes(weights)
        command = [str(oracle_tool), str(capture), str(compiled), target]
        record["compiler"]["command"] = [str(oracle_tool), "CAPTURE_DIR", "OUTPUT_DIR", target]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=180, check=False)
            error = (result.stderr or result.stdout).strip()
            hwx = compiled / "model.hwx"
            if result.returncode == 0 and hwx.is_file():
                record.update(parse_hwx(hwx.read_bytes(), target))
                record["error"] = None
                status = "decoded"
            else:
                record.update({
                    "hwx_sha256": None, "hwx_bytes": None,
                    "program_descriptor": None, "tensor_descriptors": [],
                    "constant_section": None, "task_descriptors": [],
                    "error": error or f"compiler exited {result.returncode}",
                })
                status = "rejected"
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            record.update({
                "hwx_sha256": None, "hwx_bytes": None,
                "program_descriptor": None, "tensor_descriptors": [],
                "constant_section": None, "task_descriptors": [],
                "error": str(error),
            })
            status = "rejected"
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return status


def source_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def local_run(args: argparse.Namespace) -> int:
    tool = Path(args.oracle_tool)
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    if not os.access(tool, os.X_OK):
        raise SystemExit(f"oracle tool is not executable: {tool}")
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.limit is not None:
        selected = selected[:args.limit]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
    decoded = rejected = 0
    output = Path(args.output)
    for target in args.targets:
        for item in selected:
            destination = output / target / f"{item['name']}.json"
            if destination.exists() and not args.force:
                existing = json.loads(destination.read_text())
                decoded += existing.get("error") is None
                rejected += existing.get("error") is not None
                continue
            status = run_case(item, target, output, tool, args.source_commit)
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root_result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-oracles.XXXXXX"],
        capture_output=True, text=True, check=True)
    remote_root = root_result.stdout.strip()
    if not remote_root.startswith("/tmp/mil-hwx-oracles."):
        raise RuntimeError(f"unexpected remote temporary path: {remote_root!r}")
    script = Path(__file__).resolve()
    decoder = script.with_name("h13_td.py")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", str(script), str(decoder),
                        f"{args.host}:{remote_root}/"], check=True)
        command = [
            "python3", f"{remote_root}/{script.name}", "--local",
            "--oracle-tool", args.oracle_tool, "--output", f"{remote_root}/oracles",
            "--source-commit", args.source_commit, "--targets", *args.targets,
        ]
        if args.case:
            command.extend(["--case", args.case])
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.force:
            command.append("--force")
        if args.list:
            command.append("--list")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
        if not args.list:
            with tempfile.TemporaryDirectory(prefix="mil-hwx-oracle-json-") as staging:
                subprocess.run(["scp", "-q", "-r",
                                f"{args.host}:{remote_root}/oracles/.", staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(remote_root)}"], check=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true", help="run the Apple compiler locally")
    mode.add_argument("--host", help="copy the worker to this SSH Mac and retrieve JSON")
    parser.add_argument("--targets", nargs="+", choices=sorted(SUBTYPES),
                        default=sorted(SUBTYPES))
    parser.add_argument("--oracle-tool", default=DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--case", help="shell pattern selecting case names")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local else remote_run(arguments))
