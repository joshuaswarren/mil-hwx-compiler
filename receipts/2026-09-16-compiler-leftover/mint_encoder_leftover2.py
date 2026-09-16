#!/usr/bin/env python3
"""Round 2: remint with distinct payloads and capture the respelled forms.

Round 1 established Apple's accept/refuse verdicts for the encoder leftover
families. Round 2 closes what those verdicts left open:

1. **Packing visibility.** Round 1's uniform payloads cannot reveal how Apple
   permutes a weight into the constant section, and the per-vector bias was
   constant-folded into the scalar addend register ``0xc80c``. Every linear and
   batched-matmul case remints with the round-2 batched campaign's
   ``uint16(index+1)`` payload, weight and bias both.
2. **The layer_norm peel.** Apple refuses gamma/beta on layer_norm (every
   ``affine`` probe in the norm campaign and round 1 refused), so the encoder
   lowers ``y = normalize(x) * gamma + beta`` as one norm program plus two
   broadcasts. Mint the per-channel mul/add at the encoder's rank-3 shapes,
   plus the runtime-runtime mul/add the GLU, attention residual and FFN need.
3. **Conv respells.** Round 1's conv refusals track the spellings Apple's own
   frontend never emits: ``pad_type = "custom"`` and rank-3 kernels. Mint the
   same/valid respells the planner can emit instead, in rank-4.
4. **Missing transpose directions.** Round 1 captured [375,1024]->[1024,375];
   the encoder also runs the reverse and the [1,256,375,16] rel-pos form.
5. **Chains without the ``linear`` op.** The chain corpus spells feed-forwards
   with ``matmul`` plus a separate bias add; round 1's ``linear``-op chain
   refused. Retest with the corpus spelling.

    python3 receipts/2026-09-16-compiler-leftover/mint_encoder_leftover2.py --host macstudio
"""
from __future__ import annotations

import argparse
import array
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402
from mint_chain_probes import weight_blob  # noqa: E402

SOURCE_COMMIT = "b61de468d1ca37b687d5fa371e489be16cd84ed1"
DEFAULT_TOOL = "/tmp/h13-oracle/bin/ane-compile-hwx"
WEIGHT_PATTERN = "fp16_bits((index+1)&0xFFFF)"


def index_fp16_payload(elements: int) -> bytes:
    return array.array(
        "H", ((index + 1) & 0xFFFF for index in range(elements))).tobytes()


def hwx_const_bytes(data: bytes) -> bytes:
    _, _, _, _, command_count, command_bytes, _, _ = struct.unpack_from("<8I", data)
    cursor = 32
    command_end = cursor + command_bytes
    for _ in range(command_count):
        command, size = struct.unpack_from("<2I", data, cursor)
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = om.cstring(fields[2])
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, section_cursor)
                if segment == "__TEXT" and om.cstring(entry[0]) == "__const":
                    return data[entry[4]:entry[4] + entry[3]]
                section_cursor += 80
        cursor += size
        if cursor > command_end:
            break
    raise ValueError("HWX has no __TEXT,__const section")


class Blobs:
    """BLOBFILE constants with one metadata record and payload per constant."""

    RECORD_BYTES = 24
    RECORD_CAPACITY = 4096

    def __init__(self) -> None:
        self.body: list[str] = []
        self.shapes: list[tuple[int, ...]] = []

    def offset(self, index: int) -> int:
        return 64 + index * self.RECORD_BYTES

    def constant(self, name: str, shape: tuple[int, ...]) -> str:
        kind = om.tensor_type(shape)
        index = len(self.shapes)
        self.body.append(
            f'{kind} {name} = const()[name = string("{name}"), val = {kind}'
            '(BLOBFILE(path = string("@model_path/weights.bin"), '
            f'offset = uint64({self.offset(index)})))];')
        self.shapes.append(shape)
        return name

    def const_scalars(self) -> list[str]:
        return [
            'bool f = const()[name = string("f"), val = bool(false)];',
            'bool t = const()[name = string("t"), val = bool(true)];',
        ]

    def blob_bytes(self) -> bytes:
        return weight_blob(self.shapes)

    def blob_bytes_index(self) -> bytes:
        """The same metadata with a distinct uint16(index+1) payload per
        element, spanning every constant, so Apple's weight permutation is
        visible in the recorded section."""
        data_start = (self.offset(self.RECORD_CAPACITY) + 0x3F) & ~0x3F
        payloads = []
        value = 0
        for shape in self.shapes:
            count = math_prod(shape)
            payloads.append(array.array(
                "H", ((value + index + 1) & 0xFFFF
                      for index in range(count))).tobytes())
            value += count
        blob = bytearray(data_start + sum(len(p) for p in payloads))
        struct.pack_into("<II", blob, 0, len(self.shapes), 2)
        offset = data_start
        for index, payload in enumerate(payloads):
            struct.pack_into("<IIQQ", blob, self.offset(index),
                             0xDEADBEEF, 1, len(payload), offset)
            blob[offset:offset + len(payload)] = payload
            offset += len(payload)
        return bytes(blob)

    def payload_elements(self) -> int:
        return sum(math_prod(shape) for shape in self.shapes)


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def finish_index_case(item: dict[str, Any], blobs: Blobs) -> dict[str, Any]:
    """Attach the distinct-payload blob the case actually compiled against."""
    if blobs.shapes:
        item["weights"] = blobs.blob_bytes_index()
        item["weights_description"] = {
            "storage": "BLOBFILE",
            "shapes": [list(shape) for shape in blobs.shapes],
            "record_offsets": [blobs.offset(index)
                               for index in range(len(blobs.shapes))],
            "payload_bytes": blobs.payload_elements() * 2,
            "value": WEIGHT_PATTERN,
            "pattern": "uint16_le_index_plus_one_wrapping",
        }
    else:
        item["weights"] = None
        item["weights_description"] = {"storage": "none"}
    return item


def finish_blob_case(item: dict[str, Any], blobs: Blobs) -> dict[str, Any]:
    item["weights"] = blobs.blob_bytes() if blobs.shapes else None
    if blobs.shapes:
        item["weights_description"] = {
            "storage": "BLOBFILE",
            "shapes": [list(shape) for shape in blobs.shapes],
            "record_offsets": [blobs.offset(index)
                               for index in range(len(blobs.shapes))],
            "payload_bytes": sum(math_prod(shape) * 2
                                 for shape in blobs.shapes),
            "value": "fp16 bits 0x3400 + index, one value per constant",
            "index_payload": True,
        }
    else:
        item["weights_description"] = {"storage": "none"}
    return item


def encoder_linear(rows: int, reduction: int, columns: int, bias: bool,
                   payload: str) -> dict[str, Any]:
    """Rank-3 linear with the index payload, so packing is visible."""
    blobs = Blobs()
    x_shape = (1, rows, reduction)
    y_shape = (1, rows, columns)
    weight = blobs.constant("w", (columns, reduction))
    bias_argument = ""
    if bias:
        b = blobs.constant("b", (columns,))
        bias_argument = f", bias = {b}"
    body = list(blobs.const_scalars()) + blobs.body
    body.append(
        f'{om.tensor_type(y_shape)} y = linear(weight = {weight}'
        f'{bias_argument}, x = x)[name = string("y")];')
    name = (f"encoder_linear_m{rows}_k{reduction}_n{columns}"
            f"_bias{int(bias)}_idx")
    item = om.env_case(name, "encoder_linear", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "bias": bias, "x_shape": list(x_shape),
        "w_shape": [columns, reduction], "output_shape": list(y_shape),
        "weight_mode": "index",
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return finish_index_case(item, blobs)


def head_projection(batch: int, rows: int, reduction: int, columns: int,
                    transpose_y: bool) -> dict[str, Any]:
    """B=8 head projection with the index payload."""
    blobs = Blobs()
    x_shape = (batch, rows, reduction)
    w_matrix = (columns, reduction) if transpose_y else (reduction, columns)
    w_shape = (batch,) + w_matrix
    out_shape = (batch, rows, columns)
    weight = blobs.constant("w", w_shape)
    body = list(blobs.const_scalars()) + blobs.body
    body.append(
        f'{om.tensor_type(out_shape)} y = matmul(transpose_x = f, '
        f'transpose_y = t, x = x, y = {weight})[name = string("y")];')
    name = (f"encoder_bmm_m{rows}_k{reduction}_n{columns}"
            f"_ty{int(transpose_y)}_b{batch}_idx")
    item = om.env_case(name, "encoder_bmm", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "batch": batch, "transpose_x": False, "transpose_y": transpose_y,
        "x_storage": "runtime", "w_storage": "blob",
        "x_shape": list(x_shape), "w_shape": list(w_shape),
        "output_shape": list(out_shape), "weight_mode": "index",
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return finish_index_case(item, blobs)


def encoder_broadcast(operation: str, x_shape: tuple[int, ...],
                      y_shape: tuple[int, ...], mode: str) -> dict[str, Any]:
    """Broadcast/runtime binary at a rank-3 or rank-4 encoder shape."""
    out_shape = tuple(max(a, b) for a, b in zip(x_shape, y_shape))
    body: list[str] = []
    arguments = f"{om.tensor_type(x_shape)} x"
    operand_shape = y_shape if mode == "runtime" else None
    elements = None
    if mode == "runtime":
        arguments += f", {om.tensor_type(y_shape)} y"
    else:
        source, count = om.blobfile("y", y_shape)
        body.append(source)
        elements = count
        operand_shape = y_shape
    body.append(
        f'{om.tensor_type(out_shape)} z = {operation}(x = x, y = y)'
        '[name = string("z")];')
    operand = mode if mode == "scalar" else f"{mode}_{om.dims(y_shape)}"
    name = (f"env_bcast_{operation}_{om.dims(x_shape)}_{operand}")
    return om.env_case(name, "env_broadcast", {
        "operation": operation, "shape": x_shape,
        "operand_shape": operand_shape, "operand": mode,
        "output_shape": out_shape,
    }, arguments, body, "z", elements,
        [list(y_shape)] if elements else None)


def encoder_conv(x_shape: tuple[int, ...], w_shape: tuple[int, ...],
                 out_shape: tuple[int, ...], stride: tuple[int, ...],
                 groups: int, pad_type: str, bias: bool) -> dict[str, Any]:
    """One rank-4 convolution with same/valid padding."""
    blobs = Blobs()
    weight = blobs.constant("w", w_shape)
    bias_argument = ""
    if bias:
        b = blobs.constant("b", (w_shape[0],))
        bias_argument = f"bias = {b}, "
    body = list(blobs.const_scalars()) + blobs.body
    stride_kind = f"tensor<int32, [{len(stride)}]>"
    pad = (0,) * (2 * len(stride))
    body.append(
        f'string pt = const()[name = string("pt"), val = string("{pad_type}")];')
    body.append(f'{stride_kind} st = const()[name = string("st"), '
                f'val = {stride_kind}([{", ".join(map(str, stride))}])];')
    body.append(f'tensor<int32, [{len(pad)}]> pd = const()[name = string("pd"), '
                f'val = tensor<int32, [{len(pad)}]>([{", ".join(map(str, pad))}])];')
    body.append('tensor<int32, [2]> dl = const()[name = string("dl"), '
                'val = tensor<int32, [2]>([1, 1])];')
    body.append(f'int32 gp = const()[name = string("gp"), val = int32({groups})];')
    body.append(
        f'{om.tensor_type(out_shape)} y = conv({bias_argument}dilations = dl, '
        f'groups = gp, pad = pd, pad_type = pt, strides = st, '
        f'weight = {weight}, x = x)[name = string("y")];')
    name = (f"encoder_conv_c{x_shape[1]}_n{w_shape[0]}_k{w_shape[2]}"
            f"x{w_shape[3]}_s{stride[0]}_g{groups}_bias{int(bias)}_{pad_type}")
    item = om.env_case(name, "conv_probe", {
        "kernel": w_shape[2], "input_channels": x_shape[1],
        "output_channels": w_shape[0], "spatial": x_shape[2],
        "spatial_width": x_shape[3], "stride": stride[0], "groups": groups,
        "bias": bias, "pad_type": pad_type, "input_shape": list(x_shape),
        "output_shape": list(out_shape), "weight_shape": list(w_shape),
        "rank": 4,
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return finish_blob_case(item, blobs)


def encoder_transpose(x_shape: tuple[int, ...], perm: tuple[int, ...]) -> dict[str, Any]:
    out_shape = tuple(x_shape[index] for index in perm)
    body = [
        f'tensor<int32, [{len(perm)}]> perm = const()[name = string("perm"), '
        f'val = tensor<int32, [{len(perm)}]>([{", ".join(map(str, perm))}])];',
        f'{om.tensor_type(out_shape)} y = transpose(perm = perm, '
        'x = x)[name = string("y")];',
    ]
    return om.case(
        f"encoder_transpose_r{len(perm)}_{''.join(map(str, perm))}"
        f"_{'x'.join(map(str, x_shape))}", "encoder_structure",
        {"probe": "transpose", "input_shape": list(x_shape),
         "perm": list(perm), "output_shape": list(out_shape)},
        om.program(f"{om.tensor_type(x_shape)} x", body, "y"), None,
        {"storage": "none"})


def encoder_chain_matmul() -> dict[str, Any]:
    """FFN chain in the corpus spelling: matmul, bias add, silu, matmul."""
    blobs = Blobs()
    shape = (1, 375, 1024)
    w1 = blobs.constant("w1", (4096, 1024))
    b1 = blobs.constant("b1", (4096,))
    w2 = blobs.constant("w2", (1024, 4096))
    b2 = blobs.constant("b2", (1024,))
    body = list(blobs.const_scalars()) + blobs.body
    body.extend([
        'tensor<int32, [1]> axes = const()[name = string("axes"), '
        'val = tensor<int32, [1]>([-1])];',
        'tensor<int32, [1]> axs = const()[name = string("axs"), '
        'val = tensor<int32, [1]>([1])];',
        f'tensor<fp16, [1, 375, 4096]> p1 = matmul(transpose_x = f, '
        f'transpose_y = t, x = x, y = {w1})[name = string("p1")];',
        f'tensor<fp16, [1, 375, 4096]> h = add(x = p1, y = {b1})'
        '[name = string("h")];',
        'tensor<fp16, [1, 375, 4096]> a = silu(x = h)[name = string("a")];',
        f'tensor<fp16, [1, 375, 1024]> p2 = matmul(transpose_x = f, '
        f'transpose_y = t, x = a, y = {w2})[name = string("p2")];',
        f'tensor<fp16, [1, 375, 1024]> y = add(x = p2, y = {b2})'
        '[name = string("y")];',
    ])
    item = om.case("chain_ffn_d1024_s375_matmul_silu", "chain",
                   {"probe": "ffn_matmul", "input_shape": list(shape)},
                   om.program(f"{om.tensor_type(shape)} x", body, "y"),
                   None, None)
    return finish_blob_case(item, blobs)


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # 1. Distinct-payload remints: packing and the general bias form.
    cases.append(encoder_linear(375, 1024, 1024, bias=True, payload="index"))
    cases.append(encoder_linear(375, 1024, 4096, bias=True, payload="index"))
    cases.append(encoder_linear(375, 4096, 1024, bias=True, payload="index"))
    cases.append(encoder_linear(375, 1024, 128, bias=True, payload="index"))
    cases.append(encoder_linear(375, 1024, 640, bias=True, payload="index"))
    cases.append(encoder_linear(375, 1024, 1024, bias=False, payload="index"))
    cases.append(head_projection(8, 375, 1024, 128, transpose_y=True))
    # 2. The layer_norm peel: per-channel mul/add at the encoder shapes.
    cases.append(encoder_broadcast("mul", (1, 375, 1024), (1, 1, 1024), "blob"))
    cases.append(encoder_broadcast("add", (1, 375, 1024), (1, 1, 1024), "blob"))
    cases.append(encoder_broadcast("mul", (1, 1024, 375), (1, 1, 1024), "blob"))
    cases.append(encoder_broadcast("add", (1, 1024, 375), (1, 1, 1024), "blob"))
    cases.append(encoder_broadcast("mul", (1, 375, 4096), (1, 1, 4096), "blob"))
    cases.append(encoder_broadcast("add", (1, 375, 4096), (1, 1, 4096), "blob"))
    # Runtime-runtime: the GLU product, the FFN gate product, the residual add.
    cases.append(encoder_broadcast("mul", (1, 1024, 375), (1, 1024, 375), "runtime"))
    cases.append(encoder_broadcast("mul", (1, 375, 4096), (1, 375, 4096), "runtime"))
    cases.append(encoder_broadcast("add", (1, 8, 375, 375), (1, 8, 375, 375), "runtime"))
    # The per-head position-bias add: [1,8,1,128] broadcast over rows.
    cases.append(encoder_broadcast("add", (1, 8, 375, 128), (1, 8, 1, 128), "blob"))
    # 3. Conv respells in rank-4 same/valid.
    cases.append(encoder_conv((1, 1024, 1, 375), (2048, 1024, 1, 1),
                              (1, 2048, 1, 375), (1, 1), 1, "valid", bias=False))
    cases.append(encoder_conv((1, 1024, 1, 375), (1024, 1024, 1, 1),
                              (1, 1024, 1, 375), (1, 1), 1, "valid", bias=False))
    cases.append(encoder_conv((1, 1024, 1, 375), (1024, 1, 9, 1),
                              (1, 1024, 1, 375), (1, 1), 1024, "same", bias=False))
    cases.append(encoder_conv((1, 8, 375, 750), (8, 1, 1, 1),
                              (1, 8, 375, 750), (1, 1), 8, "valid", bias=False))
    cases.append(encoder_conv((1, 1, 3000, 128), (256, 1, 3, 3),
                              (1, 256, 1500, 64), (2, 2), 1, "same", bias=True))
    cases.append(encoder_conv((1, 256, 1500, 64), (256, 1, 3, 3),
                              (1, 256, 750, 32), (2, 2), 256, "same", bias=True))
    cases.append(encoder_conv((1, 256, 750, 32), (256, 256, 1, 1),
                              (1, 256, 750, 32), (1, 1), 1, "valid", bias=True))
    cases.append(encoder_conv((1, 256, 750, 32), (256, 1, 3, 3),
                              (1, 256, 375, 16), (2, 2), 256, "same", bias=True))
    cases.append(encoder_conv((1, 256, 375, 16), (256, 256, 1, 1),
                              (1, 256, 375, 16), (1, 1), 1, "valid", bias=True))
    # 4. The missing transpose forms.
    cases.append(encoder_transpose((1, 1024, 375), (0, 2, 1)))
    cases.append(encoder_transpose((1, 375, 256, 16), (0, 2, 1, 3)))
    # 5. Chain with the corpus matmul spelling.
    cases.append(encoder_chain_matmul())
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


def run_case_retain(item: dict[str, Any], output: Path, tool: Path,
                    source_commit: str) -> tuple[str, dict[str, Any]]:
    destination = output / "h13" / f"{item['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema_version": 1,
        "case": item["name"],
        "family": item["family"],
        "parameters": item["parameters"],
        "mil": item["mil"],
        "weights": item["weights_description"],
        "target": "h13",
        "source_commit": source_commit,
        "compiler": {
            "tool": str(tool),
            "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(
                Path(om.__file__).read_bytes()).hexdigest(),
            "decoder_sha256": hashlib.sha256(
                Path(om.__file__).with_name("h13_td.py").read_bytes()).hexdigest(),
            "campaign_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "host": platform.node(),
            "platform": platform.platform(),
            "command": [str(tool), "CAPTURE_DIR", "OUTPUT_DIR", "h13"],
        },
    }
    with tempfile.TemporaryDirectory(
            prefix=f"mil-hwx-enc2-{item['name']}-") as root:
        root_path = Path(root)
        capture = root_path / "capture"
        compiled = root_path / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        weights = item["weights"]
        if isinstance(weights, int):
            weights = om.blob(om.half_payload(weights))
        (capture / "weights.bin").write_bytes(weights or b"")
        try:
            result = subprocess.run(
                [str(tool), str(capture), str(compiled), "h13"],
                capture_output=True, text=True,
                timeout=om.COMPILE_TIMEOUT_SECONDS, check=False)
            error = (result.stderr or result.stdout).strip()
            hwx = compiled / "model.hwx"
            if result.returncode == 0 and hwx.is_file():
                hwx_bytes = hwx.read_bytes()
                record.update(om.parse_hwx(hwx_bytes, "h13"))
                record["error"] = None
                const = hwx_const_bytes(hwx_bytes)
                const_path = destination.with_suffix(".const.bin")
                const_path.write_bytes(const)
                record.setdefault("constant_section", {})
                record["constant_section"]["raw_path"] = const_path.name
                record["constant_section"]["raw_sha256"] = (
                    hashlib.sha256(const).hexdigest())
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
    return status, record


def local_run(args: argparse.Namespace) -> int:
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)}")
        return 0
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    tool = Path(args.oracle_tool)
    if not os.access(tool, os.X_OK):
        raise SystemExit(f"oracle tool is not executable: {tool}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    decoded = rejected = 0
    for item in selected:
        destination = output / "h13" / f"{item['name']}.json"
        if destination.exists() and not args.force:
            record = json.loads(destination.read_text())
            status = "decoded" if record.get("error") is None else "rejected"
        else:
            status, record = run_case_retain(
                item, output, tool, args.source_commit)
        decoded += status == "decoded"
        rejected += status == "rejected"
        tasks = len(record.get("task_descriptors") or [])
        const = (record.get("constant_section") or {}).get("size")
        print(f"h13 {item['name']} {status} tasks={tasks} const={const}"
              + ("" if record.get("error") is None
                 else f" error={record['error'][:120]}"),
              flush=True)
    print(f"SUMMARY cases={len(selected)} decoded={decoded} "
          f"rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-encoder-r2.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-encoder-r2."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    research = script.parents[2] / "research"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script),
             str(research / "mint_oracles.py"),
             str(research / "h13_td.py"),
             str(research / "mint_chain_probes.py"),
             f"{args.host}:{root}/"], check=True)
        command = [
            "python3", f"{root}/{script.name}", "--local",
            "--oracle-tool", args.oracle_tool,
            "--output", f"{root}/oracles",
            "--source-commit", args.source_commit,
        ]
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
            staging = output / "_staging"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            subprocess.run(
                ["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                 str(staging)], check=True)
            for path in staging.rglob("*"):
                if path.is_file():
                    shutil.copy2(path, output / path.name)
            shutil.rmtree(staging)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    parser.add_argument("--oracle-tool", default=DEFAULT_TOOL)
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parent / "oracles"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", help="fnmatch selecting case names")
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
