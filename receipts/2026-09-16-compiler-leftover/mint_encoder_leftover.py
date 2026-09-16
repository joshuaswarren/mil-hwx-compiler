#!/usr/bin/env python3
"""Mint H13 oracles for the Parakeet encoder leftover op families.

The 2026-09-14 leftover table (mlx-omarchy receipts/2026-09-14-encoder-leftover.md)
names the op families the pinned mil-hwx-compiler still refuses at the exact
encoder forms: rank-3 linears, rank-3 unaries (silu/sigmoid), rank-3/4
normalizations (layer_norm with gamma/beta, softmax over 8 heads at [375,375]),
the nine encoder convolutions, and the concat/transpose/slice structure ops.

This campaign mints single-op Apple oracles at those exact MIL spellings so the
H13 encoders can be extended from decoded bytes instead of guesses. Family
names match the parity-suite conventions:

- ``encoder_linear``   rank-3 ``linear`` with a BLOBFILE weight (+bias variant)
- ``batched_matmul``   the eight head projections as one B=8 matmul
- ``unary``            rank-3 silu/sigmoid (parity suite selects by op name)
- ``normalization``    softmax and layer_norm at encoder shapes
- ``norm_params``      layer_norm with gamma/beta operands
- ``conv_probe``       the nine encoder convolutions (rectangular surfaces)
- ``encoder_structure``concat / transpose / slice acceptance probes
- ``chain``            FFN and norm-projection chains at d1024 s375

Run from anywhere; the tool runs on the macOS oracle host:

    python3 receipts/2026-09-16-compiler-leftover/mint_encoder_leftover.py --host macstudio

No Apple HWX bytes are retained: every record keeps the MIL, the decoded task
words, the descriptors, constant-section hashes and sizes, and the compiler's
status, exactly like every other campaign in this repository. Constant-y cases
additionally retain the raw ``__TEXT,__const`` bytes and the input payload so
the packed sections are checkable offline.
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
    """Unique-in-window fp16 bit patterns: uint16(index+1), never zero."""
    return array.array(
        "H", ((index + 1) & 0xFFFF for index in range(elements))).tobytes()


def hwx_const_bytes(data: bytes) -> bytes:
    """The __TEXT,__const body of an HWX image, in memory only."""
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
                    offset, length = entry[4], entry[3]
                    return data[offset:offset + length]
                section_cursor += 80
        cursor += size
        if cursor > command_end:
            break
    raise ValueError("HWX has no __TEXT,__const section")


# --------------------------------------------------------------------------
# MIL construction: constants with one BLOBFILE record and payload each
# --------------------------------------------------------------------------

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
        """The inline scalar/flag constants every case shares."""
        return [
            'bool f = const()[name = string("f"), val = bool(false)];',
            'bool t = const()[name = string("t"), val = bool(true)];',
        ]

    def blob_bytes(self) -> bytes:
        return weight_blob(self.shapes)


def encoder_linear(rows: int, reduction: int, columns: int, bias: bool,
                   activation: str | None = None) -> dict[str, Any]:
    """One rank-3 ``linear`` exactly as the encoder MIL spells it."""
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
            f"_bias{int(bias)}")
    if activation:
        body.append(f'{om.tensor_type(y_shape)} z = {activation}(x = y)'
                    '[name = string("z")];')
        name += f"_{activation}"
    item = om.env_case(name, "encoder_linear", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "bias": bias, "activation": activation, "x_shape": list(x_shape),
        "w_shape": [columns, reduction], "output_shape": list(y_shape),
    }, f"{om.tensor_type(x_shape)} x", body,
        "z" if activation else "y", None, None)
    return finish_blob_case(item, blobs)


def head_projection(batch: int, rows: int, reduction: int, columns: int,
                    transpose_y: bool) -> dict[str, Any]:
    """The eight head projections as one batched matmul with a packed weight.

    The encoder spells each head as its own ``linear``; one B=8 matmul over
    a ``[8, 1024, 128]`` weight eliminates both the seven sibling ops and
    the concat. Spelling mirrors the batched campaign's ``bmm_rb`` rows.
    """
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
            f"_ty{int(transpose_y)}_b{batch}")
    item = om.env_case(name, "encoder_bmm", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "batch": batch, "transpose_x": False, "transpose_y": transpose_y,
        "x_storage": "runtime", "w_storage": "blob",
        "x_shape": list(x_shape), "w_shape": list(w_shape),
        "output_shape": list(out_shape),
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return finish_blob_case(item, blobs)


def encoder_unary(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    """One rank-3 unary exactly as the encoder MIL spells it."""
    kind = om.tensor_type(shape)
    mode = ', mode = string("EXACT")' if operation == "gelu" else ""
    body = [f'{kind} y = {operation}(x = x{mode})[name = string("y")];']
    return om.case(f"unary_{operation}_{'x'.join(map(str, shape))}", "unary",
                   {"operation": operation, "shape": shape},
                   om.program(f"{kind} x", body, "y"), None,
                   {"storage": "none"})


def encoder_softmax(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    body = [
        'int32 axis = const()[name = string("axis"), val = int32(-1)];',
        f'{kind} y = softmax(x = x, axis = axis)[name = string("y")];',
    ]
    return om.case(f"softmax_{'x'.join(map(str, shape))}", "normalization",
                   {"operation": "softmax", "shape": shape},
                   om.program(f"{kind} x", body, "y"), None,
                   {"storage": "none"})


def encoder_layer_norm(shape: tuple[int, ...], params: bool) -> dict[str, Any]:
    """``layer_norm`` over the last axis, with and without gamma/beta."""
    blobs = Blobs()
    kind = om.tensor_type(shape)
    axes_kind = "tensor<int32, [1]>"
    body = list(blobs.const_scalars()) + blobs.body
    body.append(
        f'{axes_kind} axes = const()[name = string("axes"), '
        f'val = {axes_kind}([-1])];')
    gamma_argument = beta_argument = ""
    if params:
        width = shape[-1]
        gamma = blobs.constant("gamma", (width,))
        beta = blobs.constant("beta", (width,))
        gamma_argument = f", gamma = {gamma}"
        beta_argument = f", beta = {beta}"
    body.append(
        f'{kind} y = layer_norm(x = x, axes = axes, epsilon = fp32(0.00001)'
        f'{gamma_argument}{beta_argument})[name = string("y")];')
    name = (f"layer_norm_{'x'.join(map(str, shape))}"
            f"{'_params' if params else ''}")
    item = om.case(name, "norm_params" if params else "normalization",
                   {"operation": "layer_norm", "shape": shape,
                    "gamma_beta": params},
                   om.program(f"{kind} x", body, "y"), None, None)
    item["weights"] = blobs.blob_bytes() if blobs.shapes else None
    item["weights_description"] = {"storage": "none"}
    if blobs.shapes:
        item["weights_description"] = {
            "storage": "BLOBFILE",
            "shapes": [list(shape) for shape in blobs.shapes],
            "record_offsets": [64 + index * Blobs.RECORD_BYTES
                               for index in range(len(blobs.shapes))],
            "payload_bytes": sum(
                __import__("math").prod(shape) * 2
                for shape in blobs.shapes),
            "value": "fp16 bits 0x3400 + index, one value per constant",
        }
    return item


def encoder_conv(x_shape: tuple[int, ...], w_shape: tuple[int, ...],
                 out_shape: tuple[int, ...], stride: tuple[int, ...],
                 groups: int, pad: tuple[int, ...], pad_type: str,
                 bias: bool) -> dict[str, Any]:
    """One encoder convolution with its exact MIL spelling.

    Rank-3 forms carry rank-3 kernels exactly as the encoder emits them;
    rank-4 forms carry the four-int pad vector.
    """
    blobs = Blobs()
    rank = len(x_shape)
    weight = blobs.constant("w", w_shape)
    body = list(blobs.const_scalars()) + blobs.body
    stride_kind = f"tensor<int32, [{len(stride)}]>"
    pad_kind = f"tensor<int32, [{len(pad)}]>"
    body.append(
        f'string pt = const()[name = string("pt"), val = string("{pad_type}")];')
    body.append(f'{stride_kind} st = const()[name = string("st"), '
                f'val = {stride_kind}([{", ".join(map(str, stride))}]);')
    body.append(f'{pad_kind} pd = const()[name = string("pd"), '
                f'val = {pad_kind}([{", ".join(map(str, pad))}]);')
    body.append('tensor<int32, [2]> dl = const()[name = string("dl"), '
                'val = tensor<int32, [2]>([1, 1])];')
    body.append(f'int32 gp = const()[name = string("gp"), val = int32({groups})];')
    bias_argument = ""
    if bias:
        b = blobs.constant("b", (w_shape[0],))
        bias_argument = f"bias = {b}, "
    body.append(
        f'{om.tensor_type(out_shape)} y = conv({bias_argument}dilations = dl, '
        f'groups = gp, pad = pd, pad_type = pt, strides = st, '
        f'weight = {weight}, x = x)[name = string("y")];')
    name = (f"encoder_conv_c{x_shape[1]}_n{w_shape[0]}_k{w_shape[2]}"
            f"{'x' + str(w_shape[3]) if rank == 4 else ''}"
            f"_s{stride[0]}_g{groups}_bias{int(bias)}_{pad_type}")
    item = om.env_case(name, "conv_probe", {
        "kernel": w_shape[2], "input_channels": x_shape[1],
        "output_channels": w_shape[0], "spatial": x_shape[2],
        "spatial_width": x_shape[3] if rank == 4 else 1,
        "stride": stride[0], "groups": groups, "bias": bias,
        "pad_type": pad_type, "input_shape": list(x_shape),
        "output_shape": list(out_shape), "weight_shape": list(w_shape),
        "rank": rank,
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return finish_blob_case(item, blobs)


def encoder_structure(kind: str) -> dict[str, Any]:
    """Concat / transpose / slice acceptance probes at encoder forms."""
    if kind == "concat_heads":
        inputs = {"x0": (1, 375, 128), "x1": (1, 375, 128)}
        body = [
            'int32 ax = const()[name = string("ax"), val = int32(1)];',
            'tensor<fp16, [1, 2, 375, 128]> y = concat(axis = ax, '
            'x0 = x0, x1 = x1)[name = string("y")];',
        ]
        name = "encoder_concat_xn_heads"
    elif kind == "concat_values":
        inputs = {"x0": (1, 375, 128), "x1": (1, 375, 128),
                  "x2": (1, 375, 128), "x3": (1, 375, 128),
                  "x4": (1, 375, 128), "x5": (1, 375, 128),
                  "x6": (1, 375, 128), "x7": (1, 375, 128)}
        names = ", ".join(f"x{index}" for index in range(8))
        body = [
            'int32 ax = const()[name = string("ax"), val = int32(1)];',
            f'tensor<fp16, [1, 8, 375, 128]> y = concat(axis = ax, '
            f'values = [{names}])[name = string("y")];',
        ]
        name = "encoder_concat_values_heads"
    elif kind == "transpose_r3":
        inputs = {"x": (1, 375, 1024)}
        body = [
            'tensor<int32, [3]> perm = const()[name = string("perm"), '
            'val = tensor<int32, [3]>([0, 2, 1])];',
            'tensor<fp16, [1, 1024, 375]> y = transpose(perm = perm, '
            'x = x)[name = string("y")];',
        ]
        name = "encoder_transpose_r3_021"
    elif kind == "transpose_r4":
        inputs = {"x": (1, 8, 375, 128)}
        body = [
            'tensor<int32, [4]> perm = const()[name = string("perm"), '
            'val = tensor<int32, [4]>([0, 2, 1, 3])];',
            'tensor<fp16, [1, 375, 8, 128]> y = transpose(perm = perm, '
            'x = x)[name = string("y")];',
        ]
        name = "encoder_transpose_r4_0213"
    elif kind == "slice_lastdim":
        inputs = {"x": (1, 8, 375, 749)}
        body = [
            'tensor<int32, [4]> eb = const()[name = string("eb"), '
            'val = tensor<int32, [4]>([0, 0, 0, 0])];',
            'tensor<int32, [4]> ee = const()[name = string("ee"), '
            'val = tensor<int32, [4]>([1, 8, 375, 375])];',
            'tensor<fp16, [1, 8, 375, 375]> y = slice_by_index(x = x, '
            'begin = eb, end = ee)[name = string("y")];',
        ]
        name = "encoder_slice_lastdim"
    else:
        raise ValueError(kind)
    arguments = ", ".join(f"{om.tensor_type(shape)} {operand}"
                          for operand, shape in inputs.items())
    return om.case(name, "encoder_structure", {"probe": kind},
                   om.program(arguments, body, "y"), None,
                   {"storage": "none"})


def encoder_chain(kind: str) -> dict[str, Any]:
    """FFN and norm-projection chains at d1024 s375 with distinct weights."""
    blobs = Blobs()
    shape = (1, 375, 1024)
    body = list(blobs.const_scalars()) + blobs.body
    if kind == "ffn":
        w1 = blobs.constant("w1", (4096, 1024))
        b1 = blobs.constant("b1", (4096,))
        w2 = blobs.constant("w2", (1024, 4096))
        b2 = blobs.constant("b2", (1024,))
        body.extend([
            'tensor<int32, [1]> axes = const()[name = string("axes"), '
            'val = tensor<int32, [1]>([-1])];',
            f'tensor<fp16, [1, 375, 4096]> h = linear(weight = {w1}, '
            f'bias = {b1}, x = x)[name = string("h")];',
            'tensor<fp16, [1, 375, 4096]> a = silu(x = h)'
            '[name = string("a")];',
            f'tensor<fp16, [1, 375, 1024]> y = linear(weight = {w2}, '
            f'bias = {b2}, x = a)[name = string("y")];',
        ])
        name = "chain_ffn_d1024_s375_silu"
    elif kind == "lnproj":
        gamma = blobs.constant("gamma", (1024,))
        beta = blobs.constant("beta", (1024,))
        w = blobs.constant("w", (1024, 1024))
        b = blobs.constant("b", (1024,))
        body.extend([
            'tensor<int32, [1]> axes = const()[name = string("axes"), '
            'val = tensor<int32, [1]>([-1])];',
            'tensor<fp16, [1, 375, 1024]> n = layer_norm(x = x, axes = axes, '
            f'epsilon = fp32(0.00001), gamma = {gamma}, beta = {beta})'
            '[name = string("n")];',
            f'tensor<fp16, [1, 375, 1024]> y = linear(weight = {w}, '
            f'bias = {b}, x = n)[name = string("y")];',
        ])
        name = "chain_lnproj_d1024_s375_params"
    else:
        raise ValueError(kind)
    item = om.case(name, "chain", {"probe": kind,
                                   "input_shape": list(shape)},
                   om.program(f"{om.tensor_type(shape)} x", body, "y"),
                   None, None)
    return finish_blob_case(item, blobs)


def finish_blob_case(item: dict[str, Any], blobs: Blobs) -> dict[str, Any]:
    """Attach the per-constant weights blob a BLOBFILE case compiled against."""
    item["weights"] = blobs.blob_bytes() if blobs.shapes else None
    if blobs.shapes:
        payload_bytes = 0
        for shape in blobs.shapes:
            payload_bytes += math_prod(shape) * 2
        item["weights_description"] = {
            "storage": "BLOBFILE",
            "shapes": [list(shape) for shape in blobs.shapes],
            "record_offsets": [blobs.offset(index)
                               for index in range(len(blobs.shapes))],
            "payload_bytes": payload_bytes,
            "value": "fp16 bits 0x3400 + index, one value per constant",
        }
    else:
        item["weights_description"] = {"storage": "none"}
    return item


def math_prod(shape: tuple[int, ...]) -> int:
    result = 1
    for extent in shape:
        result *= extent
    return result


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # 1. Linear chains: the encoder's rank-3 linears at their exact forms.
    cases.append(encoder_linear(375, 1024, 1024, bias=True))
    cases.append(encoder_linear(375, 1024, 4096, bias=True))
    cases.append(encoder_linear(375, 4096, 1024, bias=True))
    cases.append(encoder_linear(375, 1024, 128, bias=True))
    cases.append(encoder_linear(375, 1024, 640, bias=True))
    cases.append(encoder_linear(375, 1024, 1024, bias=False))
    # 2. The eight head projections as one B=8 batched matmul.
    cases.append(head_projection(8, 375, 1024, 128, transpose_y=True))
    # 3. Rank-3 unaries.
    cases.append(encoder_unary("silu", (1, 375, 4096)))
    cases.append(encoder_unary("silu", (1, 1024, 375)))
    cases.append(encoder_unary("sigmoid", (1, 1024, 375)))
    # 4. Normalizations at encoder shapes.
    cases.append(encoder_softmax((1, 8, 375, 375)))
    cases.append(encoder_softmax((1, 1, 375, 375)))
    cases.append(encoder_layer_norm((1, 375, 1024), params=False))
    cases.append(encoder_layer_norm((1, 375, 1024), params=True))
    # 5. The nine encoder convolutions.
    cases.append(encoder_conv(
        (1, 1024, 375), (2048, 1024, 1), (1, 2048, 375),
        stride=(1,), groups=1, pad=(0, 0), pad_type="valid", bias=False))
    cases.append(encoder_conv(
        (1, 1024, 375), (1024, 1024, 1), (1, 1024, 375),
        stride=(1,), groups=1, pad=(0, 0), pad_type="valid", bias=False))
    cases.append(encoder_conv(
        (1, 1024, 375), (1024, 1, 9), (1, 1024, 375),
        stride=(1,), groups=1024, pad=(4, 4), pad_type="custom", bias=False))
    cases.append(encoder_conv(
        (1, 8, 375, 750), (8, 1, 1, 1), (1, 8, 375, 750),
        stride=(1, 1), groups=8, pad=(0, 0, 1, 0), pad_type="custom",
        bias=False))
    cases.append(encoder_conv(
        (1, 1, 3000, 128), (256, 1, 3, 3), (1, 256, 1500, 64),
        stride=(2, 2), groups=1, pad=(1, 1, 1, 1), pad_type="custom",
        bias=True))
    cases.append(encoder_conv(
        (1, 256, 1500, 64), (256, 1, 3, 3), (1, 256, 750, 32),
        stride=(2, 2), groups=256, pad=(1, 1, 1, 1), pad_type="custom",
        bias=True))
    cases.append(encoder_conv(
        (1, 256, 750, 32), (256, 256, 1, 1), (1, 256, 750, 32),
        stride=(1, 1), groups=1, pad=(0, 0, 0, 0), pad_type="valid",
        bias=True))
    cases.append(encoder_conv(
        (1, 256, 750, 32), (256, 1, 3, 3), (1, 256, 375, 16),
        stride=(2, 2), groups=256, pad=(1, 1, 1, 1), pad_type="custom",
        bias=True))
    cases.append(encoder_conv(
        (1, 256, 375, 16), (256, 256, 1, 1), (1, 256, 375, 16),
        stride=(1, 1), groups=1, pad=(0, 0, 0, 0), pad_type="valid",
        bias=True))
    # 6. Structure probes: acceptance evidence, not lowerings.
    cases.append(encoder_structure("concat_heads"))
    cases.append(encoder_structure("concat_values"))
    cases.append(encoder_structure("transpose_r3"))
    cases.append(encoder_structure("transpose_r4"))
    cases.append(encoder_structure("slice_lastdim"))
    # 7. Chains at d1024 s375.
    cases.append(encoder_chain("ffn"))
    cases.append(encoder_chain("lnproj"))
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


# --------------------------------------------------------------------------
# Run harness (mirrors the batched round-2 campaign)
# --------------------------------------------------------------------------

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
            prefix=f"mil-hwx-encoder-{item['name']}-") as root:
        root_path = Path(root)
        capture = root_path / "capture"
        compiled = root_path / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        weights = item["weights"]
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
         "mktemp -d /tmp/mil-hwx-encoder-leftover.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-encoder-leftover."):
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
