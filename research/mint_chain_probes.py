#!/usr/bin/env python3
"""Mint transformer-block chains and report Apple's task partitioning rules.

The envelope campaign (``research/oracle-envelope.md``) proved Apple keeps every
accepted form in one HWX program and partitions work by emitting more tasks. It
measured five chain shapes at a handful of sizes, which is enough to see that
``x + relu(x)`` fuses and ``x + gelu(x)`` does not, but not enough to schedule a
transformer: it never ran a projection, a bias, a scaled score matrix, a
pre-norm block, or two stacked blocks, and it never labelled the individual
tasks Apple emitted.

This campaign mints the chains a native scheduler has to place — feed-forward
with and without bias, ``layer_norm`` feeding a projection, attention with the
score scaling, residuals around whole sub-blocks, full pre-norm blocks, and
stacks of blocks up to Apple's refusal — at realistic sizes (``d_model`` in
{256, 512, 768, 1024}, heads in {4, 8, 12}, sequence in {64, 128, 256, 512},
feed-forward 4x). ``--report`` reads the decoded JSON back and answers the
scheduling questions:

* how many tasks does Apple emit, and what is each task?  Every chain task is
  labelled by matching its register words against the single-op corpus already
  in ``research/oracles``, so a label is a measurement rather than a guess.
* which intermediates get a declared surface, and which live in the scratch
  allocation below the surfaces?
* which adjacent operations fuse into one task, and which register words carry
  the fusion?
* what is the largest chain Apple accepts in one program, and what does it say
  beyond that?

    python3 research/mint_chain_probes.py --host macstudio --targets h13 h14
    python3 research/mint_chain_probes.py --report --targets h13 h14

No Apple HWX bytes are retained: the records keep the MIL, the decoded task
words, the descriptors, constant-section hashes and sizes, and the compiler's
status string, exactly like every other campaign in this directory.
"""
from __future__ import annotations

import argparse
import collections
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402

SURFACE_BASE = 0x30000000
# fp16 1/8, a stand-in for 1/sqrt(head_dim) that stays exactly representable.
SCALE_LITERAL = "fp16(0x1p-3)"
# Blob metadata: the first record sits at 64, as in every other campaign here,
# and each record is the `<IIQQ` sentinel, dtype, size, data-offset tuple the
# committed writer tests already use.
RECORD_BASE = 64
RECORD_BYTES = 24
RECORD_CAPACITY = 4096
# fp16 bit patterns 0.25 (0x3400) through just under 1.0, so each constant
# carries its own value and Apple cannot merge two of them by content.
VALUE_BASE = 0x3400
VALUE_SPAN = 0x0800
# A signature below this many agreed words is too thin to identify anything:
# the matmul staging role alone contributes hundreds of groups.
MIN_SIGNATURE_WORDS = 64


def record_offset(index: int) -> int:
    return RECORD_BASE + index * RECORD_BYTES


def weight_blob(shapes: list[tuple[int, ...]]) -> bytes:
    """One weights.bin holding a distinct payload region per constant."""
    data_start = (record_offset(RECORD_CAPACITY) + 0x3F) & ~0x3F
    payloads = [struct.pack("<H", VALUE_BASE + index % VALUE_SPAN)
                * math.prod(shape) for index, shape in enumerate(shapes)]
    blob = bytearray(data_start + sum(len(payload) for payload in payloads))
    struct.pack_into("<II", blob, 0, len(shapes), 2)
    offset = data_start
    for index, payload in enumerate(payloads):
        struct.pack_into("<IIQQ", blob, record_offset(index),
                         0xDEADBEEF, 1, len(payload), offset)
        blob[offset:offset + len(payload)] = payload
        offset += len(payload)
    return bytes(blob)


# --------------------------------------------------------------------------
# MIL construction
# --------------------------------------------------------------------------

class Chain:
    """A MIL body under construction, with unique names for every value.

    Every BLOBFILE constant gets its own metadata record and its own payload
    region with its own fp16 value. Apple's compiler stores one copy of a
    constant that repeats — a block whose four projections all read blob
    offset 64 packs 1,183,744 bytes instead of the declared 1,572,864 — so a
    shared payload would measure a deduplicated program rather than the
    transformer the block describes.
    """

    def __init__(self, inputs: dict[str, tuple[int, ...]]):
        self.inputs = inputs
        self.body: list[str] = []
        self.declared: set[str] = set()
        self.shapes: list[tuple[int, ...]] = []
        self.weight_bytes = 0
        self.ops: list[str] = []
        self.counter = 0

    # -- naming and constants ------------------------------------------
    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"{stem}{self.counter}"

    def arguments(self) -> str:
        return ", ".join(f"{om.tensor_type(shape)} {name}"
                         for name, shape in self.inputs.items())

    def once(self, name: str, source: str) -> str:
        """Declare a shared constant at most once."""
        if name not in self.declared:
            self.declared.add(name)
            self.body.append(source)
        return name

    def flag(self, value: bool) -> str:
        name = "t" if value else "f"
        return self.once(name, f'bool {name} = const()[name = string("{name}"), '
                               f'val = bool({om.boolean(value)})];')

    def axis(self) -> str:
        return self.once("ax", 'int32 ax = const()[name = string("ax"), '
                               'val = int32(-1)];')

    def axes(self) -> str:
        return self.once("axs", 'tensor<int32, [1]> axs = const()'
                                '[name = string("axs"), '
                                'val = tensor<int32, [1]>([-1])];')

    def scale(self) -> str:
        return self.once("sc", f'fp16 sc = const()[name = string("sc"), '
                               f'val = {SCALE_LITERAL}];')

    def constant(self, shape: tuple[int, ...]) -> str:
        """A BLOBFILE constant of `shape` with its own record and payload."""
        name = self.fresh("c")
        index = len(self.shapes)
        kind = om.tensor_type(shape)
        self.body.append(
            f'{kind} {name} = const()[name = string("{name}"), val = {kind}'
            '(BLOBFILE(path = string("@model_path/weights.bin"), '
            f'offset = uint64({record_offset(index)})))];')
        self.shapes.append(shape)
        self.weight_bytes += math.prod(shape) * 2
        return name

    # -- operations ----------------------------------------------------
    def emit(self, shape: tuple[int, ...], expression: str, stem: str) -> str:
        name = self.fresh(stem)
        self.body.append(f'{om.tensor_type(shape)} {name} = {expression}'
                         f'[name = string("{name}")];')
        return name

    def matmul(self, x: str, out_shape: tuple[int, ...], y: str,
               transpose_x: bool = False, transpose_y: bool = True) -> str:
        self.ops.append("matmul")
        return self.emit(out_shape,
                         f'matmul(transpose_x = {self.flag(transpose_x)}, '
                         f'transpose_y = {self.flag(transpose_y)}, x = {x}, '
                         f'y = {y})', "m")

    def project(self, x: str, x_shape: tuple[int, ...], columns: int) -> str:
        """`x @ W^T` with a BLOBFILE weight, the transformer projection."""
        weight = self.constant((columns, x_shape[-1]))
        return self.matmul(x, x_shape[:-1] + (columns,), weight)

    def unary(self, operation: str, x: str, shape: tuple[int, ...]) -> str:
        self.ops.append(operation)
        mode = ', mode = string("EXACT")' if operation == "gelu" else ""
        return self.emit(shape, f'{operation}(x = {x}{mode})', operation[0])

    def binary(self, operation: str, x: str, y: str,
               shape: tuple[int, ...]) -> str:
        self.ops.append(operation)
        return self.emit(shape, f'{operation}(x = {x}, y = {y})',
                         "a" if operation == "add" else "p")

    def scaled(self, x: str, shape: tuple[int, ...]) -> str:
        return self.binary("mul", x, self.scale(), shape)

    def bias(self, x: str, shape: tuple[int, ...], form: str) -> str:
        """Add a BLOBFILE bias, broadcast over the rows or shaped like `x`."""
        operand = shape if form == "full" else (1,) * (len(shape) - 1) + (shape[-1],)
        return self.binary("add", x, self.constant(operand), shape)

    def softmax(self, x: str, shape: tuple[int, ...]) -> str:
        self.ops.append("softmax")
        return self.emit(shape, f'softmax(x = {x}, axis = {self.axis()})', "s")

    def layer_norm(self, x: str, shape: tuple[int, ...]) -> str:
        self.ops.append("layer_norm")
        return self.emit(shape, f'layer_norm(x = {x}, axes = {self.axes()}, '
                                'epsilon = fp32(0.00001))', "n")

    # -- composites ----------------------------------------------------
    def feed_forward(self, x: str, shape: tuple[int, ...], hidden: int,
                     bias: str | None = None,
                     activation: str = "gelu") -> str:
        inner = shape[:-1] + (hidden,)
        h = self.project(x, shape, hidden)
        if bias:
            h = self.bias(h, inner, bias)
        a = self.unary(activation, h, inner)
        y = self.project(a, inner, shape[-1])
        if bias:
            y = self.bias(y, shape, bias)
        return y

    def attention(self, q: str, k: str, v: str, shape: tuple[int, ...],
                  scale: bool = True) -> str:
        scores_shape = shape[:-1] + (shape[-2],)
        scores = self.matmul(q, scores_shape, k, transpose_y=True)
        if scale:
            scores = self.scaled(scores, scores_shape)
        probabilities = self.softmax(scores, scores_shape)
        return self.matmul(probabilities, shape, v, transpose_y=False)

    def block(self, x: str, shape: tuple[int, ...], projections: bool) -> str:
        """One pre-norm transformer block."""
        width = shape[-1]
        normalized = self.layer_norm(x, shape)
        if projections:
            q, k, v = (self.project(normalized, shape, width) for _ in range(3))
        else:
            q = k = v = normalized
        attended = self.attention(q, k, v, shape)
        if projections:
            attended = self.project(attended, shape, width)
        residual = self.binary("add", x, attended, shape)
        return self.binary("add", residual,
                           self.feed_forward(self.layer_norm(residual, shape),
                                             shape, 4 * width), shape)

    def weightless_unit(self, x: str, shape: tuple[int, ...]) -> str:
        """The block's shape with no constants, for the depth ceiling sweep.

        A projected block carries 1.5 MiB of weights, so depth alone runs into
        the weight budget long before it runs into whatever limit bounds the
        program. This unit keeps the operation mix — norm, attention, two
        residual adds, an activation — and drops every weight.
        """
        normalized = self.layer_norm(x, shape)
        attended = self.attention(normalized, normalized, normalized, shape)
        residual = self.binary("add", x, attended, shape)
        return self.binary("add", residual,
                           self.unary("gelu", self.layer_norm(residual, shape),
                                      shape), shape)


def chain_case(name: str, family: str, parameters: dict[str, Any],
               chain: Chain, result: str) -> dict[str, Any]:
    description: dict[str, Any] = {"storage": "none"}
    if chain.shapes:
        description = {
            "storage": "BLOBFILE",
            "shapes": [list(shape) for shape in chain.shapes],
            "record_offsets": [record_offset(index)
                               for index in range(len(chain.shapes))],
            "payload_bytes": chain.weight_bytes,
            "value": "fp16 bits 0x3400 + index, one value per constant",
        }
    parameters = dict(parameters)
    parameters["operations"] = chain.ops
    parameters["constant_matrices"] = [list(shape) for shape in chain.shapes]
    parameters["constant_matrix_bytes"] = chain.weight_bytes
    return om.case(name, family, parameters,
                   om.program(chain.arguments(), chain.body, result),
                   None, description)


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def ffn(width: int, sequence: int, bias: str | None = None,
        activation: str = "gelu") -> dict[str, Any]:
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    result = chain.feed_forward("x", shape, 4 * width, bias, activation)
    name = f"chain_ffn_d{width}_s{sequence}"
    if activation != "gelu":
        name += f"_{activation}"
    if bias:
        name += f"_bias{bias}"
    return chain_case(name, "chain_ffn", {
        "width": width, "sequence": sequence, "hidden": 4 * width,
        "bias": bias, "activation": activation, "input_shape": shape,
    }, chain, result)


def baseline(operation: str, width: int = 256, sequence: int = 64) -> dict[str, Any]:
    """One lone operation at the pair geometry, so a diff is exact.

    The corpus covers each of these operations, but never at
    `[1, sequence, width]` with this weight shape, and a fusion word can only
    be read off a diff whose two sides agree on the geometry.
    """
    shape = (1, sequence, width)
    inputs = {"x": shape}
    if operation == "add":
        inputs["z"] = shape
    chain = Chain(inputs)
    if operation == "matmul":
        result = chain.project("x", shape, width)
    elif operation == "add":
        result = chain.binary("add", "x", "z", shape)
    elif operation == "mul_scalar":
        result = chain.scaled("x", shape)
    elif operation == "bias":
        result = chain.bias("x", shape, "bcast")
    elif operation == "layer_norm":
        result = chain.layer_norm("x", shape)
    elif operation == "softmax":
        result = chain.softmax("x", shape)
    else:
        result = chain.unary(operation, "x", shape)
    return chain_case(f"chain_base_{operation}_d{width}_s{sequence}",
                      "chain_base", {
        "operation": operation, "width": width, "sequence": sequence,
        "input_shape": shape,
    }, chain, result)


def branches(width: int = 256, sequence: int = 64,
             independent: bool = True) -> dict[str, Any]:
    """Two projections of one input, summed: does Apple interleave them?"""
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    left = chain.project("x", shape, width)
    right = chain.project("x" if independent else left, shape, width)
    result = chain.binary("add", left, right, shape)
    name = "chain_branch" if independent else "chain_serial"
    return chain_case(f"{name}_d{width}_s{sequence}", "chain_order", {
        "width": width, "sequence": sequence, "independent": independent,
        "input_shape": shape,
    }, chain, result)


def norm_projection(width: int, sequence: int) -> dict[str, Any]:
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    result = chain.project(chain.layer_norm("x", shape), shape, width)
    return chain_case(f"chain_lnproj_d{width}_s{sequence}", "chain_norm", {
        "width": width, "sequence": sequence, "input_shape": shape,
    }, chain, result)


def attention(sequence: int, head_dim: int, heads: int = 1,
              scale: bool = True) -> dict[str, Any]:
    shape = (heads, sequence, head_dim)
    chain = Chain({"q": shape, "k": shape, "v": shape})
    result = chain.attention("q", "k", "v", shape, scale)
    name = f"chain_attn_s{sequence}_dh{head_dim}"
    if heads != 1:
        name += f"_h{heads}"
    if not scale:
        name += "_noscale"
    return chain_case(name, "chain_attention", {
        "sequence": sequence, "head_dim": head_dim, "heads": heads,
        "scale": scale, "operand_shape": shape,
    }, chain, result)


def residual_unary(operation: str, width: int) -> dict[str, Any]:
    shape = (1, width, 1, 1)
    chain = Chain({"x": shape})
    result = chain.binary("add", "x", chain.unary(operation, "x", shape), shape)
    return chain_case(f"chain_res_{operation}_d{width}", "chain_residual", {
        "operation": operation, "width": width, "shape": shape,
    }, chain, result)


def residual_ffn(width: int, sequence: int) -> dict[str, Any]:
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    result = chain.binary("add", "x",
                          chain.feed_forward("x", shape, 4 * width), shape)
    return chain_case(f"chain_resffn_d{width}_s{sequence}", "chain_residual", {
        "width": width, "sequence": sequence, "hidden": 4 * width,
        "input_shape": shape,
    }, chain, result)


def residual_attention(sequence: int, width: int) -> dict[str, Any]:
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    result = chain.binary("add", "x",
                          chain.attention("x", "x", "x", shape), shape)
    return chain_case(f"chain_resattn_s{sequence}_d{width}", "chain_residual", {
        "sequence": sequence, "width": width, "input_shape": shape,
    }, chain, result)


# A full record keeps every decoded task word, which is the evidence a
# fusion rule rests on; beyond this many tasks the JSON runs to hundreds of
# megabytes and says nothing the shallower stacks have not already said, so
# deeper chains are recorded as `chain_ceiling` summaries instead.
FULL_RECORD_DEPTH = 8


def stack_family(depth: int) -> str:
    if depth == 1:
        return "chain_block"
    return "chain_stack" if depth <= FULL_RECORD_DEPTH else "chain_ceiling"


def block(width: int, sequence: int, projections: bool,
          depth: int = 1) -> dict[str, Any]:
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    value = "x"
    for _ in range(depth):
        value = chain.block(value, shape, projections)
    stem = "chain_block" if depth == 1 else f"chain_stack{depth}"
    return chain_case(f"{stem}_d{width}_s{sequence}_proj{int(projections)}",
                      stack_family(depth), {
        "width": width, "sequence": sequence, "projections": projections,
        "depth": depth, "hidden": 4 * width, "input_shape": shape,
    }, chain, value)


def deep(depth: int, width: int = 256, sequence: int = 64) -> dict[str, Any]:
    """`depth` weightless block units, the depth probe."""
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    value = "x"
    for _ in range(depth):
        value = chain.weightless_unit(value, shape)
    return chain_case(f"chain_deep{depth}_d{width}_s{sequence}",
                      stack_family(depth), {
        "width": width, "sequence": sequence, "depth": depth,
        "input_shape": shape,
    }, chain, value)


def weight_ceiling(depth: int, width: int) -> dict[str, Any]:
    """A projected stack whose constant section passes 134 MiB."""
    shape = (1, 64, width)
    chain = Chain({"x": shape})
    value = "x"
    for _ in range(depth):
        value = chain.block(value, shape, True)
    return chain_case(f"chain_ceiling_weights{depth}_d{width}", "chain_ceiling", {
        "width": width, "sequence": 64, "depth": depth, "projections": True,
        "hidden": 4 * width, "input_shape": shape,
    }, chain, value)


def envelope_bound(rows: int, reduction: int, columns: int,
                   transpose_y: bool = True) -> dict[str, Any]:
    """A feed-forward chain built on one specific matmul geometry.

    The envelope campaign found exactly two refusals among single operations:
    a rank-2 constant-weight matmul at M=128 with K=8192, and any
    `transpose_y=false` BLOBFILE weight. These probes put those geometries
    inside a chain, which answers whether a chain inherits the refusal of the
    operation it contains or whether Apple re-plans around it.
    """
    x_shape = (rows, reduction)
    inner = (rows, columns)
    chain = Chain({"x": x_shape})
    weight = chain.constant((columns, reduction) if transpose_y
                            else (reduction, columns))
    hidden = chain.matmul("x", inner, weight, transpose_y=transpose_y)
    activated = chain.unary("gelu", hidden, inner)
    result = chain.matmul(activated, x_shape,
                          chain.constant((reduction, columns)))
    return chain_case(
        f"chain_ceiling_mlp_m{rows}_k{reduction}_n{columns}"
        f"_ty{int(transpose_y)}", "chain_ceiling", {
            "rows": rows, "reduction": reduction, "columns": columns,
            "transpose_y": transpose_y, "depth": 1, "input_shape": x_shape,
        }, chain, result)


def pair(label: str, width: int = 256, sequence: int = 64) -> dict[str, Any]:
    """One adjacent-operation pair, the smallest question about fusion."""
    shape = (1, sequence, width)
    chain = Chain({"x": shape})
    if label == "mm_relu":
        result = chain.unary("relu", chain.project("x", shape, width), shape)
    elif label == "mm_gelu":
        result = chain.unary("gelu", chain.project("x", shape, width), shape)
    elif label == "mm_silu":
        result = chain.unary("silu", chain.project("x", shape, width), shape)
    elif label == "mm_bias":
        result = chain.bias(chain.project("x", shape, width), shape, "bcast")
    elif label == "mm_bias_relu":
        biased = chain.bias(chain.project("x", shape, width), shape, "bcast")
        result = chain.unary("relu", biased, shape)
    elif label == "mm_add":
        result = chain.binary("add", "x", chain.project("x", shape, width), shape)
    elif label == "mm_mm":
        result = chain.project(chain.project("x", shape, width), shape, width)
    elif label == "gelu_mm":
        result = chain.project(chain.unary("gelu", "x", shape), shape, width)
    elif label == "ln_gelu":
        result = chain.unary("gelu", chain.layer_norm("x", shape), shape)
    elif label == "ln_ln":
        result = chain.layer_norm(chain.layer_norm("x", shape), shape)
    elif label == "add_ln":
        result = chain.layer_norm(
            chain.binary("add", "x", chain.project("x", shape, width), shape),
            shape)
    elif label == "mm_softmax":
        result = chain.softmax(chain.project("x", shape, width), shape)
    elif label == "softmax_mm":
        result = chain.project(chain.softmax("x", shape), shape, width)
    elif label == "mul_mm":
        result = chain.project(chain.scaled("x", shape), shape, width)
    elif label == "mm_mul":
        result = chain.scaled(chain.project("x", shape, width), shape)
    else:
        raise ValueError(f"unknown pair {label!r}")
    return chain_case(f"chain_pair_{label}_d{width}_s{sequence}", "chain_pair", {
        "pair": label, "width": width, "sequence": sequence,
        "input_shape": shape,
    }, chain, result)


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # 0. Shape-matched single-operation baselines, so every pair below can be
    # diffed against its own parts rather than against a corpus record at a
    # different geometry.
    for operation in ("matmul", "add", "mul_scalar", "bias", "relu", "gelu",
                      "silu", "layer_norm", "softmax"):
        cases.append(baseline(operation))
    cases.append(branches(independent=True))
    cases.append(branches(independent=False))

    # 1. Adjacent-operation pairs at one small geometry. These are the rows a
    # fusion rule is read off: each one is a lone operation the corpus already
    # covers plus exactly one neighbour.
    for label in ("mm_relu", "mm_gelu", "mm_silu", "mm_bias", "mm_bias_relu",
                  "mm_add", "mm_mm", "gelu_mm", "ln_gelu", "ln_ln", "add_ln",
                  "mm_softmax", "softmax_mm", "mul_mm", "mm_mul"):
        cases.append(pair(label))

    # 2. Feed-forward, the block that dominates a transformer's task budget.
    for width in (256, 512, 768, 1024):
        for sequence in (64, 128):
            cases.append(ffn(width, sequence))
    for sequence in (256, 512):
        cases.append(ffn(256, sequence))
    cases.append(ffn(512, 256))
    for bias in ("bcast", "full"):
        cases.append(ffn(256, 64, bias=bias))
        cases.append(ffn(512, 128, bias=bias))
    cases.append(ffn(768, 128, bias="bcast"))
    cases.append(ffn(256, 64, activation="silu"))
    cases.append(ffn(256, 64, activation="relu"))

    # 3. layer_norm feeding a projection, the head of every pre-norm block.
    for width in (256, 512, 768, 1024):
        for sequence in (64, 128):
            cases.append(norm_projection(width, sequence))
    for sequence in (256, 512):
        cases.append(norm_projection(512, sequence))

    # 4. Attention with the score scaling, at the head widths d_model / heads
    # gives for the swept widths and head counts.
    for sequence in (64, 128, 256, 512):
        for head_dim in (64, 96):
            cases.append(attention(sequence, head_dim))
    for heads in (4, 8, 12):
        cases.append(attention(128, 64, heads=heads))
    cases.append(attention(128, 64, scale=False))
    cases.append(attention(256, 64, scale=False))

    # 5. Residuals: around a single activation, and around whole sub-blocks.
    for operation in ("relu", "gelu", "silu"):
        for width in (256, 1024):
            cases.append(residual_unary(operation, width))
    for width in (256, 512):
        for sequence in (64, 128):
            cases.append(residual_ffn(width, sequence))
    cases.append(residual_attention(64, 256))
    cases.append(residual_attention(128, 256))

    # 6. Whole pre-norm blocks, with and without the four projections.
    for width in (256, 512, 768):
        for sequence in (64, 128):
            cases.append(block(width, sequence, True))
    cases.append(block(256, 64, False))
    cases.append(block(512, 128, False))

    # 7. Stacked blocks, and the depth sweep that finds Apple's ceiling. The
    # projected stack carries 1.5 MiB of weights per block, so the deep end of
    # the sweep drops the projections and then halves the width to keep the
    # constant section inside the 134 MiB the envelope campaign already
    # proved Apple accepts.
    for width in (256, 512):
        for sequence in (64, 128):
            cases.append(block(width, sequence, True, depth=2))
    for depth in (3, 4, 6, 8, 12, 16, 24, 32, 48, 64):
        cases.append(block(256, 64, True, depth=depth))
    for depth in (48, 64, 96, 128):
        cases.append(block(256, 64, False, depth=depth))
    for depth in (128, 192, 256, 384, 512):
        cases.append(block(128, 64, False, depth=depth))

    # 8. The ceiling. Every chain above `FULL_RECORD_DEPTH` is a summary
    # record, which is also what separates Apple's verdict from whether this
    # repository's task splitter can read the result back: the depth-4096
    # object compiled cleanly and only the splitter failed.
    for depth in (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
        cases.append(deep(depth))
    for depth, width in ((32, 512), (64, 512), (16, 1024)):
        cases.append(weight_ceiling(depth, width))
    # Depth, task count and weight bytes are all accepted far past anything a
    # transformer needs, so the remaining bound is the per-operation
    # envelope: these probes put the refused geometries, and their accepted
    # neighbours, inside a chain.
    cases.append(envelope_bound(64, 256, 256, transpose_y=False))
    cases.append(envelope_bound(32, 2048, 2048, transpose_y=False))
    cases.append(envelope_bound(128, 2048, 2048, transpose_y=False))
    cases.append(envelope_bound(128, 8192, 8192))
    cases.append(envelope_bound(256, 8192, 8192))

    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        if not item["name"].startswith("chain_"):
            raise AssertionError(f"probe {item['name']} lacks the chain_ prefix")
        unique.setdefault(item["name"], item)
    return list(unique.values())


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------

def selected_cases(pattern: str | None) -> list[dict[str, Any]]:
    return [item for item in campaign()
            if not pattern or fnmatch.fnmatch(item["name"], pattern)]


def run_ceiling_case(item: dict[str, Any], target: str, output: Path,
                     tool: Path, commit: str) -> str:
    """Compile one ceiling probe and record acceptance apart from decoding.

    `mint_oracles.run_case` folds a decoder limit into the same `error` field
    as an Apple refusal, which is exactly the distinction a ceiling needs:
    the depth-4096 probe compiled cleanly and only this repository's task
    splitter could not read the 42 MiB task section back.
    """
    destination = output / target / f"{item['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1, "case": item["name"], "family": item["family"],
        "parameters": item["parameters"], "mil_sha256":
            hashlib.sha256(item["mil"].encode()).hexdigest(),
        "mil_bytes": len(item["mil"]), "weights": item["weights_description"],
        "target": target, "source_commit": commit,
        "compiler": {
            "tool": str(tool),
            "tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "host": platform.node(), "platform": platform.platform(),
            "command": [str(tool), "CAPTURE_DIR", "OUTPUT_DIR", target],
        },
    }
    with tempfile.TemporaryDirectory(prefix=f"mil-hwx-ceiling-{target}-") as root:
        capture, compiled = Path(root) / "capture", Path(root) / "compiled"
        capture.mkdir()
        compiled.mkdir()
        (capture / "model.mil").write_text(item["mil"])
        (capture / "weights.bin").write_bytes(item["weights"] or b"")
        started = time.monotonic()
        try:
            result = subprocess.run(
                [str(tool), str(capture), str(compiled), target],
                capture_output=True, text=True, check=False,
                timeout=om.COMPILE_TIMEOUT_SECONDS)
            status, output_text = result.returncode, (result.stderr or result.stdout)
        except subprocess.TimeoutExpired:
            status, output_text = None, "compiler timed out"
        hwx = compiled / "model.hwx"
        record["compiler"].update({
            "returncode": status, "status": output_text.strip()[:2000],
            "seconds": round(time.monotonic() - started, 1),
        })
        record["accepted"] = status == 0 and hwx.is_file()
        if record["accepted"]:
            data = hwx.read_bytes()
            record.update({"hwx_bytes": len(data),
                           "hwx_sha256": hashlib.sha256(data).hexdigest()})
            try:
                parsed = om.parse_hwx(data, target)
                record.update({
                    "program_count": parsed["program_count"],
                    "task_count": len(parsed["task_descriptors"]),
                    "constant_bytes": parsed["constant_section"]["size"],
                    "parse_error": None,
                })
            except (ValueError, KeyError, struct.error) as error:
                record["parse_error"] = str(error)
        else:
            record["hwx_bytes"] = None
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return "accepted" if record["accepted"] else "refused"


def local_run(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    selected = selected_cases(args.case)
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
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
            shapes = [tuple(shape) for shape
                      in item["parameters"]["constant_matrices"]]
            item["weights"] = weight_blob(shapes) if shapes else None
            try:
                if item["family"] == "chain_ceiling":
                    status = run_ceiling_case(item, target, output, tool,
                                              args.source_commit)
                else:
                    status = om.run_case(item, target, output, tool,
                                         args.source_commit)
            finally:
                item["weights"] = None
            decoded += status in ("decoded", "accepted")
            rejected += status in ("rejected", "refused")
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} "
          f"decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-chain-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-chain-probes."):
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
            with tempfile.TemporaryDirectory(prefix="mil-hwx-chain-json-") as staging:
                subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


# --------------------------------------------------------------------------
# Task labelling: match a chain task against the single-op corpus
# --------------------------------------------------------------------------

SINGLE_OP_FAMILIES = {
    "binary_runtime", "binary_constant", "unary", "normalization", "reduction",
    "matmul", "env_matmul", "rrmm_matmul", "rrmm_matvec", "env_broadcast",
    "rrmm_broadcast", "env_activation", "env_conv", "convolution",
    "chain_base",
}
# The `chain_base` probes in this campaign are the only single-operation
# records at the chain geometry, so their labels have to line up with the
# corpus names rather than with the MIL operation names.
BASELINE_LABELS = {
    "matmul": "matmul_const", "add": "add", "mul_scalar": "mul_bcast",
    "bias": "add_bcast", "relu": "relu", "gelu": "gelu", "silu": "silu",
    "layer_norm": "layer_norm", "softmax": "softmax",
}


def single_op_label(record: dict[str, Any]) -> str | None:
    """The one operation a corpus record contains, or None if it holds more."""
    family = record["family"]
    if family not in SINGLE_OP_FAMILIES:
        return None
    parameters = record["parameters"]
    if family == "chain_base":
        return BASELINE_LABELS[parameters["operation"]]
    if parameters.get("relu") or parameters.get("bias"):
        return None  # a fused corpus case, not a lone operation
    if family in ("matmul", "env_matmul", "rrmm_matmul", "rrmm_matvec"):
        runtime = parameters.get("w_storage", "blob") == "runtime"
        return "matmul_rt" if runtime else "matmul_const"
    if family in ("env_conv", "convolution"):
        return "conv"
    operation = parameters.get("operation")
    if not operation:
        return None
    if family in ("env_broadcast", "rrmm_broadcast"):
        same = parameters.get("shape") == parameters.get("operand_shape")
        return f"{operation}" if same else f"{operation}_bcast"
    if family == "binary_constant":
        return f"{operation}_const"
    return operation


def task_slots(task: dict[str, Any]) -> dict[str, int]:
    """Every header word and register value in one decoded task."""
    slots = {f"header[{index}]": int(word, 16)
             for index, word in enumerate(task["header_words"])}
    for block in task["blocks"].values():
        for address, value in block["words"].items():
            slots[address] = int(value, 16)
    return slots


def load_records(target: str, directory: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((directory / target).glob("*.json")):
        record = json.loads(path.read_text())
        record["path"] = str(path)
        records.append(record)
    return records


def shape_of(record: dict[str, Any]) -> tuple[int, ...]:
    parameters = record["parameters"]
    for key in ("shape", "input_shape", "x_shape"):
        if parameters.get(key):
            return tuple(parameters[key])
    return ()


def single_op_tasks(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every decoded task from a record that holds exactly one operation."""
    tasks = []
    for record in records:
        if record.get("error") is not None:
            continue
        label = single_op_label(record)
        if label is None:
            continue
        count = len(record["task_descriptors"])
        for index, task in enumerate(record["task_descriptors"]):
            if "decode_error" in task:
                continue
            tasks.append({
                "label": label, "case": record["case"], "index": index,
                "count": count, "shape": shape_of(record),
                "baseline": record["family"] == "chain_base",
                "size": task["size_bytes"], "slots": task_slots(task),
            })
    return tasks


def operation_varying_slots(tasks: list[dict[str, Any]]) -> set[str]:
    """Slots two operations disagree on at one shape, task count and index.

    Derived, not assumed: holding the shape and the task role fixed leaves
    the operation as the only difference, so any slot that moves inside such
    a group carries operation identity rather than geometry.
    """
    per_shape: dict[tuple, dict[str, dict[str, int]]] = collections.defaultdict(dict)
    for task in tasks:
        per_shape[(task["shape"], task["count"], task["index"])].setdefault(
            task["label"], task["slots"])
    varying: set[str] = set()
    for observations in per_shape.values():
        if len(observations) < 2:
            continue
        for address in set().union(*(set(slots) for slots in observations.values())):
            if len({slots.get(address) for slots in observations.values()}) > 1:
                varying.add(address)
    return varying


def build_reference(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per (operation, task count, task index) the operation-selector words.

    A corpus signature keeps only the operation-varying slots that every task
    of that operation and role agrees on, so a slot that tracks the shape
    drops out on its own rather than by assumption. A `chain_base` signature
    is kept even from one record: it is the only lone operation minted at the
    chain geometry, which is exactly what a chain task has to be compared
    against.
    """
    tasks = single_op_tasks(records)
    varying = operation_varying_slots(tasks)
    grouped: dict[tuple[str, int, int, bool], list[dict[str, int]]] = collections.defaultdict(list)
    for task in tasks:
        grouped[(task["label"], task["count"], task["index"],
                 task["baseline"])].append(task["slots"])
    signatures: dict[tuple[str, int, int, bool], dict[str, int]] = {}
    for key, observations in grouped.items():
        if len(observations) < 2 and not key[3]:
            continue
        agreed = {address: observations[0].get(address) for address in varying
                  if all(slots.get(address) == observations[0].get(address)
                         for slots in observations[1:])}
        # A thin signature matches anything: the matmul staging tasks alone
        # contribute 271 role groups, and the smallest agrees on 28 words.
        if len(agreed) >= MIN_SIGNATURE_WORDS:
            signatures[key] = agreed
    per_slot: dict[tuple[str, int | None], set[str]] = collections.defaultdict(set)
    for task in tasks:
        for address in varying:
            per_slot[(address, task["slots"].get(address))].add(task["label"])
    return {"varying": sorted(varying), "signatures": signatures,
            "per_slot": per_slot, "tasks": len(tasks),
            "labels": sorted({task["label"] for task in tasks})}


def label_task(task: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Label one chain task against the single-op signatures.

    Every operation reaching the best score is named, because operations that
    genuinely emit the same selector words — the lookup-table unaries, whose
    identity lives in the constant section instead — must not be reported as
    one arbitrary pick. A task that misses by a few words is where fusion
    shows: the report carries those words and which operation each belongs
    to.
    """
    slots = task_slots(task)
    scored: list[tuple[float, int, str, list[str]]] = []
    for key, signature in reference["signatures"].items():
        label = key[0]
        matched = [address for address, value in signature.items()
                   if slots.get(address) == value]
        missed = [address for address in signature if address not in matched]
        scored.append((len(matched) / len(signature), len(matched), label, missed))
    scored.sort(reverse=True)
    best = scored[0][0]
    winners = sorted({entry[2] for entry in scored if entry[0] == best})
    missed = min((entry[3] for entry in scored if entry[0] == best), key=len)
    runner_up = next((entry for entry in scored if entry[2] not in winners), None)
    return {
        "label": "|".join(winners),
        "score": round(best, 3),
        "exact": best == 1.0,
        "size": task["size_bytes"],
        "runner_up": runner_up[2] if runner_up else None,
        "runner_up_score": round(runner_up[0], 3) if runner_up else None,
        "fused_words": {
            address: {
                "value": f"0x{slots[address]:08x}" if address in slots else None,
                "belongs_to": sorted(reference["per_slot"].get(
                    (address, slots.get(address)), ())),
            } for address in missed},
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def surface_layout(record: dict[str, Any]) -> dict[str, Any]:
    """Declared surfaces, binding roles, and the scratch below them."""
    addresses = [int(value, 16) for value in
                 record["programs"][-1]["resource_addresses"]]
    live = [value for value in addresses if value]
    bindings = collections.Counter(descriptor["binding"]
                                   for descriptor in record["tensor_descriptors"])
    return {
        "surfaces": len(record["tensor_descriptors"]),
        "resources": len(live),
        "inputs": bindings.get(1, 0),
        "outputs": bindings.get(2, 0),
        "scratch": (min(live) - SURFACE_BASE) if live else 0,
        "surface_bytes": sum(descriptor["total_bytes"]
                             for descriptor in record["tensor_descriptors"]),
    }


def pe_words(task: dict[str, Any]) -> dict[str, str]:
    """The PE and NE block words, where a fused post-operation shows up."""
    wanted = ("pe", "ne")
    return {address: value
            for block in task["blocks"].values() if block["name"] in wanted
            for address, value in sorted(block["words"].items())}


PAIR_PARTS = {
    "mm_relu": ("matmul", "relu"), "mm_gelu": ("matmul", "gelu"),
    "mm_silu": ("matmul", "silu"), "mm_bias": ("matmul", "bias"),
    "mm_bias_relu": ("matmul", "relu"), "mm_add": ("matmul", "add"),
    "mm_mm": ("matmul", "matmul"), "gelu_mm": ("gelu", "matmul"),
    "ln_gelu": ("layer_norm", "gelu"), "ln_ln": ("layer_norm", "layer_norm"),
    "add_ln": ("matmul", "layer_norm"), "mm_softmax": ("matmul", "softmax"),
    "softmax_mm": ("softmax", "matmul"), "mul_mm": ("mul_scalar", "matmul"),
    "mm_mul": ("matmul", "mul_scalar"),
}


def analyze(target: str, directory: Path) -> dict[str, Any]:
    records = load_records(target, directory)
    reference = build_reference(records)
    chains, ceilings = [], []
    tasks_by_case: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not record["case"].startswith("chain_") or record["family"] == "chain":
            continue  # `chain` is the first campaign's two-op case
        if record["family"] == "chain_ceiling":
            ceilings.append(record)
            continue
        entry = {
            "case": record["case"], "family": record["family"],
            "parameters": record["parameters"], "error": record["error"],
        }
        if record["error"] is None:
            tasks = record["task_descriptors"]
            tasks_by_case[record["case"]] = tasks
            labels, _ = label_all(tasks, reference)
            lengths = [task["size_bytes"] for task in tasks]
            entry.update({
                "tasks": len(tasks),
                "programs": record["program_count"],
                "sizes": collections.Counter(lengths),
                "labels": labels,
                # The register words carry per-block surface offsets, so the
                # repeat shows in the task lengths rather than in the words.
                "period": role_period(lengths),
                "constant_bytes": record["constant_section"]["size"],
                "constant_nonzero": record["constant_section"]["nonzero_bytes"],
                "hwx_bytes": record["hwx_bytes"],
                "task_bytes": record["programs"][-1]["task_section"]["size"],
                **surface_layout(record),
            })
        chains.append(entry)
    return {"reference": reference, "chains": chains, "ceilings": ceilings,
            "records": len(records), "tasks_by_case": tasks_by_case}


def label_all(tasks: list[dict[str, Any]],
              reference: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple]]:
    """Label every task, reusing the answer for tasks of the same role.

    Labelling only reads the operation-varying words, so two tasks that agree
    on all of them must get the same label; a 2,048-unit chain holds 36,864
    tasks and 18 roles, and the cache is what makes that tractable.
    """
    cache: dict[tuple, dict[str, Any]] = {}
    labels, roles = [], []
    for task in tasks:
        slots = task_slots(task)
        role = tuple(slots.get(address) for address in reference["varying"])
        if role not in cache:
            cache[role] = label_task(task, reference)
        labels.append(cache[role])
        roles.append(role)
    return labels, roles


def role_period(roles: list[tuple]) -> int | None:
    """The shortest repeat length of the task-role sequence, if it repeats."""
    total = len(roles)
    for period in range(1, total // 2 + 1):
        if total % period:
            continue
        if all(roles[index] == roles[index % period] for index in range(total)):
            return period
    return None


def fusion_rows(analysis: dict[str, Any], width: int = 256,
                sequence: int = 64) -> list[dict[str, Any]]:
    """Per pair case, the words that differ from the lone first operation.

    A word whose fused value equals the lone second operation's value is the
    word that carries the fusion; a word that matches neither is new.
    """
    tasks_by_case = analysis["tasks_by_case"]
    per_slot = analysis["reference"]["per_slot"]
    rows = []
    for label, (first, second) in PAIR_PARTS.items():
        case = f"chain_pair_{label}_d{width}_s{sequence}"
        left = f"chain_base_{first}_d{width}_s{sequence}"
        right = f"chain_base_{second}_d{width}_s{sequence}"
        if not all(name in tasks_by_case for name in (case, left, right)):
            continue
        fused, lone, other = (tasks_by_case[name] for name in (case, left, right))
        for index, task in enumerate(fused[:len(lone)]):
            slots, lone_slots = task_slots(task), task_slots(lone[index])
            other_slots = task_slots(other[min(index, len(other) - 1)])
            for address in sorted(set(slots) | set(lone_slots)):
                value, was = slots.get(address), lone_slots.get(address)
                if value == was:
                    continue
                rows.append({
                    "case": case, "pair": label, "task": index,
                    "tasks": (len(fused), len(lone), len(other)),
                    "word": address,
                    "lone_first": was, "fused": value,
                    "lone_second": other_slots.get(address),
                    "adopted": value == other_slots.get(address),
                    "belongs_to": sorted(per_slot.get((address, value), ())),
                })
    return rows


def label_sequence(entry: dict[str, Any]) -> str:
    """The labelled task sequence, run-length encoded.

    A stacked chain repeats one task pattern per block, so the sequence is
    printed once and marked with its repeat count instead of 36,864 times.
    """
    labels = entry["labels"]
    period = entry["period"]
    repeats = 1
    if period and period < len(labels):
        labels, repeats = labels[:period], len(labels) // period
    parts: list[list[Any]] = []
    for label in labels:
        text = label["label"] + ("" if label["exact"] else f"~{label['score']}")
        if parts and parts[-1][0] == text:
            parts[-1][1] += 1
        else:
            parts.append([text, 1])
    sequence = " -> ".join(text if count == 1 else f"{text} x{count}"
                           for text, count in parts)
    return sequence if repeats == 1 else f"[{sequence}] x{repeats}"


def size_text(sizes: collections.Counter) -> str:
    return ",".join(f"{size}x{count}" for size, count in sorted(sizes.items()))


def ceiling_verdict(record: dict[str, Any]) -> str:
    """Apple's verdict, with a compile-budget timeout kept apart from a refusal."""
    if record["accepted"]:
        return "accepted"
    if "timed out" in (record["compiler"]["status"] or ""):
        return "timed out here"
    return "refused"


def print_report(target: str, analysis: dict[str, Any]) -> None:
    chains = analysis["chains"]
    decoded = [entry for entry in chains if entry["error"] is None]
    refused = [entry for entry in chains if entry["error"] is not None]
    reference = analysis["reference"]
    print(f"\n## {target.upper()}: {len(decoded)} decoded, {len(refused)} refused; "
          f"{len(reference['varying'])} operation-varying words and "
          f"{len(reference['signatures'])} role signatures derived from "
          f"{reference['tasks']} single-op tasks in "
          f"{analysis['records']} corpus records\n")
    print("### Decoded chains\n")
    print("| Case | Ops | Tasks | Labelled task sequence | Task sizes | "
          "Surf (in/out) | Scratch | Const bytes | Weight bytes | "
          "Task bytes | HWX bytes |")
    print("|---|---:|---:|---|---|---|---:|---:|---:|---:|---:|")
    for entry in sorted(decoded, key=lambda item: item["case"]):
        parameters = entry["parameters"]
        print(f"| `{entry['case']}` | {len(parameters['operations'])} | "
              f"{entry['tasks']} | {label_sequence(entry)} | "
              f"{size_text(entry['sizes'])} | "
              f"{entry['surfaces']} ({entry['inputs']}/{entry['outputs']}) | "
              f"{entry['scratch']} | {entry['constant_bytes']} | "
              f"{parameters['constant_matrix_bytes']} | {entry['task_bytes']} | "
              f"{entry['hwx_bytes']} |")
    if refused:
        print("\n### Refusals\n")
        print("| Case | Ops | Compiler status |")
        print("|---|---:|---|")
        for entry in sorted(refused, key=lambda item: item["case"]):
            print(f"| `{entry['case']}` | "
                  f"{len(entry['parameters']['operations'])} | "
                  f"{entry['error']} |")
    rows = fusion_rows(analysis)
    if rows:
        print("\n### Words a pair changes against its lone first operation\n")
        print("| Pair | Tasks (pair/first/second) | Task | Word | Lone first | "
              "Fused | Lone second | Adopted from second |")
        print("|---|---|---:|---|---|---|---|---|")
        for row in rows:
            def hexed(value: int | None) -> str:
                return "-" if value is None else f"0x{value:08x}"
            print(f"| `{row['pair']}` | {'/'.join(map(str, row['tasks']))} | "
                  f"{row['task']} | `{row['word']}` | {hexed(row['lone_first'])} | "
                  f"{hexed(row['fused'])} | {hexed(row['lone_second'])} | "
                  f"{'yes' if row['adopted'] else 'no'} |")
    depths = [entry for entry in decoded
              if entry["family"] in ("chain_stack", "chain_deep", "chain_block")
              and entry["error"] is None]
    if depths:
        print("\n### Depth scaling\n")
        print("| Case | Depth | Tasks | Tasks per block | Const bytes | "
              "Const per block | Task bytes | Task bytes per block | Scratch | "
              "Programs |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for entry in sorted(depths, key=lambda item: (
                item["parameters"].get("width", 0),
                item["parameters"].get("projections", True),
                item["parameters"].get("depth", 1))):
            depth = entry["parameters"].get("depth", 1)
            print(f"| `{entry['case']}` | {depth} | {entry['tasks']} | "
                  f"{entry['tasks'] / depth:.2f} | {entry['constant_bytes']} | "
                  f"{entry['constant_bytes'] // depth} | {entry['task_bytes']} | "
                  f"{entry['task_bytes'] // depth} | {entry['scratch']} | "
                  f"{entry['programs']} |")
    if analysis["ceilings"]:
        print("\n### Ceiling probes (summary records)\n")
        print("| Case | Depth | Weight bytes | Apple verdict | Compiler status "
              "| Seconds | HWX bytes | Tasks | Const bytes | Decoder |")
        print("|---|---:|---:|---|---|---:|---:|---:|---:|---|")
        for record in sorted(analysis["ceilings"],
                             key=lambda item: (item["parameters"].get("width", 0),
                                               item["parameters"]["depth"])):
            parameters = record["parameters"]
            compiler = record["compiler"]
            decoder = record.get("parse_error") or "read back"
            print(f"| `{record['case']}` | {parameters['depth']} | "
                  f"{parameters['constant_matrix_bytes']} | "
                  f"{ceiling_verdict(record)} | "
                  f"{compiler['status'] or 'silent, exit 0'} | "
                  f"{compiler['seconds']} | {record['hwx_bytes']} | "
                  f"{record.get('task_count', '-')} | "
                  f"{record.get('constant_bytes', '-')} | {decoder} |")


def report(args: argparse.Namespace) -> int:
    directory = Path(args.output)
    for target in args.targets:
        print_report(target, analyze(target, directory))
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true",
                      help="run Apple's compiler on this Mac")
    mode.add_argument("--host", help="copy the worker to this SSH Mac")
    mode.add_argument("--report", action="store_true",
                      help="analyze the decoded records already on disk")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=sorted(om.SUBTYPES))
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--case", help="shell pattern selecting case names")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.report:
        raise SystemExit(report(arguments))
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
