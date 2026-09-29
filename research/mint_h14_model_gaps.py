#!/usr/bin/env python3
"""Mint the H14 model-gap families: Qwen matvec dims, rms_norm decomposition
forms, Parakeet island B select, and island A/C batched matmul.

Four gaps block the models on H14:

* constant-weight matvec past the minted (K, N) grid -- Qwen3.8-2B decode
  wants (K, N) = 2048x5120 (qkv), 2048x3072 (gate/up, z/beta), 3072x2048
  (ffn_down), and the 1536/3072 grid between; plus M=8 above K=1024.
* rms_norm has no native MIL op -- coremltools decomposes it into abs,
  reduce_max, real_div, square, reduce_mean, add, sqrt, mul, and a final
  per-channel mul. This campaign mints exactly those sub-ops at the Qwen
  decode dims the decomposition needs, plus the whole chain once.
* Parakeet island B (``island-select-8head``) is ``select`` with a bool
  cond; the corpus has no select case on either target.
* Parakeet islands A/C are batched runtime-runtime matmuls over a leading
  head batch (rank-3 ``[8, M, K]`` and rank-4 ``[1, 8, M, K]``); the
  envelope only decoded rank-3 batch 1.

Records land under ``research/oracles/{h13,h14}/`` with a ``ga`` prefix
(model gap) so they never collide with the earlier campaigns:

    python3 research/mint_h14_model_gaps.py --host macstudio
    python3 research/mint_h14_model_gaps.py --local --list
"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("mint_oracles.py", "h13_td.py", "mint_h14_model_gaps.py")
DEFAULT_TOOL = om.DEFAULT_TOOL
REMOTE_ROOT = "oracle-mint-scratch/h14-mint/run"


def batched_matmul(rows: int, reduction: int, columns: int, batch: int,
                   heads_form: bool, transpose_y: bool = False) -> dict[str, Any]:
    """A batched runtime-runtime matmul over a leading head batch.

    ``heads_form`` spells the batch as a rank-4 ``[1, batch, M, K]`` head
    axis, the Parakeet island shape; otherwise it is rank-3 ``[B, M, K]``.
    """
    prefix = (1, batch) if heads_form else (batch,)
    x_shape = prefix + (rows, reduction)
    w_shape = prefix + ((columns, reduction) if transpose_y
                        else (reduction, columns))
    out_shape = prefix + (rows, columns)
    body = [
        f'bool tx = const()[name = string("tx"), val = bool(false)];',
        f'bool ty = const()[name = string("ty"), '
        f'val = bool({om.boolean(transpose_y)})];',
    ]
    body.append(f'{om.tensor_type(out_shape)} product = matmul(transpose_x = tx, '
                'transpose_y = ty, x = x, y = w)[name = string("product")];')
    layout = "r4heads" if heads_form else "r3"
    name = (f"gabmm_{layout}_m{rows}_k{reduction}_n{columns}"
            f"_tx0_ty{int(transpose_y)}_b{batch}")
    return om.env_case(name, "env_matmul", {
        "batch": batch, "layout": layout, "rows": rows,
        "reduction": reduction, "columns": columns,
        "transpose_x": False, "transpose_y": transpose_y,
        "x_shape": list(x_shape), "w_shape": list(w_shape),
        "output_shape": list(out_shape), "x_storage": "runtime",
        "w_storage": "runtime",
    }, f"{om.tensor_type(x_shape)} x, {om.tensor_type(w_shape)} w",
        body, "product")


def ninf_halfwords(elements: int) -> bytes:
    return struct.pack("<e", float("-inf")) * elements


def aligned_blob(payload: bytes) -> bytes:
    """weights.bin whose sub-header sits where select's blob reader looks:
    length at 72 and payload offset at 80. `om.blob` packs the length four
    bytes lower, and Apple's select refuses that layout with "Cannot
    retrieve file blob properties" while every other family accepts it."""
    data = bytearray(128 + len(payload))
    struct.pack_into("<II", data, 0, 1, 2)
    struct.pack_into("<I", data, 64, 0xDEADBEEF)
    struct.pack_into("<Q", data, 72, len(payload))
    struct.pack_into("<Q", data, 80, 128)
    data[128:] = payload
    return bytes(data)


def select(shape: tuple[int, ...], mode: str) -> dict[str, Any]:
    """A ``select`` with a bool cond; the corpus has none on either target.

    ``rr`` binds both values runtime; ``ninf`` keeps ``a`` a full-size
    BLOBFILE constant of fp16 -inf, the Parakeet mask form (a rank-0
    BLOBFILE ``a`` was refused on H13, so the fill is materialized).
    """
    kind = om.tensor_type(shape)
    arguments = f"{kind} b, tensor<bool, {om.shape_text(shape)}> cond"
    body = []
    if mode == "rr":
        arguments = f"{kind} a, " + arguments
    else:
        source, elements = om.blobfile("a", shape)
        body.append(source)
        body.append(f'{kind} y = select(a = a, b = b, cond = cond)'
                    '[name = string("y")];')
        name = f"gasel_ninf_{om.dims(shape)}"
        payload = ninf_halfwords(elements)
        return om.case(name, "env_boolean_select", {
            "operation": "select", "shape": shape, "mode": mode,
            "cond_dtype": "bool",
        }, om.program(arguments, body, "y"), aligned_blob(payload), {
            "storage": "BLOBFILE", "shape": list(shape),
            "payload_bytes": elements * 2, "value": "fp16 -inf",
            "blob_layout": "aligned",
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        })
    body.append(f'{kind} y = select(a = a, b = b, cond = cond)'
                '[name = string("y")];')
    return om.env_case(f"gasel_rrb_{om.dims(shape)}", "env_boolean_select", {
        "operation": "select", "shape": shape, "mode": mode,
        "cond_dtype": "bool",
    }, arguments, body, "y")


def rms_norm_reduce(operation: str, shape: tuple[int, ...],
                    keep_dims: bool) -> dict[str, Any]:
    """The reduce_max/reduce_mean half of the rms_norm decomposition at the
    Qwen decode shape, over every non-batch axis with keep_dims as the
    decomposition spells it."""
    axes = tuple(range(1, len(shape)))
    output = tuple(1 if index in axes else value
                   for index, value in enumerate(shape)) if keep_dims else \
        tuple(value for index, value in enumerate(shape) if index not in axes)
    axis_kind = f"tensor<int32, [{len(axes)}]>"
    values = ", ".join(map(str, axes))
    kind, out_kind = om.tensor_type(shape), om.tensor_type(output)
    mil = om.program(f"{kind} x", [
        f'{axis_kind} axes = const()[name = string("axes"), '
        f'val = {axis_kind}([{values}])];',
        f'{out_kind} y = {operation}(x = x, axes = axes, '
        f'keep_dims = {om.boolean(keep_dims)})[name = string("y")];',
    ], "y")
    return om.case(
        f"garms_reduce_{operation}_{om.dims(shape)}_kd{int(keep_dims)}",
        "reduction", {"operation": operation, "shape": shape, "axes": axes,
                      "keep_dims": keep_dims}, mil)


def rms_norm_chain(channels: int, gamma: bool) -> dict[str, Any]:
    """The whole coremltools rms_norm decomposition at one Qwen decode
    shape, spelled the way the torch frontend emits it."""
    shape = (1, channels, 1, 1)
    kind = om.tensor_type(shape)
    arguments = f"{kind} x"
    body = [
        f'{kind} abs0 = abs(x = x)[name = string("abs0")];',
        'tensor<int32, [3]> axes = const()[name = string("axes"), '
        'val = tensor<int32, [3]>([1, 2, 3])];',
        f'{om.tensor_type((1, 1, 1, 1))} amax = reduce_max(x = abs0, '
        'axes = axes, keep_dims = bool(true))[name = string("amax")];',
        f'{kind} scaled = real_div(x = x, y = amax)[name = string("scaled")];',
        f'{kind} sq = square(x = scaled)[name = string("sq")];',
        f'{om.tensor_type((1, 1, 1, 1))} mean = reduce_mean(x = sq, '
        'axes = axes, keep_dims = bool(true))[name = string("mean")];',
        'fp16 eps = const()[name = string("eps"), val = fp16(0x1p-17)];',
        f'{om.tensor_type((1, 1, 1, 1))} meps = add(x = mean, '
        'y = eps)[name = string("meps")];',
        f'{om.tensor_type((1, 1, 1, 1))} rms = sqrt(x = meps)'
        '[name = string("rms")];',
        f'{om.tensor_type((1, 1, 1, 1))} rscaled = mul(x = rms, '
        'y = amax)[name = string("rscaled")];',
        f'{kind} norm = real_div(x = x, y = rscaled)[name = string("norm")];',
    ]
    result = "norm"
    if gamma:
        source, elements = om.blobfile("gamma", shape)
        body.insert(0, source)
        body.append(f'{kind} y = mul(x = norm, y = gamma)'
                    '[name = string("y")];')
        result = "y"
        name = f"garms_chain_c{channels}_gamma"
        weights, description = elements, {
            "storage": "BLOBFILE", "shape": list(shape),
            "payload_bytes": elements * 2, "value": "fp16(0x1p-1)",
        }
    else:
        name = f"garms_chain_c{channels}"
        weights, description = None, {"storage": "none"}
    return om.case(name, "rms_norm_chain", {
        "operation": "rms_norm_decomposed", "shape": shape, "gamma": gamma,
    }, om.program(arguments, body, result), weights, description)


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Qwen3.8-2B decode matvecs past the minted grid: hidden 2048, qkv 5120,
    # gate/up and z/beta 3072, ffn_down 3072->2048, and the 1536 grid around
    # them. M=8 above K=1024 has no decoded point either.
    for reduction, columns in ((1536, 1536), (1536, 2048), (2048, 1536),
                              (1536, 3072), (3072, 1536), (2048, 3072),
                              (3072, 2048), (3072, 3072), (2048, 5120)):
        cases.append(om.matmul(reduction, columns, 1, True))
    cases.append(om.matmul(2048, 2048, 8, True))
    # rms_norm decomposition forms at the decode shape.
    for operation in ("reduce_max", "reduce_mean"):
        cases.append(rms_norm_reduce(operation, (1, 2048, 1, 1), True))
    cases.append(om.env_broadcast("real_div", (1, 2048, 1, 1), (1, 1, 1, 1)))
    cases.append(om.env_broadcast("mul", (1, 2048, 1, 1), (1, 1, 1, 1)))
    cases.append(om.unary("sqrt", 1))
    cases.append(om.binary_runtime("add", (1, 1, 1, 1)))
    cases.append(om.binary_constant("mul", 2048, "blob"))
    cases.append(rms_norm_chain(2048, False))
    cases.append(rms_norm_chain(2048, True))
    # Island B select forms.
    cases.append(select((1, 64, 1, 1), "rr"))
    cases.append(select((1, 8, 375, 375), "rr"))
    cases.append(select((1, 8, 375, 375), "ninf"))
    cases.append(select((1, 1024, 375), "rr"))
    # Islands A/C: batched runtime-runtime matmuls at the Parakeet geometry.
    cases.append(batched_matmul(375, 128, 375, 8, False))
    cases.append(batched_matmul(375, 128, 749, 8, False))
    cases.append(batched_matmul(375, 375, 128, 8, False))
    cases.append(batched_matmul(375, 128, 375, 8, True))
    cases.append(batched_matmul(375, 128, 749, 8, True))
    cases.append(batched_matmul(375, 375, 128, 8, True))
    for batch in (2, 4, 16):
        cases.append(batched_matmul(375, 128, 749, batch, False))
    cases.append(batched_matmul(375, 128, 749, 8, False, transpose_y=True))
    return cases


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         f"mkdir -p ~/{REMOTE_ROOT} && echo ~/{REMOTE_ROOT}"],
        capture_output=True, text=True, check=True)
    remote_root = root.stdout.strip().splitlines()[-1]
    script = Path(__file__).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name in SCRIPTS:
            subprocess.run(["scp", "-q", str(script.parent / name),
                            f"{args.host}:{remote_root}/"], check=True)
        command = [
            "python3", f"{remote_root}/{script.name}", "--local",
            "--oracle-tool", args.oracle_tool, "--output",
            f"{remote_root}/oracles", "--source-commit", args.source_commit,
            "--targets", *args.targets,
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
            with tempfile.TemporaryDirectory(
                    prefix="mil-hwx-model-gap-json-") as staging:
                subprocess.run(["scp", "-q", "-r",
                                f"{args.host}:{remote_root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(remote_root)}"], check=False)


def local_run(args: argparse.Namespace) -> int:
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
    tool = Path(args.oracle_tool)
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    if not os.access(tool, os.X_OK):
        raise SystemExit(f"oracle tool is not executable: {tool}")
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
            status = om.run_case(item, target, output, tool, args.source_commit)
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} "
          f"decoded={decoded} rejected={rejected}")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host", help="SSH Mac that runs Apple's compiler")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=sorted(om.SUBTYPES))
    parser.add_argument("--oracle-tool", default=DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--case", help="shell pattern selecting case names")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
