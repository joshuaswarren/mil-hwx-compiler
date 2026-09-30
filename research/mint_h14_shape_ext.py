#!/usr/bin/env python3
"""Mint the H14 shape-extension families for blockers 1, 2, 3, 5.

Four gap families block the 38 Qwen programs on H14:

* elementwise unary at (2048,1,1), (4096,1,1), (6144,1,1) for the ops the
  decoder currently rejects (sigmoid, silu, sqrt, rsqrt, relu, tanh, exp,
  leaky_relu) -- the z-gate (sigmoid), SwiGLU (silu), and rms_norm (sqrt)
  lane in particular.
* elementwise binary + broadcast at (16,1,1) for the per-head softplus /
  decay gates, and the C×C broadcast at C in {16, 4096, 6144} for the
  residual / elementwise chain.
* norm frontend shapes the rms_norm pow-form, l2-norm pow-form, and
  attention softmax call: reduce_sum/mean/max at (1,2048,1,1) axis=2 and
  (1,16,128,1) axis=2 / 2_3; softmax at (16,50) axis=-1 / 1 and at
  (1,16,50,1) axis=3.
* matvec decoder points: (2048,6144), (6144,2048), (2048,2048),
  (6144,6144), (375,1024), (1024,1024) M=1 transpose_y=true.

Records land under research/oracles/{h13,h14}/ with a gx prefix (shape
extension) so they never collide with the earlier campaigns:

    python3 research/mint_h14_shape_ext.py --host macstudio
    python3 research/mint_h14_shape_ext.py --local --list
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
SCRIPTS = ("mint_oracles.py", "h13_td.py", "mint_h14_shape_ext.py")
DEFAULT_TOOL = om.DEFAULT_TOOL
REMOTE_ROOT = "oracle-mint-scratch/h14-mint/shapes"


# --- Unary --------------------------------------------------------------------

def unary_oracle(operation: str, channels: int) -> dict[str, Any]:
    """A unary op at a Qwen channel-flat shape (1, C, 1, 1)."""
    return om.unary(operation, channels)


# --- Binary -------------------------------------------------------------------

def binary_runtime_oracle(operation: str, shape: tuple[int, ...]) -> dict[str, Any]:
    return om.binary_runtime(operation, shape)


def broadcast_oracle(operation: str, x_shape: tuple[int, ...],
                     z_shape: tuple[int, ...]) -> dict[str, Any]:
    """A runtime broadcast where (1, C, 1, 1) × (1, C, 1, 1) is the typical
    Qwen residual / SwiGLU form, also covering (1, C, 1, 1) × (1, 1, 1, 1)."""
    return om.env_broadcast(operation, x_shape, z_shape)


# --- Norm ---------------------------------------------------------------------

def softmax_oracle(shape: tuple[int, ...], axis: int) -> dict[str, Any]:
    """softmax at (16,50) or (1,16,50,1), the Qwen attention decode shape."""
    return om.normalization("softmax", shape)


def reduce_oracle(operation: str, shape: tuple[int, ...],
                  axes: tuple[int, ...], keep_dims: bool) -> dict[str, Any]:
    """reduce_sum/mean/max at the shape the rms_norm / l2_norm pow form needs."""
    axis_kind = f"tensor<int32, [{len(axes)}]>"
    values = ", ".join(map(str, axes))
    kind = om.tensor_type(shape)
    out = tuple(1 if i in axes else v for i, v in enumerate(shape)) \
        if keep_dims else tuple(v for i, v in enumerate(shape) if i not in axes)
    out_kind = om.tensor_type(out)
    mil = om.program(f"{kind} x", [
        f'{axis_kind} axes = const()[name = string("axes"), '
        f'val = {axis_kind}([{values}])];',
        f'{out_kind} y = {operation}(x = x, axes = axes, '
        f'keep_dims = {om.boolean(keep_dims)})[name = string("y")];',
    ], "y")
    name = f"gxreduce_{operation}_{om.dims(shape)}_ax{'_'.join(map(str,axes))}_kd{int(keep_dims)}"
    return om.case(name, "reduction", {"operation": operation, "shape": shape,
                                       "axes": axes, "keep_dims": keep_dims}, mil)


# --- Matvec -------------------------------------------------------------------

def matvec_oracle(reduction: int, columns: int) -> dict[str, Any]:
    """M=1 matvec with transpose_y=true (Apple's only accepted orientation)."""
    return om.matmul(reduction, columns, 1, True)


# --- Campaign -----------------------------------------------------------------

def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # Blocker 1: unary at (2048,1,1), (4096,1,1), (6144,1,1).
    unary_ops = ("sigmoid", "silu", "sqrt", "rsqrt", "relu", "tanh", "exp",
                 "leaky_relu", "abs")
    for op in unary_ops:
        for channels in (2048, 4096, 6144):
            cases.append(unary_oracle(op, channels))

    # Blocker 2: binary at (16,1,1) and (1,16,1,1).
    for op in ("add", "mul", "sub", "maximum", "minimum", "real_div"):
        cases.append(binary_runtime_oracle(op, (1, 16, 1, 1)))

    # Blocker 2: C×C broadcast at C ∈ {16, 4096, 6144}.
    for op in ("add", "mul", "sub", "maximum", "minimum", "real_div"):
        for channels in (16, 4096, 6144):
            shape = (1, channels, 1, 1)
            cases.append(broadcast_oracle(op, shape, shape))

    # Blocker 2: (1, 2048, 1, 1) × (1, 2048, 1, 1) and (1, 6144, 1, 1) × (1, 6144, 1, 1).
    for op in ("add", "mul", "sub", "maximum", "minimum", "real_div"):
        for channels in (2048, 6144):
            shape = (1, channels, 1, 1)
            cases.append(broadcast_oracle(op, shape, shape))

    # Blocker 3: softmax at (16,50) axis=-1 / 1 / 2 (rank-2), (1,16,50,1) axis=3 (rank-4).
    for shape, axis in (((16, 50), -1), ((16, 50), 1), ((1, 16, 50, 1), 3),
                        ((1, 16, 50, 1), 2), ((1, 16, 1, 50), 3),
                        ((1, 1, 50, 1), 3)):
        cases.append(softmax_oracle(shape, axis))

    # Blocker 3: reduce at (1,2048,1,1) axis=2 (the rms_norm pow-form lane),
    # and at (1,16,128,1) axis=2 / 3 / (2,3) (l2-norm pow-form, head axis).
    for shape, axes, kd in (
            ((1, 2048, 1, 1), (2,), True),
            ((1, 2048, 1, 1), (3,), True),
            ((1, 2048, 1, 1), (2, 3), True),
            ((1, 16, 128, 1), (2,), True),
            ((1, 16, 128, 1), (3,), True),
            ((1, 16, 128, 1), (2, 3), True),
            ((1, 16, 1, 128), (2,), True),
            ((1, 16, 1, 128), (3,), True)):
        for op in ("reduce_sum", "reduce_mean", "reduce_max"):
            cases.append(reduce_oracle(op, shape, axes, kd))

    # Blocker 5: matvec points.
    for reduction, columns in ((2048, 6144), (6144, 2048), (2048, 2048),
                              (6144, 6144), (375, 1024), (1024, 1024),
                              (1024, 375), (4096, 4096), (1024, 4096),
                              (4096, 1024), (1024, 2048), (2048, 1024),
                              (1024, 6144), (6144, 1024)):
        cases.append(matvec_oracle(reduction, columns))

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
                    prefix="mil-hwx-shape-ext-json-") as staging:
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
