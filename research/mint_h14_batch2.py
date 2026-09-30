#!/usr/bin/env python3
"""Mint the H14 Qwen-batch2 families F1+F3+F4+F5.

F1 — broadcast [1,16,128,128] × [16,1,1] (B-state decay; 2 cases mul+add).
F3 — rank-4 b16 batched matmul rows=1 and rows=128 outer-product (2 cases).
F4 — softmax attention pipeline at [16,50] (mul+add+softmax chain; 1 case).
F5 — 3-op chain sigmoid→mul→add at [1,16,128,1] (C-readout chain; 1 case).
F2 helpers — rank-3 reduce_sum/mean/max at [1,16,128] axes=[1] and the
     rank-4 twin axes=[2] (Apple accepts both).

Apple-side H14 compiler at /private/tmp/ane-compile-hwx on macstudio.
"""
from __future__ import annotations
import argparse
import fnmatch
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("mint_oracles.py", "h13_td.py", "mint_h14_batch2.py")
DEFAULT_TOOL = "/private/tmp/ane-compile-hwx"
REMOTE_ROOT = "oracle-mint-scratch/h14-mint/batch2"


# --- F1: broadcast [1,16,128,128] × [16,1,1] ---------------------------------

def f1_broadcast(operation: str) -> dict[str, Any]:
    """State-decay broadcast mul/add at head-major [1, 16, 128, 128] ×
    [16, 1, 1]. Same Apple decoder family as the existing per-head
    channel binaries at (1, 16, 1, 1) × (1, 16, 1, 1) (already in the
    elementwise envelope from H14Shapes); this is the same operation with
    a runtime-shape target to verify Apple accepts the head-major layout.
    """
    return om.env_broadcast(operation, (1, 16, 128, 128), (16, 1, 1))


# --- F3: rank-4 b16 batched matmul ---------------------------------------------

def f3_b16_r4_matmul(rows: int, reduction: int, columns: int,
                     transpose_y: bool, transpose_x: bool = False,
                     name_suffix: str = "") -> dict[str, Any]:
    """Qwen state-block matmul: [1, 16, rows, reduction] × [1, 16, reduction,
    columns] rank-4 b16, runtime × runtime. Parakeet decoded rows=375;
    this batch adds rows=1 (state-decayed k @ state), rows=128 outer
    product (k^T @ delta outer product, with transpose_x=true gives
    rows=128 outer-product form).
    """
    return om.env_matmul(rows, reduction, columns, batch=16,
                         transpose_x=transpose_x, transpose_y=transpose_y,
                         x_storage="runtime", w_storage="runtime")


# --- F4: softmax attention pipeline at [16,50] --------------------------------

def f4_attn_pipeline() -> dict[str, Any]:
    """The Qwen gated-attention chain at [16, 50]:
        mul(scores, 1/sqrt(128) = 0x3958) → add(scores, mask) → softmax(-1)
    The bare softmax [16,50] already compiled at the merged build; this
    chains the upstream mul+add to it. The mask broadcast is at
    (1, 1, 1, 50) → (16, 1, 1, 50); the scale mul is (16, 50) × (16, 50).
    """
    arguments = (f"{om.tensor_type((1, 16, 50, 1))} x, "
                 f"{om.tensor_type((1, 16, 50, 1))} scale, "
                 f"{om.tensor_type((1, 1, 50, 1))} mask")
    body = [
        f'{om.tensor_type((1, 16, 50, 1))} scaled = mul(x = x, y = scale)'
        f'[name = string("scaled")];',
        f'{om.tensor_type((1, 16, 50, 1))} masked = add(x = scaled, y = mask)'
        f'[name = string("masked")];',
        f'{om.tensor_type((1, 16, 50, 1))} y = softmax(axis = 3, x = masked)'
        f'[name = string("y")];',
    ]
    return om.env_case("env_chain_softmax_attn_h16_s50",
                       "env_chain",
                       {"shape": (1, 16, 50, 1), "axis": 3, "upstream": "mul+add"},
                       arguments, body, "y")


# --- F5: 3-op chain sigmoid → mul → add at [16, 128] -------------------------

def f5_c_readout_chain() -> dict[str, Any]:
    """The C-readout chained tail: sigmoid(gate) → mul(o, gate) → add(h_residual).
    At [1, 16, 128, 1] rank-4 channel-flat. The chain encoder currently
    supports only one producer + relu, so the 3-op form needs a separate
    Apple decode.
    """
    arguments = (f"{om.tensor_type((1, 16, 128, 1))} o, "
                 f"{om.tensor_type((1, 16, 128, 1))} gate, "
                 f"{om.tensor_type((1, 16, 128, 1))} h_residual")
    body = [
        f'{om.tensor_type((1, 16, 128, 1))} sg = sigmoid(x = gate)'
        f'[name = string("sg")];',
        f'{om.tensor_type((1, 16, 128, 1))} gated = mul(x = o, y = sg)'
        f'[name = string("gated")];',
        f'{om.tensor_type((1, 16, 128, 1))} h = add(x = gated, y = h_residual)'
        f'[name = string("h")];',
    ]
    return om.env_case("env_chain_creadout_h16_s128",
                       "env_chain",
                       {"shape": (1, 16, 128, 1), "ops": "sigmoid+mul+add"},
                       arguments, body, "h")


# --- F2 helpers: rank-3 reduce_sum + rank-4 twin axis -------------------------

def f2_reduce_rank3(operation: str, channels: int) -> dict[str, Any]:
    """Qwen per-head reduce at rank-3 form: [1, 16, channels] axes=[1] keep_dims.
    Apple accepts this; the rank-4 twin [1, 16, channels, 1] axes=[2,3] is
    the Apple-rejected shape we rewrite around. The rank-4 axes=[2]
    sibling is also accepted; both rank-3 and rank-4 axes=[2] are
    pre-existing oracles (gxreduce_*).
    """
    shape = (1, 16, channels)
    mil = om.program(
        f"{om.tensor_type(shape)} x",
        [f'{om.tensor_type(shape)} y = {operation}(axes = [1], keep_dims = true, x = x)'
         f'[name = string("y")];'],
        "y",
    )
    name = f"f2_{operation}_1x16x{channels}_ax1_kd1"
    return om.case(name, "f2_rank3_reduce", {
        "operation": operation, "shape": list(shape), "axes": [1],
        "keep_dims": True,
    }, mil)


def f2_reduce_rank4_twin(operation: str, channels: int) -> dict[str, Any]:
    """Qwen per-head reduce at rank-4 Apple-accepted form:
    [1, 16, channels, 1] axes=[2] keep_dims (Apple accepts; axes=[2,3] is
    rejected).
    """
    shape = (1, 16, channels, 1)
    mil = om.program(
        f"{om.tensor_type(shape)} x",
        [f'{om.tensor_type(shape)} y = {operation}(axes = [2], keep_dims = true, x = x)'
         f'[name = string("y")];'],
        "y",
    )
    name = f"f2_{operation}_1x16x{channels}x1_ax2_kd1"
    return om.case(name, "f2_rank4_reduce_twin", {
        "operation": operation, "shape": list(shape), "axes": [2],
        "keep_dims": True,
    }, mil)


# --- Campaign -----------------------------------------------------------------

def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # F1: broadcast [1,16,128,128] × [16,1,1] for state decay (mul + add).
    for op in ("mul", "add"):
        cases.append(f1_broadcast(op))

    # F3: rank-4 b16 batched matmul, three programs.
    cases.append(f3_b16_r4_matmul(1, 128, 128, transpose_y=False))
    cases.append(f3_b16_r4_matmul(128, 1, 128, transpose_y=True, transpose_x=True))

    # F4: softmax attention pipeline at [16, 50].
    cases.append(f4_attn_pipeline())

    # F5: 3-op chain sigmoid → mul → add at [1, 16, 128, 1].
    cases.append(f5_c_readout_chain())

    # F2 helpers — Apple-decoded probes for the rank-3 form.
    for op in ("reduce_sum", "reduce_mean", "reduce_max"):
        cases.append(f2_reduce_rank3(op, 128))
        cases.append(f2_reduce_rank4_twin(op, 128))

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
            "--targets", "h14",
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
                    prefix="mil-hwx-batch2-json-") as staging:
                subprocess.run(["scp", "-q", "-r",
                                f"{args.host}:{remote_root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        if not args.keep_remote:
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
    if not os.access(tool, os.EX_OK):
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
    parser.add_argument("--targets", nargs="+", default=["h14"])
    parser.add_argument("--oracle-tool", default=DEFAULT_TOOL)
    parser.add_argument("--output", default=str(ROOT / "research" / "oracles" / "batch2"))
    parser.add_argument("--case", help="shell pattern selecting case names")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    parser.add_argument("--keep-remote", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
