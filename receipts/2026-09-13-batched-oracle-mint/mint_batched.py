#!/usr/bin/env python3
"""Mint H13 batched-matmul oracles with Apple's ane-compile-hwx.

Does not retain HWX bytes. Writes decoded JSON via mint_oracles.run_case.
Batch is kept on BOTH operands, including BLOBFILE y — unlike env_matmul,
which drops the batch axis from constant weights.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402


SOURCE_COMMIT = "12d5f05bae67e239251e5e1afd54344987d03de1"
DEFAULT_TOOL = "/tmp/h13-oracle/bin/ane-compile-hwx"


def batched_matmul(rows: int, reduction: int, columns: int, batch: int,
                   transpose_x: bool = False, transpose_y: bool = False,
                   w_storage: str = "runtime",
                   x_shape: tuple[int, ...] | None = None,
                   w_shape: tuple[int, ...] | None = None,
                   out_shape: tuple[int, ...] | None = None,
                   layout: str = "r3") -> dict[str, Any]:
    """One batched matmul. Constant y keeps the batch axis on the blob."""
    if x_shape is None:
        x_matrix = (reduction, rows) if transpose_x else (rows, reduction)
        w_matrix = (columns, reduction) if transpose_y else (reduction, columns)
        x_shape = (batch,) + x_matrix
        w_shape = (batch,) + w_matrix
        out_shape = (batch, rows, columns)
    assert w_shape is not None and out_shape is not None
    body = [
        f'bool tx = const()[name = string("tx"), val = bool({om.boolean(transpose_x)})];',
        f'bool ty = const()[name = string("ty"), val = bool({om.boolean(transpose_y)})];',
    ]
    arguments, elements, shapes = [], None, []
    for operand, shape, storage in (("x", x_shape, "runtime"),
                                    ("w", w_shape, w_storage)):
        if storage == "runtime":
            arguments.append(f"{om.tensor_type(shape)} {operand}")
        else:
            source, count = om.blobfile(operand, shape)
            body.append(source)
            elements = max(elements or 0, count)
            shapes.append(shape)
    body.append(
        f'{om.tensor_type(out_shape)} product = matmul(transpose_x = tx, '
        'transpose_y = ty, x = x, y = w)[name = string("product")];')
    form = f"{layout}r{w_storage[0]}"
    name = (f"bmm_{form}_m{rows}_k{reduction}_n{columns}"
            f"_tx{int(transpose_x)}_ty{int(transpose_y)}_b{batch}")
    if layout != "r3":
        name += f"_{layout}"
    return om.env_case(name, "batched_matmul", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "batch": batch, "transpose_x": transpose_x, "transpose_y": transpose_y,
        "x_storage": "runtime", "w_storage": w_storage,
        "x_shape": list(x_shape), "w_shape": list(w_shape),
        "output_shape": list(out_shape), "layout": layout,
        "batch_x_base_elements": rows * reduction,
        "batch_y_base_elements": reduction * columns,
        "batch_out_base_elements": rows * columns,
    }, ", ".join(arguments), body, "product", elements, shapes)


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # Encoder attention chains from the survey, ty/tx false.
    geometries = (
        (375, 128, 749),  # scores: [375,128] x [128,749]
        (375, 375, 128),  # attn-output: [375,375] x [375,128]
    )
    for rows, reduction, columns in geometries:
        for batch in (2, 4, 8, 16):
            for w_storage in ("runtime", "blob"):
                cases.append(batched_matmul(
                    rows, reduction, columns, batch, w_storage=w_storage))
    # Fold flags at encoder B=8, scores, runtime-runtime only.
    cases.append(batched_matmul(375, 128, 749, 8, transpose_y=True))
    cases.append(batched_matmul(375, 128, 749, 8, transpose_x=True))
    # Rank-4 encoder forms, B=8.
    for w_storage in ("runtime", "blob"):
        cases.append(batched_matmul(
            375, 128, 749, 8, w_storage=w_storage, layout="r4heads",
            x_shape=(1, 8, 375, 128), w_shape=(1, 8, 128, 749),
            out_shape=(1, 8, 375, 749)))
        cases.append(batched_matmul(
            375, 375, 128, 8, w_storage=w_storage, layout="r4heads",
            x_shape=(1, 8, 375, 375), w_shape=(1, 8, 375, 128),
            out_shape=(1, 8, 375, 128)))
    # Specified mixed layout [1,375,8,128] x [1,8,128,D]; may reject.
    cases.append(batched_matmul(
        8, 128, 749, 8, layout="r4mixed",
        x_shape=(1, 375, 8, 128), w_shape=(1, 8, 128, 749),
        out_shape=(1, 375, 8, 749)))
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


def dma_bases(task: dict[str, Any]) -> dict[str, str | None]:
    src = task.get("blocks", {}).get("0x13800", {}).get("words", {})
    dst = task.get("blocks", {}).get("0x17800", {}).get("words", {})
    return {
        "src_cfg": src.get("0x13800"),
        "src_off": src.get("0x13808"),
        "src_row": src.get("0x1380c"),
        "src_plane": src.get("0x13810"),
        "dst_off": dst.get("0x17804"),
        "dst_row": dst.get("0x17808"),
    }


def structure_report(record: dict[str, Any]) -> dict[str, Any]:
    params = record["parameters"]
    batch = params["batch"]
    error = record.get("error")
    if error:
        return {"case": record["case"], "status": "rejected",
                "batch": batch, "error": error[:400]}
    descriptor = record.get("program_descriptor") or {}
    tensors = record.get("tensor_descriptors") or []
    tasks = record.get("task_descriptors") or []
    shapes = [td.get("shape") for td in tensors]
    strides = [td.get("strides") for td in tensors]
    totals = [td.get("total_bytes") for td in tensors]
    bases = [dma_bases(task) for task in tasks]
    has_batch_dim = any(isinstance(shape, list) and batch in shape
                        for shape in shapes)
    plane = params["rows"] * params["reduction"] * 2
    # Flattened: leading batch missing and one operand plane equals B*rows*red.
    flat_plane = any(total == batch * plane for total in totals if total)
    distinct_src = len({item["src_off"] for item in bases})
    constant = record.get("constant_section") or {}
    verdict = "batched"
    if not has_batch_dim and (flat_plane or descriptor.get("task_count") == 1):
        verdict = "flattened"
    elif not has_batch_dim and distinct_src <= 1 and batch > 1:
        verdict = "unclear"
    return {
        "case": record["case"],
        "status": "decoded",
        "batch": batch,
        "verdict": verdict,
        "task_count": descriptor.get("task_count"),
        "program_count": record.get("program_count"),
        "hwx_bytes": record.get("hwx_bytes"),
        "tensor_shapes": shapes,
        "tensor_strides": strides,
        "tensor_total_bytes": totals,
        "has_batch_dim": has_batch_dim,
        "distinct_src_off": distinct_src,
        "dma_bases": bases,
        "resource_addresses": descriptor.get("resource_addresses"),
        "constant_bytes": constant.get("size"),
        "constant_sha256": constant.get("sha256"),
        "constant_nonzero_bytes": constant.get("nonzero_bytes"),
        "w_storage": params["w_storage"],
        "w_shape": params["w_shape"],
        "layout": params.get("layout"),
    }


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
    reports = []
    decoded = rejected = 0
    for item in selected:
        destination = output / "h13" / f"{item['name']}.json"
        if destination.exists() and not args.force:
            record = json.loads(destination.read_text())
            status = "decoded" if record.get("error") is None else "rejected"
        else:
            status = om.run_case(item, "h13", output, tool, args.source_commit)
            record = json.loads(destination.read_text())
        decoded += status == "decoded"
        rejected += status == "rejected"
        report = structure_report(record)
        reports.append(report)
        print(f"h13 {item['name']} {status} verdict={report.get('verdict', 'rejected')} "
              f"tasks={report.get('task_count')} const={report.get('constant_bytes')}",
              flush=True)
        shutil.copy2(destination, output / destination.name)
    summary = {
        "cases": len(selected), "decoded": decoded, "rejected": rejected,
        "reports": reports,
    }
    (output / "structure_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"SUMMARY cases={len(selected)} decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-batched-oracles.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-batched-oracles."):
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
            for path in staging.rglob("*.json"):
                if path.name == "structure_summary.json":
                    shutil.copy2(path, output / path.name)
                else:
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
        Path(__file__).resolve().parents[2] / "research/oracles/h13/batched"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case", help="fnmatch selecting case names")
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local else remote_run(arguments))
