#!/usr/bin/env python3
"""Mint H13 pad/tile oracles with Apple's ane-compile-hwx.

Encoder forms from tests/test_h13_layout_cli.py plus the scores-shaped
[1,8,375,749]→[1,8,375,750] row-growth. Goal: materialized vs descriptor.
Rejections are findings.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402
import mint_batched_round2 as r2  # noqa: E402

SOURCE_COMMIT = r2.SOURCE_COMMIT
DEFAULT_TOOL = r2.DEFAULT_TOOL


def shape_name(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape)) if shape else "scalar"


def amt_name(values: tuple[int, ...]) -> str:
    return "".join(map(str, values))


def int32_const(name: str, values: tuple[int, ...]) -> str:
    kind = f"tensor<int32, [{len(values)}]>"
    body = ", ".join(map(str, values))
    return f'{kind} {name} = const()[name = string("{name}"), val = {kind}([{body}])];'


def pad_runtime(shape: tuple[int, ...], amounts: tuple[int, ...],
                result: tuple[int, ...], mode: str = "constant") -> dict[str, Any]:
    xkind = om.tensor_type(shape)
    ykind = om.tensor_type(result)
    name = f"pad_r_{shape_name(shape)}_to_{shape_name(result)}_a{amt_name(amounts)}"
    return om.env_case(name, "pad", {
        "operation": "pad", "shape": list(shape), "result_shape": list(result),
        "amounts": list(amounts), "mode": mode, "x_storage": "runtime",
        "weight_mode": "none",
    }, f"{xkind} x", [
        int32_const("pad_amounts", amounts),
        f'{ykind} p = pad(mode = string("{mode}"), pad = pad_amounts, x = x)[name = string("p")];',
        f'{ykind} y = relu(x = p)[name = string("y")];',
    ], "y")


def pad_const_x(shape: tuple[int, ...], amounts: tuple[int, ...],
                result: tuple[int, ...]) -> dict[str, Any]:
    xkind = om.tensor_type(shape)
    ykind = om.tensor_type(result)
    source, count = om.blobfile("x", shape)
    name = f"pad_b_{shape_name(shape)}_to_{shape_name(result)}_a{amt_name(amounts)}_idx"
    item = om.env_case(name, "pad", {
        "operation": "pad", "shape": list(shape), "result_shape": list(result),
        "amounts": list(amounts), "mode": "constant", "x_storage": "blob",
        "weight_mode": "index",
    }, "", [
        source,
        int32_const("pad_amounts", amounts),
        f'{ykind} p = pad(mode = string("constant"), pad = pad_amounts, x = x)[name = string("p")];',
        f'{ykind} y = relu(x = p)[name = string("y")];',
    ], "y", count, [shape])
    payload = r2.index_fp16_payload(count)
    item["weights"] = om.blob(payload)
    item["weights_description"] = {
        "storage": "BLOBFILE", "shapes": [list(shape)],
        "payload_bytes": count * 2, "value": r2.WEIGHT_PATTERN,
        "pattern": "uint16_le_index_plus_one_wrapping",
    }
    item["_input_payload"] = payload
    return item


def tile_runtime(shape: tuple[int, ...], reps: tuple[int, ...],
                 result: tuple[int, ...]) -> dict[str, Any]:
    xkind = om.tensor_type(shape)
    ykind = om.tensor_type(result)
    name = f"tile_r_{shape_name(shape)}_reps_{amt_name(reps)}"
    return om.env_case(name, "tile", {
        "operation": "tile", "shape": list(shape), "result_shape": list(result),
        "reps": list(reps), "x_storage": "runtime", "weight_mode": "none",
    }, f"{xkind} x", [
        int32_const("reps", reps),
        f'{ykind} t = tile(reps = reps, x = x)[name = string("t")];',
        f'{ykind} y = relu(x = t)[name = string("y")];',
    ], "y")


def tile_const_x(shape: tuple[int, ...], reps: tuple[int, ...],
                 result: tuple[int, ...]) -> dict[str, Any]:
    ykind = om.tensor_type(result)
    source, count = om.blobfile("x", shape)
    name = f"tile_b_{shape_name(shape)}_reps_{amt_name(reps)}_idx"
    item = om.env_case(name, "tile", {
        "operation": "tile", "shape": list(shape), "result_shape": list(result),
        "reps": list(reps), "x_storage": "blob", "weight_mode": "index",
    }, "", [
        source,
        int32_const("reps", reps),
        f'{ykind} t = tile(reps = reps, x = x)[name = string("t")];',
        f'{ykind} y = relu(x = t)[name = string("y")];',
    ], "y", count, [shape])
    payload = r2.index_fp16_payload(count)
    item["weights"] = om.blob(payload)
    item["weights_description"] = {
        "storage": "BLOBFILE", "shapes": [list(shape)],
        "payload_bytes": count * 2, "value": r2.WEIGHT_PATTERN,
        "pattern": "uint16_le_index_plus_one_wrapping",
    }
    item["_input_payload"] = payload
    return item


def campaign() -> list[dict[str, Any]]:
    cases = [
        # Identity controls (our compiler treats these as views).
        pad_runtime((1, 64, 1, 1), (0, 0, 0, 0, 0, 0, 0, 0), (1, 64, 1, 1)),
        tile_runtime((1, 64, 1, 1), (1, 1, 1, 1), (1, 64, 1, 1)),
        # Encoder pad: one-row growth of attention scores.
        pad_runtime((1, 1, 749, 375), (0, 0, 0, 0, 0, 0, 1, 0), (1, 1, 750, 375)),
        pad_const_x((1, 1, 749, 375), (0, 0, 0, 0, 0, 0, 1, 0), (1, 1, 750, 375)),
        # Parent scores-shaped [1,8,375,749]→[1,8,375,750], both last-axis styles.
        pad_runtime((1, 8, 375, 749), (0, 0, 0, 0, 0, 0, 1, 0), (1, 8, 375, 750)),
        pad_runtime((1, 8, 375, 749), (0, 0, 0, 0, 0, 0, 0, 1), (1, 8, 375, 750)),
        pad_const_x((1, 8, 375, 749), (0, 0, 0, 0, 0, 0, 0, 1), (1, 8, 375, 750)),
        # Encoder tile: mask broadcast reps [1,375,1].
        tile_runtime((1, 1, 375), (1, 375, 1), (1, 375, 375)),
        tile_const_x((1, 1, 375), (1, 375, 1), (1, 375, 375)),
        tile_runtime((1, 8, 1, 375), (1, 1, 375, 1), (1, 8, 375, 375)),
    ]
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


def report(record: dict[str, Any], packing: dict[str, Any] | None) -> dict[str, Any]:
    params = record["parameters"]
    error = record.get("error")
    if error:
        return {
            "case": record["case"], "status": "rejected",
            "operation": params.get("operation"),
            "shape": params.get("shape"),
            "result_shape": params.get("result_shape"),
            "amounts": params.get("amounts") or params.get("reps"),
            "error": error[:400],
        }
    descriptor = record.get("program_descriptor") or {}
    tasks = record.get("task_descriptors") or []
    tensors = record.get("tensor_descriptors") or []
    constant = record.get("constant_section") or {}
    shapes = [td.get("shape") for td in tensors]
    totals = [td.get("total_bytes") for td in tensors]
    in_elems = 1
    for dim in params.get("shape") or [1]:
        in_elems *= dim
    out_elems = 1
    for dim in params.get("result_shape") or [1]:
        out_elems *= dim
    # Descriptor trick: output tensor bytes == input bytes (view).
    # Materialized: output bytes grow with the padded/tiled extent.
    verdict = "unclear"
    if totals:
        if any(total == out_elems * 2 and out_elems != in_elems for total in totals if total):
            verdict = "materialized"
        elif all((total or 0) <= in_elems * 2 + 64 for total in totals) and out_elems != in_elems:
            verdict = "descriptor_or_small"
        elif out_elems == in_elems:
            verdict = "identity"
    return {
        "case": record["case"],
        "status": "decoded",
        "verdict": verdict,
        "operation": params.get("operation"),
        "shape": params.get("shape"),
        "result_shape": params.get("result_shape"),
        "amounts": params.get("amounts") or params.get("reps"),
        "task_count": descriptor.get("task_count"),
        "task_sizes": [task.get("size_bytes") for task in tasks],
        "program_count": record.get("program_count"),
        "hwx_bytes": record.get("hwx_bytes"),
        "tensor_shapes": shapes,
        "tensor_total_bytes": totals,
        "constant_bytes": constant.get("size"),
        "constant_sha256": constant.get("sha256"),
        "const_raw_path": constant.get("raw_path"),
        "packing": packing,
    }


def local_run(args: argparse.Namespace) -> int:
    import os
    import platform
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
            packing = (record.get("constant_section") or {}).get("packing")
        else:
            status, record, packing = r2.run_case_retain(
                item, output, tool, args.source_commit)
        decoded += status == "decoded"
        rejected += status == "rejected"
        item_report = report(record, packing)
        reports.append(item_report)
        print(
            f"h13 {item['name']} {status} verdict={item_report.get('verdict', 'rejected')} "
            f"tasks={item_report.get('task_count')} const={item_report.get('constant_bytes')}",
            flush=True)
        if destination.exists():
            shutil.copy2(destination, output / destination.name)
        for sidecar in (
                destination.with_suffix(".const.bin"),
                destination.with_suffix(".weights.bin")):
            if sidecar.exists():
                shutil.copy2(sidecar, output / sidecar.name)
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
         "mktemp -d /tmp/mil-hwx-pad-tile-oracles.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-pad-tile-oracles."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    research = script.parents[2] / "research"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script),
             str(script.with_name("mint_batched_round2.py")),
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
        Path(__file__).resolve().parents[2] / "research/oracles/h13/layout"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local else remote_run(arguments))
