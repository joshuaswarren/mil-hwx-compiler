#!/usr/bin/env python3
"""Mint H13 less/floor/select/floor_div oracles with Apple's ane-compile-hwx.

Mint list from receipts/2026-09-13-h13-registry-boolean-ops.md.
Non-uniform constants except the requested -inf and scalar-2.0 forms.
Apple callback_status != 0 is a finding, not a failure.
"""
from __future__ import annotations

import argparse
import array
import fnmatch
import json
import math
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402
import mint_batched_round2 as r2  # noqa: E402

SOURCE_COMMIT = r2.SOURCE_COMMIT
DEFAULT_TOOL = r2.DEFAULT_TOOL


def mask01_payload(elements: int) -> bytes:
    """fp16 0/1 mask, odd indices 1.0 (0x3c00)."""
    return array.array("H", (0x3C00 if index & 1 else 0 for index in range(elements))).tobytes()


def ninf_payload(elements: int) -> bytes:
    return array.array("H", (0xFC00,) * elements).tobytes()


def shape_name(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape)) if shape else "scalar"


def attach_blob(item: dict[str, Any], payload: bytes, description: dict[str, Any]) -> dict[str, Any]:
    item["weights"] = om.blob(payload)
    item["weights_description"] = description
    item["_input_payload"] = payload
    return item


def op_case(name: str, family: str, parameters: dict[str, Any],
            arguments: str, body: list[str], result: str,
            elements: int | None = None,
            shapes: list[tuple[int, ...]] | None = None) -> dict[str, Any]:
    return om.env_case(name, family, parameters, arguments, body, result,
                       elements, shapes)


def less_runtime(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    name = f"less_rr_{shape_name(shape)}"
    return op_case(name, "less", {
        "operation": "less", "shape": list(shape),
        "x_storage": "runtime", "y_storage": "runtime",
        "weight_mode": "none",
    }, f"{kind} x, {kind} y", [
        f'{kind} z = less(x = x, y = y)[name = string("z")];',
    ], "z")


def less_const_y(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    source, count = om.blobfile("y", shape)
    name = f"less_rb_{shape_name(shape)}_idx"
    item = op_case(name, "less", {
        "operation": "less", "shape": list(shape),
        "x_storage": "runtime", "y_storage": "blob",
        "weight_mode": "index",
    }, f"{kind} x", [
        source,
        f'{kind} z = less(x = x, y = y)[name = string("z")];',
    ], "z", count, [shape])
    return attach_blob(item, r2.index_fp16_payload(count), {
        "storage": "BLOBFILE", "shapes": [list(shape)],
        "payload_bytes": count * 2, "value": r2.WEIGHT_PATTERN,
        "pattern": "uint16_le_index_plus_one_wrapping",
    })


def bool_type(shape: tuple[int, ...]) -> str:
    return f"tensor<bool, {om.shape_text(shape)}>"


def less_bool_runtime(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    out = bool_type(shape)
    name = f"less_bool_rr_{shape_name(shape)}"
    return op_case(name, "less", {
        "operation": "less", "shape": list(shape), "result": "bool",
        "x_storage": "runtime", "y_storage": "runtime", "weight_mode": "none",
    }, f"{kind} x, {kind} y", [
        f'{out} z = less(x = x, y = y)[name = string("z")];',
    ], "z")


def less_bool_cast(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    out = bool_type(shape)
    name = f"less_bool_cast_{shape_name(shape)}"
    return op_case(name, "less", {
        "operation": "less", "shape": list(shape), "result": "bool_cast_fp16",
        "x_storage": "runtime", "y_storage": "runtime", "weight_mode": "none",
    }, f"{kind} x, {kind} y", [
        'string dt = const()[name = string("dt"), val = string("fp16")];',
        f'{out} m = less(x = x, y = y)[name = string("m")];',
        f'{kind} z = cast(dtype = dt, x = m)[name = string("z")];',
    ], "z")



def floor_runtime(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    name = f"floor_r_{shape_name(shape)}"
    return op_case(name, "floor", {
        "operation": "floor", "shape": list(shape),
        "x_storage": "runtime", "weight_mode": "none",
    }, f"{kind} x", [
        f'{kind} y = floor(x = x)[name = string("y")];',
    ], "y")


def floor_const(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    source, count = om.blobfile("x", shape)
    name = f"floor_b_{shape_name(shape)}_idx"
    item = op_case(name, "floor", {
        "operation": "floor", "shape": list(shape),
        "x_storage": "blob", "weight_mode": "index",
    }, "", [
        source,
        f'{kind} y = floor(x = x)[name = string("y")];',
    ], "y", count, [shape])
    return attach_blob(item, r2.index_fp16_payload(count), {
        "storage": "BLOBFILE", "shapes": [list(shape)],
        "payload_bytes": count * 2, "value": r2.WEIGHT_PATTERN,
        "pattern": "uint16_le_index_plus_one_wrapping",
    })


def select_runtime(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    name = f"select_rrr_{shape_name(shape)}"
    return op_case(name, "select", {
        "operation": "select", "shape": list(shape),
        "a_storage": "runtime", "b_storage": "runtime", "cond_storage": "runtime",
        "weight_mode": "none",
    }, f"{kind} a, {kind} b, {kind} cond", [
        f'{kind} y = select(a = a, b = b, cond = cond)[name = string("y")];',
    ], "y")


def select_ninf_a(shape: tuple[int, ...], scalar_a: bool) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    a_shape: tuple[int, ...] = () if scalar_a else shape
    source, count = om.blobfile("a", a_shape)
    tag = "scalar" if scalar_a else "tensor"
    name = f"select_ninf_{tag}_{shape_name(shape)}"
    item = op_case(name, "select", {
        "operation": "select", "shape": list(shape),
        "a_storage": "blob", "a_value": "-inf", "a_rank": len(a_shape),
        "b_storage": "runtime", "cond_storage": "runtime",
        "weight_mode": "neg_inf",
    }, f"{kind} b, {kind} cond", [
        source,
        f'{kind} y = select(a = a, b = b, cond = cond)[name = string("y")];',
    ], "y", count, [a_shape])
    return attach_blob(item, ninf_payload(count), {
        "storage": "BLOBFILE", "shapes": [list(a_shape)],
        "payload_bytes": count * 2, "value": "fp16(-inf)",
        "pattern": "uint16_le_0xfc00",
    })


def select_bool_cond(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    cond = bool_type(shape)
    name = f"select_rrb_{shape_name(shape)}"
    return op_case(name, "select", {
        "operation": "select", "shape": list(shape),
        "a_storage": "runtime", "b_storage": "runtime", "cond_storage": "bool_runtime",
        "weight_mode": "none",
    }, f"{kind} a, {kind} b, {cond} cond", [
        f'{kind} y = select(a = a, b = b, cond = cond)[name = string("y")];',
    ], "y")


def select_ninf_bool_cond(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    cond = bool_type(shape)
    source, count = om.blobfile("a", shape)
    name = f"select_ninf_bool_{shape_name(shape)}"
    item = op_case(name, "select", {
        "operation": "select", "shape": list(shape),
        "a_storage": "blob", "a_value": "-inf",
        "b_storage": "runtime", "cond_storage": "bool_runtime",
        "weight_mode": "neg_inf",
    }, f"{kind} b, {cond} cond", [
        source,
        f'{kind} y = select(a = a, b = b, cond = cond)[name = string("y")];',
    ], "y", count, [shape])
    return attach_blob(item, ninf_payload(count), {
        "storage": "BLOBFILE", "shapes": [list(shape)],
        "payload_bytes": count * 2, "value": "fp16(-inf)",
        "pattern": "uint16_le_0xfc00",
    })


def floor_div_scalar2(shape: tuple[int, ...], composed: bool) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    form = "real_div_then_floor" if composed else "native"
    tag = "comp" if composed else "native"
    name = f"floor_div_{tag}_{shape_name(shape)}_s2"
    body = ['fp16 two = const()[name = string("two"), val = fp16(2.0)];']
    if composed:
        body.extend([
            f'{kind} q = real_div(x = x, y = two)[name = string("q")];',
            f'{kind} y = floor(x = q)[name = string("y")];',
        ])
    else:
        body.append(
            f'{kind} y = floor_div(x = x, y = two)[name = string("y")];')
    return op_case(name, "floor_div", {
        "operation": "floor_div", "shape": list(shape), "form": form,
        "y_storage": "scalar", "y_value": "2.0", "weight_mode": "scalar2",
    }, f"{kind} x", body, "y")


def floor_div_native_const2(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    two = om.tensor_type(shape)
    values = ", ".join(["fp16(2.0)"] * math.prod(shape))
    name = f"floor_div_native_{shape_name(shape)}_c2"
    return op_case(name, "floor_div", {
        "operation": "floor_div", "shape": list(shape),
        "form": "native", "y_storage": "inline", "y_value": "2.0",
        "weight_mode": "const2",
    }, f"{kind} x", [
        f'{two} two = const()[name = string("two"), val = {two}([{values}])];',
        f'{kind} y = floor_div(x = x, y = two)[name = string("y")];',
    ], "y")


def floor_div_native_runtime(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    name = f"floor_div_native_{shape_name(shape)}_rr"
    return op_case(name, "floor_div", {
        "operation": "floor_div", "shape": list(shape),
        "form": "native", "y_storage": "runtime", "weight_mode": "none",
    }, f"{kind} x, {kind} y", [
        f'{kind} z = floor_div(x = x, y = y)[name = string("z")];',
    ], "z")


def floor_div_composed_const2(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    two = om.tensor_type(shape)
    values = ", ".join(["fp16(2.0)"] * math.prod(shape))
    name = f"floor_div_comp_{shape_name(shape)}_c2"
    return op_case(name, "floor_div", {
        "operation": "floor_div", "shape": list(shape),
        "form": "real_div_then_floor", "y_storage": "inline", "y_value": "2.0",
        "weight_mode": "const2",
    }, f"{kind} x", [
        f'{two} two = const()[name = string("two"), val = {two}([{values}])];',
        f'{kind} q = real_div(x = x, y = two)[name = string("q")];',
        f'{kind} y = floor(x = q)[name = string("y")];',
    ], "y")


def floor_div_composed_runtime(shape: tuple[int, ...]) -> dict[str, Any]:
    kind = om.tensor_type(shape)
    name = f"floor_div_comp_{shape_name(shape)}_rr"
    return op_case(name, "floor_div", {
        "operation": "floor_div", "shape": list(shape),
        "form": "real_div_then_floor", "y_storage": "runtime",
        "weight_mode": "none",
    }, f"{kind} x, {kind} y", [
        f'{kind} q = real_div(x = x, y = y)[name = string("q")];',
        f'{kind} z = floor(x = q)[name = string("z")];',
    ], "z")


def campaign() -> list[dict[str, Any]]:
    cases = [
        # less: encoder channel-mask extents, flat, plus generic NCHW.
        less_runtime((375, 1, 1)),
        less_runtime((750, 1, 1)),
        less_runtime((1500, 1, 1)),
        less_runtime((1, 64, 1, 1)),
        less_runtime((1, 512, 1, 1)),
        less_runtime((1, 375)),
        less_const_y((1, 375)),
        less_const_y((375, 1, 1)),
        # floor: [1]-shaped length scalar plus generic.
        floor_runtime((1,)),
        floor_runtime((1, 64, 1, 1)),
        floor_runtime((1, 512, 1, 1)),
        floor_const((1,)),
        # select: -inf a (scalar + tensor) and runtime-a, encoder + generic.
        select_ninf_a((1, 8, 375, 375), scalar_a=True),
        select_ninf_a((1, 8, 375, 375), scalar_a=False),
        select_runtime((1, 8, 375, 375)),
        select_runtime((1, 1024, 375)),
        select_runtime((1, 64, 1, 1)),
        # floor_div: native vs real_div+floor composition.
        floor_div_native_const2((1,)),
        floor_div_native_runtime((1,)),
        floor_div_native_runtime((1, 64, 1, 1)),
        floor_div_composed_const2((1,)),
        floor_div_composed_runtime((1,)),
        floor_div_composed_runtime((1, 64, 1, 1)),

        # Retries after fp16 less/select and inline-tensor 2.0 were callback_status=1.
        less_bool_runtime((375, 1, 1)),
        less_bool_runtime((750, 1, 1)),
        less_bool_runtime((1500, 1, 1)),
        less_bool_runtime((1, 64, 1, 1)),
        less_bool_cast((1, 375)),
        select_bool_cond((1, 8, 375, 375)),
        select_bool_cond((1, 64, 1, 1)),
        select_ninf_bool_cond((1, 8, 375, 375)),
        floor_div_scalar2((1,), composed=False),
        floor_div_scalar2((1,), composed=True),
        floor_div_scalar2((1, 64, 1, 1), composed=False),
        floor_div_scalar2((1, 64, 1, 1), composed=True),

    ]
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


def boolean_report(record: dict[str, Any],
                   packing: dict[str, Any] | None) -> dict[str, Any]:
    params = record["parameters"]
    error = record.get("error")
    if error:
        return {
            "case": record["case"], "status": "rejected",
            "operation": params.get("operation"),
            "shape": params.get("shape"),
            "form": params.get("form"),
            "error": error[:400],
        }
    descriptor = record.get("program_descriptor") or {}
    tasks = record.get("task_descriptors") or []
    constant = record.get("constant_section") or {}
    return {
        "case": record["case"],
        "status": "decoded",
        "operation": params.get("operation"),
        "shape": params.get("shape"),
        "form": params.get("form"),
        "task_count": descriptor.get("task_count"),
        "task_sizes": [task.get("size_bytes") for task in tasks],
        "program_count": record.get("program_count"),
        "hwx_bytes": record.get("hwx_bytes"),
        "tensor_shapes": [td.get("shape") for td in (record.get("tensor_descriptors") or [])],
        "constant_bytes": constant.get("size"),
        "constant_sha256": constant.get("sha256"),
        "const_raw_path": constant.get("raw_path"),
        "weight_mode": params.get("weight_mode"),
        "packing": packing,
    }


def local_run(args: argparse.Namespace) -> int:
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)}")
        return 0
    import os
    import platform
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
        report = boolean_report(record, packing)
        reports.append(report)
        print(
            f"h13 {item['name']} {status} tasks={report.get('task_count')} "
            f"const={report.get('constant_bytes')}",
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
         "mktemp -d /tmp/mil-hwx-boolean-oracles.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-boolean-oracles."):
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
        Path(__file__).resolve().parents[2] / "research/oracles/h13/boolean"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--source-commit", default=SOURCE_COMMIT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local else remote_run(arguments))
