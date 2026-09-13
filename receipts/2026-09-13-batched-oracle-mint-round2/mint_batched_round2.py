#!/usr/bin/env python3
"""Mint H13 batched-matmul gap-closing oracles with Apple's ane-compile-hwx.

Round 2 closes three capture gaps from receipts/2026-09-13-h13-batched-matmul.md:

1. Packed-const y with value=f(index) fp16 bits, raw __const bytes retained.
2. Fold-flag tx=1 / ty=1 variants at B=8 (209-task form) for attn-output.
3. V-projection geometry rows=375, red=128, cols=375, ty=1 and ty=0 control.

Does not retain HWX bytes. Writes decoded JSON plus sidecar .const.bin /
.weights.bin for every constant-y case.
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

SOURCE_COMMIT = "7fd06ac89f63694571795f2f9e26e75b78e58550"
DEFAULT_TOOL = "/tmp/h13-oracle/bin/ane-compile-hwx"
WEIGHT_PATTERN = "fp16_bits((index+1)&0xFFFF)"


def index_fp16_payload(elements: int) -> bytes:
    """Unique-in-window fp16 bit patterns: uint16(index+1), never zero.

    Wraps at 65536. Includes subnormals (0x0001..0x03ff) and is far from
    the uniform 0x3800 payload used in round 1.
    """
    return array.array("H", ((index + 1) & 0xFFFF for index in range(elements))).tobytes()


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
                    offset, length = entry[4], entry[3]
                    return data[offset:offset + length]
                section_cursor += 80
        cursor += size
        if cursor > command_end:
            break
    raise ValueError("HWX has no __TEXT,__const section")


def packing_stats(const: bytes) -> dict[str, Any]:
    words = const[: min(len(const), 8192)]
    count = len(words) // 2
    values = struct.unpack_from(f"<{count}H", words) if count else ()
    unique = len(set(values))
    return {
        "prefix_bytes": len(words),
        "prefix_u16": count,
        "unique_u16_in_prefix": unique,
        "prefix_uniform_0x3800": bool(values) and all(value == 0x3800 for value in values),
        "prefix_all_zero": bool(values) and all(value == 0 for value in values),
        "first_16_u16": [f"0x{value:04x}" for value in values[:16]],
    }


def batched_matmul(rows: int, reduction: int, columns: int, batch: int,
                   transpose_x: bool = False, transpose_y: bool = False,
                   w_storage: str = "runtime",
                   x_shape: tuple[int, ...] | None = None,
                   w_shape: tuple[int, ...] | None = None,
                   out_shape: tuple[int, ...] | None = None,
                   layout: str = "r3",
                   weight_mode: str = "uniform") -> dict[str, Any]:
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
    if weight_mode == "index":
        name += "_idx"
    item = om.env_case(name, "batched_matmul", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "batch": batch, "transpose_x": transpose_x, "transpose_y": transpose_y,
        "x_storage": "runtime", "w_storage": w_storage,
        "x_shape": list(x_shape), "w_shape": list(w_shape),
        "output_shape": list(out_shape), "layout": layout,
        "weight_mode": weight_mode,
        "batch_x_base_elements": rows * reduction,
        "batch_y_base_elements": reduction * columns,
        "batch_out_base_elements": rows * columns,
    }, ", ".join(arguments), body, "product", elements, shapes)
    if weight_mode == "index" and elements:
        payload = index_fp16_payload(elements)
        item["weights"] = om.blob(payload)
        item["weights_description"] = {
            "storage": "BLOBFILE",
            "shapes": [list(shape) for shape in shapes],
            "payload_bytes": elements * 2,
            "value": WEIGHT_PATTERN,
            "pattern": "uint16_le_index_plus_one_wrapping",
            "includes_subnormals": True,
            "zero_free": True,
        }
        item["_input_payload"] = payload
    return item


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    geometries = (
        (375, 128, 749),  # scores
        (375, 375, 128),  # attn-output
    )
    # Gap 1: remint packed-const y with discriminating f(index) weights.
    for rows, reduction, columns in geometries:
        for batch in (2, 4, 8, 16):
            cases.append(batched_matmul(
                rows, reduction, columns, batch, w_storage="blob",
                weight_mode="index"))
    for w_shape, x_shape, out_shape, rows, reduction, columns in (
            ((1, 8, 128, 749), (1, 8, 375, 128), (1, 8, 375, 749), 375, 128, 749),
            ((1, 8, 375, 128), (1, 8, 375, 375), (1, 8, 375, 128), 375, 375, 128)):
        cases.append(batched_matmul(
            rows, reduction, columns, 8, w_storage="blob", layout="r4heads",
            x_shape=x_shape, w_shape=w_shape, out_shape=out_shape,
            weight_mode="index"))
    # Gap 2: fold-flag tx=1 / ty=1 at B=8 for attn-output (scores already minted).
    cases.append(batched_matmul(375, 375, 128, 8, transpose_y=True))
    cases.append(batched_matmul(375, 375, 128, 8, transpose_x=True))
    # Fold-flag const-y (scores) so packing under ty=1 / tx=1 is checkable.
    cases.append(batched_matmul(
        375, 128, 749, 8, transpose_y=True, w_storage="blob", weight_mode="index"))
    cases.append(batched_matmul(
        375, 128, 749, 8, transpose_x=True, w_storage="blob", weight_mode="index"))
    # Gap 3: V-projection 375x128x375, ty=1 and ty=0 control, B=8.
    cases.append(batched_matmul(375, 128, 375, 8))
    cases.append(batched_matmul(375, 128, 375, 8, transpose_y=True))
    cases.append(batched_matmul(
        375, 128, 375, 8, w_storage="blob", weight_mode="index"))
    cases.append(batched_matmul(
        375, 128, 375, 8, transpose_y=True, w_storage="blob", weight_mode="index"))
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


def structure_report(record: dict[str, Any],
                     packing: dict[str, Any] | None = None) -> dict[str, Any]:
    params = record["parameters"]
    batch = params["batch"]
    error = record.get("error")
    if error:
        return {"case": record["case"], "status": "rejected",
                "batch": batch, "error": error[:400],
                "gap": params.get("gap")}
    descriptor = record.get("program_descriptor") or {}
    tensors = record.get("tensor_descriptors") or []
    tasks = record.get("task_descriptors") or []
    shapes = [td.get("shape") for td in tensors]
    strides = [td.get("strides") for td in tensors]
    totals = [td.get("total_bytes") for td in tensors]
    bases = [dma_bases(task) for task in tasks]
    has_batch_dim = any(isinstance(shape, list) and batch in shape
                        for shape in shapes)
    distinct_src = len({item["src_off"] for item in bases})
    constant = record.get("constant_section") or {}
    verdict = "batched"
    if not has_batch_dim and descriptor.get("task_count") == 1:
        verdict = "flattened"
    elif not has_batch_dim and distinct_src <= 1 and batch > 1:
        verdict = "unclear"
    report = {
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
        "resource_addresses": descriptor.get("resource_addresses"),
        "constant_bytes": constant.get("size"),
        "constant_sha256": constant.get("sha256"),
        "constant_nonzero_bytes": constant.get("nonzero_bytes"),
        "const_raw_path": constant.get("raw_path"),
        "w_storage": params["w_storage"],
        "w_shape": params["w_shape"],
        "layout": params.get("layout"),
        "weight_mode": params.get("weight_mode"),
        "tx": params.get("transpose_x"),
        "ty": params.get("transpose_y"),
        "rows": params.get("rows"),
        "reduction": params.get("reduction"),
        "columns": params.get("columns"),
    }
    if packing:
        report["packing"] = packing
    return report


def run_case_retain(item: dict[str, Any], output: Path, tool: Path,
                    source_commit: str) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    destination = output / "h13" / f"{item['name']}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = item.pop("_input_payload", None)
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
    packing = None
    with tempfile.TemporaryDirectory(prefix=f"mil-hwx-h13-{item['name']}-") as root:
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
                record["constant_section"]["raw_sha256"] = hashlib.sha256(const).hexdigest()
                packing = packing_stats(const)
                record["constant_section"]["packing"] = packing
                if payload is not None:
                    weights_path = destination.with_suffix(".weights.bin")
                    weights_path.write_bytes(payload)
                    record["weights"]["raw_path"] = weights_path.name
                    record["weights"]["sha256"] = hashlib.sha256(payload).hexdigest()
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
    return status, record, packing


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
            packing = (record.get("constant_section") or {}).get("packing")
        else:
            status, record, packing = run_case_retain(
                item, output, tool, args.source_commit)
        decoded += status == "decoded"
        rejected += status == "rejected"
        report = structure_report(record, packing)
        reports.append(report)
        print(
            f"h13 {item['name']} {status} verdict={report.get('verdict', 'rejected')} "
            f"tasks={report.get('task_count')} const={report.get('constant_bytes')} "
            f"uniform3800={None if not packing else packing.get('prefix_uniform_0x3800')}",
            flush=True)
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
         "mktemp -d /tmp/mil-hwx-batched-oracles-r2.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-batched-oracles-r2."):
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
            for path in staging.rglob("*"):
                if not path.is_file():
                    continue
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
