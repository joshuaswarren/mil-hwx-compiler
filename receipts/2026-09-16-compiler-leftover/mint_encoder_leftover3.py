#!/usr/bin/env python3
"""Round 3: controls that separate conv/broadcast spelling from geometry.

Round 2's refusals cluster on the encoder's rectangular conv surfaces and on
per-channel constants aligned to the middle axis. This round mints:

1. square controls through the same builder (a known-decoding grid geometry),
   so a refusal is attributable to the geometry, not the spelling;
2. per-channel broadcasts aligned to the H axis (``[1,1024,375] * [1,1024,1]``);
3. the depthwise k9 and grouped padconv respells the planner can express,
   including H-major spellings;
4. a square depthwise control.

    python3 receipts/2026-09-16-compiler-leftover/mint_encoder_leftover3.py --host macstudio
"""
from __future__ import annotations

import argparse
import fnmatch
import json
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
import mint_encoder_leftover2 as base  # noqa: E402

SOURCE_COMMIT = base.SOURCE_COMMIT
DEFAULT_TOOL = base.DEFAULT_TOOL


def campaign() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    # 1. Square controls through the same builder.
    cases.append(base.encoder_conv((1, 4, 8, 8), (8, 4, 1, 1), (1, 8, 8, 8),
                                   (1, 1), 1, "valid", bias=False))
    cases.append(base.encoder_conv((1, 1, 8, 8), (4, 1, 3, 3), (1, 4, 4, 4),
                                   (2, 2), 1, "same", bias=True))
    cases.append(base.encoder_conv((1, 8, 8, 8), (8, 1, 1, 1), (1, 8, 8, 8),
                                   (1, 1), 8, "valid", bias=False))
    cases.append(base.encoder_conv((1, 16, 8, 8), (16, 1, 9, 9),
                                   (1, 16, 8, 8), (1, 1), 16, "same",
                                   bias=False))
    # 2. Per-channel broadcasts aligned to the H axis.
    cases.append(base.encoder_broadcast("mul", (1, 1024, 375), (1, 1024, 1),
                                        "blob"))
    cases.append(base.encoder_broadcast("add", (1, 1024, 375), (1, 1024, 1),
                                        "blob"))
    # 3. The depthwise and pointwise respells the planner can express.
    cases.append(base.encoder_conv((1, 1024, 375, 1), (2048, 1024, 1, 1),
                                   (1, 2048, 375, 1), (1, 1), 1, "valid",
                                   bias=False))
    cases.append(base.encoder_conv((1, 1024, 375, 1), (1024, 1, 9, 9),
                                   (1, 1024, 375, 1), (1, 1), 1024, "same",
                                   bias=False))
    cases.append(base.encoder_conv((1, 1024, 375, 1), (1024, 1, 1, 9),
                                   (1, 1024, 375, 1), (1, 1), 1024, "same",
                                   bias=False))
    unique: dict[str, dict[str, Any]] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


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
    if not tool.is_file():
        raise SystemExit(f"oracle tool is missing: {tool}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    decoded = rejected = 0
    for item in selected:
        destination = output / "h13" / f"{item['name']}.json"
        if destination.exists() and not args.force:
            record = json.loads(destination.read_text())
            status = "decoded" if record.get("error") is None else "rejected"
        else:
            status, record = base.run_case_retain(
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
         "mktemp -d /tmp/mil-hwx-encoder-r3.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-encoder-r3."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    research = script.parents[2] / "research"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script),
             str(script.with_name("mint_encoder_leftover2.py")),
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
