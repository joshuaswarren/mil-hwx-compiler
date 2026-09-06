#!/usr/bin/env python3
"""Mint the H13 softmax, layer_norm, and reduction probe grid against H14, and
emit the H14 encoder template table from what decoded.

The case grid is `research/mint_norm_probes.py`'s, imported rather than
restated: the flat [1, C, 1, 1] channel sweep, the sequence and attention-score
geometries, the spatial shapes, layer_norm with and without gamma/beta, and the
channel-versus-spatial reductions including the `keep_dims = false` forms.
Cases are renamed with the `h14norm_` prefix so they never collide with the H13
grid in a shared oracle directory.

No Apple HWX bytes are retained; this reuses ``mint_oracles.run_case``, which
records the same decoded words, descriptors, and section hashes as the shipped
campaign.

The ``--emit-templates`` mode reads the decoded JSON back and writes
``plugins/H14/H14NormTemplates.inc``: the exact task stream Apple emitted for
each covered (operation, surface, axes, keep_dims), with no HWX container
bytes.
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
import tempfile
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402
import mint_norm_probes as h13probes  # noqa: E402
from generate_h14_templates import task_stream, word_rows  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONSTANT_ALIGNMENT = 0x40
PREFIX = "h14norm_"
REMOTE_TEMPLATE = "/tmp/mil-hwx-h14-norm-probes.XXXXXX"
SCRIPTS = ("mint_oracles.py", "h13_td.py", "mint_norm_probes.py",
           "generate_h14_templates.py")


def rename(case: dict[str, Any]) -> dict[str, Any]:
    """The H13 probe name under this campaign's own prefix, with the H13
    `norm_`/`reduce_probe_` prefixes replaced so the operation still leads."""
    name = case["name"]
    for prefix, replacement in (("norm_", ""), ("reduce_probe_", "reduce_")):
        if name.startswith(prefix):
            return {**case, "name": PREFIX + replacement + name[len(prefix):]}
    raise AssertionError(f"probe {name} has no known H13 prefix")


def campaign() -> list[dict[str, Any]]:
    cases = [rename(case) for case in h13probes.campaign()]
    names = [case["name"] for case in cases]
    if len(names) != len(set(names)):
        raise AssertionError("H14 norm probe campaign has duplicate case names")
    return cases


def local_run(args: argparse.Namespace) -> int:
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    tool = Path(args.oracle_tool)
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)} targets={len(args.targets)}")
        return 0
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
            status = om.run_case(item, target, output, tool, args.source_commit)
            decoded += status == "decoded"
            rejected += status == "rejected"
            print(f"{target} {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected) * len(args.targets)} "
          f"decoded={decoded} rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host, f"mktemp -d {REMOTE_TEMPLATE}"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith(REMOTE_TEMPLATE[:-6]):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script),
             *(str(script.with_name(name)) for name in SCRIPTS),
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
            with tempfile.TemporaryDirectory(prefix="mil-hwx-h14-norm-json-") as staging:
                subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


def agree(previous: dict[str, Any], record: dict[str, Any]) -> str | None:
    """The first decoded field two same-key records disagree on. Program
    descriptors are compared on their shared keys: the first H14 campaign
    predates the recorder's `task_section` field, so the shipped softmax and
    layer_norm records carry one key fewer than these probes."""
    for field in ("task_descriptors", "constant_section", "tensor_descriptors"):
        if previous[field] != record[field]:
            return field
    left, right = previous["program_descriptor"], record["program_descriptor"]
    for key in set(left) & set(right):
        if left[key] != right[key]:
            return f"program_descriptor.{key}"
    return None


def selected(target: str) -> list[dict[str, Any]]:
    """One record per (operation, surface, axes, keep_dims) the encoder keys on,
    with every duplicate proven to carry identical Apple output."""
    records: dict[tuple, dict[str, Any]] = {}
    for path in sorted((ROOT / "research/oracles" / target).glob("*.json")):
        record = json.loads(path.read_text())
        if not h13probes.covered(record):
            continue
        key = h13probes.template_key(record)
        previous = records.setdefault(key, record)
        if previous is not record:
            difference = agree(previous, record)
            if difference is not None:
                raise SystemExit(
                    f"{record['case']} and {previous['case']} share a template "
                    f"key but differ in {difference}")
    return [records[key] for key in sorted(records)]


def emit(records: list[dict[str, Any]], out) -> None:
    print("// Generated from decoded H14 softmax, layer_norm, and reduction "
          "oracle task words", file=out)
    print("// by research/mint_h14_norm_probes.py --emit-templates. No HWX "
          "container bytes;", file=out)
    print("// regenerate after re-minting the oracles.", file=out)
    tables = h13probes.kernel_tables()
    symbols: dict[tuple[int, ...], str] = {}
    rows = []
    for record in records:
        words = tuple(task_stream(record))
        descriptor = record["program_descriptor"]
        constant = int(descriptor["constant_address"], 16) - \
            int(descriptor["text_address"], 16)
        if constant != h13probes.align(len(words) * 4, CONSTANT_ALIGNMENT):
            raise SystemExit(
                f"{record['case']} constant offset is not stream-aligned")
        symbol = symbols.get(words)
        if symbol is None:
            symbol = f"kH14NormText{len(symbols)}"
            symbols[words] = symbol
            print(f"static constexpr std::uint32_t {symbol}[] = {{", file=out)
            print(word_rows(list(words)), file=out)
            print("};", file=out)
        operation, input_shape, axes, kept = h13probes.template_key(record)
        output_shape = h13probes.canonical_shape(h13probes.result_shape(record))
        trailer = descriptor["trailing_words"]
        rows.append(
            f"    {{{h13probes.OPERATION_ENUM[operation]}, "
            f"{{{input_shape[0]}, {input_shape[1]}, {input_shape[2]}}}, "
            f"{{{output_shape[0]}, {output_shape[1]}, {output_shape[2]}}}, "
            f"0x{h13probes.axis_mask(axes):02x}, {'true' if kept else 'false'}, "
            f"NormConstants::{h13probes.constant_kind(record, tables)}, "
            f"{symbol}, std::size({symbol}), {descriptor['task_count']}, "
            f"{record['constant_section']['size']}, "
            f"0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}, "
            f"0x{int(trailer[18], 16):08x}}},"
            f"  // {record['case']}")
    print("static constexpr OracleNormTemplate kH14NormTasks[] = {", file=out)
    for row in rows:
        print(row, file=out)
    print("};", file=out)


def emit_templates(args: argparse.Namespace) -> int:
    records = selected(args.targets[0])
    if not records:
        raise SystemExit("no decoded H14 normalization or reduction oracles found")
    path = Path(args.template_output)
    if args.check:
        buffer = tempfile.TemporaryFile("w+")
        emit(records, buffer)
        buffer.seek(0)
        if (path.read_text() if path.exists() else "") != buffer.read():
            raise SystemExit(f"{path} is stale; regenerate it")
        print(f"H14 norm templates: up to date ({path}, {len(records)} entries)")
        return 0
    with path.open("w") as out:
        emit(records, out)
    print(f"{path}: {len(records)} norm/reduce templates")
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    mode.add_argument("--emit-templates", action="store_true")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=["h14"])
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--template-output",
                        default=str(ROOT / "plugins/H14/H14NormTemplates.inc"))
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="with --emit-templates, fail when the file is stale")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.emit_templates:
        raise SystemExit(emit_templates(arguments))
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
