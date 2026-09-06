#!/usr/bin/env python3
"""Mint H13 matmul oracles with known weight patterns to recover Apple's packing.

The campaign in ``mint_oracles.py`` uses a uniform fp16 0.5 weight, which cannot
distinguish one constant-section permutation from another. These probes give
every weight element a distinct, exactly representable fp16 bit pattern, so the
recorded constant-section hashes (and, for sections no larger than 256 bytes,
the decoded index/value table) pin the packing permutation exactly. No Apple
HWX bytes are retained; this script only reuses ``mint_oracles.run_case``.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mint_oracles as om  # noqa: E402


def index_bits(row: int, column: int, reduction: int) -> int:
    """A distinct small integer per element, exact in fp16 up to 2048."""
    value = row * reduction + column + 1
    if value > 2048:
        raise ValueError("index pattern needs at most 2048 elements")
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def mask_bits(row: int, column: int, _reduction: int) -> int:
    """fp16 in [0.125, 0.25) with both bytes nonzero, so zero halfwords show up
    in the recorded nonzero-byte counts."""
    return 0x3001 | (((row * 31 + column * 7) & 0x01ff) << 1)


PATTERNS = {"index": index_bits, "mask": mask_bits}

# The Apple compiler copies the weight file from byte 0, so the 128-byte blob
# header occupies the first 64 halfwords of the constant section. A "logical"
# halfword index therefore counts from the start of the file, and payload
# halfword ``i`` is logical index ``i + 64``.
HEADER_HALFWORDS = 64
ONE_HOT_BITS = 0x3555


def payload(reduction: int, columns: int, pattern: str) -> bytes:
    data = bytearray(columns * reduction * 2)
    if pattern == "zero":
        return bytes(data)
    if pattern.startswith("onehot"):
        logical = int(pattern[len("onehot"):])
        struct.pack_into("<H", data, (logical - HEADER_HALFWORDS) * 2,
                         ONE_HOT_BITS)
        return bytes(data)
    if pattern.startswith("rowstain") or pattern.startswith("colstain"):
        selector = int(pattern[len("rowstain"):])
        for row in range(columns):
            for column in range(reduction):
                keep = (row == selector if pattern.startswith("rowstain")
                        else column == selector)
                if keep:
                    struct.pack_into("<H", data, (row * reduction + column) * 2,
                                     ONE_HOT_BITS)
        return bytes(data)
    generate = PATTERNS[pattern]
    for row in range(columns):
        for column in range(reduction):
            struct.pack_into("<H", data, (row * reduction + column) * 2,
                             generate(row, column, reduction))
    return bytes(data)


def weight_file(data: bytes, container: str) -> tuple[bytes, int]:
    """The bytes written as weights.bin and the MIL BLOBFILE offset.

    The Apple compiler in this harness takes the weight pointer as the mapped
    file base, so the 128-byte blob header of the ``blob`` container lands in
    the constant section. ``pad`` keeps the blob offset but zeroes the header,
    and ``raw`` drops the wrapper, so both give a section with no header bytes.
    """
    if container == "blob":
        return om.blob(data), 64
    if container == "pad":
        return bytes(128) + data, 64
    if container == "raw":
        return data, 0
    raise ValueError(f"unknown weight container {container!r}")


def probe(reduction: int, columns: int, rows: int, pattern: str,
          container: str = "blob") -> dict:
    """A ``transpose_y=true`` matmul whose [columns, reduction] weight carries a
    known pattern; ``mint_oracles.matmul`` is uniform and cannot do this."""
    x_kind = om.tensor_type((rows, reduction))
    y_kind = om.tensor_type((rows, columns))
    w_kind = om.tensor_type((columns, reduction))
    data = payload(reduction, columns, pattern)
    contents, offset = weight_file(data, container)
    body = [
        'bool f = const()[name = string("f"), val = bool(false)];',
        'bool ty = const()[name = string("ty"), val = bool(true)];',
        f'{w_kind} w = const()[name = string("w"), val = {w_kind}'
        '(BLOBFILE(path = string("@model_path/weights.bin"), '
        f'offset = uint64({offset})))];',
        f'{y_kind} product = matmul(transpose_x = f, transpose_y = ty, '
        'x = x, y = w)[name = string("product")];',
    ]
    suffix = "" if container == "blob" else f"_{container}"
    name = f"matvec_probe_{pattern}{suffix}_m{rows}_k{reduction}_n{columns}"
    return om.case(name, "matvec_probe", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "transpose_y": True, "pattern": pattern, "container": container,
    }, om.program(f"{x_kind} x", body, "product"), contents, {
        "storage": "BLOBFILE", "container": container, "blob_offset": offset,
        "shape": [columns, reduction],
        "payload_bytes": len(data), "value": f"probe pattern {pattern}",
        "payload_sha256": hashlib.sha256(data).hexdigest(),
        "file_sha256": hashlib.sha256(contents).hexdigest(),
        "probe_driver_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
    })


def campaign() -> list[dict]:
    cases = []
    # Sections no larger than 256 bytes: the record decodes every nonzero fp16
    # word, which reads the permutation out directly.
    for reduction, columns in ((16, 8), (32, 4), (64, 2), (8, 16), (128, 1),
                               (16, 4), (64, 1)):
        cases.append(probe(reduction, columns, 1, "index"))
    # Sections no larger than 64 KiB: the record adds 2 KiB chunk hashes and
    # per-chunk nonzero-byte counts, which localize block structure.
    for reduction, columns in ((256, 32), (256, 64), (512, 32), (1024, 16),
                               (256, 128), (128, 256), (64, 64), (32, 256)):
        cases.append(probe(reduction, columns, 1, "mask"))
    # The shipped grid: whole-section SHA-256 proves the packing end to end.
    for reduction in (256, 512, 1024):
        for columns in (256, 512, 1024):
            cases.append(probe(reduction, columns, 1, "mask"))
    cases.append(probe(256, 1024, 64, "mask"))
    cases.append(probe(512, 512, 8, "mask"))
    # One nonzero halfword per case: the destination is recovered by hashing
    # every candidate position, which reads the permutation out exactly.
    for reduction, columns in ((256, 32), (256, 64), (512, 32), (1024, 16),
                               (256, 128), (128, 256)):
        for logical in (64, 65, 66, 79, 80, 95, 96, 127, 128, 129, 130, 159,
                        160, 255, 256, 257, 288, 320, 512, 1023, 1024, 2048,
                        4096, 8191):
            if logical < reduction * columns:
                cases.append(probe(reduction, columns, 1, f"onehot{logical}"))
    # An all-zero weight: the only nonzero bytes in the section come from the
    # 128-byte blob header the compiler copies, which locates that block.
    for reduction, columns in ((256, 32), (256, 64), (512, 32), (1024, 16),
                               (256, 128), (128, 256), (64, 64), (32, 256)):
        cases.append(probe(reduction, columns, 1, "zero"))
    # One stained weight row or column per case: the recorded per-chunk nonzero
    # byte counts show how that row or column spreads across the section.
    for reduction, columns in ((256, 32), (256, 64), (512, 32)):
        for row in (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63):
            if row < columns:
                cases.append(probe(reduction, columns, 1, f"rowstain{row}"))
        for column in (0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255):
            if column < reduction:
                cases.append(probe(reduction, columns, 1, f"colstain{column}"))
    # Container variants: a zeroed or absent blob header removes the header
    # bytes from the section, which makes one-hot destinations unambiguous.
    for container in ("pad", "raw"):
        for reduction, columns in ((256, 32), (1024, 16)):
            cases.append(probe(reduction, columns, 1, "zero", container))
            cases.append(probe(reduction, columns, 1, "mask", container))
            for logical in (0, 1, 2, 31, 32, 63, 64, 65, 127, 128, 255, 256,
                            257, 511, 512, 1023, 1024, 2048, 4096, 8191):
                if logical < reduction * columns:
                    cases.append(probe(reduction, columns, 1,
                                       f"onehot{logical + 64}", container))
    # Wide-N shapes with a small reduction: the section stays under 64 KiB, so
    # the chunk hashes stay available in the N > 256 regime.
    for reduction, columns in ((32, 512), (64, 512), (32, 1024), (16, 1024)):
        cases.append(probe(reduction, columns, 1, "zero"))
        cases.append(probe(reduction, columns, 1, "mask"))
        for logical in (64, 65, 66, 79, 80, 95, 96, 127, 128, 129, 255, 256,
                        512, 1024, 2048, 4096, 8191):
            if logical < reduction * columns:
                cases.append(probe(reduction, columns, 1, f"onehot{logical}"))
    # A dense one-hot sweep over the first weight rows of one wide-N shape:
    # this pins the destination of each row group directly.
    for row in range(72):
        cases.append(probe(64, 512, 1, f"onehot{row * 64}"))
    for row in (0, 1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127,
                128, 255, 256, 257, 383, 384, 511):
        for column in (0, 1, 15, 16, 31, 63):
            cases.append(probe(64, 512, 1, f"onehot{row * 64 + column}"))
    unique: dict[str, dict] = {}
    for item in cases:
        previous = unique.setdefault(item["name"], item)
        if previous["weights"] != item["weights"]:
            raise AssertionError(f"probe {item['name']} defined two ways")
    return list(unique.values())


def local_run(args: argparse.Namespace) -> int:
    tool = Path(args.oracle_tool)
    if platform.system() != "Darwin":
        raise SystemExit("--local requires macOS")
    selected = [item for item in campaign()
                if not args.case or fnmatch.fnmatch(item["name"], args.case)]
    if args.list:
        for item in selected:
            print(item["name"])
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
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-probes."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", str(script), str(script.with_name("mint_oracles.py")),
                        str(script.with_name("h13_td.py")), f"{args.host}:{root}/"],
                       check=True)
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
            with tempfile.TemporaryDirectory(prefix="mil-hwx-probe-json-") as staging:
                subprocess.run(["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                                staging], check=True)
                shutil.copytree(staging, output, dirs_exist_ok=True)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    parser.add_argument("--targets", nargs="+", choices=sorted(om.SUBTYPES),
                        default=["h13"])
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
