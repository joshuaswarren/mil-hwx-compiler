#!/usr/bin/env python3
"""Mint H14 matmul oracles with known weights to prove Apple's H14 packing.

The shipped campaign in ``mint_oracles.py`` uses a uniform fp16 0.5 weight, so
its constant-section hashes cannot tell one permutation from another. These
probes give every weight element a distinct or positionally unique fp16 bit
pattern, so the recorded hashes -- and, for sections no larger than 256 bytes,
the decoded index/value table -- pin the H14 packing exactly. Every case name
is prefixed ``h14mv_`` and only ``target=h14`` is minted.

The Apple compiler in this harness takes the weight pointer as the mapped file
base, so the 128-byte blob header of ``mint_oracles.blob`` occupies the first
64 halfwords of the packed section. ``packed_section`` models that, which is
why ``--verify`` can rebuild every recorded hash offline.

Mint (research only; no HWX bytes are stored):

```sh
python3 research/mint_h14_matvec_probes.py --host macstudio
```

Re-check the checked-in probe records against the packing model offline:

```sh
python3 research/mint_h14_matvec_probes.py --verify
```
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

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / "research/oracles/h14"
# The Apple compiler copies the weight file from byte 0, so the 128-byte blob
# header occupies the first 64 halfwords of the constant section. A "logical"
# halfword index counts from the start of the file: payload halfword ``i`` is
# logical index ``i + 64``.
HEADER_HALFWORDS = 64
ONE_HOT_BITS = 0x3555
# Apple never emits a matvec constant section below 1 KiB.
MINIMUM_SECTION_BYTES = 1024
# The decoded grid: every (M, K, N) Apple accepted as one two-task H14 program.
GRID_ROWS = (1, 2, 8, 64)
GRID_SIDES = (256, 512, 1024)


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


def payload(reduction: int, columns: int, pattern: str) -> bytes:
    """The [columns, reduction] row-major fp16 weight one pattern describes."""
    data = bytearray(columns * reduction * 2)
    if pattern == "zero":
        return bytes(data)
    if pattern.startswith("onehot"):
        logical = int(pattern[len("onehot"):])
        struct.pack_into("<H", data, (logical - HEADER_HALFWORDS) * 2,
                         ONE_HOT_BITS)
        return bytes(data)
    generate = PATTERNS[pattern]
    for row in range(columns):
        for column in range(reduction):
            struct.pack_into("<H", data, (row * reduction + column) * 2,
                             generate(row, column, reduction))
    return bytes(data)


def section_bytes(reduction: int, columns: int) -> int:
    """Apple's constant-section length: the weight bytes, but never under
    1 KiB. Proven by the `h14mv_index_*_n16` probes, whose 64..512-byte
    weights all land in a 1024-byte section."""
    return max(MINIMUM_SECTION_BYTES, reduction * columns * 2)


def packed_section(reduction: int, columns: int, source: bytes) -> bytes:
    """Apple's constant section for ``source``, the weight file from byte 0.

    ``group`` weight rows interleave at halfword granularity, each group of
    rows occupies one plane, the planes are ordered by the low four bits of
    the group index, and the plane stride is the section split evenly across
    the planes -- which is exactly ``reduction * group`` halfwords whenever
    the weight fills the section, as every shipped geometry does.
    """
    if not reduction or columns < 16 or columns % 16 or \
            (columns > 256 and columns % 256):
        raise ValueError("packing needs 16 columns per row group, 256 per "
                         "plane group above 256 columns")
    if len(source) != reduction * columns * 2:
        raise ValueError("source must hold columns * reduction halfwords")
    group = min(16, columns // 16)
    planes = columns // group
    packed = bytearray(section_bytes(reduction, columns))
    stride = len(packed) // 2 // planes
    for column in range(columns):
        plane = column // group
        destination_plane = (plane % 16) * (planes // 16) + plane // 16
        destination = (destination_plane * stride + column % group) * 2
        offset = column * reduction * 2
        for index in range(reduction):
            packed[destination:destination + 2] = \
                source[offset + index * 2:offset + index * 2 + 2]
            destination += group * 2
    return bytes(packed)


def expected_section(record: dict) -> bytes:
    """The constant section a probe record must carry, rebuilt offline."""
    parameters = record["parameters"]
    reduction, columns = parameters["reduction"], parameters["columns"]
    data = payload(reduction, columns, parameters["pattern"])
    contents = om.blob(data)
    return packed_section(reduction, columns, contents[:len(data)])


def probe(reduction: int, columns: int, rows: int, pattern: str) -> dict:
    """A ``transpose_y=true`` matmul whose [columns, reduction] weight carries
    a known pattern, which a uniform weight cannot distinguish."""
    x_kind = om.tensor_type((rows, reduction))
    y_kind = om.tensor_type((rows, columns))
    w_kind = om.tensor_type((columns, reduction))
    data = payload(reduction, columns, pattern)
    contents = om.blob(data)
    body = [
        'bool f = const()[name = string("f"), val = bool(false)];',
        'bool ty = const()[name = string("ty"), val = bool(true)];',
        f'{w_kind} w = const()[name = string("w"), val = {w_kind}'
        '(BLOBFILE(path = string("@model_path/weights.bin"), '
        'offset = uint64(64)))];',
        f'{y_kind} product = matmul(transpose_x = f, transpose_y = ty, '
        'x = x, y = w)[name = string("product")];',
    ]
    name = f"h14mv_{pattern}_m{rows}_k{reduction}_n{columns}"
    return om.case(name, "matvec_probe", {
        "rows": rows, "reduction": reduction, "columns": columns,
        "transpose_y": True, "pattern": pattern, "container": "blob",
    }, om.program(f"{x_kind} x", body, "product"), contents, {
        "storage": "BLOBFILE", "container": "blob", "blob_offset": 64,
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
    # word, so the permutation is read out element by element.
    for reduction, columns in ((16, 16), (8, 16), (4, 16), (2, 16)):
        cases.append(probe(reduction, columns, 1, "index"))
    # The shipped grid: whole-section SHA-256 over a varied pattern proves the
    # packing end to end for every (K, N) the parity encoder claims.
    for reduction in GRID_SIDES:
        for columns in GRID_SIDES:
            cases.append(probe(reduction, columns, 1, "mask"))
    # Row coverage: the packing must not depend on M.
    for rows in GRID_ROWS[1:]:
        cases.append(probe(256, 256, rows, "mask"))
    cases.append(probe(256, 1024, 64, "mask"))
    cases.append(probe(1024, 256, 8, "mask"))
    # An all-zero weight: the only nonzero section bytes come from the blob
    # header the compiler copies, which locates that block after packing.
    for reduction, columns in ((256, 32), (256, 64), (1024, 16), (128, 256),
                               (64, 512), (32, 1024)):
        cases.append(probe(reduction, columns, 1, "zero"))
        cases.append(probe(reduction, columns, 1, "mask"))
    # One nonzero halfword per case: its recorded destination pins the row
    # group, the interleave stride, and the plane order independently of any
    # hash agreement.
    for reduction, columns in ((256, 64), (1024, 16), (128, 256), (64, 512),
                               (32, 1024)):
        for logical in (64, 65, 66, 79, 80, 95, 96, 127, 128, 129, 130, 255,
                        256, 257, 512, 1024, 2048, 4096, 8191):
            if logical - HEADER_HALFWORDS < reduction * columns:
                cases.append(probe(reduction, columns, 1, f"onehot{logical}"))
    unique: dict[str, dict] = {}
    for item in cases:
        previous = unique.setdefault(item["name"], item)
        if previous["weights"] != item["weights"]:
            raise AssertionError(f"probe {item['name']} defined two ways")
    return list(unique.values())


def verify(pattern: str | None = None) -> int:
    """Rebuild every checked-in probe section from the packing model."""
    records = [json.loads(path.read_text())
               for path in sorted(ORACLES.glob("h14mv_*.json"))]
    if not records:
        raise SystemExit(f"no h14mv_* probe records under {ORACLES}")
    checked = rejected = 0
    for record in records:
        if pattern and not fnmatch.fnmatch(record["case"], pattern):
            continue
        if record.get("error") is not None:
            rejected += 1
            continue
        section = record["constant_section"]
        expected = expected_section(record)
        if len(expected) != section["size"]:
            raise SystemExit(f"{record['case']}: modelled {len(expected)} "
                             f"bytes, Apple recorded {section['size']}")
        digest = hashlib.sha256(expected).hexdigest()
        if digest != section["sha256"]:
            raise SystemExit(f"{record['case']}: modelled hash {digest} but "
                             f"Apple recorded {section['sha256']}")
        # Apple's 1 KiB minimum keeps every matvec section above the 256-byte
        # limit of the decoded index/value table, so the whole-section hash
        # plus the nonzero-byte count is the strongest available check.
        if sum(1 for byte in expected if byte) != section["nonzero_bytes"]:
            raise SystemExit(f"{record['case']}: nonzero byte count differs")
        checked += 1
    print(f"H14 matvec probes: {checked} sections rebuilt byte-for-byte "
          f"({rejected} Apple rejections)")
    return 0


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
    for item in selected:
        destination = output / "h14" / f"{item['name']}.json"
        if destination.exists() and not args.force:
            existing = json.loads(destination.read_text())
            decoded += existing.get("error") is None
            rejected += existing.get("error") is not None
            continue
        status = om.run_case(item, "h14", output, tool, args.source_commit)
        decoded += status == "decoded"
        rejected += status == "rejected"
        print(f"h14 {item['name']} {status}", flush=True)
    print(f"SUMMARY cases={len(selected)} decoded={decoded} "
          f"rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-h14-probes.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-h14-probes."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["scp", "-q", str(script),
                        str(script.with_name("mint_oracles.py")),
                        str(script.with_name("h13_td.py")),
                        f"{args.host}:{root}/"], check=True)
        command = ["python3", f"{root}/{script.name}", "--local",
                   "--oracle-tool", args.oracle_tool,
                   "--output", f"{root}/oracles",
                   "--source-commit", args.source_commit]
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
            with tempfile.TemporaryDirectory(prefix="mil-hwx-h14-json-") as staging:
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
    mode.add_argument("--verify", action="store_true",
                      help="rebuild the checked-in probe sections offline")
    parser.add_argument("--oracle-tool", default=om.DEFAULT_TOOL)
    parser.add_argument("--output", default="research/oracles")
    parser.add_argument("--case")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default=om.source_commit())
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    if arguments.verify:
        raise SystemExit(verify(arguments.case))
    raise SystemExit(local_run(arguments) if arguments.local
                     else remote_run(arguments))
