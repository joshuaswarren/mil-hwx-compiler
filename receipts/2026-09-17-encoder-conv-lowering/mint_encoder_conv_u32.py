"""Round 3: break the F1 dense colgroup alias with a 32-bit-index payload.

The round-2 F1 capture (encoder_conv_idx_c1024_n2048_k1x1_s1_g1_bias0_valid_wmaj)
carries the ``uint16_le_index_plus_one_wrapping`` payload, under which every
halfword is invariant under ``e -> e + 64*1024``: the section cannot qualify
the order of colgroups 64 apart (chunk order is invisible, plane-group order
only mod 4).  This minter re-mints the exact same geometry with a payload of
32-bit-index pairs: halfwords 2k/2k+1 hold the little-endian uint32 ``k+1``
(pair index plus one), so a colgroup shift of 64*1024 halfwords moves the
pair counter by 32768 and changes the stored low halfword everywhere and the
high halfword on carry — the alias is gone.

Distinctness gate (Main's bar, quantitative, applied before any derivation):
the payload must differ under the 65536-halfword shift at at least half the
positions, and the captured section must show it — read as uint32 pairs it
must hold at least half the expected distinct 32-bit values at some byte
alignment.

    python3 mint_encoder_conv_u32.py --host macstudio --force \\
        --output oracles-u32
"""
from __future__ import annotations

import argparse
import array
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "2026-09-16-compiler-leftover"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R2))
sys.path.insert(0, str(HERE.parents[1] / "research"))

import mint_encoder_conv_idx as idx  # noqa: E402
import mint_encoder_leftover2 as r2  # noqa: E402
import mint_oracles as om  # noqa: E402

PATTERN = "uint32le_pair_index_plus_one"
ALIAS_HALFWORDS = 64 * 1024
GATE_FRACTION = 0.5


def payload_bytes(elements: int) -> bytes:
    """One running uint32(pair+1) stream across the whole payload area."""
    halfwords = array.array("H", bytes(2 * elements))
    for pair in range((elements + 1) // 2):
        value = pair + 1
        halfwords[2 * pair] = value & 0xFFFF
        if 2 * pair + 1 < elements:
            halfwords[2 * pair + 1] = (value >> 16) & 0xFFFF
    return halfwords.tobytes()


def payload_halfwords(elements: int) -> array.array:
    return array.array("H", payload_bytes(elements))


def distinctness_gate(halfwords: array.array) -> dict:
    """Main's distinctness gate on the minted payload itself."""
    total = len(halfwords) - ALIAS_HALFWORDS
    if total <= 0:
        raise ValueError("payload shorter than the alias distance")
    differing = sum(
        1 for index in range(total)
        if halfwords[index] != halfwords[index + ALIAS_HALFWORDS])
    even = sum(
        1 for index in range(0, total, 2)
        if halfwords[index] != halfwords[index + ALIAS_HALFWORDS])
    stats = {
        "alias_halfwords": ALIAS_HALFWORDS,
        "compared_halfwords": total,
        "differing": differing,
        "differing_fraction": differing / total,
        "differing_even_positions": even,
        "gate_fraction": GATE_FRACTION,
        "passed": differing >= GATE_FRACTION * total,
    }
    if not stats["passed"]:
        raise ValueError(f"distinctness gate failed: {stats}")
    return stats


def finish_u32_case(item: dict, blobs: r2.Blobs) -> dict:
    elements = blobs.payload_elements()
    halfwords = payload_halfwords(elements)
    gate = distinctness_gate(halfwords)
    # The BLOBFILE container blob_bytes_index builds: 8-byte header, a
    # 4096-slot 24-byte record table, then the 64-byte-aligned payload area
    # carrying one u32-pair stream across every constant.
    data_start = (blobs.offset(blobs.RECORD_CAPACITY) + 0x3F) & ~0x3F
    blob = bytearray(data_start + len(halfwords) * 2)
    struct.pack_into("<II", blob, 0, len(blobs.shapes), 2)
    stream = 0
    cursor = data_start
    for index, shape in enumerate(blobs.shapes):
        count = r2.math_prod(shape)
        payload = halfwords[stream:stream + count]
        struct.pack_into("<IIQQ", blob, blobs.offset(index),
                         0xDEADBEEF, 1, count * 2, cursor)
        blob[cursor:cursor + count * 2] = payload.tobytes()
        stream += count
        cursor += count * 2
    item["weights"] = bytes(blob)
    item["weights_description"] = {
        "storage": "BLOBFILE",
        "shapes": [list(shape) for shape in blobs.shapes],
        "record_offsets": [blobs.offset(index)
                           for index in range(len(blobs.shapes))],
        "payload_bytes": elements * 2,
        "value": "uint32 little-endian (pair index + 1), one pair per two "
                 "halfwords, running across every constant",
        "pattern": PATTERN,
        "distinctness_gate": gate,
    }
    item["distinctness_gate"] = gate
    return item


def encoder_conv_idx_u32(x_shape, w_shape, out_shape, stride, groups,
                         pad_type, bias, tag):
    """idx's builder with the u32-pair finisher; MIL stays byte-identical."""
    original = r2.finish_index_case
    r2.finish_index_case = finish_u32_case
    try:
        return idx.encoder_conv_idx(x_shape, w_shape, out_shape, stride,
                                    groups, pad_type, bias, tag)
    finally:
        r2.finish_index_case = original


def campaign():
    # The F1 in-projection geometry, W-major rank-4 respell, verbatim from
    # round 2 - only the payload scheme differs.
    return [encoder_conv_idx_u32(
        (1, 1024, 1, 375), (2048, 1024, 1, 1), (1, 2048, 1, 375),
        (1, 1), 1, "valid", False, "wmaj_u32")]


def local_run(args: argparse.Namespace) -> int:
    selected = [item for item in campaign()
                if not args.case or args.case in item["name"]]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)}")
        return 0
    if sys.platform != "darwin":
        raise SystemExit("--local requires macOS")
    tool = Path(args.oracle_tool)
    if not tool.is_file():
        raise SystemExit(f"oracle tool is missing: {tool}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    decoded = rejected = 0
    for item in selected:
        status, record = r2.run_case_retain(item, output, tool,
                                            args.source_commit)
        decoded += status == "decoded"
        rejected += status == "rejected"
        tasks = len(record.get("task_descriptors") or [])
        const = (record.get("constant_section") or {}).get("size")
        print(f"h13 {item['name']} {status} tasks={tasks} const={const}"
              + ("" if record.get("error") is None
                 else f" error={record['error'][:120]}"), flush=True)
    print(f"SUMMARY cases={len(selected)} decoded={decoded} "
          f"rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    import shlex
    import shutil
    import subprocess
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-encoder-u32.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-encoder-u32."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    research = script.parents[2] / "research"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script),
             str(HERE / "mint_encoder_conv_idx.py"),
             str(R2 / "mint_encoder_leftover2.py"),
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
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
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


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    parser.add_argument("--oracle-tool",
                        default="/tmp/h13-oracle/bin/ane-compile-hwx")
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parent / "oracles-u32"))
    parser.add_argument("--case", help="substring selecting case names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default="073c7b8")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.host:
        return remote_run(args)
    return local_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
