#!/usr/bin/env python3
"""Materialized tile: runtime-operand captures lower byte-exact.

Every runtime-operand capture in research/oracles/h13/layout/tile_*.json
verifies byte-exact after the selector remap the ANEC encoder always
applies: the captured words carry Apple's channel numbering (input 4,
result 5), the emitted stream carries the canonical allocation (input 5,
result 4). The all-ones-reps form stays the free host-side alias, and the
blob-operand form stays rejected: its constant-section lane embedding has
no byte-exact reproducer, so only the runtime-operand templates lower.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")
captures = sorted(
    Path(__file__).resolve().parents[1].glob("research/oracles/h13/layout/tile_*.json"))

# Captured input/result channels -> canonical ANEC allocation.
TILE_REMAP = {5: 4, 4: 5}


def reconstruct_task(td):
    words = [int(w, 16) for w in td["header_words"]]
    flat = {}
    for block in td["blocks"].values():
        for a, v in block["words"].items():
            flat[int(a, 16)] = int(v, 16)
    for rec in td["records"]:
        words.append(int(rec["header"], 16))
        base = int(rec["address"], 16)
        for index in range(rec["count"]):
            words.append(flat[base + index * 4])
    return words


def capture_stream(record):
    words_list = [reconstruct_task(t) for t in record["task_descriptors"]]
    stream = bytearray()
    offset = 0
    for index, words in enumerate(words_list):
        blob = struct.pack(f"<{len(words)}I", *words)
        if len(stream) < offset + len(blob):
            stream.extend(b"\0" * (offset + len(blob) - len(stream)))
        linked = list(words)
        linked[0] = (linked[0] & ~0x00FF0000) | 0x00400000
        for shift in (0, 6, 12):
            channel = (linked[8] >> shift) & 0x1F
            if channel in TILE_REMAP:
                linked[8] = (linked[8] & ~(0x1F << shift)) | \
                    (TILE_REMAP[channel] << shift)
        stream[offset:offset + len(blob)] = struct.pack(
            f"<{len(linked)}I", *linked)
        offset = words[7] if index + 1 < len(words_list) else 0
    return bytes(stream)


def compile_source(root, name, text, expected_code=None, expected_message=None,
                   format="anec"):
    path = root / f"{name}.mil"
    path.write_text(text)
    output = root / name
    result = subprocess.run(
        [compiler, "--mil", str(path), "--model-root", str(root),
         "--output", str(output), "--target", "H13", "--format", format],
        capture_output=True, text=True, timeout=300, check=False)
    if expected_code is None:
        assert result.returncode == 0, result.stderr
        return output
    assert result.returncode == 65, result.stderr
    assert expected_code in result.stderr, result.stderr
    if expected_message:
        assert expected_message in result.stderr, result.stderr
    return None


def deterministic(root, name, text, format="anec"):
    first = compile_source(root, f"{name}-a", text, format=format)
    second = compile_source(root, f"{name}-b", text, format=format)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()}
    return first


def validate(root, package):
    result = subprocess.run([sys.executable, inspector, str(package)],
                            capture_output=True, text=True, timeout=60,
                            check=False)
    assert result.returncode == 0, result.stderr


def anec_task_stream(path):
    data = path.read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    return data[0x1000:0x1000 + size]


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)

    verified = 0
    for capture in captures:
        record = json.loads(capture.read_text())
        assert not record.get("error"), capture.name
        assert record.get("task_descriptors"), capture.name
        stem = capture.stem
        reps = record["parameters"]["reps"]
        shape = record["parameters"]["shape"]
        if all(rep == 1 for rep in reps):
            # All-ones repeats are the identity view: the host-side alias,
            # not a program. Apple materializes the same form; the alias
            # is the exact same surface either way.
            package = deterministic(root, f"tile-{stem}", record["mil"])
            manifest = json.loads((package / "manifest.json").read_text())
            tiled = record["mil"].split("= tile(")[0].split()[-1]
            assert manifest["tensors"][tiled]["aliasOf"], stem
            validate(root, package)
            verified += 1
            continue
        if record["parameters"].get("x_storage") == "blob":
            # The blob template's lane embedding has no reproducer.
            compile_source(root, f"tile-{stem}", record["mil"],
                           expected_code="h13.unsupported-tile",
                           expected_message="no byte-exact reproducer")
            verified += 1
            continue
        package = deterministic(root, f"tile-{stem}", record["mil"])
        manifest = json.loads((package / "manifest.json").read_text())
        programs = manifest["programs"]
        tile_program = next(p for p in programs
                            if p["encoder"] == "apple-parity-tile")
        assert tile_program["taskDescriptors"] == \
            len(record["task_descriptors"]), stem
        index = programs.index(tile_program)
        assert anec_task_stream(
            package / f"program-{index}.anec") == \
            capture_stream(record), stem
        const_bin = capture.parent / f"{stem}.const.bin"
        if const_bin.exists():
            anec = (package / f"program-{index}.anec").read_bytes()
            task_size = struct.unpack_from("<Q", anec, 16)[0]
            consts_off = (task_size + 127) // 128 * 128
            consts_size = struct.unpack_from("<Q", anec, 24)[0]
            emitted = anec[0x1000 + consts_off:
                          0x1000 + consts_off + consts_size]
            assert emitted == const_bin.read_bytes(), stem
        validate(root, package)
        verified += 1
    assert verified == 4, f"expected all four tile captures, got {verified}"

    print("h13 tile cli: PASS")
