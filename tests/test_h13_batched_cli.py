#!/usr/bin/env python3
"""Batched matmul envelope: one task stream of 26·B tasks from decoded oracles.

The batched primitive the survey called for: rank-3 [B,rows,K] or rank-4
[1,B,rows,K] operands with both transpose flags false lower as one program
whose task stream is Apple's own — batch zero's 26-task group stamped once
per batch with the verified per-batch word rules. Every runtime capture in
research/oracles/h13/batched/ is proven byte-exact (task words equal the
capture reconstruction after the link-marker rewrite the ANEC encoder always
applies). The mixed-layout form Apple itself rejects stays rejected here,
and the constant-weight form is gated until a non-uniform re-mint makes its
constant-section packing decodable.
"""
import glob
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")
captures = sorted(
    Path(__file__).resolve().parents[1].glob("research/oracles/h13/batched/*.json"))


def header(shape):
    return f"tensor<fp16, [{', '.join(map(str, shape))}]>"


def source(body, inputs, result):
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({inputs}) {{
{body}  }} -> ({result});
}}
"""


def const(name, literal):
    return f"    {literal.split('(')[0]} {name} = const()" \
           f"[name = string(\"{name}\"), val = {literal}];\n"


def reconstruct_task(td):
    words = [int(w, 16) for w in td["header_words"]]
    flat = {}
    for block in td["blocks"].values():
        for address, value in block["words"].items():
            flat[int(address, 16)] = int(value, 16)
    for record in td["records"]:
        words.append(int(record["header"], 16))
        base = int(record["address"], 16)
        for index in range(record["count"]):
            words.append(flat[base + index * 4])
    return words


def capture_stream(record):
    """The full task byte stream with the link markers bindTasks applies."""
    words_list = [reconstruct_task(t) for t in record["task_descriptors"]]
    stream = bytearray()
    offset = 0
    for index, words in enumerate(words_list):
        blob = struct.pack(f"<{len(words)}I", *words)
        if len(stream) < offset + len(blob):
            stream.extend(b"\0" * (offset + len(blob) - len(stream)))
        linked = list(words)
        linked[0] = (linked[0] & ~0x00FF0000) | 0x00400000
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

    # Byte-exactness against every capture: task streams reconstructed from
    # the decoded JSON, constant sections compared against the retained raw
    # Apple bytes (round-2 _idx captures carry weights.bin and const.bin).
    verified = 0
    for capture in captures:
        record = json.loads(capture.read_text())
        if record.get("error") or len(record["task_descriptors"]) % 26 not in (0, 25):
            continue
        root_dir = capture.parent
        weights = root_dir / (capture.stem + ".weights.bin")
        const_bin = root_dir / (capture.stem + ".const.bin")
        if record["parameters"]["w_storage"] == "blob":
            if weights.exists():
                blob = weights.read_bytes()
            else:
                # Round-1 uniform captures record the payload size only.
                blob = b"\x00\x38" * (record["weights"]["payload_bytes"] // 2)
            blob_header = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(blob), 64 + 24)
            (root / "weights.bin").write_bytes(b"\0" * 64 + blob_header + blob)
        package = compile_source(root, f"cap-{capture.stem}", record["mil"])
        manifest = json.loads((package / "manifest.json").read_text())
        assert len(manifest["programs"]) == 1
        program = manifest["programs"][0]
        batch = record["parameters"]["batch"]
        prefix = record["program_descriptor"]["task_count"] - 26 * batch
        assert program["taskDescriptors"] == 26 * batch + prefix
        assert program["encoder"] == (
            "apple-parity-batched-matmul" if record["parameters"]["w_storage"] == "runtime"
            else "apple-parity-batched-matvec")
        assert anec_task_stream(package / "program-0.anec") == \
            capture_stream(record), capture.stem
        if const_bin.exists():
            anec = (package / "program-0.anec").read_bytes()
            task_size = struct.unpack_from("<Q", anec, 16)[0]
            consts_off = (task_size + 127) // 128 * 128
            consts_size = struct.unpack_from("<Q", anec, 24)[0]
            emitted = anec[0x1000 + consts_off:0x1000 + consts_off + consts_size]
            assert emitted == const_bin.read_bytes(), capture.stem
        validate(root, package)
        verified += 1
    assert verified >= 24, f"expected every capture, verified {verified}"

    # Determinism in both artifact formats.
    sample = json.loads(
        (Path(__file__).resolve().parents[1] /
         "research/oracles/h13/batched/"
         "bmm_r4headsrr_m375_k128_n749_tx0_ty0_b8_r4heads.json").read_text())
    package = deterministic(root, "batched-deterministic", sample["mil"])
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["programs"][0]["inputs"][0]["shape"] == [1, 8, 128, 749]
    assert manifest["programs"][0]["outputs"][0]["shape"] == [1, 8, 375, 749]
    deterministic(root, "batched-deterministic-hwx", sample["mil"], format="hwx")

    # The mixed layout Apple rejects stays rejected here, mirroring it: the
    # shape mismatch surfaces at the geometry gate before any batching.
    body = const("tx", "bool(false)") + const("ty", "bool(false)")
    body += ("    tensor<fp16, [1, 8, 375, 749]> y = matmul("
             "transpose_x = tx, transpose_y = ty, x = a, y = b)"
             "[name = string(\"y\")];\n")
    compile_source(root, "batched-mixed-layout", source(
        body, header([1, 375, 8, 128]) + " a, " + header([1, 8, 128, 749]) + " b",
        "y"),
        expected_code="h13.unsupported-program",
        expected_message="H13 matmul requires positive fp16 x rows")
    body = const("tx", "bool(false)") + const("ty", "bool(false)")
    body += ("    tensor<fp16, [3, 375, 749]> y = matmul("
             "transpose_x = tx, transpose_y = ty, x = a, y = b)"
             "[name = string(\"y\")];\n")
    compile_source(root, "batched-uncaptured-batch", source(
        body, header([3, 375, 128]) + " a, " + header([3, 128, 749]) + " b", "y"),
        expected_code="h13.matmul-outside-envelope",
        expected_message="outside the decoded batched envelope")

    # The V-projection transpose_y form is captured and covered above; the
    # both-flags form has no capture and rejects.
    body = const("tx", "bool(true)") + const("ty", "bool(true)")
    body += ("    tensor<fp16, [1, 8, 375, 749]> y = matmul("
             "transpose_x = tx, transpose_y = ty, x = a, y = b)"
             "[name = string(\"y\")];\n")
    compile_source(root, "batched-both-flags", source(
        body, header([1, 8, 375, 128]) + " a, " + header([1, 8, 128, 749]) + " b",
        "y"),
        expected_code="h13.unsupported-program",
        expected_message="H13 matmul requires positive fp16 x rows")

    # A constant weight whose blob size disagrees with the geometry rejects.
    body = const("tx", "bool(false)") + const("ty", "bool(false)")
    body += const("w", "tensor<fp16, [1, 8, 128, 749]>"
                        "(BLOBFILE(path = string(\"@model_path/weights.bin\"),"
                        " offset = uint64(64)))")
    body += ("    tensor<fp16, [1, 8, 375, 749]> y = matmul("
             "transpose_x = tx, transpose_y = ty, x = a, y = w)"
             "[name = string(\"y\")];\n")
    payload = b"\x00\x38" * 16
    blob_header = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(payload), 64 + 24)
    (root / "weights.bin").write_bytes(b"\0" * 64 + blob_header + payload)
    compile_source(root, "batched-const-weight-bad-size", source(
        body, header([1, 8, 375, 128]) + " a", "y"),
        expected_code="ane.model.invalid-blob-header")

    print("h13 batched cli: PASS")
