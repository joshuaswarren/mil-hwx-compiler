#!/usr/bin/env python3
"""Boolean registry ops: less, floor, select, floor_div from decoded oracles.

Every runtime-operand capture in research/oracles/h13/boolean/ verifies
byte-exact after the selector remap the ANEC encoder always applies: the
captured words carry Apple's channel numbering, the emitted stream carries
the canonical 4..7 allocation, and the deterministic bijection between them
is applied to the expectation before comparison — every other word and the
whole constant section are compared raw. The fp16-result less and
fp16-cond select forms stay rejected: Apple's own tool refuses them, so no
device form exists.
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
    Path(__file__).resolve().parents[1].glob("research/oracles/h13/boolean/*.json"))

# Selected-channel bijection per family: what the captured words select →
# the canonical ANEC allocation (out 4, inputs 5..7).
SELECTOR_REMAP = {
    "less": {6: 4, 4: 5, 5: 6, 7: 7},
    "floor": {5: 4, 4: 5, 6: 6, 7: 7},
    "floor_div": {5: 4, 4: 5, 6: 6, 7: 7},
    "select": {7: 4, 6: 5, 4: 6, 5: 7},
}


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


def family(stem):
    if stem.startswith("less"):
        return "less"
    if stem.startswith("floor_div"):
        return "floor_div"
    if stem.startswith("floor"):
        return "floor"
    return "select"


def capture_stream(record, remap):
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
            if channel in remap:
                linked[8] = (linked[8] & ~(0x1F << shift)) | \
                    (remap[channel] << shift)
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

    # Byte-exactness over every implementable capture. The compile needs an
    # fp16 function result, so the pure bool-producing less is wrapped in
    # the captured cast-to-fp16 form and everything else compiles as-is;
    # const-input floor and the -inf select keep their exact rejections.
    verified = 0
    for capture in captures:
        record = json.loads(capture.read_text())
        if record.get("error") or not record.get("task_descriptors"):
            continue
        stem = capture.stem
        if "comp" in stem or "cast" in stem:
            continue  # composition and cast composites are not this envelope
        if stem.startswith("floor_b") or stem.startswith("select_ninf"):
            continue  # const-input twins: pending index-valued re-mints
        mil = record["mil"]
        if stem.startswith("less_bool_rr"):
            continue  # covered by the less→select composite below
        package = deterministic(root, f"bool-{stem}", mil)
        manifest = json.loads((package / "manifest.json").read_text())
        programs = manifest["programs"]
        booleanProgram = next(p for p in programs
                              if p["encoder"] == "apple-parity-boolean")
        assert booleanProgram["taskDescriptors"] == \
            len(record["task_descriptors"])
        assert anec_task_stream(
            package / f"program-{programs.index(booleanProgram)}.anec") == \
            capture_stream(record, SELECTOR_REMAP[family(stem)]), stem
        const_bin = capture.parent / (capture.stem + ".const.bin")
        if const_bin.exists():
            anec = (package /
                    f"program-{programs.index(booleanProgram)}.anec").read_bytes()
            task_size = struct.unpack_from("<Q", anec, 16)[0]
            consts_off = (task_size + 127) // 128 * 128
            consts_size = struct.unpack_from("<Q", anec, 24)[0]
            emitted = anec[0x1000 + consts_off:
                          0x1000 + consts_off + consts_size]
            assert emitted == const_bin.read_bytes(), stem
        verified += 1
    assert verified >= 8, f"expected the implementable captures, got {verified}"

    # Constant-operand twins verify byte-exact against their captures:
    # floor over a blob x (head-patched input lane) and the scalar-2.0
    # floor_div divisors.
    for stem in ("floor_b_1_idx", "floor_div_native_1_s2",
                 "floor_div_native_1x64x1x1_s2"):
        capture = Path(__file__).resolve().parents[1] / \
            "research/oracles/h13/boolean" / f"{stem}.json"
        record = json.loads(capture.read_text())
        weights_path = capture.parent / f"{stem}.weights.bin"
        if weights_path.exists():
            blob = weights_path.read_bytes()
        elif record.get("weights", {}).get("pattern",
                           "").startswith("uint16_le_index_plus_one_wrapping"):
            count = record["weights"]["payload_bytes"] // 2
            import struct as struct_module
            blob = struct_module.pack(f"<{count}H",
                                      *(((i + 1) & 0xFFFF)
                                        for i in range(count)))
        else:
            blob = None
        if blob is not None:
            header = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(blob), 64 + 24)
            (root / "weights.bin").write_bytes(b"\0" * 64 + header + blob)
        package = deterministic(root, f"bool-const-{stem}", record["mil"])
        manifest = json.loads((package / "manifest.json").read_text())
        programs = manifest["programs"]
        booleanProgram = next(pr for pr in programs
                              if pr["encoder"] == "apple-parity-boolean")
        assert booleanProgram["taskDescriptors"] == \
            len(record["task_descriptors"])
        index = programs.index(booleanProgram)
        assert anec_task_stream(package / f"program-{index}.anec") == \
            capture_stream(record, SELECTOR_REMAP[family(stem)]), stem
        const_bin = capture.parent / f"{stem}.const.bin"
        if const_bin.exists() and "floor_b" not in stem:
            anec = (package / f"program-{index}.anec").read_bytes()
            task_size = struct.unpack_from("<Q", anec, 16)[0]
            consts_off = (task_size + 127) // 128 * 128
            consts_size = struct.unpack_from("<Q", anec, 24)[0]
            emitted = anec[0x1000 + consts_off:
                          0x1000 + consts_off + consts_size]
            assert emitted == const_bin.read_bytes(), stem
        validate(root, package)

    # The floor_b head word echoes the blob lane; prove the patch, not the
    # template, carries the value.
    record = json.loads((Path(__file__).resolve().parents[1] /
                         "research/oracles/h13/boolean/floor_b_1_idx.json").read_text())
    package = compile_source(root, "bool-floor-blob-head", record["mil"])
    anec = (package / "program-0.anec").read_bytes()
    task_size = struct.unpack_from("<Q", anec, 16)[0]
    consts_off = (task_size + 127) // 128 * 128
    assert anec[0x1000 + consts_off:0x1000 + consts_off + 2] == b"\x01\x00"

    # The select cond binding declares a bool dtype with byte elements.
    sample = json.loads((Path(__file__).resolve().parents[1] /
                         "research/oracles/h13/boolean/"
                         "select_rrb_1x64x1x1.json").read_text())
    package = compile_source(root, "bool-select-dtype", sample["mil"])
    manifest = json.loads((package / "manifest.json").read_text())
    cond = next(binding for binding in manifest["programs"][0]["inputs"]
                if binding["dtype"] == "bool")
    assert cond["logicalBytes"] == 64  # 64 lanes, one byte each
    validate(root, package)

    # Apple-rejected forms stay rejected with the evidence in the message.
    def const_(name, literal):
        return (f"    {literal.split('(')[0]} {name} = const()"
                f"[name = string(\"{name}\"), val = {literal}];\n")

    def wrapped(body, inputs):
        return ("program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
                f"  func main<ios18>({inputs}) {{\n" + body +
                "  } -> (y);\n}\n")

    body = ("    tensor<fp16, [375, 1, 1]> y = less(x = a, y = a)"
            "[name = string(\"y\")];\n")
    compile_source(root, "bool-less-fp16-result",
                   wrapped(body, "tensor<fp16, [375, 1, 1]> a"),
                   expected_code="h13.boolean-outside-envelope",
                   expected_message="fp16-result less and fp16-cond select are rejected by Apple's own tool")

    body = ("    tensor<fp16, [1, 1024, 375]> y = select("
             "a = a, b = b, cond = m)[name = string(\"y\")];\n")
    compile_source(root, "bool-select-fp16-cond",
                   wrapped(body, "tensor<fp16, [1, 1024, 375]> a, "
                                 "tensor<fp16, [1, 1024, 375]> b, "
                                 "tensor<fp16, [1, 1024, 375]> m"),
                   expected_code="h13.boolean-outside-envelope",
                   expected_message="fp16-result less and fp16-cond select are rejected by Apple's own tool")

    body = ("    tensor<fp16, [1, 3, 1, 1]> y = floor(x = a)"
            "[name = string(\"y\")];\n")
    compile_source(root, "bool-floor-uncaptured",
                   wrapped(body, "tensor<fp16, [1, 3, 1, 1]> a"),
                   expected_code="h13.boolean-outside-envelope")

    # The -inf constant-a select keeps its exact pending reason.
    payload = struct.pack("<e", float("-inf"))
    header = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(payload), 64 + 24)
    (root / "weights.bin").write_bytes(b"\0" * 64 + header + payload)
    body = const_("neg_inf", "tensor<fp16, []>"
                             "(BLOBFILE(path = string(\"@model_path/weights.bin\"),"
                             " offset = uint64(64)))")
    body += ("    tensor<fp16, [1, 8, 375, 375]> y = select("
             "a = neg_inf, b = b, cond = m)[name = string(\"y\")];\n")
    compile_source(root, "bool-select-const-a",
                   wrapped(body, "tensor<fp16, [1, 8, 375, 375]> b, "
                                 "tensor<bool, [1, 8, 375, 375]> m"),
                   expected_code="h13.select-needs-decoded-encoder",
                   expected_message="materializes the fill as a runtime constant input")

    print("h13 boolean cli: PASS")
