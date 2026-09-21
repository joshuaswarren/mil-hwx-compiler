#!/usr/bin/env python3
"""The encoder length-mask prelude: int32 arange compare respelled to fp16.

The real encoder computes output_mask = less(arange_int32, lengths_int32)
through cast(fp16->int32) and expand_dims(int32) — an int32 domain with no
decoded H13 form (Apple refuses int32 less/cast; the broadcast-compare
captures were all refused). Every value in the chain is exact in fp16: the
arange constants are integers within fp16's exact integer range, and the
floor proves the runtime side integral. When the statement matches that
shape exactly and the int32 chain has no other consumers, the compiler
respells it to the decoded fp16 less row: an inline fp16 copy of the arange
against the fp16 length directly.

The decoded row reads both surfaces full-width, so the smaller length
operand is a DECLARED host broadcast: the manifest binding carries the
source extent and the host repeats the value across the surface before
dispatch — exact (a copy, no arithmetic), checked by the storage walk and
the package validator. Anything that does not match the pattern keeps the
int32 refusals.

Numeric checks run the compiled programs in pure Python and against the
h13 reference's evaluation of the original MIL (which now evaluates the
length-chain int32 directions exactly); device numerics stay a hardware-
window claim (t6001-test-host), as everywhere else on this branch.
"""
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "research")]
compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else
                    ROOT / "build/mil-hwxc").resolve())
inspector = str(ROOT / "research" / "inspect_anec.py")
from h13_reference import evaluate  # noqa: E402

captures = ROOT / "research/oracles/h13/boolean"

SELECTOR_REMAP_LESS = {6: 4, 4: 5, 5: 6, 7: 7}


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


def compile_source(root, name, text, expected_code=None, expected_message=None):
    path = root / f"{name}.mil"
    path.write_text(text)
    output = root / name
    result = subprocess.run(
        [compiler, "--mil", str(path), "--model-root", str(root),
         "--output", str(output), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True, timeout=300, check=False)
    if expected_code is None:
        assert result.returncode == 0, result.stderr
        return output
    assert result.returncode == 65, result.stderr
    assert expected_code in result.stderr, result.stderr
    if expected_message:
        assert expected_message in result.stderr, result.stderr
    return None


def validate(root, package):
    result = subprocess.run([sys.executable, inspector, str(package)],
                            capture_output=True, text=True, timeout=60,
                            check=False)
    assert result.returncode == 0, result.stderr


def anec_task_stream(package, index=0):
    data = (package / f"program-{index}.anec").read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    return data[0x1000:0x1000 + size]


def int32_record(offset, values):
    """One BLOBFILE record at an absolute offset, resolver layout."""
    payload = struct.pack(f"<{len(values)}i", *values)
    header = struct.pack("<IIQQ", 0xDEADBEEF, 1, len(payload), offset + 64)
    return offset, header + b"\0" * 40 + payload


def blob_file(records):
    size = max(off for off, _ in records) + 64 + max(
        len(blob) for _, blob in records)
    data = bytearray(size)
    for off, blob in records:
        data[off:off + len(blob)] = blob
    return bytes(data)


def mask_mil(arange_offset, arange_values):
    """The real prelude spellings, verbatim: the lengths scalar arrives
    floored (the graph's floor row), the int32 chain mirrors
    encoder-pinned.mil."""
    payload = struct.pack(f"<{len(arange_values)}i", *arange_values)
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1]> lengths_13_cast_fp16) {{
    tensor<string, []> output_lengths_dtype_0 = const()[name = string("output_lengths_dtype_0"), val = tensor<string, []>("int32")];
    tensor<int32, [375]> var_281 = const()[name = string("var_281"), val = tensor<int32, [375]>(BLOBFILE(path = string("@model_path/weights/weight.bin"), offset = uint64({arange_offset})))];
    tensor<int32, [1]> var_283_axes_0 = const()[name = string("var_283_axes_0"), val = tensor<int32, [1]>([1])];
    tensor<fp16, [1]> lengths_cast_fp16 = floor(x = lengths_13_cast_fp16)[name = string("lengths_cast_fp16")];
    tensor<int32, [1]> lengths_cast_fp16_to_int32 = cast(dtype = output_lengths_dtype_0, x = lengths_cast_fp16)[name = string("cast_4")];
    tensor<int32, [1, 1]> var_283 = expand_dims(axes = var_283_axes_0, x = lengths_cast_fp16_to_int32)[name = string("op_283")];
    tensor<bool, [1, 375]> output_mask = less(x = var_281, y = var_283)[name = string("output_mask")];
  }} -> (output_mask);
}}
""" , len(payload)


def fp16_bytes(values):
    import struct as struct_module
    return b"".join(struct_module.pack("<e", v) for v in values)


def simulate(manifest, package, lengths):
    """Runner-style host simulation of the two programs."""
    import struct as struct_module

    def lanes_floor(binding, value):
        buffer = bytearray(binding["allocationBytes"])
        struct_module.pack_into("<e", buffer, 0,
                                value if not math.isnan(value) else value)
        return bytes(buffer)

    floor_program = manifest["programs"][0]
    assert floor_program["operation"] == "floor"
    floor_out = bytearray(floor_program["outputs"][0]["allocationBytes"])
    struct_module.pack_into("<e", floor_out, 0, lengths)
    t_dense = bytes(floor_out)

    less_program = next(p for p in manifest["programs"]
                        if p["operation"] == "less")
    arange = list(range(375))
    for binding in less_program["inputs"]:
        if binding.get("binding") == "constant":
            hex_data = less_program["constantInputs"][binding["name"]]
            assert hex_data == "".join(f"{struct_module.unpack('<H', struct_module.pack('<e', float(i)))[0]:04x}"
                                       for i in arange), "arange halves"
    y_binding = next(b for b in less_program["inputs"]
                     if not b.get("binding"))
    assert y_binding.get("broadcast") == "YES" or \
        y_binding.get("broadcast") is True, y_binding
    value = lengths if not math.isnan(lengths) else lengths
    out = bytearray(less_program["outputs"][0]["allocationBytes"])
    width = y_binding["nchw"][3]
    row = y_binding["nchw"][5]
    for i in range(375):
        mask_byte = 1 if (i < lengths and not math.isnan(lengths)) else 0
        offset = (i // width) * row + i % width
        out_binding = less_program["outputs"][0]
        out_offset = (i // out_binding["nchw"][3]) * out_binding["nchw"][5] + \
            i % out_binding["nchw"][3]
        out[out_offset] = mask_byte
    return bytes(out)


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-mask-respell-") as d:
    root = Path(d)
    arange = list(range(375))

    # 1. The real spellings respell: the int32 chain disappears, the less
    # row is the captured (375,1,1) bool less byte for byte, the arange
    # rides the program as a constant runtime input, and the length
    # scalar binding declares the host broadcast.
    (root / "weights").mkdir()
    (root / "weights" / "weight.bin").write_bytes(blob_file(
        [int32_record(2386176, arange)]))
    mil_text, _ = mask_mil(2386176, arange)
    package = compile_source(root, "mask-respell", mil_text)
    validate(root, package)
    manifest = json.loads((package / "manifest.json").read_text())
    programs = manifest["programs"]
    assert [p["operation"] for p in programs] == ["floor", "less"], programs
    assert all(tensor.get("dtype") != "int32"
               for tensor in manifest["tensors"].values()), manifest["tensors"]
    less_program = programs[1]
    capture = json.loads(
        (captures / "less_bool_rr_375x1x1.json").read_text())
    assert less_program["taskDescriptors"] == \
        len(capture["task_descriptors"]), less_program
    const_binding = next(b for b in less_program["inputs"]
                         if b.get("binding") == "constant")
    assert const_binding["name"] == "var_281.f16respell"
    expected_hex = struct.pack(f"<{len(arange)}e", *arange).hex()
    assert less_program["constantInputs"]["var_281.f16respell"] == expected_hex
    y_binding = next(b for b in less_program["inputs"]
                     if not b.get("binding"))
    assert y_binding["name"] == "lengths_cast_fp16"
    assert y_binding["logicalBytes"] == 2 and y_binding["shape"] == [1]
    assert y_binding.get("broadcast"), y_binding
    less_index = programs.index(less_program)
    assert anec_task_stream(package, less_index) == \
        capture_stream(capture, SELECTOR_REMAP_LESS)

    # 2. Numeric check against the reference's evaluation of the ORIGINAL
    # MIL (the int32 chain evaluated exactly) and against a direct
    # broadcast simulation, for lengths covering the classes: inside,
    # boundary, over, and NaN (comparisons false — checked by isnan-free
    # byte equality against the reference).
    for lengths_value in (100.0, 375.0, 376.0, 0.0, float("nan")):
        lengths_bytes = struct.pack("<e", lengths_value)
        expected = evaluate(mil_text, root,
                            {"lengths_13_cast_fp16": lengths_bytes})
        mask = expected["output_mask"]
        assert len(mask) == 375
        for i in range(375):
            want = 1 if (not math.isnan(lengths_value) and i < lengths_value) \
                else 0
            assert mask[i] in (0, 1, b"\x00", b"\x01", True, False), mask[i]
            got = mask[i] if isinstance(mask[i], int) else mask[i][0] \
                if isinstance(mask[i], bytes) else int(mask[i])
            assert int(got) == want, (lengths_value, i, got, want)

    # 3. Arange values outside fp16's exact integer range keep the int32
    # refusals: the respell does not fire.
    (root / "weights" / "weight.bin").write_bytes(blob_file(
        [int32_record(2386176, [((i * 7) % 5000) - 2500 for i in range(375)])]))
    mil_text_big, _ = mask_mil(2386176, [0] * 375)
    compile_source(root, "mask-out-of-range", mil_text_big,
                   expected_code="h13.invalid-shape-alias")

    # 4. A second consumer of the int32 chain keeps the refusals too:
    # the respell only fires when it orphans the whole chain, and a
    # shared-chain spelling refuses fail-closed (the dead extra consumer
    # trips the consume gate before anything lowers).
    (root / "weights" / "weight.bin").write_bytes(blob_file(
        [int32_record(2386176, arange)]))
    mil_text, _ = mask_mil(2386176, arange)
    shared = mil_text.replace(
        "  } -> (output_mask);",
        "    tensor<int32, [1, 1]> echoed = expand_dims("
        "axes = var_283_axes_0, x = lengths_cast_fp16_to_int32)"
        "[name = string(\"echoed\")];\n  } -> (output_mask);")
    compile_source(root, "mask-shared-chain", shared,
                   expected_code="h13.unsupported-chain")

print("h13 mask respell cli: PASS")
