#!/usr/bin/env python3
"""floor_div(x, exact fp16 1.0) respell and the -inf select fill promotion.

Two encoder respells on the decoded boolean rows:

1. floor_div(x, fp16 1.0) == floor(x) universally — dividing by fp16 1.0
   is the IEEE identity for every input class (values, -0, +/-inf, NaN
   carry unchanged; no rounding), so flooring the identity is the original
   operation. The divisor operand must drop at the operand-construction
   point: the Floor program binds x only, and a retained y desyncs the
   program-input walk (the vector range crash this suite pins). The
   captured scalar-2.0 floor_div rows must stay byte-identical, and every
   other divisor keeps its refusal.

2. A rank-0 fp16 -inf const select fill is packing-invariant — every lane
   materializes to the same half (0xFC00) — so it promotes onto the
   runtime-a select rows with the fill bound as a constant runtime input.
   The captured const-a row's retained section cannot discriminate the
   packing for arbitrary constants, so any other constant keeps its
   refusal.

The numeric pipeline check runs the compiled programs in pure Python
(value classes, not just bytes): negative fractional, -0 (sign bit),
+/-inf, and NaN — NaN by isnan, never by equality. This is a host
simulation of the decoded rows' semantics; device numerics stay a
hardware-window claim (t6001-test-host), exactly like the LN-peel numeric bundle.
"""
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else
                    ROOT / "build/mil-hwxc").resolve())
inspector = str(ROOT / "research" / "inspect_anec.py")
captures = ROOT / "research/oracles/h13/boolean"

SELECTOR_REMAP = {
    "floor": {5: 4, 4: 5, 6: 6, 7: 7},
    "floor_div": {5: 4, 4: 5, 6: 6, 7: 7},
    "select": {7: 4, 4: 5, 5: 6, 6: 7},
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


def deterministic(root, name, text):
    first = compile_source(root, f"{name}-a", text)
    second = compile_source(root, f"{name}-b", text)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()}
    return first


def validate(root, package):
    result = subprocess.run([sys.executable, inspector, str(package)],
                            capture_output=True, text=True, timeout=60,
                            check=False)
    assert result.returncode == 0, result.stderr


def anec_task_stream(package, index=0):
    data = (package / f"program-{index}.anec").read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    return data[0x1000:0x1000 + size]


def emitted_constants(package, index=0):
    data = (package / f"program-{index}.anec").read_bytes()
    task_size = struct.unpack_from("<Q", data, 16)[0]
    consts_off = (task_size + 127) // 128 * 128
    consts_size = struct.unpack_from("<Q", data, 24)[0]
    return data[0x1000 + consts_off:0x1000 + consts_off + consts_size]


def wrapped(body, inputs, result="y"):
    return ("program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
            f"  func main<ios18>({inputs}) {{\n" + body +
            f"  }} -> ({result});\n}}\n")


def unit_divisor(shape="[]"):
    return (f'    tensor<fp16, {shape}> one = const()[name = string("one"), '
            f'val = tensor<fp16, {shape}>(fp16(1.0))];\n')


def blob_const(name, decl_type, blob_literal):
    return (f"    {decl_type} {name} = const()"
            f'[name = string("{name}"), val = {blob_literal}];\n')


def f16_bits(value):
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def bits_f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def floor_class(value):
    """IEEE floor with the value classes carried: -0 keeps its sign,
    +/-inf and NaN pass through."""
    if math.isnan(value) or math.isinf(value) or value == 0.0:
        return value
    return float(math.floor(value))


def pack_fp16_surface(binding, dense):
    """Dense fp16 bytes -> the binding's physical surface."""
    width, row = binding["nchw"][3], binding["nchw"][5]
    physical = bytearray(binding["allocationBytes"])
    for element in range(len(dense) // 2):
        offset = (element // width) * row + (element % width) * 2
        physical[offset:offset + 2] = dense[element * 2:element * 2 + 2]
    return bytes(physical)


def unpack_fp16_surface(binding, physical):
    width, row = binding["nchw"][3], binding["nchw"][5]
    dense = bytearray(binding["logicalBytes"])
    for element in range(len(dense) // 2):
        offset = (element // width) * row + (element % width) * 2
        dense[element * 2:element * 2 + 2] = physical[offset:offset + 2]
    return bytes(dense)


def pack_bool_surface(binding, dense):
    width, row = binding["nchw"][3], binding["nchw"][5]
    physical = bytearray(binding["allocationBytes"])
    for element in range(len(dense)):
        physical[(element // width) * row + element % width] = dense[element]
    return bytes(physical)


def fp16_lane_offset(binding, element):
    # nchw is [N, C, H, W, plane, row]: W lanes per row, each row 64-byte
    # aligned (the (64,1,1) surfaces hold W=1, so one lane per row).
    width, row = binding["nchw"][3], binding["nchw"][5]
    return (element // width) * row + (element % width) * 2


def bool_lane_offset(binding, element):
    width, row = binding["nchw"][3], binding["nchw"][5]
    return (element // width) * row + (element % width)


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-fdiv-respell-") as d:
    root = Path(d)

    # 1. The respell is byte-identical to the direct floor spelling at
    # every decoded floor geometry, and rides the captured floor rows.
    for rank_shape, chw in ((("[1]"), (1, 1, 1)), (("[1, 64]"), (64, 1, 1)),
                            (("[1, 512]"), (512, 1, 1))):
        fdiv = wrapped(unit_divisor() +
                       f'    tensor<fp16, {rank_shape}> y = floor_div('
                       f'x = x, y = one)[name = string("y")];\n',
                       f"tensor<fp16, {rank_shape}> x")
        direct = wrapped(f'    tensor<fp16, {rank_shape}> y = floor('
                         f'x = x)[name = string("y")];\n',
                         f"tensor<fp16, {rank_shape}> x")
        fdiv_package = deterministic(root, f"respell-{chw[0]}", fdiv)
        floor_package = deterministic(root, f"direct-{chw[0]}", direct)
        assert anec_task_stream(fdiv_package) == \
            anec_task_stream(floor_package), chw
        fdiv_manifest = json.loads(
            (fdiv_package / "manifest.json").read_text())
        program = fdiv_manifest["programs"][0]
        assert program["operation"] == "floor_div"
        assert program["encoder"] == "apple-parity-boolean"
        assert len(program["inputs"]) == 1, program["inputs"]
        capture = json.loads(
            (captures / f"floor_r_{'1' if chw[0] == 1 else f'1x{chw[0]}x1x1'}.json")
            .read_text())
        assert anec_task_stream(fdiv_package) == \
            capture_stream(capture, SELECTOR_REMAP["floor"]), chw
        validate(root, fdiv_package)

    # 2. The scalar-2.0 floor_div rows keep their exact captured bytes.
    for stem in ("floor_div_native_1_s2", "floor_div_native_1x64x1x1_s2"):
        record = json.loads((captures / f"{stem}.json").read_text())
        package = deterministic(root, f"div2-{stem}", record["mil"])
        manifest = json.loads((package / "manifest.json").read_text())
        program = manifest["programs"][0]
        assert program["taskDescriptors"] == len(record["task_descriptors"])
        assert anec_task_stream(package) == \
            capture_stream(record, SELECTOR_REMAP["floor_div"]), stem
        assert emitted_constants(package) == \
            (captures / f"{stem}.const.bin").read_bytes(), stem
        validate(root, package)

    # 3. Unsupported divisors keep the exact refusal: 3.0 in both scalar
    # spellings, a non-fp16 dtype, a shaped tensor const, and a blob-backed
    # divisor.
    for name, divisor in (
            ("div3-scalar", "const()[name = string(\"d\"), val = fp16(3.0)]"),
            ("div3-tensor", "const()[name = string(\"d\"), "
                            "val = tensor<fp16, []>(fp16(3.0))]")):
        body = f"    tensor<fp16, []> d = {divisor};\n"
        body += ("    tensor<fp16, [1]> y = floor_div(x = x, y = d)"
                 '[name = string("y")];\n')
        compile_source(root, name, wrapped(body, "tensor<fp16, [1]> x"),
                       expected_code="h13.invalid-constant-input",
                       expected_message="floor_div lowers the captured scalar-2.0 divisor only")
    body = ('    tensor<fp32, []> d = const()[name = string("d"), '
            'val = fp32(1.0)];\n'
            '    tensor<fp16, [1]> y = floor_div(x = x, y = d)'
            '[name = string("y")];\n')
    compile_source(root, "div-fp32", wrapped(body, "tensor<fp16, [1]> x"),
                   expected_code="h13.invalid-constant-input")
    payload = struct.pack("<e", 3.0)
    header = struct.pack("<IxxxxQQ", 0xDEADBEEF, len(payload), 64 + 24)
    (root / "weights.bin").write_bytes(b"\0" * 64 + header + payload)
    body = blob_const("d", "tensor<fp16, []>",
                      "tensor<fp16, []>(BLOBFILE("
                      "path = string(\"@model_path/weights.bin\"), "
                      "offset = uint64(64)))")
    body += ("    tensor<fp16, [1]> y = floor_div(x = x, y = d)"
             '[name = string("y")];\n')
    compile_source(root, "div3-blob", wrapped(body, "tensor<fp16, [1]> x"),
                   expected_code="h13.invalid-constant-input")

    # 3b. The real encoder carries the divisor as a rank-0 fp16 BLOBFILE
    # constant (normalized.bin; Main pins payload 1.0). The respell
    # resolves it through the verified blob reader at the real offsets —
    # 576/1216 are the actual var_23 divisor records in the pinned graph,
    # 704/1344 the normalized.bin pair Main named — and stays byte-
    # identical to the direct floor spelling. A blob payload of anything
    # else keeps its refusal.
    one = struct.pack("<e", 1.0)
    two = struct.pack("<e", 2.0)
    three = struct.pack("<e", 3.0)

    def blob_file(records):
        # Records at absolute offsets, resolver layout: 24-byte header at
        # the BLOBFILE offset, payload at offset + 64.
        size = max(off for off, _ in records) + 64 + 2
        data = bytearray(size)
        for off, payload in records:
            data[off:off + 24] = struct.pack("<IIQQ", 0xDEADBEEF, 1,
                                             len(payload), off + 64)
            data[off + 64:off + 66] = payload
        return bytes(data)

    (root / "normalized.bin").write_bytes(blob_file(
        [(off, one) for off in (576, 1216, 704, 1344)]))
    for off in (576, 1216, 704, 1344):
        body = blob_const("d", "tensor<fp16, []>",
                          f"tensor<fp16, []>(BLOBFILE("
                          f"path = string(\"@model_path/normalized.bin\"), "
                          f"offset = uint64({off})))")
        body += ("    tensor<fp16, [1]> y = floor_div(x = x, y = d)"
                 '[name = string("y")];\n')
        package = deterministic(root, f"div1-blob-{off}",
                                wrapped(body, "tensor<fp16, [1]> x"))
        floor_package = deterministic(
            root, f"div1-blob-floor-{off}",
            wrapped('    tensor<fp16, [1]> y = floor(x = x)'
                    '[name = string("y")];\n',
                    "tensor<fp16, [1]> x"))
        assert anec_task_stream(package) == \
            anec_task_stream(floor_package), off
        manifest = json.loads((package / "manifest.json").read_text())
        assert len(manifest["programs"][0]["inputs"]) == 1, off
        validate(root, package)
    (root / "normalized2.bin").write_bytes(blob_file(
        [(704, two), (1344, three)]))
    for off in (704, 1344):
        body = blob_const("d", "tensor<fp16, []>",
                          f"tensor<fp16, []>(BLOBFILE("
                          f"path = string(\"@model_path/normalized2.bin\"), "
                          f"offset = uint64({off})))")
        body += ("    tensor<fp16, [1]> y = floor_div(x = x, y = d)"
                 '[name = string("y")];\n')
        compile_source(root, f"div-blob-refuse-{off}",
                       wrapped(body, "tensor<fp16, [1]> x"),
                       expected_code="h13.invalid-constant-input",
                       expected_message="floor_div lowers the captured scalar-2.0 divisor only")

    # 4. Pipeline: floor_div(x, 1.0) -> t; select(-inf fill, t, cond) -> y.
    # The -inf fill is a rank-0 fp16 blob constant promoted onto the
    # runtime-a select row; the fill rides the program as a constant
    # runtime input.
    pipeline = wrapped(
        unit_divisor() +
        '    tensor<fp16, [1, 64, 1, 1]> t = floor_div(x = x, y = one)'
        '[name = string("t")];\n' +
        blob_const("neg_inf", "tensor<fp16, []>",
                   "tensor<fp16, []>(BLOBFILE("
                   "path = string(\"@model_path/weights.bin\"), "
                   "offset = uint64(64)))") +
        '    tensor<fp16, [1, 64, 1, 1]> y = select(a = neg_inf, b = t, '
        'cond = m)[name = string("y")];\n',
        "tensor<fp16, [1, 64, 1, 1]> x, tensor<bool, [1, 64, 1, 1]> m")
    fill = struct.pack("<e", float("-inf"))
    (root / "weights.bin").write_bytes(
        b"\0" * 64 + struct.pack("<IxxxxQQ", 0xDEADBEEF, len(fill), 64 + 24)
        + fill)
    package = deterministic(root, "pipeline", pipeline)
    validate(root, package)
    manifest = json.loads((package / "manifest.json").read_text())
    programs = manifest["programs"]
    fdiv_program = programs[0]
    select_index = next(i for i, p in enumerate(programs)
                        if p["operation"] == "select")
    select_program = programs[select_index]
    assert fdiv_program["operation"] == "floor_div"
    assert fdiv_program["encoder"] == "apple-parity-boolean"
    assert len(fdiv_program["inputs"]) == 1
    assert select_program["encoder"] == "apple-parity-boolean"
    assert select_program["inputs"][0].get("binding") == "constant"
    assert select_program["constantInputs"]["neg_inf"] == "00fc" * 64
    assert manifest["tensors"]["neg_inf"]["role"] == "constant"
    rrb = json.loads(
        (captures / "select_rrb_1x64x1x1.json").read_text())
    assert anec_task_stream(package, select_index) == \
        capture_stream(rrb, SELECTOR_REMAP["select"])

    # Numeric pipeline simulation over the compiled programs: value
    # classes through the respell and the promoted fill.
    class_lanes = [-2.7, -0.5, -0.0, 0.0, 0.5, 1.9, -1.2, 2048.0, -2048.0,
                   2047.5, float("inf"), float("-inf"), float("nan"), 3.99]
    # The device sees fp16 inputs: round first (2047.5 becomes 2048.0),
    # so every floor below is exact.
    x_lanes = [bits_f16(f16_bits(v)) for v in class_lanes]
    x_lanes = [x_lanes[i % len(x_lanes)] for i in range(64)]
    cond = [i % 3 == 0 for i in range(64)]  # both polarities across classes
    x_dense = b"".join(struct.pack("<e", v) for v in x_lanes)
    cond_dense = bytes(1 if m else 0 for m in cond)

    fdiv_in = pack_fp16_surface(fdiv_program["inputs"][0], x_dense)
    fdiv_out_binding = fdiv_program["outputs"][0]
    t_physical = pack_fp16_surface(
        fdiv_out_binding,
        b"".join(struct.pack("<e", floor_class(v)) for v in x_lanes))
    # the simulated floor row must reproduce that surface lane for lane
    t_lanes = [bits_f16(struct.unpack_from(
                   "<H", fdiv_in, fp16_lane_offset(fdiv_program["inputs"][0], i))[0])
               for i in range(64)]
    t_lanes = [floor_class(v) for v in t_lanes]
    t_dense = unpack_fp16_surface(fdiv_out_binding, t_physical)
    assert t_dense == b"".join(struct.pack("<e", v) for v in t_lanes)

    select_in = []
    for binding in select_program["inputs"]:
        if binding.get("binding") == "constant":
            raw = bytes.fromhex(select_program["constantInputs"][binding["name"]])
            select_in.append(pack_fp16_surface(binding, raw))
        elif binding["dtype"] == "bool":
            select_in.append(pack_bool_surface(binding, cond_dense))
        else:
            select_in.append(t_physical)
    a_surface, b_surface, cond_surface = select_in
    out_binding = select_program["outputs"][0]
    cond_binding = select_program["inputs"][2]
    expected = []
    out_physical = bytearray(out_binding["allocationBytes"])
    for i in range(64):
        taken = a_surface if cond_surface[bool_lane_offset(cond_binding, i)] \
            else b_surface
        offset = fp16_lane_offset(out_binding, i)
        expected.append(bits_f16(struct.unpack_from("<H", taken, offset)[0]))
        out_physical[offset:offset + 2] = struct.pack("<e", expected[-1])

    # Class assertions: NaN by isnan (NaN != NaN), -0 by sign bit, the
    # -inf fill lanes by identity, inf passthrough by equality.
    nan_positions = [i for i, v in enumerate(x_lanes) if math.isnan(v)]
    assert nan_positions, "pipeline must exercise NaN"
    for i in nan_positions:
        # the respell carries NaN through to the intermediate, by isnan
        assert math.isnan(t_lanes[i]), i
        if not cond[i]:
            assert math.isnan(expected[i]), i
        else:
            assert expected[i] == float("-inf"), i
    assert any(cond[i] for i in range(64))
    neg_zero_positions = [i for i, v in enumerate(x_lanes)
                          if v == 0.0 and struct.pack("<e", v)[1] & 0x80]
    assert neg_zero_positions, "pipeline must exercise -0"
    for i in neg_zero_positions:
        # the respell itself carries the sign through to the intermediate
        assert struct.pack("<e", t_lanes[i]) == b"\x00\x80", i
        if not cond[i]:
            assert struct.pack("<e", expected[i]) == b"\x00\x80", i
    inf_positions = [i for i, v in enumerate(x_lanes)
                     if math.isinf(v) and v > 0]
    for i in inf_positions:
        if not cond[i]:
            assert expected[i] == float("inf"), i
    fill_positions = [i for i in range(64) if cond[i]]
    assert fill_positions, "pipeline must exercise the -inf fill"
    for i in fill_positions:
        assert expected[i] == float("-inf"), i
    negative_fraction = [i for i, v in enumerate(x_lanes)
                         if v < 0 and not math.isinf(v)
                         and not math.isnan(v) and v != int(v)]
    assert negative_fraction, "pipeline must exercise negative fractions"
    for i in negative_fraction:
        if not cond[i]:
            assert expected[i] == float(math.floor(x_lanes[i])), i
    simulated = unpack_fp16_surface(out_binding, bytes(out_physical))
    assert simulated == b"".join(struct.pack("<e", v) for v in expected)

    # 5. Non-(-inf) and shaped constants keep the exact promotion refusal.
    body = blob_const("fill", "tensor<fp16, []>", "fp16(0.0)")
    body += ('    tensor<fp16, [1, 64, 1, 1]> y = select(a = fill, b = b, '
             'cond = m)[name = string("y")];\n')
    compile_source(root, "select-zero-fill",
                   wrapped(body, "tensor<fp16, [1, 64, 1, 1]> b, "
                                 "tensor<bool, [1, 64, 1, 1]> m"),
                   expected_code="h13.select-needs-decoded-encoder",
                   expected_message="materializes the fill as a runtime constant input")
    shaped_fill = struct.pack("<4e", float("-inf"), 1.0, float("-inf"), 2.0)
    (root / "weights.bin").write_bytes(
        b"\0" * 64 +
        struct.pack("<IxxxxQQ", 0xDEADBEEF, len(shaped_fill), 64 + 24) +
        shaped_fill)
    body = blob_const("fill", "tensor<fp16, [4]>",
                      "tensor<fp16, [4]>(BLOBFILE("
                      "path = string(\"@model_path/weights.bin\"), "
                      "offset = uint64(64)))")
    body += ('    tensor<fp16, [1, 64, 1, 1]> y = select(a = fill, b = b, '
             'cond = m)[name = string("y")];\n')
    compile_source(root, "select-shaped-fill",
                   wrapped(body, "tensor<fp16, [1, 64, 1, 1]> b, "
                                 "tensor<bool, [1, 64, 1, 1]> m"),
                   expected_code="h13.select-needs-decoded-encoder")

    # 6. The real encoder statements, verbatim spellings and offsets from
    # the pinned encoder graph, each as its own graph (the real graph
    # consumes floor_div through the int32 mask chain, which stays blocked
    # on the cast family): floor_div with the var_23 divisor record
    # (normalized.bin offset 576, payload 1.0 per Main), and the
    # attention-mask select with the var_8 -inf fill record (offset
    # 31460800) against a bool cond at [1, 8, 375, 375]. Real blob
    # offsets, both respells.
    (root / "encoder_normalized.bin").write_bytes(blob_file(
        [(576, one), (1216, one),
         (31460800, struct.pack("<e", float("-inf"))),
         (33558272, struct.pack("<e", float("-inf")))]))
    fdiv_body = blob_const(
        "var_23_promoted_to_fp16", "tensor<fp16, []>",
        "tensor<fp16, []>(BLOBFILE("
        "path = string(\"@model_path/encoder_normalized.bin\"), "
        "offset = uint64(576)))") + \
        '    tensor<fp16, [1]> y = floor_div(x = x, ' \
        'y = var_23_promoted_to_fp16)' \
        '[name = string("floor_div_1_cast_fp16")];\n'
    package = deterministic(root, "encoder-fdiv-statement",
                            wrapped(fdiv_body, "tensor<fp16, [1]> x"))
    validate(root, package)
    manifest = json.loads((package / "manifest.json").read_text())
    assert manifest["programs"][0]["operation"] == "floor_div"
    assert len(manifest["programs"][0]["inputs"]) == 1
    select_body = blob_const(
        "var_8_to_fp16", "tensor<fp16, []>",
        "tensor<fp16, []>(BLOBFILE("
        "path = string(\"@model_path/encoder_normalized.bin\"), "
        "offset = uint64(31460800)))") + \
        '    tensor<fp16, [1, 8, 375, 375]> y = select(' \
        'a = var_8_to_fp16, b = b, cond = var_373)' \
        '[name = string("attention_mask_9_cast_fp16")];\n'
    package = deterministic(root, "encoder-select-statement",
                            wrapped(select_body,
                                    "tensor<fp16, [1, 8, 375, 375]> b, "
                                    "tensor<bool, [1, 8, 375, 375]> var_373"))
    validate(root, package)
    manifest = json.loads((package / "manifest.json").read_text())
    select_program = manifest["programs"][0]
    assert select_program["inputs"][0].get("binding") == "constant"
    assert select_program["constantInputs"]["var_8_to_fp16"] == \
        "00fc" * 1125000
    rrb_375 = json.loads(
        (captures / "select_rrb_1x8x375x375.json").read_text())
    assert anec_task_stream(package) == \
        capture_stream(rrb_375, SELECTOR_REMAP["select"])

print("h13 floor_div respell cli: PASS")
