#!/usr/bin/env python3
"""Compares emitted H14 task streams word-for-word with decoded Apple oracles."""
import contextlib
import hashlib
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_td import decode_task, split_h14_tasks  # noqa: E402
import inspect_hwx  # noqa: E402
import mint_h14_matvec_probes as probes  # noqa: E402
import mint_oracles  # noqa: E402
from generate_h14_templates import (selected_matvec_oracles,  # noqa: E402
                                    selected_oracles)

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
TILE_BYTES = 0x4000
MATVEC_ENCODER = "apple-parity-matvec"
ELEMENTWISE_ENCODER = "h14-oracle-parity"


def encoder(oracle):
    return ELEMENTWISE_ENCODER if oracle["family"] in {
        "binary_runtime", "binary_constant", "unary"} else MATVEC_ENCODER


def write_weights(oracle, root):
    """Recreates the exact weights.bin Apple's compiler saw for this case."""
    description = oracle.get("weights", {})
    if description.get("storage") != "BLOBFILE":
        return
    parameters = oracle["parameters"]
    if oracle["family"] == "matvec_probe":
        payload = probes.payload(parameters["reduction"], parameters["columns"],
                                 parameters["pattern"])
        assert hashlib.sha256(payload).hexdigest() == \
            description["payload_sha256"], oracle["case"]
    else:
        assert description["value"] == "fp16(0x1p-1)", oracle["case"]
        payload = mint_oracles.half_payload(description["payload_bytes"] // 2)
    (root / "weights.bin").write_bytes(mint_oracles.blob(payload))


def grid_probe_oracles():
    """The known-weight probes whose geometry the parity encoder covers, so the
    packed constant section is proven against a nonuniform Apple weight."""
    covered = {(oracle["parameters"]["rows"], oracle["parameters"]["reduction"],
                oracle["parameters"]["columns"])
               for oracle in selected_matvec_oracles()}
    selected = []
    for path in sorted((ROOT / "research/oracles/h14").glob("h14mv_*.json")):
        oracle = json.loads(path.read_text())
        parameters = oracle["parameters"]
        if oracle.get("error") is None and \
                (parameters["rows"], parameters["reduction"],
                 parameters["columns"]) in covered:
            selected.append(oracle)
    return selected


def compile_oracle(oracle, output, artifact_format):
    mil = output.parent / f"{oracle['case']}-{artifact_format}.mil"
    mil.write_text(oracle["mil"])
    write_weights(oracle, output.parent)
    result = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(output.parent),
         "--target", "H14", "--format", artifact_format, "--output", str(output)],
        capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, \
        f"{oracle['case']} {artifact_format}: {result.stdout}{result.stderr}"
    return json.loads((output / "manifest.json").read_text())


def aligned_blob(payload):
    """A weights.bin whose sub-header fields sit exactly where
    `ANEBlobResolver` reads them, so the resolved constant is `payload` alone.
    `mint_oracles.blob` packs the length and payload offset four bytes lower,
    which is why Apple's campaign sections start with the blob header."""
    data = bytearray(128 + len(payload))
    struct.pack_into("<II", data, 0, 1, 2)
    struct.pack_into("<I", data, 64, 0xDEADBEEF)
    struct.pack_into("<Q", data, 72, len(payload))
    struct.pack_into("<Q", data, 80, 128)
    data[128:] = payload
    return bytes(data)


def check_transposed_weights(root):
    """Apple rejects `transpose_y=false`, so no oracle covers it. Compiling the
    same non-square matrix in both forms must give one identical artifact, and
    its constants must be the packing the [N, K] weight implies."""
    rows, reduction, columns = 2, 512, 256
    weight = bytearray(columns * reduction * 2)
    transposed = bytearray(len(weight))
    for column in range(columns):
        for index in range(reduction):
            bits = 0x3000 | ((column * 131 + index * 17) & 0x3FF)
            struct.pack_into("<H", weight, (column * reduction + index) * 2, bits)
            struct.pack_into("<H", transposed,
                             (index * columns + column) * 2, bits)
    artifacts = {}
    for flag, payload, weight_shape in (
            ("true", weight, (columns, reduction)),
            ("false", transposed, (reduction, columns))):
        model = root / f"transpose-y-{flag}"
        model.mkdir()
        (model / "weights.bin").write_bytes(aligned_blob(bytes(payload)))
        x_kind = f"tensor<fp16, [{rows}, {reduction}]>"
        y_kind = f"tensor<fp16, [{rows}, {columns}]>"
        w_kind = f"tensor<fp16, [{weight_shape[0]}, {weight_shape[1]}]>"
        (model / "model.mil").write_text(mint_oracles.program(f"{x_kind} x", [
            'bool f = const()[name = string("f"), val = bool(false)];',
            f'bool ty = const()[name = string("ty"), val = bool({flag})];',
            f'{w_kind} w = const()[name = string("w"), val = {w_kind}'
            '(BLOBFILE(path = string("@model_path/weights.bin"), '
            'offset = uint64(64)))];',
            f'{y_kind} product = matmul(transpose_x = f, transpose_y = ty, '
            'x = x, y = w)[name = string("product")];'], "product"))
        result = subprocess.run(
            [str(COMPILER), "--mil", str(model / "model.mil"),
             "--model-root", str(model), "--target", "H14", "--format", "anec",
             "--output", str(model / "out")],
            capture_output=True, text=True, timeout=60, check=False)
        assert result.returncode == 0, \
            f"transpose_y={flag}: {result.stdout}{result.stderr}"
        artifacts[flag] = (model / "out/program-0.anec").read_bytes()
    assert artifacts["true"] == artifacts["false"], \
        "transpose_y=false does not reproduce the transpose_y=true artifact"
    _, constants, _, _ = anec_contents(artifacts["true"])
    assert constants == probes.packed_section(reduction, columns, bytes(weight)), \
        "the emitted constants are not the modelled packing of the [N, K] weight"


def unpack_commands(hwx):
    """Decode the H14 fields research/mint_oracles.py records from Apple HWX."""
    magic, _, subtype, _, count, command_bytes, _, _ = struct.unpack_from("<8I", hwx)
    assert magic == 0xBEEFFACE and subtype == 5, (hex(magic), subtype)
    cursor = 32
    end = cursor + command_bytes
    sections = {}
    program = None
    tensors = []
    for _ in range(count):
        command, size = struct.unpack_from("<2I", hwx, cursor)
        kind = struct.unpack_from("<I", hwx, cursor + 8)[0] if size >= 12 else None
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", hwx, cursor)
            segment = fields[2].split(b"\0", 1)[0].decode()
            section_cursor = cursor + 72
            for _ in range(fields[-2]):
                values = struct.unpack_from("<16s16s2Q8I", hwx, section_cursor)
                section = values[0].split(b"\0", 1)[0].decode()
                sections[(segment, section)] = {
                    "address": values[2], "size": values[3], "offset": values[4]}
                section_cursor += 80
        elif command == 4 and kind == 4 and size >= 0x838:
            program = {
                "kind": kind, "command_size": size,
                "text_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x10)[0]),
                "constant_address": hex(
                    struct.unpack_from("<Q", hwx, cursor + 0x20)[0]),
                "resource_addresses": [
                    hex(value) for value in
                    struct.unpack_from("<5Q", hwx, cursor + 0x30)],
                "text_words": struct.unpack_from("<I", hwx, cursor + 0x824)[0],
                "task_count": struct.unpack_from("<I", hwx, cursor + 0x830)[0],
                "trailing_words": [
                    f"0x{value:08x}" for value in struct.unpack_from(
                        f"<{(size - 0x810) // 4}I", hwx, cursor + 0x810)],
            }
        elif command == 4 and kind == 3 and size >= 0x78:
            tensors.append({
                "binding": struct.unpack_from("<I", hwx, cursor + 0x14)[0],
                "element_code": struct.unpack_from("<I", hwx, cursor + 0x24)[0],
                "shape": list(struct.unpack_from("<4I", hwx, cursor + 0x28)),
                "strides": list(struct.unpack_from("<4Q", hwx, cursor + 0x50)),
                "total_bytes": struct.unpack_from("<Q", hwx, cursor + 0x70)[0],
            })
        cursor += size
    assert cursor == end
    return sections, program, tensors


def anec_contents(payload):
    content_bytes, first_task_bytes, task_count, stream_bytes, constant_bytes, \
        inputs, outputs = struct.unpack_from("<QIIQQII", payload)
    stream = payload[0x1000:0x1000 + stream_bytes]
    tasks = split_h14_tasks(stream)
    constant_offset = (stream_bytes + 0x3F) & ~0x3F
    constants = payload[0x1000 + constant_offset:
                        0x1000 + constant_offset + constant_bytes]
    header = {"first_task_bytes": first_task_bytes, "task_count": task_count,
              "stream_bytes": stream_bytes, "inputs": inputs, "outputs": outputs,
              "content_bytes": content_bytes}
    return tasks, constants, constant_offset, header


def assert_tasks(actual, oracle):
    expected = oracle["task_descriptors"]
    assert len(actual) == len(expected), \
        f"{oracle['case']}: {len(actual)} tasks, oracle has {len(expected)}"
    for index, task in enumerate(actual):
        assert decode_task(task, "h14") == expected[index], \
            f"{oracle['case']} task {index} words differ from the oracle"


def assert_constants(constants, oracle):
    expected = oracle["constant_section"]
    assert len(constants) == expected["size"], oracle["case"]
    assert hashlib.sha256(constants).hexdigest() == expected["sha256"], \
        f"{oracle['case']} constant section differs from the oracle"


def check_anec(oracle, output, manifest):
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    payload = (output / record["file"]).read_bytes()
    tasks, constants, constant_offset, header = anec_contents(payload)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)
    text_bytes = oracle["program_descriptor"]["text_words"] * 4
    assert header["stream_bytes"] == text_bytes, oracle["case"]
    assert header["task_count"] == oracle["program_descriptor"]["task_count"], \
        oracle["case"]
    assert header["first_task_bytes"] == oracle["task_descriptors"][0]["size_bytes"], \
        oracle["case"]
    assert header["inputs"] == len(oracle["tensor_descriptors"]) - 1, oracle["case"]
    assert header["outputs"] == 1, oracle["case"]
    assert record["constantOffset"] == constant_offset == \
        (text_bytes + 0x3F) & ~0x3F, oracle["case"]
    assert header["content_bytes"] == constant_offset + len(constants), \
        oracle["case"]


def check_hwx(oracle, output, manifest):
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    path = output / record["file"]
    payload = path.read_bytes()
    sections, program, tensors = unpack_commands(payload)
    assert program == oracle["program_descriptor"], oracle["case"]
    assert tensors == oracle["tensor_descriptors"], oracle["case"]
    text = sections[("__TEXT", "__text")]
    constants = sections[("__TEXT", "__const")]
    assert text["size"] == program["text_words"] * 4, oracle["case"]
    assert constants["size"] == oracle["constant_section"]["size"], oracle["case"]
    assert constants["address"] - text["address"] == record["constantOffset"], \
        oracle["case"]
    # The surface allocations Apple's text address implies: 16 KiB tiles per
    # tensor, starting at the scratch base the descriptor records in slot 4.
    allocated = sum((descriptor["total_bytes"] + TILE_BYTES - 1) // TILE_BYTES
                    * TILE_BYTES for descriptor in tensors)
    assert text["address"] == int(program["resource_addresses"][4], 16) + allocated, \
        oracle["case"]
    assert_tasks(split_h14_tasks(payload[text["offset"]:
                                        text["offset"] + text["size"]]), oracle)
    assert_constants(payload[constants["offset"]:
                             constants["offset"] + constants["size"]], oracle)
    report = io.StringIO()
    with contextlib.redirect_stdout(report):
        inspect_hwx.main(str(path))
    text_report = report.getvalue()
    assert "name=H14 isa=11" in text_report, oracle["case"]
    assert f"h14_tasks count={len(oracle['task_descriptors'])}" in text_report, \
        oracle["case"]


def main():
    elementwise = selected_oracles()
    matvec = selected_matvec_oracles()
    probe = grid_probe_oracles()
    assert len(elementwise) == 87, \
        f"expected 87 decoded elementwise oracles, found {len(elementwise)}"
    assert len(matvec) == 36, \
        f"expected 36 decoded matvec oracles, found {len(matvec)}"
    grid = {(oracle["parameters"]["reduction"], oracle["parameters"]["columns"])
            for oracle in probe}
    assert grid == {(reduction, columns)
                    for reduction in probes.GRID_SIDES
                    for columns in probes.GRID_SIDES}, sorted(grid)
    oracles = elementwise + matvec + probe
    with tempfile.TemporaryDirectory(prefix="h14-parity-") as directory:
        root = Path(directory)
        for oracle in oracles:
            for artifact_format, check in (("anec", check_anec), ("hwx", check_hwx)):
                output = root / f"{oracle['case']}-{artifact_format}"
                manifest = compile_oracle(oracle, output, artifact_format)
                assert len(manifest["programs"]) == 1, \
                    f"{oracle['case']}: {len(manifest['programs'])} programs"
                check(oracle, output, manifest)
        check_transposed_weights(root)
    probes.verify()
    print(f"H14 oracle parity: PASS ({len(oracles)} cases, "
          f"{len(elementwise)} elementwise, {len(matvec)} matvec, "
          f"{len(probe)} known-weight matvec probes over {len(grid)} (K, N) "
          f"grid points, {len(oracles) * 2} artifacts)")


if __name__ == "__main__":
    main()
