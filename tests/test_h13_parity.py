#!/usr/bin/env python3
"""Compares emitted H13 task streams word-for-word with decoded Apple oracles."""
import collections
import contextlib
import hashlib
import io
import json
import struct
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from h13_td import decode_task, split_h13_tasks  # noqa: E402
from inspect_hwx import h13_anec  # noqa: E402
import mint_oracles  # noqa: E402

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "build/mil-hwxc").resolve()
ORACLES = ROOT / "research/oracles/h13"
RUNTIME_BINARY = {"add", "mul", "maximum", "minimum", "sub"}
UNARY = {"abs", "exp", "gelu", "leaky_relu", "relu", "rsqrt", "sigmoid", "silu",
         "sqrt", "tanh"}
BROADCAST_FAMILIES = {"env_broadcast", "rrmm_broadcast"}
MATMUL_FAMILIES = {"env_matmul", "rrmm_matmul", "rrmm_matvec"}
DEFAULT_ENCODER = "h13-oracle-parity"


def selected_oracles():
    selected = []
    for path in sorted(ORACLES.glob("*.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is not None:
            continue
        parameters = oracle.get("parameters", {})
        operation = parameters.get("operation")
        family = oracle.get("family")
        if family == "binary_runtime" and operation in RUNTIME_BINARY:
            selected.append(oracle)
        elif family == "binary_constant" and parameters.get("constant") == "scalar":
            selected.append(oracle)
        elif family == "unary" and operation in UNARY:
            selected.append(oracle)
        elif family == "matmul":
            selected.append(oracle)
        elif family in ("normalization", "reduction"):
            selected.append(oracle)
        elif family in BROADCAST_FAMILIES:
            selected.append(oracle)
        elif family in MATMUL_FAMILIES and parameters["x_storage"] == "runtime":
            # A constant `x` leaves `y` as the runtime operand; no encoder
            # lowers that form, so it stays outside the envelope.
            selected.append(oracle)
    return selected


def encoder(oracle):
    family = oracle["family"]
    if family in MATMUL_FAMILIES:
        return "apple-parity-matmul" \
            if oracle["parameters"]["w_storage"] == "runtime" \
            else "apple-parity-matvec"
    if family in BROADCAST_FAMILIES:
        return "apple-parity-broadcast"
    return {"matmul": "apple-parity-matvec",
            "normalization": "apple-parity-norm",
            "reduction": "apple-parity-norm"}.get(family, DEFAULT_ENCODER)


def write_weights(oracle, root):
    """Recreates the exact weights.bin the campaign compiled against."""
    description = oracle.get("weights", {})
    if description.get("storage") != "BLOBFILE":
        return
    assert description["value"] == "fp16(0x1p-1)", oracle["case"]
    elements = description["payload_bytes"] // 2
    (root / "weights.bin").write_bytes(mint_oracles.blob(
        mint_oracles.half_payload(elements)))


def compile_oracle(oracle, output, artifact_format):
    mil = output.parent / f"{oracle['case']}-{artifact_format}.mil"
    mil.write_text(oracle["mil"])
    write_weights(oracle, output.parent)
    result = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(output.parent),
         "--target", "H13", "--format", artifact_format, "--output", str(output)],
        capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, \
        f"{oracle['case']} {artifact_format}: {result.stdout}{result.stderr}"
    return json.loads((output / "manifest.json").read_text())


def expected_task_stream_bytes(oracle):
    tasks = oracle["task_descriptors"]
    offset = 0
    for index, task in enumerate(tasks[:-1]):
        offset = int(task["header_words"][7], 16)
        assert offset >= 0, index
    return offset + tasks[-1]["size_bytes"]


def unpack_commands(hwx):
    _, _, _, _, count, command_bytes, _, _ = struct.unpack_from("<8I", hwx)
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
        elif command == 4 and kind == 1:
            program = {
                "code": struct.unpack_from("<I", hwx, cursor + 0x0C)[0],
                "command_size": size,
                "text_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x10)[0]),
                "constant_address": hex(struct.unpack_from("<Q", hwx, cursor + 0x18)[0]),
                "resource_addresses": [
                    hex(struct.unpack_from("<Q", hwx, cursor + 0x30 + index * 8)[0])
                    for index in range(5)],
                "kind": kind,
                "task_words_minus_one": struct.unpack_from("<I", hwx, cursor + 0x818)[0],
                "task_count": struct.unpack_from("<I", hwx, cursor + 0x81C)[0],
            }
        elif command == 4 and kind == 3:
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
    _, first_task_bytes, task_count, task_stream_bytes, constant_bytes, _, _ = \
        struct.unpack_from("<QIIQQII", payload)
    stream = payload[0x1000:0x1000 + task_stream_bytes]
    tasks = split_h13_tasks(stream, first_task_bytes // 4 - 1, task_count)
    constant_offset = (task_stream_bytes + 0x3F) & ~0x3F
    constants = payload[0x1000 + constant_offset:
                        0x1000 + constant_offset + constant_bytes]
    return tasks, constants, constant_offset


def assert_tasks(actual, oracle):
    expected = oracle["task_descriptors"]
    assert len(actual) == len(expected), \
        f"{oracle['case']}: {len(actual)} tasks, oracle has {len(expected)}"
    for index, task in enumerate(actual):
        assert decode_task(task, "h13") == expected[index], \
            f"{oracle['case']} task {index} words differ from the oracle"


def assert_constants(constants, oracle):
    expected = oracle["constant_section"]
    assert len(constants) == expected["size"], oracle["case"]
    assert hashlib.sha256(constants).hexdigest() == expected["sha256"], \
        f"{oracle['case']} constant section differs from the oracle"


def check_anec(oracle, output, manifest):
    assert len(manifest["programs"]) == 1, \
        f"{oracle['case']}: {len(manifest['programs'])} programs, expected 1"
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    payload = (output / record["file"]).read_bytes()
    tasks, constants, constant_offset = anec_contents(payload)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)
    assert record["constantOffset"] == constant_offset, oracle["case"]
    assert constant_offset == (expected_task_stream_bytes(oracle) + 0x3F) & ~0x3F, \
        oracle["case"]


def check_hwx(oracle, output, manifest):
    assert len(manifest["programs"]) == 1, \
        f"{oracle['case']}: {len(manifest['programs'])} programs, expected 1"
    record = manifest["programs"][0]
    assert record["encoder"] == encoder(oracle), oracle["case"]
    payload = (output / record["file"]).read_bytes()
    sections, program, tensors = unpack_commands(payload)
    expected = dict(oracle["program_descriptor"])
    # The envelope campaign records which slice of __TEXT/__text each program
    # descriptor owns; a single-program object owns all of it.
    section = expected.pop("task_section", None)
    assert program == expected, oracle["case"]
    assert section in (None, {"offset": 0,
                              "size": expected_task_stream_bytes(oracle)}), \
        oracle["case"]
    assert tensors == oracle["tensor_descriptors"], oracle["case"]
    assert sections[("__TEXT", "__text")]["size"] == \
        expected_task_stream_bytes(oracle), oracle["case"]
    assert sections[("__TEXT", "__const")]["size"] == \
        oracle["constant_section"]["size"], oracle["case"]
    with contextlib.redirect_stdout(io.StringIO()):
        extracted, _ = h13_anec(payload)
    tasks, constants, _ = anec_contents(extracted)
    assert_tasks(tasks, oracle)
    assert_constants(constants, oracle)


def main():
    oracles = selected_oracles()
    families = collections.Counter(oracle["family"] for oracle in oracles)
    expected = {"binary_runtime": 50, "binary_constant": 12, "unary": 25,
                "matmul": 36, "normalization": 105, "reduction": 114,
                "env_broadcast": 93, "env_matmul": 110}
    for family in ("rrmm_broadcast", "rrmm_matmul", "rrmm_matvec"):
        if families[family]:
            expected[family] = families[family]
    assert families == collections.Counter(expected), \
        f"decoded parity oracles per family: {dict(families)}"
    with tempfile.TemporaryDirectory(prefix="h13-parity-") as directory:
        root = Path(directory)
        for oracle in oracles:
            for artifact_format, check in (("anec", check_anec), ("hwx", check_hwx)):
                output = root / f"{oracle['case']}-{artifact_format}"
                check(oracle, output, compile_oracle(oracle, output, artifact_format))
                # The constant-weight grid reaches 134 MiB per artifact, so a
                # case is discarded as soon as it has been compared.
                shutil.rmtree(output)
            for stale in root.glob(f"{oracle['case']}-*.mil"):
                stale.unlink()
    matmul = families["matmul"] + families["env_matmul"] + \
        families["rrmm_matmul"] + families["rrmm_matvec"]
    broadcast = families["env_broadcast"] + families["rrmm_broadcast"]
    print(f"H13 oracle parity: PASS ({len(oracles)} cases, "
          f"{matmul} matmul, {broadcast} broadcast, "
          f"{families['normalization']} softmax/layer_norm, "
          f"{families['reduction']} reduction, {len(oracles) * 2} artifacts)")


if __name__ == "__main__":
    main()
