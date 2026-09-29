#!/usr/bin/env python3
"""Generate the H14 elementwise parity templates from the decoded oracles.

Writes `plugins/H14/H14ElementwiseTemplates.inc` from
`research/oracles/h14/*.json`. Only decoded task words, program-descriptor
metadata, and constant-section contents are emitted; no HWX container bytes
are read or stored.

Regenerate and compare with:

```sh
python3 research/generate_h14_templates.py --check
```
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from h13_td import decode_task  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / "research/oracles/h14"
OUTPUT = ROOT / "plugins/H14/H14ElementwiseTemplates.inc"
MATVEC_OUTPUT = ROOT / "plugins/H14/H14MatvecTemplates.inc"
ISLAND_OUTPUT = ROOT / "plugins/H14/H14IslandTemplates.inc"
CONSTANT_ALIGNMENT = 0x40

# Task-stream framing: a 16-byte zero prefix that decodes as a zero-size task,
# then every task 16-byte aligned with the last task left unpadded.
TASK_ALIGNMENT = 16
STREAM_PREFIX_BYTES = 16

BINARY_OPERATIONS = {"add": 0, "mul": 1, "maximum": 2, "minimum": 3, "sub": 4,
                     "real_div": 5}
UNARY_OPERATIONS = {"abs": 0, "exp": 1, "gelu": 2, "leaky_relu": 3, "relu": 4,
                    "rsqrt": 5, "sigmoid": 6, "silu": 7, "sqrt": 8, "tanh": 9,
                    # Apple emits a different program per gelu mode, so each
                    # approximation is its own encoder operation.
                    "gelu_sigmoid_approximation": 10,
                    "gelu_tanh_approximation": 11}
GELU_MODES = {"EXACT": "gelu",
              "SIGMOID_APPROXIMATION": "gelu_sigmoid_approximation",
              "TANH_APPROXIMATION": "gelu_tanh_approximation"}
RUNTIME_BINARY = {"add", "mul", "maximum", "minimum", "sub", "real_div"}
UNARY = set(UNARY_OPERATIONS)
KINDS = {"binary_runtime": "BinaryRuntime", "binary_constant": "BinaryScalar",
         "unary": "Unary"}


def classify(oracle: dict[str, Any]) -> tuple[str, str, tuple[int, int, int],
                                              tuple[int, int, int]] | None:
    """The encoder key of a decoded elementwise oracle — kind, operation, the
    CHW result surface, and the CHW surface of the second operand — or None
    when the case sits outside what the H14 encoder models.

    Two envelope forms stay outside: a batched surface (`N > 1`), whose
    descriptor records one batch's bytes against a rank-4 shape the encoder's
    surface formula does not spell, and a `BLOBFILE` broadcast operand, whose
    constant section is not the operand payload (see research/h14-td-fields.md).
    """
    if oracle.get("error") is not None:
        return None
    parameters = oracle.get("parameters", {})
    operation = parameters.get("operation")
    family = oracle.get("family")
    shape = parameters.get("shape")
    if family not in ("binary_runtime", "binary_constant", "unary",
                      "env_activation", "env_broadcast") or not shape:
        return None
    if shape[0] != 1:
        return None
    chw = (shape[1], shape[2], shape[3])
    if family in ("unary", "env_activation"):
        if operation == "gelu":
            operation = GELU_MODES.get(parameters.get("mode") or "EXACT")
        return ("Unary", operation, chw, chw) if operation in UNARY else None
    if family == "binary_constant":
        return ("BinaryScalar", operation, chw, chw) \
            if parameters.get("constant") == "scalar" and \
            operation in BINARY_OPERATIONS else None
    if operation not in RUNTIME_BINARY:
        return None
    if family == "binary_runtime":
        return ("BinaryRuntime", operation, chw, chw)
    if parameters["output_shape"] != shape:
        raise ValueError(f"{oracle['case']}: broadcast result is not the x shape")
    operand = parameters.get("operand")
    if operand == "scalar":
        return ("BinaryScalar", operation, chw, chw)
    if operand != "runtime":
        return None
    operand_shape = parameters["operand_shape"]
    if operand_shape[0] != 1:
        return None
    return ("BinaryRuntime", operation, chw,
            (operand_shape[1], operand_shape[2], operand_shape[3]))


def selected_oracles() -> list[dict[str, Any]]:
    """Every decoded H14 case the elementwise encoder covers: the shipped
    binary, scalar-constant, and unary campaign plus the envelope campaign's
    activation shapes and runtime broadcast forms."""
    return [oracle for oracle in
            (json.loads(path.read_text()) for path in sorted(ORACLES.glob("*.json")))
            if classify(oracle) is not None]


def task_words(task: dict[str, Any]) -> list[int]:
    """Rebuild one task's exact word stream from its decoded records."""
    words = [int(word, 16) for word in task["header_words"]]
    registers = {int(address, 16): int(value, 16)
                 for block in task["blocks"].values()
                 for address, value in block["words"].items()}
    for record in task["records"]:
        header = int(record["header"], 16)
        base = (header & 0x7FFF) * 4
        if header & 0x80000000:
            mask = (header >> 15) & 0xFFFF
            addresses = (base,) + tuple(
                base + (bit + 1) * 4 for bit in range(16) if mask & (1 << bit))
        else:
            addresses = tuple(
                base + index * 4 for index in range(((header >> 15) & 0x3F) + 1))
        if len(addresses) != record["count"]:
            raise ValueError(f"record {record['header']} address count differs")
        words.append(header)
        words.extend(registers[address] for address in addresses)
    stream = struct.pack(f"<{len(words)}I", *words)
    if len(stream) != task["size_bytes"] or decode_task(stream, "h14") != task:
        raise ValueError("rebuilt task does not decode back to the oracle")
    return words


def task_stream(oracle: dict[str, Any]) -> list[int]:
    """Assemble the __TEXT/__text word stream for one oracle case."""
    words = [0] * (STREAM_PREFIX_BYTES // 4)
    tasks = oracle["task_descriptors"]
    for index, task in enumerate(tasks):
        words.extend(task_words(task))
        if index + 1 != len(tasks):
            while len(words) % (TASK_ALIGNMENT // 4):
                words.append(0)
    declared = oracle["program_descriptor"]["text_words"]
    if len(words) != declared:
        raise ValueError(
            f"{oracle['case']}: assembled {len(words)} text words, program "
            f"descriptor declares {declared}")
    return words


def constant_runs(oracle: dict[str, Any]) -> list[tuple[int, int, int]]:
    """Nonzero fp16 halfword runs (index, bits, count) of the constant section."""
    section = oracle["constant_section"]
    halfwords: dict[int, int] = {}
    if section["nonzero_bytes"]:
        words = section["nonzero_fp16_words"]
        if words is None:
            # Oracles above 256 bytes keep only hashes; the scalar real_div
            # divisor reciprocal splat is the one such parity case.
            elements = oracle["parameters"]["shape"][1]
            for index in range(elements):
                halfwords[elements + index] = 0x4000
        else:
            for word in words:
                value = word["value"]
                if value == "nan":
                    bits = 0x7E00
                elif value == "+inf":
                    bits = 0x7C00
                elif value == "-inf":
                    bits = 0xFC00
                else:
                    bits = struct.unpack("<H", struct.pack("<e", value))[0]
                halfwords[word["index"]] = bits
    data = bytearray(section["size"])
    for index, bits in halfwords.items():
        struct.pack_into("<H", data, index * 2, bits)
    digest = hashlib.sha256(data).hexdigest()
    if digest != section["sha256"]:
        raise ValueError(
            f"{oracle['case']}: rebuilt constants hash {digest} but the oracle "
            f"records {section['sha256']}")
    runs: list[tuple[int, int, int]] = []
    for index in sorted(halfwords):
        bits = halfwords[index]
        if runs and runs[-1][1] == bits and runs[-1][0] + runs[-1][2] == index:
            runs[-1] = (runs[-1][0], bits, runs[-1][2] + 1)
        else:
            runs.append((index, bits, 1))
    return runs


def word_rows(words: list[int], indent: str = "    ") -> str:
    rows = []
    for start in range(0, len(words), 8):
        row = ", ".join(f"0x{word:08x}" for word in words[start:start + 8])
        rows.append(f"{indent}{row},")
    return "\n".join(rows)


def template_fields(oracle: dict[str, Any]) -> tuple:
    """Everything the emitted template carries, so two oracles that share an
    encoder key are proven to carry identical Apple output."""
    trailer = oracle["program_descriptor"]["trailing_words"]
    if int(trailer[18], 16):
        raise ValueError(
            f"{oracle['case']}: descriptor word 0x858 is "
            f"{trailer[18]}, which the elementwise encoder does not carry")
    return (tuple(task_stream(oracle)), tuple(constant_runs(oracle)),
            oracle["constant_section"]["size"],
            oracle["program_descriptor"]["task_count"],
            int(trailer[20], 16), int(trailer[28], 16))


def generate() -> str:
    templates: dict[tuple, tuple[dict[str, Any], tuple]] = {}
    for oracle in selected_oracles():
        key = classify(oracle)
        fields = template_fields(oracle)
        previous = templates.setdefault(key, (oracle, fields))
        if previous[1] != fields:
            raise ValueError(
                f"{oracle['case']} and {previous[0]['case']} share the encoder "
                f"key {key} but Apple emitted different programs")
    lines = ["// Generated from decoded H14 oracle task words by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, key in enumerate(templates):
        oracle, (stream, runs, constantBytes, taskCount, records,
                 unresolved) = templates[key]
        kind, operation, shape, operand = key
        code = (UNARY_OPERATIONS if kind == "Unary"
                else BINARY_OPERATIONS)[operation]
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t kH14Text{index}[] = {{")
        lines.append(word_rows(list(stream)))
        lines.append("};")
        constants = "nullptr, 0"
        if runs:
            lines.append(f"static constexpr ConstantRun kH14Constants{index}[] = {{")
            for start, bits, count in runs:
                lines.append(f"    {{{start}, 0x{bits:04x}, {count}}},")
            lines.append("};")
            constants = (f"kH14Constants{index}, "
                         f"std::size(kH14Constants{index})")
        lines.append("")
        entries.append(
            f"    {{ElementwiseKind::{kind}, {code}, "
            f"{{{shape[0]}, {shape[1]}, {shape[2]}}}, "
            f"{{{operand[0]}, {operand[1]}, {operand[2]}}}, kH14Text{index}, "
            f"std::size(kH14Text{index}), {taskCount}, "
            f"{constants}, {constantBytes}, "
            f"0x{records:08x}, 0x{unresolved:08x}}},")
    lines.append("static constexpr OracleTaskTemplate kElementwiseTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def selected_matvec_oracles() -> list[dict[str, Any]]:
    """The H14 `transpose_y=true` matmul geometries Apple decoded, ordered by
    rows, reduction, then columns."""
    selected = []
    for path in sorted(ORACLES.glob("matmul_m*_ty1.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is None and oracle["family"] == "matmul":
            selected.append(oracle)
    selected.sort(key=lambda item: (item["parameters"]["rows"],
                                    item["parameters"]["reduction"],
                                    item["parameters"]["columns"]))
    return selected


def generate_matvec() -> str:
    oracles = selected_matvec_oracles()
    lines = ["// Generated from decoded H14 matmul oracle task words by",
             "// research/generate_h14_templates.py. No HWX container bytes.",
             ""]
    entries = []
    for index, oracle in enumerate(oracles):
        parameters = oracle["parameters"]
        descriptor = oracle["program_descriptor"]
        rows = parameters["rows"]
        reduction = parameters["reduction"]
        columns = parameters["columns"]
        stream = task_stream(oracle)
        constant_offset = (int(descriptor["constant_address"], 16) -
                           int(descriptor["text_address"], 16))
        aligned = (len(stream) * 4 + CONSTANT_ALIGNMENT - 1) & \
            ~(CONSTANT_ALIGNMENT - 1)
        if constant_offset != aligned:
            raise ValueError(
                f"{oracle['case']}: constants sit at {constant_offset}, not "
                f"the {aligned} the aligned text stream implies")
        if oracle["constant_section"]["size"] != reduction * columns * 2:
            raise ValueError(
                f"{oracle['case']}: constant section is not K*N halfwords")
        lines.append(f"// {oracle['case']}")
        lines.append(f"static constexpr std::uint32_t kH14MatvecText{index}[] = {{")
        lines.append(word_rows(stream))
        lines.append("};")
        lines.append("")
        trailer = descriptor["trailing_words"]
        if int(trailer[18], 16):
            raise ValueError(
                f"{oracle['case']}: descriptor word 0x858 is {trailer[18]}, "
                "which the matvec encoder does not carry")
        entries.append(
            f"    {{{rows}, {reduction}, {columns}, kH14MatvecText{index}, "
            f"std::size(kH14MatvecText{index}), {descriptor['task_count']}, "
            f"0x{int(trailer[20], 16):08x}, 0x{int(trailer[28], 16):08x}}},")
    lines.append("static constexpr OracleMatvecTemplate kMatvecTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def island_oracles() -> list[dict[str, Any]]:
    """The decoded select and batched-matmul records the island tables cover."""
    selected = []
    for path in sorted(ORACLES.glob("ga*.json")):
        oracle = json.loads(path.read_text())
        if oracle.get("error") is not None:
            continue
        if oracle["family"] in ("env_boolean_select", "env_matmul") and \
                (oracle["case"].startswith("gasel_") or
                 oracle["case"].startswith("gabmm_")):
            selected.append(oracle)
    return selected


def island_tensor_entries(oracle: dict[str, Any]) -> list[str]:
    """One `{index, {nchw...}, allocation, elementCode, elementBytes}` literal
    per decoded descriptor, inputs in the decoded order and the output last.
    The decoded strides are [batch, plane, row, element] bytes."""
    entries = []
    for index, descriptor in enumerate(oracle["tensor_descriptors"]):
        shape = descriptor["shape"]
        strides = descriptor["strides"]
        element_bytes = strides[3]
        element_code = {1: 3, 2: 5}[element_bytes]
        channel = 4 if index + 1 == len(oracle["tensor_descriptors"]) \
            else 5 + index
        allocation = (descriptor["total_bytes"] + 0x3FFF) & ~0x3FFF
        entries.append(
            f"    {{{channel}, {{{shape[0]}, {shape[1]}, {shape[2]}, "
            f"{shape[3]}, {strides[1]}, {strides[2]}}}, {allocation}, "
            f"{element_code}, {element_bytes}}},")
    return entries


def generate_island() -> str:
    """Tables for the Parakeet island families: five-task `select` with a bool
    cond and the batched runtime-runtime matmuls. Both encode as whole-program
    replays: every task word, the constant section runs, the three descriptor
    words the campaign resolves no formula for, and the surface layouts come
    from the decoded oracle."""
    lines = ["// Generated from decoded H14 select and batched-matmul oracle",
             "// task words by research/generate_h14_templates.py.",
             ""]
    selects = []
    batched = []
    for oracle in island_oracles():
        parameters = oracle["parameters"]
        if oracle["family"] == "env_boolean_select":
            if parameters["mode"] != "rr":
                # The const-fill form's 2.2 MB section is the -inf fill at a
                # 376-halfword row stride plus the table; no run encoder, no
                # template. The runtime form is what the island binds.
                continue
        elif parameters["batch"] == 1:
            # Rank-3 batch-1 runtime matmuls stay outside: the two-task
            # constant-weight encoder owns the rank-2 grid and nothing covers
            # these yet.
            continue
        stream = task_stream(oracle)
        runs = constant_runs(oracle)
        descriptor = oracle["program_descriptor"]
        trailer = descriptor["trailing_words"]
        common = (tuple(stream), tuple(runs),
                  oracle["constant_section"]["size"],
                  descriptor["task_count"], int(trailer[20], 16),
                  int(trailer[28], 16), int(trailer[18], 16))
        tensors = island_tensor_entries(oracle)
        if oracle["family"] == "env_boolean_select":
            key_shape = oracle["tensor_descriptors"][0]["shape"]
            selects.append((parameters, common, tensors, oracle["case"],
                            key_shape))
        else:
            batched.append((parameters, common, tensors, oracle["case"]))
    lines.append("struct OracleIslandTensor {")
    lines.append("    std::uint32_t index;")
    lines.append("    std::array<std::uint64_t, 6> nchw;")
    lines.append("    std::uint64_t allocationBytes;")
    lines.append("    std::uint32_t elementCode;")
    lines.append("    std::uint32_t elementBytes;")
    lines.append("};")
    lines.append("")
    lines.append("struct OracleIslandTemplate {")
    lines.append("    const std::uint32_t *text;")
    lines.append("    std::size_t textWords;")
    lines.append("    std::uint32_t taskCount;")
    lines.append("    const ConstantRun *constants;")
    lines.append("    std::size_t constantRuns;")
    lines.append("    std::size_t constantBytes;")
    lines.append("    std::uint32_t programRecordCount;")
    lines.append("    std::uint32_t unresolvedDescriptorWord;")
    lines.append("    std::uint32_t scratchDescriptorWord;")
    lines.append("    std::array<OracleIslandTensor, 4> tensors;")
    lines.append("    std::uint32_t inputCount;")
    lines.append("};")
    lines.append("")

    def emit_common(index, common):
        stream, runs, constant_bytes, task_count, records, unresolved, \
            scratch = common
        lines.append(f"static constexpr std::uint32_t kH14IslandText{index}[] = {{")
        lines.append(word_rows(list(stream)))
        lines.append("};")
        constants = "nullptr, 0"
        if runs:
            lines.append(f"static constexpr ConstantRun "
                         f"kH14IslandConstants{index}[] = {{")
            for start, bits, count in runs:
                lines.append(f"    {{{start}, 0x{bits:04x}, {count}}},")
            lines.append("};")
            constants = (f"kH14IslandConstants{index}, "
                         f"std::size(kH14IslandConstants{index})")
        return (f"kH14IslandText{index}, std::size(kH14IslandText{index}), "
                f"{task_count}, {constants}, {constant_bytes}, "
                f"0x{records:08x}, 0x{unresolved:08x}, 0x{scratch:08x}")

    entries = []
    for index, (parameters, common, tensors, case, key_shape) in \
            enumerate(selects):
        lines.append(f"// {case}")
        fields = emit_common(index, common)
        shape = parameters["shape"]
        const_fill = 1 if parameters["mode"] == "ninf" else 0
        lines.append(f"static constexpr std::array<OracleIslandTensor, 4> "
                     f"kH14SelectTensors{index} = " + "{{")
        lines.extend(tensors)
        lines.append("}};")
        lines.append("")
        shape_lit = "{%d, %d, %d}" % (key_shape[1], key_shape[2],
                                      key_shape[3])
        entries.append("    {" + shape_lit + ", " + str(const_fill) + ", "
                       + fields + ", kH14SelectTensors%d}" % index + ","),
    lines.append("struct OracleSelectTemplate {")
    lines.append("    ElementwiseShape shape;")
    lines.append("    bool constFill;")
    lines.append("    const std::uint32_t *text;")
    lines.append("    std::size_t textWords;")
    lines.append("    std::uint32_t taskCount;")
    lines.append("    const ConstantRun *constants;")
    lines.append("    std::size_t constantRuns;")
    lines.append("    std::size_t constantBytes;")
    lines.append("    std::uint32_t programRecordCount;")
    lines.append("    std::uint32_t unresolvedDescriptorWord;")
    lines.append("    std::uint32_t scratchDescriptorWord;")
    lines.append("    std::array<OracleIslandTensor, 4> tensors;")
    lines.append("};")
    lines.append("static constexpr OracleSelectTemplate kH14SelectTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")

    entries = []
    for index, (parameters, common, tensors, case) in enumerate(batched):
        lines.append(f"// {case}")
        fields = emit_common(len(selects) + index, common)
        lines.append(f"static constexpr std::array<OracleIslandTensor, 3> "
                     f"kH14BatchedTensors{index} = " + "{{")
        lines.extend(tensors)
        lines.append("}};")
        lines.append("")
        shape_lit = "{%d, %d, %d, %d, %d, %d}" % (
            parameters["rows"], parameters["reduction"],
            parameters["columns"], parameters["batch"],
            1 if parameters["layout"] == "r4heads" else 0,
            1 if parameters["transpose_y"] else 0)
        entries.append("    {" + shape_lit + ", " + fields
                       + ", kH14BatchedTensors%d}" % index + ",")
    lines.append("struct OracleBatchedMatmulTemplate {")
    lines.append("    BatchedMatmulShape shape;")
    lines.append("    const std::uint32_t *text;")
    lines.append("    std::size_t textWords;")
    lines.append("    std::uint32_t taskCount;")
    lines.append("    const ConstantRun *constants;")
    lines.append("    std::size_t constantRuns;")
    lines.append("    std::size_t constantBytes;")
    lines.append("    std::uint32_t programRecordCount;")
    lines.append("    std::uint32_t unresolvedDescriptorWord;")
    lines.append("    std::uint32_t scratchDescriptorWord;")
    lines.append("    std::array<OracleIslandTensor, 3> tensors;")
    lines.append("};")
    lines.append("static constexpr OracleBatchedMatmulTemplate "
                 "kH14BatchedMatmulTasks[] = {")
    lines.extend(entries)
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="fail when a checked-in file is stale")
    arguments = parser.parse_args()
    for path, generated in ((OUTPUT, generate()),
                            (MATVEC_OUTPUT, generate_matvec()),
                            (ISLAND_OUTPUT, generate_island())):
        if arguments.check:
            if (path.read_text() if path.exists() else "") != generated:
                raise SystemExit(f"{path} is stale; regenerate it")
            print(f"H14 templates: up to date ({path})")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(generated)
        print(f"wrote {path} ({len(generated)} bytes)")


if __name__ == "__main__":
    main()
