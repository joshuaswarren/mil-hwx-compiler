#!/usr/bin/env python3
"""Generate the H13 boolean-op template tables from oracle captures.

Each decoded capture in research/oracles/h13/boolean/ is one fixed task
stream keyed by (family, shape, storage); the generator reconstructs the
raw task words from the decoded register-stream JSON, verifies the
stream against itself (link chain, uniform structure), and emits one
entry per capture with the constant-section bytes retained verbatim
where the capture carries them.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "research" / "oracles" / "h13" / "boolean"


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


def derive(path):
    record = json.loads(path.read_text())
    if record.get("error") or not record.get("task_descriptors"):
        return None
    tasks = [reconstruct_task(t) for t in record["task_descriptors"]]
    # The link chain must terminate and each next pointer must out-run the
    # current task; the stream words embed the pointers so the encoder
    # replays them verbatim.
    offset = 0
    for index, words in enumerate(tasks):
        nxt = words[7]
        if index + 1 == len(tasks):
            if nxt:
                raise SystemExit(f"{path.name}: final task links on")
        elif nxt <= offset:
            raise SystemExit(f"{path.name}: task {index} link regresses")
        offset = nxt
    flat = [w for task in tasks for w in task]
    const_bin = path.parent / (path.stem + ".const.bin")
    constants = const_bin.read_bytes() if const_bin.exists() else None
    if constants is not None and len(constants) != record["constant_section"]["size"]:
        raise SystemExit(f"{path.name}: const.bin size mismatch")
    if constants is None:
        constants = bytes(record["constant_section"]["size"])
    input_shape = None
    for descriptor in record.get("tensor_descriptors") or []:
        if descriptor.get("binding") == 1:
            input_shape = list(descriptor["shape"])
            break
    return {
        "words": flat,
        "sizes": [len(task) for task in tasks],
        "constants": constants,
        "params": record["parameters"],
        "input_shape": input_shape,
        "mil": record["mil"],
    }


def words_literal(name, words):
    lines = [f"static const uint32_t {name}[] = {{"]
    for off in range(0, len(words), 6):
        chunk = words[off:off + 6]
        lines.append("    " + ", ".join(f"0x{w:08x}" for w in chunk) + ",")
    lines.append("};")
    return lines


def main():
    entries = []
    for path in sorted(CAPTURES.glob("*.json")):
        entry = derive(path)
        if entry is None:
            continue
        stem = path.stem
        if stem.startswith("floor_div_comp") or stem.startswith("less_bool_cast"):
            continue  # native floor_div and the pure less are the implemented forms
        broadcast_cond = False
        if stem.startswith("less"):
            kind, const_input = "Less", False
        elif stem.startswith("floor_div_native"):
            kind, const_input = "FloorDiv", "_s2" in stem
        elif stem.startswith("floor_"):
            kind, const_input = "Floor", stem.startswith("floor_b")
        elif stem.startswith("candidate_select"):
            kind, const_input = "Select", False
            broadcast_cond = True
        elif stem.startswith("candidate_transpose_bool"):
            kind, const_input = "TransposeBool", False
        elif stem.startswith("candidate_cast_f16_to_b_1x1x375"):
            kind, const_input = "CastFp16ToBool", False
        elif stem.startswith("candidate_cast_f16_to_b_1x375x375"):
            kind, const_input = "CastFp16ToBool", False
        elif stem.startswith("candidate_cast_b_to_f16_1x375x375"):
            kind, const_input = "CastBoolToFp16", False
        elif stem.startswith("candidate_mul_rr"):
            continue  # elementwise binary op: not a boolean-table row
        elif stem.startswith("select_ninf"):
            kind, const_input = "Select", True
        elif stem.startswith("select"):
            kind, const_input = "Select", False
        elif stem.startswith("cast_b_to_f16"):
            kind, const_input = "CastBoolToFp16", False
        elif stem.startswith("logical_not"):
            kind, const_input = "LogicalNot", False
        else:
            raise SystemExit(f"{stem}: unclassified capture")
        entry["kind"] = kind
        entry["constInput"] = const_input
        entry["broadcastCond"] = broadcast_cond
        entry["name"] = stem
        entries.append(entry)
    print(f"verified {len(entries)} entries")

    out = ["// Generated by research/generate_boolean_tables.py from",
           "// research/oracles/h13/boolean/ — do not edit by hand.",
           "#pragma once",
           "#include <cstdint>",
           "#include <cstddef>",
           "",
           "struct H13BooleanTemplate {",
           "    H13BooleanKind kind;",
           "    bool constInput;",
           "    bool broadcastCond;",
           "    std::uint32_t channels, height, width;",
           "    const std::uint32_t *words;",
           "    std::size_t wordCount;",
           "    const std::uint8_t *constants;",
           "    std::size_t constantBytes;",
           "    const std::uint32_t *taskWords;",
           "    std::size_t taskCount;",
           "};",
           ""]
    table = []
    for index, entry in enumerate(entries):
        wname = f"kBooleanWords{index}"
        cname = f"kBooleanConst{index}"
        sname = f"kBooleanSizes{index}"
        out.extend(words_literal(wname, entry["words"]))
        sizes = ", ".join(str(s) for s in entry["sizes"])
        out.append(f"static const uint32_t {sname}[] = {{{sizes}}};")
        data = ", ".join(f"0x{b:02x}" for b in entry["constants"])
        out.append(f"static const uint8_t {cname}[] = {{{data}}};")
        params = entry["params"]
        # Prefer the decoded input binding: the mask-oracle records carry
        # no parameters.shape, but every decoded record binds its bool x
        # as tensor_descriptors[0].
        shape = None
        descriptors = entry.get("input_shape") or []
        if descriptors:
            shape = list(descriptors)
        if not shape:
            shape = list(params.get("shape") or params.get("x_shape") or [1])
        while len(shape) < 3:
            shape = [1] + shape
        while len(shape) > 3:
            if shape[0] != 1:
                raise SystemExit(f"{entry['name']}: odd rank {shape}")
            shape = shape[1:]
        channels, height, width = shape
        kind = {"Less": "H13BooleanKind::Less", "Floor": "H13BooleanKind::Floor",
                "Select": "H13BooleanKind::Select",
                "FloorDiv": "H13BooleanKind::FloorDiv",
                "CastBoolToFp16": "H13BooleanKind::CastBoolToFp16",
                "CastFp16ToBool": "H13BooleanKind::CastFp16ToBool",
                "TransposeBool": "H13BooleanKind::TransposeBool",

                "LogicalNot": "H13BooleanKind::LogicalNot"}[entry["kind"]]
        table.append(
            "    {%s, %s, %s, %d, %d, %d, %s, std::size(%s), %s, std::size(%s), %s, std::size(%s)},"
            % (kind, "true" if entry["constInput"] else "false",
               "true" if entry["broadcastCond"] else "false",
               channels, height, width,
               wname, wname, cname, cname, sname, sname))
    out.append("static const H13BooleanTemplate kBooleanTasks[] = {")
    out.extend("    " + row for row in table)
    out.append("};")
    destination = ROOT / "plugins" / "H13" / "H13BooleanTemplates.inc"
    destination.write_text("\n".join(out) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
