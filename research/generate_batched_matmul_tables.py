#!/usr/bin/env python3
"""Generate the H13 batched-matmul envelope tables from oracle captures.

Reconstructs each capture's raw task words from the decoded register-stream
JSON and derives the batch structure empirically: a capture is an optional
prefix task followed by one 26-task group per batch, and every group word
either moves affinely with the batch index or carries per-batch literals.
The generator re-derives every batch of every capture from the emitted
model and aborts on any mismatch.

For packed-constant captures the round-2 `_idx` files retain the input
weights and Apple's constant section, so the packing is derived and
verified byte-exactly against the raw bytes: a 64-halfword kernel header
followed by the weight matrix (transposed for ty=1 forms) cut into
N-wide chunks placed at the padded row stride with a -64 phase, and the
final 64 source halves unwritten. The 128-byte headers cluster by packed
shape and batch; each distinct header is emitted.
"""
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "research" / "oracles" / "h13" / "batched"
TASKS_PER_BATCH = 26
PACK_OFFSET = 64


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


def parse_case(name):
    parts = name.replace(".json", "").split("_")
    batch = None
    for part in parts:
        if part.startswith("b") and part[1:].isdigit():
            batch = int(part[1:])
    tx = 1 if "tx1" in parts else 0
    ty = 1 if "ty1" in parts else 0
    storage = "Runtime" if "rr" in parts[1] else "Packed"
    return int(parts[2][1:]), int(parts[3][1:]), int(parts[4][1:]), \
        tx, ty, storage, batch


def derive(path):
    record = json.loads(path.read_text())
    if record.get("error") or not record.get("task_descriptors"):
        return None
    params = record["parameters"]
    tasks = [reconstruct_task(t) for t in record["task_descriptors"]]
    batch = params["batch"]
    prefix = None
    if len(tasks) % TASKS_PER_BATCH == 0:
        groups = tasks
    elif (len(tasks) - 1) % TASKS_PER_BATCH == 0:
        prefix = tasks[0]
        groups = tasks[1:]
    else:
        print(f"skip {path.name}: {len(tasks)} tasks fits no prefix+26·B form",
              file=sys.stderr)
        return None
    if len(groups) != TASKS_PER_BATCH * batch:
        print(f"skip {path.name}: group count {len(groups)} is not "
              f"{TASKS_PER_BATCH}×{batch}", file=sys.stderr)
        return None
    group = groups[:TASKS_PER_BATCH]
    affine = [{} for _ in range(TASKS_PER_BATCH)]
    literal = [{} for _ in range(TASKS_PER_BATCH)]
    for i in range(TASKS_PER_BATCH):
        base = group[i]
        first = groups[TASKS_PER_BATCH + i]
        if len(base) != len(first):
            print(f"skip {path.name}: task {i} size changes per batch",
                  file=sys.stderr)
            return None
        for w, (b0, b1) in enumerate(zip(base, first)):
            if b0 != b1:
                affine[i][w] = b1 - b0
        for b in range(1, batch):
            actual = groups[b * TASKS_PER_BATCH + i]
            for w, delta in list(affine[i].items()):
                if actual[w] != base[w] + b * delta:
                    affine[i].pop(w)
        for b in range(1, batch):
            actual = groups[b * TASKS_PER_BATCH + i]
            for w in range(len(base)):
                if actual[w] != base[w] + b * affine[i].get(w, 0):
                    literal[i].setdefault(w, {})[b] = actual[w]
    for b in range(1, batch):
        for i in range(TASKS_PER_BATCH):
            actual = groups[b * TASKS_PER_BATCH + i]
            rebuilt = [group[i][w] + b * affine[i].get(w, 0)
                       for w in range(len(group[i]))]
            for w, values in literal[i].items():
                if b in values:
                    rebuilt[w] = values[b]
            if rebuilt != actual:
                bad = {w: (rebuilt[w], actual[w]) for w in range(len(actual))
                       if rebuilt[w] != actual[w]}
                raise SystemExit(
                    f"{path.name}: batch {b} task {i} model mismatch: {bad}")
    entry = {
        "prefix": prefix,
        "group": group,
        "affine": affine,
        "literal": literal,
        "params": params,
        "constant_size": record["constant_section"]["size"],
        "scratch": int(record["program_descriptor"]["resource_addresses"][0],
                       16) - 0x30000000,
    }
    # Binding order is a property of the captured object: for single-input
    # (packed) captures the gap between the first two resources names which
    # surface sits first. Runtime captures bind three surfaces and keep the
    # output-first order the runtime path already emits.
    entry["input_first"] = False
    if storage == "Packed" if False else entry["params"]["w_storage"] == "blob":
        res = record["program_descriptor"]["resource_addresses"]
        gap = int(res[1], 16) - int(res[0], 16)
        x_total = record["tensor_descriptors"][0]["total_bytes"]
        y_total = record["tensor_descriptors"][1]["total_bytes"]
        if gap == ((x_total + 0x3FFF) & ~0x3FFF):
            entry["input_first"] = True
        elif gap != ((y_total + 0x3FFF) & ~0x3FFF):
            raise SystemExit(f"{path.name}: unrecognized resource layout")
    const_bin = path.parent / (path.stem + ".const.bin")
    weights_bin = path.parent / (path.stem + ".weights.bin")
    if const_bin.exists() and weights_bin.exists():
        weights = weights_bin.read_bytes()
        const = const_bin.read_bytes()
        if len(const) != entry["constant_size"]:
            raise SystemExit(f"{path.name}: const.bin size mismatch")
        # Derive and verify the packing: header + phased chunks over the
        # (transposed for ty=1) weight matrix.
        pack_rows = params["reduction"]
        pack_cols = params["columns"]
        if params["transpose_y"]:
            pack_rows, pack_cols = pack_cols, pack_rows
        padded = (pack_cols + 31) // 32 * 32
        wi = list(struct.unpack(f"<{len(weights)//2}H", weights))
        ci = list(struct.unpack(f"<{len(const)//2}H", const))
        if len(ci) != batch * pack_rows * padded:
            raise SystemExit(
                f"{path.name}: packed size {len(ci)} != "
                f"{batch}×{pack_rows}×{padded}")
        if wi == ci and len(ci) == batch * pack_rows * padded:
            # The head-projection capture stores the weight blob verbatim:
            # no kernel header, no phase, identity packing.
            entry["identity"] = True
            entry["pack_rows"] = pack_rows
            entry["pack_cols"] = pack_cols
            return entry
        dest = [0] * len(ci)
        dest[:PACK_OFFSET] = ci[:PACK_OFFSET]
        for r in range(batch * pack_rows):
            if r == 0:
                lo, hi, at = 0, pack_cols - PACK_OFFSET, PACK_OFFSET
            else:
                lo, hi = r * pack_cols - PACK_OFFSET, (r + 1) * pack_cols - PACK_OFFSET
                at = r * padded
            dest[at:at + (hi - lo)] = wi[lo:hi]
        if dest != ci:
            bad = next(i for i in range(len(ci)) if dest[i] != ci[i])
            raise SystemExit(
                f"{path.name}: packing model mismatch at half {bad}")
        entry["header"] = const[:PACK_OFFSET * 2]
        entry["pack_rows"] = pack_rows
        entry["pack_cols"] = pack_cols
    return entry


def words_literal(name, words):
    lines = [f"static const uint32_t {name}[] = {{"]
    for off in range(0, len(words), 6):
        chunk = words[off:off + 6]
        lines.append("    " + ", ".join(f"0x{w:08x}" for w in chunk) + ",")
    lines.append("};")
    return lines


def main():
    entries = {}
    headers = {}
    for path in sorted(CAPTURES.glob("*.json"),
                       key=lambda p: (not p.stem.endswith("_idx"), p.name)):
        rows, red, cols, tx, ty, storage, batch = parse_case(path.name)
        entry = derive(path)
        if entry is None:
            continue
        key = (rows, red, cols, tx, ty, storage, batch)
        if key in entries:
            other = entries[key]
            if (other["prefix"] != entry["prefix"] or
                    other["group"] != entry["group"] or
                    other["affine"] != entry["affine"] or
                    other["literal"] != entry["literal"]):
                raise SystemExit(f"{path.name}: disagrees with its twin")
            if "header" in entry and "header" not in other:
                entries[key] = entry
            continue
        entries[key] = entry
        if "header" in entry:
            headers.setdefault(entry["header"], []).append(key)
    print(f"verified {len(entries)} entries, {len(headers)} headers")

    out = ["// Generated by research/generate_batched_matmul_tables.py from",
           "// research/oracles/h13/batched/ — do not edit by hand.",
           "#pragma once",
           "enum class BatchedWeight : std::uint8_t { Runtime, Packed };",
           "struct H13BatchedWordRule {",
           "    std::uint32_t word;",
           "    std::size_t literalCount;",
           "    std::int64_t delta;",
           "    std::size_t literalTotal;",
           "    const std::uint32_t *literals;",
           "};",
           "struct H13BatchedTaskRule {",
           "    std::uint32_t task;",
           "    const H13BatchedWordRule *rules;",
           "    std::size_t ruleCount;",
           "};",
           "struct OracleBatchedMatmulTemplate {",
           "    std::uint32_t rows, reduction, columns;",
           "    std::uint8_t transposeX, transposeY;",
           "    BatchedWeight storage;",
           "    std::uint32_t batch;",
           "    const std::uint32_t *prefixWords;",
           "    std::size_t prefixWordCount;",
           "    const std::uint32_t *groupWords;",
           "    std::size_t groupWordCount;",
           "    const H13BatchedTaskRule *taskRules;",
           "    std::size_t taskRuleCount;",
           "    std::size_t constantBytes;",
           "    const std::uint8_t *packHeader;",
           "    std::uint32_t packRows, packCols;",
           "    std::uint8_t identityPacking;",
           "    std::uint32_t scratchBytes;",
           "    std::uint8_t inputFirst;",
           "};", ""]
    header_ids = {}
    for index, header in enumerate(sorted(headers)):
        name = f"kBatchedPackHeader{index}"
        data = ", ".join(f"0x{b:02x}" for b in header)
        out.append(f"static const uint8_t {name}[] = {{{data}}};")
        header_ids[header] = name
    out.append("")
    table = []
    index = 0
    for key in sorted(entries):
        rows, red, cols, tx, ty, storage, batch = key
        entry = entries[key]
        name = f"kBatchedGroup{index}"
        out.extend(words_literal(
            name, [w for task in entry["group"] for w in task]))
        out.append("")
        prefix_ref, prefix_count = "nullptr", "0"
        if entry["prefix"] is not None:
            pname = f"kBatchedPrefix{index}"
            out.extend(words_literal(pname, entry["prefix"]))
            out.append("")
            prefix_ref, prefix_count = pname, f"std::size({pname})"
        patch_parts = []
        for i in range(TASKS_PER_BATCH):
            affine = entry["affine"][i]
            literal = entry["literal"][i]
            if not affine and not literal:
                continue
            members = []
            for w, delta in sorted(affine.items()):
                if w not in literal:
                    members.append(f"{{{w}, 0, {delta:+d}, 0, nullptr}}")
            for w, values in sorted(literal.items()):
                column = f"kBatchedLit{index}_{i}_{w}"
                data = ", ".join(
                    f"0x{values.get(b, entry['group'][i][w]):08x}"
                    for b in range(1, batch))
                out.append(f"static const uint32_t {column}[] = {{{data}}};")
                members.append(f"{{{w}, 1, 0, {batch - 1}, {column}}}")
            out.append(f"static const H13BatchedWordRule kBatchedRules{index}_{i}[] = {{{', '.join(members)}}};")
            patch_parts.append(f"{{{i}, kBatchedRules{index}_{i}, {len(members)}}}")
        if patch_parts:
            out.append(f"static const H13BatchedTaskRule kBatchedTaskRules{index}[] = {{{', '.join(patch_parts)}}};")
            rules = f"kBatchedTaskRules{index}, {len(patch_parts)}"
        else:
            rules = "nullptr, 0"
        header = entry.get("header")
        identity = entry.get("identity", False)
        header_ref = header_ids.get(header, "nullptr") if header else "nullptr"
        pack = (f"{entry.get('pack_rows', 0)}, {entry.get('pack_cols', 0)}"
                if header or identity else "0, 0")
        table.append(
            "    {%d, %d, %d, %d, %d, BatchedWeight::%s, %d, %s, %s, %s, "
            "std::size(%s), %s, %d, %s, %s, %d, %u, %d},"
            % (rows, red, cols, tx, ty, storage, batch,
               prefix_ref, prefix_count, name, name, rules,
               entry["constant_size"], header_ref, pack,
               1 if identity else 0, entry.get("scratch", 0),
               1 if entry.get("input_first") else 0))
        index += 1
    out.append("static constexpr OracleBatchedMatmulTemplate kBatchedTasks[] = {")
    out.extend("    " + row for row in table)
    out.append("};")
    destination = ROOT / "plugins" / "H13" / "H13BatchedMatmulTemplates.inc"
    destination.write_text("\n".join(out) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
