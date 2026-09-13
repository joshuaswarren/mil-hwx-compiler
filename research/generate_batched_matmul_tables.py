#!/usr/bin/env python3
"""Generate the H13 batched-matmul envelope tables from oracle captures.

Reconstructs each capture's raw task words from the decoded register-stream
JSON and derives the batch structure per task word: a word either moves
affinely with the batch index (batch-zero value plus b times a fixed delta)
or carries per-batch literals (task markers, link pointers). The generator
re-derives every batch of every capture from the emitted model and aborts on
any mismatch, then writes plugins/H13/H13BatchedMatmulTemplates.inc.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPTURES = ROOT / "research" / "oracles" / "h13" / "batched"
TASKS_PER_BATCH = 26


def reconstruct_task(td):
    words = [int(w, 16) for w in td["header_words"]]
    flat = {}
    for block in td["blocks"].values():
        for a, v in block["words"].items():
            flat[int(a, 16)] = int(v, 16)
    for rec in td["records"]:
        words.append(int(rec["header"], 16))
        base = int(rec["address"], 16)
        for i in range(rec["count"]):
            words.append(flat[base + i * 4])
    return words


def parse_case(name):
    parts = name.replace(".json", "").split("_")
    batch = None
    for part in parts:
        if part.startswith("b") and part[1:].isdigit():
            batch = int(part[1:])
    storage = "Runtime" if "rr" in parts[1] else "Packed"
    return int(parts[2][1:]), int(parts[3][1:]), int(parts[4][1:]), \
        storage, batch


def derive(path):
    record = json.loads(path.read_text())
    if record.get("error") or not record.get("task_descriptors"):
        return None
    params = record["parameters"]
    tasks = [reconstruct_task(t) for t in record["task_descriptors"]]
    batch = params["batch"]
    if len(tasks) != TASKS_PER_BATCH * batch:
        print(f"skip {path.name}: {len(tasks)} tasks is not "
              f"{TASKS_PER_BATCH}×{batch}", file=sys.stderr)
        return None
    group = tasks[:TASKS_PER_BATCH]
    affine = [{} for _ in range(TASKS_PER_BATCH)]
    literal = [{} for _ in range(TASKS_PER_BATCH)]
    for i in range(TASKS_PER_BATCH):
        base = group[i]
        first = tasks[TASKS_PER_BATCH + i]
        if len(base) != len(first):
            print(f"skip {path.name}: task {i} size changes per batch",
                  file=sys.stderr)
            return None
        for w, (b0, b1) in enumerate(zip(base, first)):
            if b0 != b1:
                affine[i][w] = b1 - b0
        for b in range(1, batch):
            actual = tasks[b * TASKS_PER_BATCH + i]
            for w, delta in list(affine[i].items()):
                if actual[w] != base[w] + b * delta:
                    affine[i].pop(w)
        for b in range(1, batch):
            actual = tasks[b * TASKS_PER_BATCH + i]
            for w in range(len(base)):
                expected = base[w] + b * affine[i].get(w, 0)
                if actual[w] != expected:
                    literal[i].setdefault(w, {})[b] = actual[w]
    # Full verification: rebuild every batch exactly.
    for b in range(1, batch):
        for i in range(TASKS_PER_BATCH):
            actual = tasks[b * TASKS_PER_BATCH + i]
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
    tensors = record["tensor_descriptors"]
    return {
        "group": group,
        "affine": affine,
        "literal": literal,
        "params": params,
        "strides": [t.get("strides") for t in tensors],
        "shapes": [t.get("shape") for t in tensors],
        "bindings": [t.get("binding") for t in tensors],
        "constant_size": record["constant_section"]["size"],
        "task_sizes": [len(t) for t in group],
    }


def words_literal(name, words):
    lines = [f"static const uint32_t {name}[] = {{"]
    for off in range(0, len(words), 6):
        chunk = words[off:off + 6]
        lines.append("    " + ", ".join(f"0x{w:08x}" for w in chunk) + ",")
    lines.append("};")
    return lines


def main():
    entries = {}
    for path in sorted(CAPTURES.glob("*.json")):
        rows, red, cols, storage, batch = parse_case(path.name)
        entry = derive(path)
        if entry is None:
            continue
        key = (rows, red, cols, storage, batch)
        if key in entries:
            other = entries[key]
            if other["group"] != entry["group"] or \
                    other["affine"] != entry["affine"] or \
                    other["literal"] != entry["literal"]:
                raise SystemExit(f"{path.name}: rank-4 disagrees with rank-3")
            continue
        entries[key] = entry
    print(f"verified {len(entries)} entries")

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
           "    BatchedWeight storage;",
           "    std::uint32_t batch;",
           "    const std::uint32_t *groupWords;",
           "    std::size_t groupWordCount;",
           "    const H13BatchedTaskRule *taskRules;",
           "    std::size_t taskRuleCount;",
           "    std::size_t constantBytes;",
           "};", ""]
    table = []
    index = 0
    for key in sorted(entries):
        rows, red, cols, storage, batch = key
        entry = entries[key]
        name = f"kBatchedGroup{index}"
        out.extend(words_literal(
            name, [w for task in entry["group"] for w in task]))
        out.append("")
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
        table.append(
            "    {%d, %d, %d, BatchedWeight::%s, %d, %s, std::size(%s), %s, %d},"
            % (rows, red, cols, storage, batch, name, name, rules,
               entry["constant_size"]))
        index += 1
    out.append("static constexpr OracleBatchedMatmulTemplate kBatchedTasks[] = {")
    out.extend("    " + row for row in table)
    out.append("};")
    destination = ROOT / "plugins" / "H13" / "H13BatchedMatmulTemplates.inc"
    destination.write_text("\n".join(out) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
