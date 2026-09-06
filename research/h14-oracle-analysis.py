#!/usr/bin/env python3
"""Generate the H14 task-descriptor field report from checked-in oracle JSON."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
ORACLES = ROOT / "oracles"
MAP_COMMIT = "ce54664e787976b646c450ceabed1731b506a4cd"
REPO_COMMIT = "72a03870db43e03c91e362bcdbd6e7b438fef3db"

BLOCKS = (
    ("Common", "common", 0x0000, 19, 0x00000, 16),
    ("L2", "l2", 0x0500, 25, 0x04800, 18),
    ("PE", "pe", 0x0900, 5, 0x08800, 4),
    ("NE", "ne", 0x0D00, 5, 0x0C800, 5),
    ("TileDMA source", "tile_dma_src", 0x1100, 53, 0x13800, 28),
    ("TileDMA destination", "tile_dma_dst", 0x1500, 10, 0x17800, 7),
    ("KernelDMA", "kernel_dma_src", 0x1900, 70, 0x1F800, 62),
)

COMMON_NAMES = [
    "InDim (packed W+H)", "InDepth", "ChannelCfg", "InChannels",
    "OutChannels", "OutDim (packed W+H)", "OutDepth", "OCGSize",
    "ConvCfg", "ConvCfg3d", "GroupConvCfg", "TileCfg", "TileOverlap",
    "NECfg", "Cfg", "TaskInfo", "DPE", "Spare0", "Spare1",
]
L2_NAMES = [
    "Control", "Src1Cfg", "Src2Cfg", "Src1Base", "Src1ChannelStride",
    "Src1RowStride", "Src1DepthStride", "Src1GroupStride", "Src2Base",
    "Src2ChannelStride", "Src2RowStride", "Src2DepthStride",
    "Src2GroupStride", "ResultCfg", "ResultBase", "ResultChannelStride",
    "ResultRowStride", "ResultDepthStride", "ResultGroupStride",
    "SrcAndResultWrapCfg", "Src1WrapStart", "Src2WrapStart", "L2Reserved",
    "ResultWrapIndex", "ResultWrapStartOffset",
]
PE_NAMES = ["PEConfig", "BiasScale", "PreScale", "FinalScale", "Quant"]
NE_NAMES = ["KernelCfg", "MacCfg", "MatrixVectorBias", "AccBias", "PostScale"]
TILE_SRC_NAMES: list[str | None] = [
    "Src1DMAConfig", "Src2DMAConfig", "Src1DMAConfigExt", "Src2DMAConfigExt",
    "Src1BaseAddrLo", "Src1BaseAddrHi", "Src1RowStride", "Src1PlaneStride",
    "Src1DepthStride", "Src1GroupStride", "Src2BaseAddrLo", "Src2BaseAddrHi",
    "Src2RowStride", "Src2PlaneStride", "Src2DepthStride", "Src2GroupStride",
    *(["TileDmaSrcReserved"] * 4), "Src1Fmt", "Src2Fmt",
    *(["TileDmaSrcReserved"] * 4),
    "PixelOffset[0]", "PixelOffset[1]", "PixelOffset[2]", "PixelOffset[3]",
    "TileDmaSrcReserved", *([None] * 20), "Spare0", "Spare1",
]
TILE_DST_NAMES = [
    "DstDMAConfig", "DstReserved", "DstBaseAddrLo", "DstBaseAddrHi",
    "DstRowStride", "DstPlaneStride", "DstDepthStride", "DstGroupStride",
    "DstFmt", "Spare0",
]
KERNEL_NAMES = [
    "MasterConfig", "AlignedCoeffSizePerCh", "Prefetch", "Reserved0",
    "Reserved1", "Reserved2",
    *[f"CoeffDMAConfig{i}" for i in range(16)],
    *[f"CoeffBaseAddr{i}" for i in range(16)],
    *[f"CoeffBfrSize{i}" for i in range(16)],
    "BiasDMAConfig", "BiasBaseAddr", "BiasReserved0", "BiasReserved1",
    "PostScaleDMAConfig", "PostScaleBaseAddr", "PostScaleReserved0",
    "PostScaleReserved1", "SparseBlockSizeCfg", "Reserved", "Reserved",
    "Reserved", "Spare0", "Spare1", "Reserved", "Reserved",
]
NAMES = {
    "common": COMMON_NAMES,
    "l2": L2_NAMES,
    "pe": PE_NAMES,
    "ne": NE_NAMES,
    "tile_dma_src": TILE_SRC_NAMES,
    "tile_dma_dst": TILE_DST_NAMES,
    "kernel_dma_src": KERNEL_NAMES,
}

MEANINGS = {
    0x0000: "W bits 0:14; H bits 16:30; pad bits 15,31",
    0x0004: "input depth",
    0x0008: "input/source-2/output formats",
    0x000C: "input channels",
    0x0010: "output channels",
    0x0014: "W bits 0:14; H bits 16:30; pad bits 15,31",
    0x0900: "pool mode, operation, nonlinear mode",
    0x1100: "enable/cache/dependency; DataSetId predicted at bits 8:15",
    0x1104: "enable/cache/dependency; DataSetId predicted at bits 8:15",
    0x1500: "enable/cache/L2 mode; DataSetId predicted at bits 8:15",
    0x19F8: "BlockSize bits 0:7 in the detailed map",
    0x1A00: "reserved in the detailed map; SparseKernelBlockSize in map prose",
}

FORMULAS = {
    0x0000: "one-task elementwise/unary t0: `(H<<16)|W`; matvec t0: `(1<<16)|K`, t1: `(1<<16)|M`",
    0x000C: "one-task elementwise/unary t0: `C`; matvec t0: `M`, t1: `K`",
    0x0010: "one-task elementwise/unary t0: `C`; matvec t0: `M`, t1: `N`",
    0x0014: "one-task elementwise/unary t0: `(H<<16)|W`; matvec t0: `(1<<16)|K`, t1: `(1<<16)|M`",
    0x002C: "square spatial sweep: `S` where `H=W=S`",
    0x0900: "binary op: add `0x80000`, mul `0x80004`, max `0x80008`, min `0x8000c`, sub `0xc0000`",
}

BITFIELDS = {
    "L2": "Addresses/strides use 16-byte units in the freedomtan map.",
    "TileDMA source": "Base/stride fields use 64-byte units; format and pixel-offset fields are packed.",
    "TileDMA destination": "Base/stride fields use 64-byte units; DstFmt is packed.",
    "KernelDMA": "Config words contain enable/cache/DataSetId/UserTag; base and size fields use 64-byte units.",
}


def load(target: str) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted((ORACLES / target).glob("*.json"))]


def decoded(docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [doc for doc in docs if doc.get("task_descriptors")]


def word_observations(docs: Iterable[dict[str, Any]]) -> tuple[
        dict[int, list[tuple[int, str]]], int, set[int]]:
    observations: dict[int, list[tuple[int, str]]] = defaultdict(list)
    addresses: set[int] = set()
    task_count = 0
    for doc in docs:
        for task_index, task in enumerate(doc["task_descriptors"]):
            task_count += 1
            for block in task["blocks"].values():
                for address, value in block["words"].items():
                    numeric = int(address, 16)
                    addresses.add(numeric)
                    observations[numeric].append(
                        (int(value, 16), f"{doc['case']}#t{task_index}"))
    return observations, task_count, addresses


def value_summary(items: list[tuple[int, str]], task_count: int) -> str:
    if not items:
        return f"not written (0/{task_count} tasks)"
    values = sorted({value for value, _ in items})
    rendered = [f"`0x{value:08x}`" for value in values]
    body = ", ".join(rendered)
    return f"{body} ({len(items)}/{task_count} tasks)"


def evidence(items: list[tuple[int, str]]) -> str:
    if not items:
        return "all 172 decoded H14 cases"
    by_value: dict[int, str] = {}
    for value, case in items:
        by_value.setdefault(value, case)
    values = sorted(by_value)
    selected = values if len(values) <= 3 else [values[0], values[len(values) // 2], values[-1]]
    return ", ".join(f"`{by_value[value]}`" for value in selected)


def h13_relation(
        h13_docs: list[dict[str, Any]], h14_docs: list[dict[str, Any]],
        block_name: str, offset: int, h13_count: int) -> str:
    if offset >= h13_count * 4:
        return "new H14 slot"
    same = changed = h13_only = h14_only = 0
    h13_by_case = {doc["case"]: doc for doc in h13_docs}
    for right in h14_docs:
        left = h13_by_case[right["case"]]
        for task_index in range(max(len(left["task_descriptors"]), len(right["task_descriptors"]))):
            lv = rv = None
            if task_index < len(left["task_descriptors"]):
                for block in left["task_descriptors"][task_index]["blocks"].values():
                    if block["name"] == block_name:
                        address = next(base for _, name, _, _, base, _ in BLOCKS if name == block_name) + offset
                        value = block["words"].get(f"0x{address:05x}")
                        lv = None if value is None else int(value, 16)
            if task_index < len(right["task_descriptors"]):
                for block in right["task_descriptors"][task_index]["blocks"].values():
                    if block["name"] == block_name:
                        address = next(base for _, name, base, _, _, _ in BLOCKS if name == block_name) + offset
                        value = block["words"].get(f"0x{address:05x}")
                        rv = None if value is None else int(value, 16)
            if lv is None and rv is None:
                continue
            if lv is None:
                h14_only += 1
            elif rv is None:
                h13_only += 1
            elif lv == rv:
                same += 1
            else:
                changed += 1
    parts = []
    if same:
        parts.append(f"same {same}")
    if changed:
        parts.append(f"changed {changed}")
    if h13_only:
        parts.append(f"H13-only {h13_only}")
    if h14_only:
        parts.append(f"H14-only {h14_only}")
    return "; ".join(parts) or "unwritten on both"


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def family_blockers(docs: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    values: dict[tuple[int, int], set[int]] = defaultdict(set)
    headers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for doc in docs:
        for task_index, task in enumerate(doc["task_descriptors"]):
            for index, value in enumerate(task["header_words"]):
                headers[task_index, index].add(int(value, 16))
            for block in task["blocks"].values():
                for address, value in block["words"].items():
                    values[task_index, int(address, 16)].add(int(value, 16))
    words = [
        f"t{task}:`0x{address:04x}`" for (task, address), variants in sorted(values.items())
        if len(variants) > 1 and address not in FORMULAS
    ]
    header_words = [
        f"t{task}:h{index}" for (task, index), variants in sorted(headers.items())
        if len(variants) > 1 and index != 0
    ]
    return words, header_words


def table(headers: list[str], rows: Iterable[Iterable[str]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return output


def main() -> None:
    h13_all, h14_all = load("h13"), load("h14")
    h13, h14 = decoded(h13_all), decoded(h14_all)
    assert len(h13_all) == len(h14_all) == 235
    assert len(h13) == len(h14) == 172
    assert {d["case"] for d in h13} == {d["case"] for d in h14}
    assert {d["source_commit"] for d in h13_all + h14_all} == {REPO_COMMIT}
    observations, task_count, addresses = word_observations(h14)
    assert task_count == 265

    known = {base + index * 4 for _, _, base, count, _, _ in BLOCKS for index in range(count)}
    outside = sorted(addresses - known)
    formulas_verified = {
        0x0000: [("binary_add_1x64x1x1", 0x00010001), ("binary_add_1x64x8x8", 0x00080008), ("binary_add_1x3x224x224", 0x00E000E0)],
        0x000C: [("binary_add_1x64x1x1", 64), ("binary_add_1x128x1x1", 128)],
        0x0010: [("binary_add_1x64x1x1", 64), ("binary_add_1x128x1x1", 128)],
        0x0014: [("binary_add_1x64x1x1", 0x00010001), ("binary_add_1x64x8x8", 0x00080008), ("binary_add_1x3x224x224", 0x00E000E0)],
        0x002C: [("binary_add_1x64x1x1", 1), ("binary_add_1x64x8x8", 8), ("binary_add_1x3x224x224", 224)],
        0x0900: [("binary_add_1x64x1x1", 0x00080000), ("binary_mul_1x64x1x1", 0x00080004), ("binary_sub_1x64x1x1", 0x000C0000)],
    }
    by_case = {doc["case"]: doc for doc in h14}
    for address, cases in formulas_verified.items():
        for case, expected in cases:
            found = [value for task in by_case[case]["task_descriptors"] for block in task["blocks"].values()
                     for key, value in block["words"].items() if int(key, 16) == address]
            assert found and int(found[0], 16) == expected, (address, case, found)

    def task_words(doc: dict[str, Any], task_index: int) -> dict[int, int]:
        return {int(address, 16): int(value, 16)
                for block in doc["task_descriptors"][task_index]["blocks"].values()
                for address, value in block["words"].items()}

    elementwise_ops = {"add": 0x80000, "mul": 0x80004, "maximum": 0x80008,
                       "minimum": 0x8000C, "sub": 0xC0000}
    for doc in h14:
        if (doc["family"] == "unary" or
                doc["family"] == "binary_runtime" and
                doc["parameters"].get("operation") in elementwise_ops):
            _, channels, height, width = doc["parameters"]["shape"]
            words = task_words(doc, 0)
            assert words[0x0000] == words[0x0014] == (height << 16) | width
            assert words[0x000C] == words[0x0010] == channels
            if doc["family"] == "binary_runtime":
                assert words[0x002C] == height == width
                assert words[0x0900] == elementwise_ops[doc["parameters"]["operation"]]
        elif doc["family"] == "matmul":
            rows = doc["parameters"]["rows"]
            reduction = doc["parameters"]["reduction"]
            columns = doc["parameters"]["columns"]
            first, second = task_words(doc, 0), task_words(doc, 1)
            assert first[0x0000] == first[0x0014] == (1 << 16) | reduction
            assert first[0x000C] == first[0x0010] == rows
            assert second[0x0000] == second[0x0014] == (1 << 16) | rows
            assert second[0x000C] == reduction and second[0x0010] == columns
    lines = [
        "# H14 task-descriptor fields and H13→H14 emitter delta", "",
        "This file is generated from the checked-in JSON. Edit `research/h14-oracle-analysis.py`, not this file.", "",
        "Regenerate and compare it with:", "",
        "```sh",
        "python3 research/h14-oracle-analysis.py > /tmp/h14-td-fields.md",
        "cmp research/h14-td-fields.md /tmp/h14-td-fields.md",
        "```", "",
        "## Scope and evidence", "",
        f"The analysis ran against repository commit `{REPO_COMMIT}`. All 470 oracle records name that source commit; 172 H13 and 172 H14 records decoded, and 63 per target record Apple rejection. The external names come from freedomtan's `h14_register_map.md` at `{MAP_COMMIT}`. The tables cover all {task_count} decoded H14 tasks. An absent write is distinct from a zero write.", "",
    ]
    family_counts = Counter(doc["family"] for doc in h14)
    lines += table(["Family", "Decoded H14 cases"], ([name, str(count)] for name, count in sorted(family_counts.items())))
    lines += ["", "## Block coverage and resolution", "",
              "A word is **evidence-resolved** when it is invariant, never written, or has a verified one-parameter formula for at least one named family below. A word is **unresolved** when two or more values occur and no sampled family explains them. This is an emitter-readiness count, not a claim that an invariant or partly resolved word has a complete hardware meaning.", ""]
    resolution_rows = []
    for title, name, base, count, _, _ in BLOCKS:
        resolved = sum(len({v for v, _ in observations[base + i * 4]}) <= 1 or base + i * 4 in FORMULAS for i in range(count))
        resolution_rows.append([title, f"`0x{base:04x}`", str(count), str(resolved), str(count - resolved)])
    lines += table(["Block", "H14 old base", "Words", "Resolved", "Unresolved"], resolution_rows)
    lines += ["", "## Map predictions tested against the oracles", ""]
    prediction_rows = [
        ["Nine-word header / DTID at h8", "Refuted for emitted H14 tasks. Each decoded task has eight header words; the word at index 8 is the first dense/scatter record header, not DTID.", "`binary_add_1x64x1x1`, `matmul_m1_k256_n512_ty1`; all 265 decoded H14 tasks parse this way"],
        ["Packed 15-bit W/H", "Confirmed at Common `0x0000` and `0x0014`: `1→0x00010001`, `8→0x00080008`, `224→0x00e000e0`.", "`binary_add_1x64x1x1`, `binary_add_1x64x8x8`, `binary_add_1x3x224x224`"],
        ["Non-Common `+0x3c00` remap", "Refuted as an H14 stream encoding rule. Every H14 record uses old bases `0x0500`..`0x1900`; no record uses `0x4100`..`0x5500`. The delta is only the map's old-to-modern presentation transform.", "all 172 decoded H14 cases"],
        ["TileDMA DataSetId bits 8:15", "Location is consistent but not experimentally confirmed: source-1/source-2/destination config words carry `0x0e` in bits 8:15 throughout this campaign; no MIL parameter controls the ID.", "`binary_add_1x64x1x1`, `matmul_m1_k256_n512_ty1`, `softmax_1x64x8x8`"],
        ["KernelDMA DataSetId bits 8:15", "Not confirmed: all written `CoeffDMAConfig0..15` words have zero in bits 8:15. Their bits 16:23 vary with task position/allocation instead.", "`conv_k1_c1024_n1024_s1_bias0`, `matmul_m1_k256_n512_ty1`, `binary_pow_1x1024x1x1`"],
        ["SparseKernelBlockSize", "The detailed map's `0x19f8` prediction is refuted: that word is never written. The prose prediction at `0x1a00` is supported: the same pow case writes `0x00000000` in t1 and `0x00000080` in t2. The campaign does not establish the value formula.", "`binary_pow_1x1024x1x1#t1`, `binary_pow_1x1024x1x1#t2`"],
        ["Block extents", "Common 19, L2 25, PE 5, NE 5, TileDMA source 53, TileDMA destination 10, and KernelDMA 70 match decoded writes. This makes `0x11d4` and `0x1528` outside the mapped stream blocks despite the map listing them; `0x1524` is written.", "`binary_add_1x64x1x1`"],
    ]
    lines += table(["Prediction", "Result", "Oracle evidence"], prediction_rows)

    lines += ["", "## Task-stream delta", ""]
    h14_dense = h14_scatter = h13_records = 0
    for doc in h13:
        h13_records += sum(len(task["records"]) for task in doc["task_descriptors"])
    for doc in h14:
        for task in doc["task_descriptors"]:
            for record in task["records"]:
                if int(record["header"], 16) & 0x80000000:
                    h14_scatter += 1
                else:
                    h14_dense += 1
    task_rows = [
        ["Header", "10 words", "8 words", "Drop H13 next-pointer/linked-size conventions; emit H14 `TID | task_words<<16` in h0 and seven control words."],
        ["Task size", "First: program `task_words_minus_one+1`; later: prior h1 bits 16:24 plus one", "h0 bits 16:26 give exact words", "Compute each H14 task independently. `binary_add_1x64x1x1` is 61 words; `matmul_m1_k256_n512_ty1` is 38 then 85."],
        ["Link/alignment", "h7 is next section-relative byte offset; final h7=0", "No link; tasks are 16-byte aligned and zero-size 16-byte prefixes/padding are skipped", "Replace linked traversal with aligned sequential tasks."],
        ["Register records", f"{h13_records} dense records; header `((count-1)<<26)|byte_address`", f"{h14_dense} dense + {h14_scatter} scatter; base is a word index in bits 0:14", "Dense: count-minus-one bits 15:20. Scatter: bit31 plus a 16-bit following-word mask in bits 15:30."],
        ["Block addresses", "Common `0x00000`; others `0x04800`..`0x1f800`", "Common `0x0000`; others `0x0500`..`0x1900`", "Change every non-Common base; do not apply the modern `+0x3c00` display remap."],
        ["Task decomposition", "1/2/3/5/6-task objects: 98/59/13/1/1", "1/2/3/5/6-task objects: 98/60/12/1/1", "Do not assume identical task count: `matmul_m64_k256_n1024_ty1` is H13 3 tasks and H14 2."],
    ]
    lines += table(["Property", "H13", "H14", "Emitter delta"], task_rows)

    lines += ["", "## Program and tensor descriptor delta", ""]
    assert all(a["tensor_descriptors"] == b["tensor_descriptors"] for a, b in zip(h13, h14))
    for doc in h14:
        pd = doc["program_descriptor"]
        delta = int(pd["constant_address"], 16) - int(pd["text_address"], 16)
        assert delta == align_up(pd["text_words"] * 4, 64)
    for doc in h13:
        pd = doc["program_descriptor"]
        tasks = doc["task_descriptors"]
        used = tasks[0]["size_bytes"] if len(tasks) == 1 else int(tasks[-2]["header_words"][7], 16) + tasks[-1]["size_bytes"]
        assert int(pd["constant_address"], 16) - int(pd["text_address"], 16) == align_up(used, 128)
    descriptor_rows = [
        ["Program command kind/size", "kind 1, `0x880` bytes", "kind 4, `0x890` bytes", "All 172 pairs"],
        ["Text address", "resource-dependent", "resource-dependent", "Same in 135/172 pairs; H14 is lower in 37 because resources moved"],
        ["Text→const offset", "128-byte alignment of linked-task extent", "`align_up(text_words*4, 64)`", "All 172 pairs; add C64 is `0x200` vs `0x140`"],
        ["Task metadata", "code 538, count, first size-minus-one", "count and `text_words`; no H13 code/first-size fields", "`binary_add_1x64x1x1`, `matmul_m1_k256_n512_ty1`"],
        ["Resources", "slots 0..2 carry allocation addresses as needed; slots 3..4 are zero", "slots 0..3 are zero; slot 4 is `0x30000000` in every case", "All 172 pairs"],
        ["H14 32-word trailer", "absent", "text address, kind=4, text words, task count, constants/sentinels, function name `main`, plus unresolved words 18/20/28", "All H14 cases; compare `binary_add_1x64x1x1` and `matmul_m1_k256_n512_ty1`"],
        ["Tensor descriptors", "binding, element code, shape, strides, total bytes", "byte-for-byte same decoded fields", "All 172 paired cases; element code is 5 for campaign fp16 tensors"],
    ]
    lines += table(["Field", "H13", "H14", "Evidence/result"], descriptor_rows)

    lines += ["", "## Complete H14 block tables", "",
              "The map column reports freedomtan's name, even where observed behavior challenges that name. The H13 relation compares the same block-relative slot for the same MIL case and task index. `same`, `changed`, and one-sided counts are write comparisons, not semantic equivalence.", ""]
    for title, name, base, count, h13_base, h13_count in BLOCKS:
        lines += [f"### {title} (`0x{base:04x}`, {count} words)", ""]
        if title in BITFIELDS:
            lines += [BITFIELDS[title], ""]
        rows = []
        names = NAMES[name]
        assert len(names) == count, (name, len(names), count)
        for index in range(count):
            address = base + index * 4
            items = observations[address]
            distinct = {value for value, _ in items}
            status = FORMULAS.get(address)
            if status is None:
                status = "invariant/omitted" if len(distinct) <= 1 else "unresolved"
            modern = address if name == "common" else address + 0x3C00
            map_name = names[index] or "—"
            meaning = MEANINGS.get(address, "")
            if meaning:
                map_name += f" — {meaning}"
            rows.append([
                f"`0x{address:04x}`", f"`0x{modern:04x}`", map_name,
                value_summary(items, task_count), status,
                f"`0x{h13_base + index * 4:05x}`" if index < h13_count else "—",
                h13_relation(h13, h14, name, index * 4, h13_count), evidence(items),
            ])
        lines += table(["H14 old", "Modern", "Map name/meaning", "Observed values", "Formula/status", "H13 address", "Paired value relation", "Oracle evidence"], rows)
        lines.append("")

    lines += ["## Records outside the declared blocks", ""]
    outside_rows = [[f"`0x{address:04x}`"] for address in outside] or [[f"None across all {task_count} decoded H14 tasks"]]
    lines += table(["Address"], outside_rows)

    elementwise = [d for d in h14 if d["family"] == "binary_runtime" and d["parameters"]["operation"] in {"add", "mul", "maximum", "minimum", "sub"}]
    unary_docs = [d for d in h14 if d["family"] == "unary"]
    matvec = [d for d in h14 if d["family"] == "matmul"]
    ew_words, ew_headers = family_blockers(elementwise)
    unary_words, unary_headers = family_blockers(unary_docs)
    mm_words, mm_headers = family_blockers(matvec)
    ew_shapes = sorted({"×".join(map(str, d["parameters"]["shape"])) for d in elementwise})
    unary_by_op: dict[str, set[int]] = defaultdict(set)
    for doc in unary_docs:
        unary_by_op[doc["parameters"]["operation"]].add(doc["parameters"]["shape"][1])
    unary_envelope = "; ".join(f"{op}: C={','.join(map(str, sorted(channels)))}" for op, channels in sorted(unary_by_op.items()))
    plan_rows = [
        ["1", "Elementwise add/mul/max/min/sub", "Template-encodable at the ten sampled fp16 shapes: " + ", ".join(ew_shapes), "Resolve geometry-varying " + ", ".join(ew_words) + "; header " + ", ".join(ew_headers) + ". Operation word `0x0900` and Common geometry are resolved.", "`binary_add_1x64x1x1`, `binary_mul_1x128x16x16`, `binary_sub_1x3x224x224`"],
        ["2", "Unary", "Template-encodable only at sampled channels. " + unary_envelope, "Resolve operation-dependent " + ", ".join(unary_words) + "; header " + ", ".join(unary_headers) + ". Each sampled object is one task, but the campaign lacks independent H/W sweeps.", "`unary_abs_c64`, `unary_abs_c4096`, `unary_exp_c512`, `unary_sqrt_c512`"],
        ["3", "Matvec (`transpose_y=true` matmul)", "Template-encodable grid M={1,2,8,64}, K={256,512,1024}, N={256,512,1024}; all 36 combinations decoded as two H14 tasks", "Resolve " + ", ".join(mm_words) + "; header " + ", ".join(mm_headers) + ". The 16 coefficient base/size words dominate the gap. `transpose_y=false` has no accepted oracle.", "`matmul_m1_k256_n256_ty1`, `matmul_m64_k1024_n1024_ty1`; `matmul_m1_k256_n256_ty0` is rejected"],
    ]
    lines += ["", "## Ranked H14 backend implementation plan", "",
              "“Template-encodable” means an emitter can reproduce a checked-in point exactly. It does not mean interpolation is safe. The unresolved lists contain every word that varies within the named evidence family without a formula above.", ""]
    lines += table(["Rank", "Family", "Evidence envelope now", "Exact blockers to general encoding", "Oracle evidence"], plan_rows)
    lines += ["", "Implement the shared H14 container and task-record writer first, then copy exact templates for smoke probes, then derive the listed changing words one family at a time. Do not start with convolution, reductions, normalization, real_div, or pow: their extra task graphs add unresolved allocation and SparseKernelBlockSize behavior beyond the three ranked families.", "",
              "No extra oracle was minted for this report. The checked-in campaign already supplies at least three spatial points, seven channel points, five operation points, and the full 4×3×3 matvec grid needed for the formulas and blockers stated here."]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
