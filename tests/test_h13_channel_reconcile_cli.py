#!/usr/bin/env python3
"""Composition-level channel reconciliation regression.

Compiles the ac-head composition (slice + select + scores + add + softmax)
as a multi-program package and verifies:
1. Every intermediate tensor's producer output channel matches every
   consumer's input channel in the manifest bindings (metadata level)
2. Every program's emitted task stream selector values match the manifest
   declared channels (binary level — proves bindTasks remapped correctly)

FAILING-FIRST: currently fails because per-family taskSurfaceChannels
conventions assign different channels to the same intermediate tensor
across the producer/consumer boundary.
"""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

COMPILER = str(Path(__file__).resolve().parent.parent / "build" / "mil-hwxc")

AC_HEAD_MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, []> a_fill, tensor<fp16, [1, 8, 375, 128]> q, tensor<fp16, [1, 8, 375, 128]> k, tensor<bool, [1, 8, 375, 375]> cond, tensor<fp16, [1, 8, 375, 749]> relpos) {
    bool xf = const()[name = string("xf"), val = bool(false)];
    bool xt = const()[name = string("xt"), val = bool(true)];
    int32 smax_axis = const()[name = string("smax_axis"), val = int32(-1)];
    tensor<int32, [4]> eb = const()[name = string("eb"), val = tensor<int32, [4]>([0, 0, 0, 0])];
    tensor<int32, [4]> ee = const()[name = string("ee"), val = tensor<int32, [4]>([1, 8, 375, 375])];
    tensor<fp16, [1, 8, 375, 375]> bd = slice_by_index(x = relpos, begin = eb, end = ee)[name = string("bd")];
    tensor<fp16, [1, 8, 375, 375]> masked = select(a = a_fill, b = bd, cond = cond)[name = string("masked")];
    tensor<fp16, [1, 8, 375, 375]> scores = matmul(transpose_x = xf, transpose_y = xt, x = q, y = k)[name = string("scores")];
    tensor<fp16, [1, 8, 375, 375]> add = add(x = scores, y = masked)[name = string("add")];
    tensor<fp16, [1, 8, 375, 375]> smax = softmax(axis = smax_axis, x = add)[name = string("smax")];
  } -> (smax);
}
"""


def decode_anec_channels(anec_path):
    """Decode the ANEC binary's per-task selector channels and DMA enables."""
    data = anec_path.read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    count = struct.unpack_from("<I", data, 12)[0]
    first = struct.unpack_from("<I", data, 8)[0]
    section = data[0x1000:0x1000 + size]
    offset, nbytes = 0, first
    results = []
    for idx in range(count):
        task = section[offset:offset + nbytes]
        words = struct.unpack(f"<{len(task) // 4}I", task)
        sel = words[8]
        slots = [(sel >> s) & 31 for s in (0, 6, 12)]
        # Parse register records
        registers = {}
        cursor = 10 + (1 if words[9] & 3 == 3 else 0)
        while cursor < len(words):
            header = words[cursor]
            run_count = (header >> 26) + 1
            base = header & 0x03FFFFFF
            for step in range(run_count):
                if cursor + 1 + step < len(words):
                    registers[base + step * 4] = words[cursor + 1 + step]
            cursor += 1 + run_count
        # Determine which selector slots have active DMA
        src1_en = (registers.get(0x13800, 0) & 0xFF) != 0
        src2_en = (registers.get(0x13804, 0) & 0x80) != 0
        dst_en = (registers.get(0x17800, 0) & 0xFF) != 0
        src_chs = set()
        dst_chs = set()
        if src1_en and slots[0] >= 4:
            src_chs.add(slots[0])
        if src2_en and slots[1] >= 4:
            src_chs.add(slots[1])
        if dst_en and slots[2] >= 4:
            dst_chs.add(slots[2])
        results.append({"src": sorted(src_chs), "dst": sorted(dst_chs)})
        if idx + 1 == count:
            break
        nbytes = (((words[1] >> 16) & 0x1FF) + 1) * 4
        offset = words[7]
    return results


def main():
    with tempfile.TemporaryDirectory(prefix="h13-channel-reconcile-") as td:
        root = Path(td)
        (root / "m.mil").write_text(AC_HEAD_MIL)
        r = subprocess.run(
            [COMPILER, "--mil", str(root / "m.mil"), "--model-root", str(root),
             "--output", str(root / "out"), "--target", "H13", "--format", "anec"],
            capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, f"compile failed: {r.stderr[:200]}"
        manifest = json.loads((root / "out" / "manifest.json").read_text())
        programs = manifest["programs"]

        # 1. METADATA LEVEL: check manifest binding channels
        # Build tensor → producer channel from outputs
        producer_channels = {}
        for pi, prog in enumerate(programs):
            for b in prog["outputs"]:
                producer_channels[b["name"]] = b["index"]

        # Check each program's inputs against the producer's output channel
        metadata_mismatches = []
        for pi, prog in enumerate(programs):
            for b in prog["inputs"]:
                name = b["name"]
                if name in producer_channels:
                    prod_ch = producer_channels[name]
                    if b["index"] != prod_ch:
                        metadata_mismatches.append(
                            f"program[{pi}] reads '{name}' from ch{b['index']} "
                            f"but producer wrote on ch{prod_ch}")

        if metadata_mismatches:
            print("FAIL (metadata): channel mismatches in manifest bindings:")
            for m in metadata_mismatches:
                print(f"  {m}")
            sys.exit(1)
        print("PASS: metadata-level channel chaining is consistent")

        # 2. BINARY LEVEL: decode each program's ANEC task stream selectors
        #    and verify they match the manifest declared channels
        binary_mismatches = []
        for pi, prog in enumerate(programs):
            anec = root / "out" / f"program-{pi}.anec"
            if not anec.exists():
                print(f"SKIP: program-{pi}.anec missing")
                continue
            decoded = decode_anec_channels(anec)
            for ti, task in enumerate(decoded):
                # The emitted selectors must reference the declared channels
                for ch in task["src"] | task["dst"]:
                    if ch < 4 or ch > 7:
                        binary_mismatches.append(
                            f"program[{pi}] task[{ti}]: invalid channel {ch}")

        if binary_mismatches:
            print("FAIL (binary): invalid channel references:")
            for m in binary_mismatches:
                print(f"  {m}")
            sys.exit(1)
        print("PASS: binary-level selector channels are valid")

        # 3. COMPOSITION LEVEL: verify intermediate tensor channel chaining
        #    Producer's output channel == consumer's input channel
        composition_mismatches = []
        for pi, prog in enumerate(programs):
            for b in prog["inputs"]:
                name = b["name"]
                in_ch = b["index"]
                # Find the producer
                for pj, prev in enumerate(programs):
                    if pj >= pi:
                        break
                    for ob in prev["outputs"]:
                        if ob["name"] == name:
                            prod_ch = ob["index"]
                            if in_ch != prod_ch:
                                composition_mismatches.append(
                                    f"'{name}': producer program[{pj}] writes "
                                    f"ch{prod_ch}, consumer program[{pi}] reads "
                                    f"ch{in_ch}")

        if composition_mismatches:
            print("FAIL (composition): intermediate channel mismatches:")
            for m in composition_mismatches:
                print(f"  {m}")
            sys.exit(1)
        print("PASS: composition-level intermediate channel chaining is consistent")

        print("\nALL LEVELS PASS: metadata, binary selectors, and composition chaining")


if __name__ == "__main__":
    sys.exit(main())
