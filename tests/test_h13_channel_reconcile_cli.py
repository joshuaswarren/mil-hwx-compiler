#!/usr/bin/env python3
"""Composition-level channel reconciliation regression.

Compiles the ac-head composition (slice + select + scores + add + softmax)
as a multi-program package and verifies that every intermediate tensor's
producer output channel matches every consumer's input channel.

FAILING-FIRST: currently fails on 3 edges (bd: slice->select, masked:
select->add, add: add->softmax) because the per-family taskSurfaceChannels
conventions assign different channels. The shared channel routing fix must
make ALL edges pass.
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


def decode_selector_channels(anec_path):
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
        registers = {}
        cursor = 10 + (1 if words[9] & 3 == 3 else 0)
        while cursor < len(words):
            header = words[cursor]
            run_count = (header >> 26) + 1
            base = header & 0x03FFFFFF
            for step in range(run_count):
                registers[base + step * 4] = words[cursor + 1 + step]
            cursor += 1 + run_count
        src_chs = set()
        dst_chs = set()
        if slots[0] >= 4 and (registers.get(0x13800, 0) & 0xFF):
            src_chs.add(slots[0])
        if slots[1] >= 4 and (registers.get(0x13804, 0) & 0x80):
            src_chs.add(slots[1])
        if slots[2] >= 4 and (registers.get(0x17800, 0) & 0xFF):
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
        print(f"programs: {len(programs)}")

        # Build the tensor-to-channel map from the manifest
        tensor_channels = {}
        tensor_producer = {}
        for pi, prog in enumerate(programs):
            for b in prog["inputs"]:
                name = b["name"]
                tensor_channels.setdefault(name, set()).add(b["index"])
            for b in prog["outputs"]:
                name = b["name"]
                tensor_channels.setdefault(name, set()).add(b["index"])
                tensor_producer[name] = (pi, b["index"])

        # Check chaining
        broken = []
        for name in ("bd", "masked", "scores", "add", "smax"):
            if name not in tensor_producer:
                continue
            prod_prog, prod_ch = tensor_producer[name]
            consumer_chs = set()
            for pj, prog in enumerate(programs):
                if pj == prod_prog:
                    continue
                for b in prog["inputs"]:
                    if b["name"] == name:
                        consumer_chs.add((pj, b["index"]))
            mismatched = [(pj, ch) for pj, ch in consumer_chs if ch != prod_ch]
            if mismatched:
                broken.append((name, prod_ch, mismatched))

        if broken:
            print("FAIL: broken intermediate channel chains:")
            for name, prod_ch, mismatches in broken:
                print(f"  {name}: producer ch{prod_ch}, consumers on {mismatches}")
            sys.exit(1)

        print("PASS: all intermediate channels chain correctly")


if __name__ == "__main__":
    sys.exit(main())
