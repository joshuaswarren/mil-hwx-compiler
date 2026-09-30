#!/usr/bin/env python3
"""Parse captured hwx files and emit C++ template code for the
H14 chain / elementwise / island template tables."""
import sys, struct, json, hashlib
from pathlib import Path

sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om

INPUT = Path('/tmp/hwx_batch5')

MAGIC = 0xfeedfacf
SUBTYPES = {"h13": 5, "h14": 6}

def parse_hwx(data):
    if len(data) < 32:
        raise ValueError("truncated")
    magic, _, subtype, _, command_count, command_bytes, _, _ = struct.unpack_from("<8I", data)
    assert magic in (MAGIC, 0xbeefface), f"magic 0x{magic:08x}"
    sections = {}
    tensors = []
    programs = []
    cursor = 32
    command_end = cursor + command_bytes
    for cmd_idx in range(command_count):
        cmd, size = struct.unpack_from("<2I", data, cursor)
        kind = struct.unpack_from("<I", data, cursor + 8)[0] if size >= 12 else None
        if cmd == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = fields[2].rstrip(b"\x00").decode()
            sec_cursor = cursor + 72
            for _ in range(fields[-2]):
                entry = struct.unpack_from("<16s16s2Q8I", data, sec_cursor)
                section = entry[0].rstrip(b"\x00").decode()
                sections[(segment, section)] = {"address": entry[2], "size": entry[3], "offset": entry[4]}
                sec_cursor += 80
        elif cmd == 4 and kind == 3:
            tensors.append({
                "binding": struct.unpack_from("<I", data, cursor + 0x14)[0],
                "shape": list(struct.unpack_from("<4I", data, cursor + 0x28)),
                "total_bytes": struct.unpack_from("<Q", data, cursor + 0x70)[0],
            })
        elif cmd == 4 and kind == 4:
            programs.append({
                "text_address": struct.unpack_from("<Q", data, cursor + 0x10)[0],
                "constant_address": struct.unpack_from("<Q", data, cursor + 0x20)[0],
                "text_words": struct.unpack_from("<I", data, cursor + 0x824)[0],
                "task_count": struct.unpack_from("<I", data, cursor + 0x830)[0],
            })
        cursor += size
    text = sections[("__TEXT", "__text")]
    constants = sections[("__TEXT", "__const")]
    return {
        "text_offset": text["offset"], "text_size": text["size"],
        "const_offset": constants["offset"], "const_size": constants["size"],
        "programs": programs, "tensors": tensors,
        "raw": data,
    }


def emit_text_words(name, info, indent=4):
    raw = info["raw"][info["text_offset"]:info["text_offset"] + info["text_size"]]
    words = list(struct.unpack(f"<{len(raw)//4}I", raw))
    out = [f"static constexpr std::uint32_t {name}[] = {{"]
    for start in range(0, len(words), 8):
        chunk = " ".join(f"0x{w:08x}," for w in words[start:start + 8])
        out.append(" " * indent + chunk)
    out.append("};")
    return "\n".join(out)


def emit_const_runs(name, info):
    """Walk the first 128 bytes of constant section, emit ConstantRun for each
    nonzero halfword run."""
    raw = info["raw"][info["const_offset"]:info["const_offset"] + 128]
    halfwords = list(struct.unpack(f"<{len(raw)//2}H", raw))
    runs = []
    i = 0
    while i < len(halfwords):
        if halfwords[i] != 0:
            j = i
            while j < len(halfwords) and halfwords[j] == halfwords[i]:
                j += 1
            runs.append((i, halfwords[i], j - i))
            i = j
        else:
            i += 1
    out = [f"static constexpr ConstantRun {name}[] = {{"]
    for idx, bits, count in runs:
        out.append(f"    {{{idx}, 0x{bits:04x}, {count}}},")
    out.append("};")
    return "\n".join(out)


def main():
    cases = [
        ("env_chain_creadout_h16_s128", "F5 chain (3-task)"),
        ("env_bcast_mul_1x16x1x128_runtime_1x16x1x1", "F1b mul (head scale)"),
        ("env_bcast_mul_1x16x1x1_runtime_1x16x1x128", "F1b mul (head broadcast)"),
        ("env_bcast_add_1x16x1x128_runtime_1x16x1x1", "F1b add (head scale)"),
        ("env_bcast_add_1x16x1x1_runtime_1x16x1x128", "F1b add (head broadcast)"),
    ]
    for case, desc in cases:
        path = INPUT / f"{case}.hwx"
        if not path.is_file():
            print(f"  {case}: missing")
            continue
        data = path.read_bytes()
        info = parse_hwx(data)
        n_words = info["text_size"] // 4
        print(f"=== {case} ({desc}) ===")
        print(f"  bytes={len(data)}, text_words={n_words}, tasks={info['programs'][0]['task_count']}, const_bytes={info['const_size']}")
        print(f"  tensors={len(info['tensors'])}, shapes={[t['shape'] for t in info['tensors']]}")
        print(emit_text_words(f"k_{case}_text", info))
        print(emit_const_runs(f"k_{case}_const", info))
        print()


if __name__ == "__main__":
    main()