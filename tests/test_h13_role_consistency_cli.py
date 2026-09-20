"""Within-program manifest role consistency: declared bindings vs decoded
selectors + DMA.

Main-directed contract (2026-09-20): the runtime worker stages every tensor
by its manifest binding index into that channel's BO, then the ANEC task
reads/writes the channels its own selectors + DMA registers name. Each
runtime handle is per-program, so the contract is WITHIN one program:

    for every program: declared input channels == selector+DMA-named
    source channels, declared output channels == selector+DMA-named
    destination channels.

Cross-program channel equality is NOT a contract (runtime_worker.cpp
load_programs gives each program its own handle/chans; execute_program
copies through the manifest tensor names).

This regression walks a representative family matrix and fails on the
first program whose declarations disagree with the decoded stream roles.
"""

import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

COMPILER = sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc"

SAME_ADD = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 8, 375, 375]> x, tensor<fp16, [1, 8, 375, 375]> y) {
    tensor<fp16, [1, 8, 375, 375]> z = add(x = x, y = y)[name = string("z")];
  } -> (z);
}
"""

BCAST_ADD = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 256, 8, 8]> x, tensor<fp16, [1, 256, 1, 1]> y) {
    tensor<fp16, [1, 256, 8, 8]> z = add(x = x, y = y)[name = string("z")];
  } -> (z);
}
"""





COMPILE_CASES = [
    ("same-shape add", SAME_ADD),
]

REFUSE_CASES = [
    ("broadcasting add", BCAST_ADD,
     "h13.runtime-broadcast-needs-materialized-operand"),
]


def compile_package(root, name, mil):
    (root / f"{name}.mil").write_text(mil)
    out = root / name
    run = subprocess.run(
        [COMPILER, "--mil", str(root / f"{name}.mil"), "--model-root",
         str(root), "--output", str(out), "--target", "H13",
         "--format", "anec"],
        capture_output=True, text=True, timeout=300, check=False)
    return run, out


def decode_program_roles(anec_path):
    """Decode every task's selector+DMA roles from one ANEC program.

    Returns (sources, destinations) as sorted channel lists covering all
    tasks: channels the stream reads and writes, per the guard registers
    at 0x13800 (src enable) / 0x17800 (dst enable).
    """
    data = anec_path.read_bytes()
    size = struct.unpack_from("<Q", data, 16)[0]
    section = data[0x1000:0x1000 + size]
    words = struct.unpack(f"<{len(section) // 4}I", section)
    sources, destinations = set(), set()
    cursor = 10 + (1 if words[9] & 3 == 3 else 0)
    task_selectors = [words[8]]
    while cursor < len(words):
        header = words[cursor]
        run_count = (header >> 26) + 1
        base = header & 0x03FFFFFF
        cursor += 1 + run_count
        # Subsequent task headers start with their own selector word at a
        # fixed stride; keep the first task's convention as the baseline
        # and extend with any later selector words discovered.
        if base == 8 and run_count >= 1:
            task_selectors.append(words[cursor - run_count])
    for selectors in task_selectors:
        registers = {}
        c2 = 10 + (1 if words[9] & 3 == 3 else 0)
        while c2 < len(words):
            h = words[c2]
            rc = (h >> 26) + 1
            b = h & 0x03FFFFFF
            for step in range(rc):
                registers[b + step * 4] = words[c2 + 1 + step]
            c2 += 1 + rc
        for shift, role in ((0, "src"), (6, "src"), (12, "dst")):
            channel = (selectors >> shift) & 31
            if channel < 4:
                continue
            if role == "dst":
                if registers.get(0x17800, 0) & 0xFF:
                    destinations.add(channel)
            elif registers.get(0x13800, 0) & 0xFF:
                sources.add(channel)
    return sorted(sources), sorted(destinations)


failures = []
with tempfile.TemporaryDirectory(prefix="h13-role-consistency-") as directory:
    root = Path(directory)
    for label, mil in COMPILE_CASES:
        run, out = compile_package(root, label.replace(" ", "-"), mil)
        if run.returncode != 0:
            failures.append(f"{label}: compile refused: {run.stderr.strip()[:160]}")
            continue
        manifest = json.loads((out / "manifest.json").read_text())
        for prog in manifest["programs"]:
            src_ch, dst_ch = decode_program_roles(out / prog["file"])
            declared_in = sorted(b["index"] for b in prog["inputs"])
            declared_out = sorted(b["index"] for b in prog["outputs"])
            if declared_in != src_ch:
                failures.append(
                    f"{label} [{prog['encoder']}]: declared inputs "
                    f"{declared_in} != decoded sources {src_ch}")
            if declared_out != dst_ch:
                failures.append(
                    f"{label} [{prog['encoder']}]: declared outputs "
                    f"{declared_out} != decoded destinations {dst_ch}")
    for label, mil, code in REFUSE_CASES:
        run, _ = compile_package(root, label.replace(" ", "-"), mil)
        if run.returncode == 0:
            failures.append(f"{label}: compiled but must be refused ({code})")
        elif code not in run.stderr:
            failures.append(
                f"{label}: refused with the wrong contract: "
                f"{run.stderr.strip()[:160]}")

if failures:
    for f in failures:
        print(f"FAIL: {f}")
    sys.exit(1)
print("h13 role consistency cli: PASS "
      f"({len(COMPILE_CASES)} compiled cases declarations==decoded roles, "
      f"{len(REFUSE_CASES)} refused cases carry the documented contract)")
