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





ENV_BCAST = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 64, 8, 8]> x, tensor<fp16, [1, 64, 1, 1]> y) {
    tensor<fp16, [1, 64, 8, 8]> z = add(x = x, y = y)[name = string("z")];
  } -> (z);
}
"""

COMPILE_CASES = [
    ("same-shape add", SAME_ADD),
    ("broadcasting add", BCAST_ADD),
    ("env broadcasting add", ENV_BCAST),
]

REFUSE_CASES = []


def compile_package(root, name, mil):
    (root / f"{name}.mil").write_text(mil)
    out = root / name
    run = subprocess.run(
        [COMPILER, "--mil", str(root / f"{name}.mil"), "--model-root",
         str(root), "--output", str(out), "--target", "H13",
         "--format", "anec"],
        capture_output=True, text=True, timeout=300, check=False)
    return run, out


import sys as _sys
sys_path = str(Path(__file__).resolve().parent.parent / "research")
if sys_path not in _sys.path:
    _sys.path.insert(0, sys_path)
from h13_td import split_h13_tasks  # noqa: E402
from inspect_anec import h13_task_registers  # noqa: E402

H13_DMA_SELECTORS = ((0x13800, 0), (0x13804, 6), (0x17800, 12))
# Enable values decoded from the whole linked task streams of the oracle
# corpus and our emitted packages: src surface reads are 0x33881/0x33880,
# the matvec-class L2 staging read is 0x48880; dst surface writes are
# 0xc1 (375-square class) and 0x40000c1, and 0xc0 writes L2 only.
# 0x8880 marks a disabled slot. Validated per task by
# test_h13_role_consistency_cli.py against both single- and multi-task
# positive controls.
H13_SRC_SURFACE = (0x00033881, 0x00033880)
H13_SRC_L2 = 0x00048880
H13_DST_SURFACE = (0x000000c1, 0x040000c1)
H13_DST_L2 = 0x000000c0
H13_SLOT_DISABLED = (0x00000000, 0x00008880)


def decode_program_roles_anec(section, task_words_minus_one, task_count):
    """Decode surface vs L2 roles per task with exact task boundaries.

    Returns (surface_sources, surface_destinations, l2_reads, l2_writes,
    per_task) where per_task records each task's selector-named channels,
    its DMA enable values, and whether that task moves data through L2.
    A selector slot counts as a surface role only when its own task's
    paired enable register carries a surface-enable value; a named slot
    whose enable is absent (or an enable whose slot is unnamed) is
    reported per task so the classification stays auditable.
    """
    tasks = split_h13_tasks(section, task_words_minus_one, task_count)
    surface_src, surface_dst = set(), set()
    l2_reads, l2_writes = 0, 0
    per_task = []
    for index, task in enumerate(tasks):
        registers = h13_task_registers(task)
        words = struct.unpack(f"<{len(task) // 4}I", task)
        selectors = words[8]
        roles = []
        for enable_address, shift in H13_DMA_SELECTORS:
            channel = (selectors >> shift) & 31
            enable = registers.get(enable_address, 0)
            dst_slot = enable_address == 0x17800
            if enable in H13_SLOT_DISABLED:
                continue
            if dst_slot:
                if enable in H13_DST_SURFACE and channel >= 4:
                    surface_dst.add(channel)
                    roles.append(("dst", f"ch{channel}", hex(enable)))
                elif enable == H13_DST_L2:
                    l2_writes += 1
                    roles.append(("l2-write", "-", hex(enable)))
                else:
                    roles.append(("dst?", f"ch{channel}", hex(enable)))
            else:
                if enable in H13_SRC_SURFACE and channel >= 4:
                    surface_src.add(channel)
                    roles.append(("src", f"ch{channel}", hex(enable)))
                elif enable == H13_SRC_L2:
                    l2_reads += 1
                    roles.append(("l2-read", "-", hex(enable)))
                else:
                    roles.append(("src?", f"ch{channel}", hex(enable)))
        per_task.append({"task": index, "selectors": hex(selectors),
                         "roles": roles})
    return (sorted(surface_src), sorted(surface_dst),
            l2_reads, l2_writes, per_task)


def decode_package_program(anec_path):
    data = anec_path.read_bytes()
    first = struct.unpack_from("<I", data, 8)[0]
    count = struct.unpack_from("<I", data, 12)[0]
    size = struct.unpack_from("<Q", data, 16)[0]
    section = data[0x1000:0x1000 + size]
    return decode_program_roles_anec(section, first // 4 - 1, count)


def decode_oracle_program(oracle):
    prog = oracle["programs"][0]
    section = bytes.fromhex(prog["task_section"])
    return decode_program_roles_anec(
        section, prog["task_words_minus_one"], prog["task_count"])


def run():
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
                src_ch, dst_ch, l2r, l2w, per = decode_package_program(
                    out / prog["file"])
                declared_in = sorted(b["index"] for b in prog["inputs"])
                declared_out = sorted(b["index"] for b in prog["outputs"])
                if declared_in != src_ch:
                    failures.append(
                        f"{label} [{prog['encoder']}]: declared inputs "
                        f"{declared_in} != decoded surface sources {src_ch} "
                        f"(l2_reads={l2r})")
                if declared_out != dst_ch:
                    failures.append(
                        f"{label} [{prog['encoder']}]: declared outputs "
                        f"{declared_out} != decoded surface destinations "
                        f"{dst_ch} (l2_writes={l2w})")
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


if __name__ == "__main__":
    run()
