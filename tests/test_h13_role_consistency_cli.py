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

SCOPE (Main-labeled, 2026-09-20): this is a BINDING-ROLE ladder, not an
arithmetic-correctness ladder -- it compares no numeric output. It proves
deliverability only: what the runtime must stage is what the decoded
stream reads/writes. Numeric correctness rests on the parity suite's
896-case BYTE comparison against Apple's oracles (byte-parity is a narrow
claim: the task streams match Apple's, not that any device run agrees
with a reference), and on device gates. The arithmetic rungs here lock
the binding contract per op; they do not certify arithmetic results.

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





_BINARY_HEADER = "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"


def _binary_runtime(op):
    return (_BINARY_HEADER +
            "  func main<ios18>(tensor<fp16, [1, 64, 1, 1]> x, "
            "tensor<fp16, [1, 64, 1, 1]> y) {\n"
            f"    tensor<fp16, [1, 64, 1, 1]> z = {op}(x = x, y = y)"
            "[name = string(\"z\")];\n"
            "  } -> (z);\n}\n")


ENV_BCAST = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 64, 8, 8]> x, tensor<fp16, [1, 64, 1, 1]> y) {
    tensor<fp16, [1, 64, 8, 8]> z = add(x = x, y = y)[name = string("z")];
  } -> (z);
}
"""

# The three Apple-refused encoder forms decompose exactly on their value
# domains onto primitives that both Apple's tool decoded and this compiler
# lowers. Preconditions (range-proven, receipt 2026-09-20):
# - int32 less = fp16 less: lengths/arange are non-negative integers <= 375,
#   exactly representable in fp16 (<= 2048), so the comparison result is
#   identical in both domains.
# - logical_and(a, b) = less(0.5, mul(cast a, cast b)): bools are {0,1};
#   cast and mul are exact on {0,1}; the 0.5 threshold separates exactly.
# - reduce_min over a {0,1} mask = 1 - max(1 - x): the complement rides
#   mul(-1.0)+add(1.0) (integer fp16, exact on {0,1}), reduce_max fp16 is
#   decoded, and the outer 1-m is sub with the CONSTANT ON Y -- the exact
#   sub_scalar_1 spelling Apple decoded. No const-on-x sub is needed;
#   every intermediate is an exact integer fp16 value.
# Geometry note: cast rides its decoded [1,1,width,1] band, less rides the
# decoded CHW (375,1,1) band; the reshape between them is a view (no data
# movement, no numeric content).
DECOMP_LOGICAL_AND = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<bool, [1, 1, 375, 1]> a, tensor<bool, [1, 1, 375, 1]> b) {
    tensor<fp16, [1, 1, 375, 1]> fa = cast(dtype = string("fp16"), x = a)[name = string("fa")];
    tensor<fp16, [1, 1, 375, 1]> fb = cast(dtype = string("fp16"), x = b)[name = string("fb")];
    tensor<int32, [3]> shp = const()[name = string("shp"), val = tensor<int32, [3]>([375, 1, 1])];
    tensor<fp16, [375, 1, 1]> far = reshape(shape = shp, x = fa)[name = string("far")];
    tensor<fp16, [375, 1, 1]> fbr = reshape(shape = shp, x = fb)[name = string("fbr")];
    tensor<fp16, [375, 1, 1]> p = mul(x = far, y = fbr)[name = string("p")];
    tensor<fp16, [375, 1, 1]> half = const()[name = string("half"), val = tensor<fp16, [375, 1, 1]>([fp16(0.5), fp16(0.5), fp16(0.5)])];
    tensor<bool, [375, 1, 1]> y = less(x = p, y = half)[name = string("y")];
  } -> (y);
}
"""

DECOMP_REDUCE_MIN = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 1, 375, 375]> x) {
    fp16 neg = const()[name = string("neg"), val = fp16(-1.0)];
    fp16 one = const()[name = string("one"), val = fp16(1.0)];
    tensor<fp16, [1, 1, 375, 375]> nx = mul(x = x, y = neg)[name = string("nx")];
    tensor<fp16, [1, 1, 375, 375]> cx = add(x = nx, y = one)[name = string("cx")];
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([2])];
    tensor<fp16, [1, 1, 1, 375]> m = reduce_max(axes = axes, keep_dims = bool(true), x = cx)[name = string("m")];
    tensor<fp16, [1, 1, 1, 375]> y = sub(x = one, y = m)[name = string("y")];
  } -> (y);
}
"""

COMPILE_CASES = [
    ("same-shape add", SAME_ADD),
    ("broadcasting add", BCAST_ADD),
    ("env broadcasting add", ENV_BCAST),
] + [
    (f"runtime {op} [1,64,1,1]", _binary_runtime(op))
    for op in ("add", "mul", "maximum", "minimum", "sub")
] + [
    ("decomposed logical_and", DECOMP_LOGICAL_AND),
]

# The reduce_min chain is blocked at exactly one row: sub with the
# CONSTANT ON X (h13.nonfoldable-binary). Apple DECODED that form
# (research/oracles/h13/sub_scalar_1_1x1x1x375.json -- the byte-source
# exists), so landing it is queued emitted-template work, not a search.
# The test asserts the refusal contract until the row lands.
BLOCKED_CASES = [
    ("decomposed reduce_min", DECOMP_REDUCE_MIN,
     "h13.nonfoldable-binary"),
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
        for label, mil, code in REFUSE_CASES + BLOCKED_CASES:
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
          f"{len(REFUSE_CASES)} refused cases carry the documented contract, "
          f"{len(BLOCKED_CASES)} blocked case pinned to its queued row)")


if __name__ == "__main__":
    run()
