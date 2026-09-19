#!/usr/bin/env python3
"""Probe every remaining encoder op form against the H13 target.

Census of the Parakeet encoder statements whose H13 dispatch status is not
already settled by a device-certified island (matmul/conv/linear/norm/unary/
layout) or a decoded oracle table (less/floor/floor_div/select/tile/pad).
Each probe is a minimal MIL mirroring the exact operand kinds (const vs
runtime), dtypes, and shapes the encoder source uses.

  python3 research/encoder_form_census.py [path/to/mil-hwxc]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

COMPILER = Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve()


def program(arguments: str, body: list[str], result: str) -> str:
    lines = [
        "program(1.3)",
        "[buildInfo = dict<string, string>({})]",
        "{",
        f"  func main<ios18>({arguments}) {{",
        *(f"    {line}" for line in body),
        f"  }} -> ({result});",
        "}",
    ]
    return "\n".join(lines) + "\n"


def const_f16(name: str, shape: tuple[int, ...], value: float = 0.5) -> str:
    elems = 1
    for dim in shape:
        elems *= dim
    vals = ", ".join(["0x3800"] * elems)  # fp16 0.5
    shape_s = ", ".join(str(d) for d in shape)
    return (f'tensor<fp16, [{shape_s}]> {name} = const()[name = string("{name}"), '
            f'val = tensor<fp16, [{shape_s}]>({vals})];')


def const_i32(name: str, shape: tuple[int, ...], value: int = 2) -> str:
    elems = 1
    for dim in shape:
        elems *= dim
    vals = ", ".join([str(value)] * elems)
    shape_s = ", ".join(str(d) for d in shape)
    return (f'tensor<int32, [{shape_s}]> {name} = const()[name = string("{name}"), '
            f'val = tensor<int32, [{shape_s}]>({vals})];')


def probe(name: str, arguments: str, body: list[str], result: str) -> dict:
    return {"name": name, "mil": program(arguments, body, result)}


def arange_const_i32(name: str, count: int) -> str:
    vals = ", ".join(str(i) for i in range(count))
    return (f'tensor<int32, [{count}]> {name} = const()[name = string("{name}"), '
            f'val = tensor<int32, [{count}]>({vals})];')


def arange_const_f16(name: str, count: int) -> str:
    vals = ", ".join(["0x3800"] * count)
    return (f'tensor<fp16, [{count}]> {name} = const()[name = string("{name}"), '
            f'val = tensor<fp16, [{count}]>({vals})];')


PROBES: list[dict] = []

# --- less (4 encoder statements) ---
for width in (1500, 750, 375):
    PROBES.append(probe(
        f"less_cmask_f16_{width}",
        f"tensor<fp16, [1, 1]> y",
        [arange_const_f16("x", width),
         f'tensor<bool, [1, {width}]> out = less(x = x, y = y)[name = string("out")];'],
        "out"))
# output_mask: int32 comparison
PROBES.append(probe(
    "less_output_mask_i32",
    "tensor<int32, [1, 1]> y",
    [arange_const_i32("x", 375),
     'tensor<bool, [1, 375]> out = less(x = x, y = y)[name = string("out")];'],
    "out"))

# --- floor (3) ---
PROBES.append(probe(
    "floor_f16_1",
    "tensor<fp16, [1]> x",
    ['tensor<fp16, [1]> out = floor(x = x)[name = string("out")];'],
    "out"))

# --- floor_div (3: one int32, two fp16) ---
PROBES.append(probe(
    "floor_div_int32",
    "tensor<int32, [1]> x",
    [const_i32("y", (), 2),
     'tensor<int32, [1]> out = floor_div(x = x, y = y)[name = string("out")];'],
    "out"))
PROBES.append(probe(
    "floor_div_f16",
    "tensor<fp16, [1]> x",
    [const_f16("y", ()),
     'tensor<fp16, [1]> out = floor_div(x = x, y = y)[name = string("out")];'],
    "out"))

# --- tile (1: the encoder mask form) ---
PROBES.append(probe(
    "tile_mask_1x1x375",
    "tensor<bool, [1, 1, 375]> x",
    [const_i32("reps", (3,), 1),  # placeholder, replaced below
     ],
    "out"))

# tile needs literal reps [1, 375, 1]
PROBES[-1] = probe(
    "tile_mask_1x1x375",
    "tensor<bool, [1, 1, 375]> x",
    ['tensor<int32, [3]> reps = const()[name = string("reps"), val = tensor<int32, [3]>([1, 375, 1])];',
     'tensor<bool, [1, 375, 375]> out = tile(reps = reps, x = x)[name = string("out")];'],
    "out")

# --- logical_and / logical_not (1 each) ---
PROBES.append(probe(
    "logical_and_mask",
    "tensor<bool, [1, 375, 375]> x, tensor<bool, [1, 375, 375]> y",
    ['tensor<bool, [1, 375, 375]> out = logical_and(x = x, y = y)[name = string("out")];'],
    "out"))
PROBES.append(probe(
    "logical_not_mask",
    "tensor<bool, [1, 1, 375, 375]> x",
    ['tensor<bool, [1, 1, 375, 375]> out = logical_not(x = x)[name = string("out")];'],
    "out"))

# --- reduce_min (1) ---
PROBES.append(probe(
    "reduce_min_mask",
    "tensor<int32, [1, 1, 375, 375]> x",
    ['tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>(2)];',
     'tensor<bool, []> keep = const()[name = string("keep"), val = bool(false)];',
     'tensor<int32, [1, 1, 375]> out = reduce_min(axes = axes, keep_dims = keep, x = x)[name = string("out")];'],
    "out"))

# --- casts (11 encoder statements, deduped to distinct dtype pairs/forms) ---
CAST_FORMS = [
    ("cast_f32_to_f16_in", "tensor<fp32, [1, 3000, 128]> x", "fp16"),
    ("cast_i32_to_f16_1", "tensor<int32, [1]> x", "fp16"),
    ("cast_b_to_f16_1500x1", "tensor<bool, [1, 1, 1500, 1]> x", "fp16"),
    ("cast_f16_to_i32_1", "tensor<fp16, [1]> x", "int32"),
    ("cast_b_to_i32_375x375", "tensor<bool, [1, 1, 375, 375]> x", "int32"),
    ("cast_i32_to_b_375", "tensor<int32, [1, 1, 375]> x", "bool"),
    ("cast_f16_to_f32_out", "tensor<fp16, [1, 375, 640]> x", "fp32"),
    ("cast_b_to_i32_375", "tensor<bool, [1, 375]> x", "int32"),
]
for pname, arg, dtype in CAST_FORMS:
    PROBES.append(probe(
        pname,
        arg,
        [f'tensor<{dtype}, []> dt = const()[name = string("dt"), val = string("{dtype}")];',
         f'tensor<{dtype}, []> out = cast(dtype = dt, x = x)[name = string("out")];'],
        "out"))

# --- select (2 runtime forms; the 8-head one is certified, listed for contrast) ---
PROBES.append(probe(
    "select_am_1x1024x375",
    "tensor<fp16, [1, 1024, 375]> b, tensor<bool, [1, 1, 375]> cond",
    [const_f16("a", ()),
     'tensor<fp16, [1, 1024, 375]> out = select(a = a, b = b, cond = cond)[name = string("out")];'],
    "out"))

# --- two-op chains: interior casts feeding consumers (island spellings) ---
PROBES.append(probe(
    "chain_castf2i_less",
    "tensor<fp16, [1]> x",
    ['tensor<int32, []> dt = const()[name = string("dt"), val = string("int32")];',
     'tensor<int32, [1]> xi = cast(dtype = dt, x = x)[name = string("xi")];',
     arange_const_i32("ar", 375),
     'tensor<int32, [1]> xi1 = const()[name = string("xi1"), val = tensor<int32, [1]>(1)];'
     ,
     'tensor<bool, [1, 375]> out = less(x = ar, y = xi)[name = string("out")];'],
    "out"))

PROBES.append(probe(
    "chain_b2i_rmin_i2b_select",
    "tensor<bool, [1, 1, 375, 375]> cond_mask, tensor<fp16, [1, 1024, 375]> b",
    [const_f16("a", ()),
     'tensor<int32, []> dti = const()[name = string("dti"), val = string("int32")];',
     'tensor<int32, [1, 1, 375, 375]> ci = cast(dtype = dti, x = cond_mask)[name = string("ci")];',
     'tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>(2)];',
     'tensor<bool, []> keep = const()[name = string("keep"), val = bool(false)];',
     'tensor<int32, [1, 1, 375]> rmin = reduce_min(axes = axes, keep_dims = keep, x = ci)[name = string("rmin")];',
     'tensor<bool, []> dtb = const()[name = string("dtb"), val = string("bool")];',
     'tensor<bool, [1, 1, 375]> am = cast(dtype = dtb, x = rmin)[name = string("am")];',
     'tensor<fp16, [1, 1024, 375]> out = select(a = a, b = b, cond = am)[name = string("out")];'],
    "out"))


def run_probe(root: Path, spec: dict) -> dict:
    mil = root / f"{spec['name']}.mil"
    mil.write_text(spec["mil"])
    out = root / f"out-{spec['name']}"
    proc = subprocess.run(
        [str(COMPILER), "--mil", str(mil), "--model-root", str(root),
         "--output", str(out), "--target", "H13", "--format", "anec"],
        capture_output=True, text=True, timeout=120)
    message = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
    diag = message[-1] if message else ""
    return {
        "name": spec["name"],
        "exit": proc.returncode,
        "diagnostic": diag,
        "lowered": proc.returncode == 0,
    }


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = [run_probe(root, spec) for spec in PROBES]
    lowered = sum(1 for row in rows if row["lowered"])
    for row in rows:
        mark = "PASS" if row["lowered"] else "REFUSE"
        print(f"{mark:7s} {row['name']:28s} {row['diagnostic'][:110]}")
    print(f"\n{lowered}/{len(rows)} encoder forms lower under H13")
    (Path(__file__).resolve().parent / "results" / "encoder-form-census.json").parent.mkdir(
        parents=True, exist_ok=True)
    (Path(__file__).resolve().parent / "results" / "encoder-form-census.json").write_text(
        json.dumps({"compiler": str(COMPILER), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
