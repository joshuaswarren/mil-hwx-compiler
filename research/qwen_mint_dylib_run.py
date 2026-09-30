#!/usr/bin/env python3
"""Mint Qwen matvec/batched-matmul shapes via the e5rt dispatch dylib.

Run via ssh to macstudio. The dylib call uses ane_e5rt_program_compile
from /Users/joshuawarren/src/ane-af-split-wt/aneforge/_lib/.

Light, niced, scratch dir.
"""
import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def matvec_mil(rows, k, n, ty):
    # When transpose_y=true, the declared weight shape is [N, K] (post-transpose
    # the matmul sees [K, N]). When transpose_y=false, declared shape is [N, K].
    y_decl = [n, k]
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3520.4.1"}}, {{"coremlc-version", "3520.5.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [{rows}, {k}]> x) {{
        tensor<fp16, [{y_decl[0]}, {y_decl[1]}]> w = const()[name = string("w"), val = tensor<fp16, [{y_decl[0]}, {y_decl[1]}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
        tensor<fp16, [{rows}, {n}]> product = matmul(transpose_x = bool(false), transpose_y = bool({str(ty).lower()}), x = x, y = w)[name = string("product")];
    }} -> (product);
}}
'''


def batched_mil(rows, k, n, batch, rank4, ty):
    x_lead = f"[1, {batch}, {rows}, {k}]" if rank4 else f"[{batch}, {rows}, {k}]"
    # weight declared shape: when transpose_y=true the BLOBFILE shape is [N, K]
    # when transpose_y=false the BLOBFILE shape is [K, N] — but the matmul "y"
    # argument's effective shape becomes [K, N] in either case after the
    # transpose semantics are applied by MIL's matmul. We follow the
    # ANEForge convention: BLOBFILE is the post-matmul-y shape.
    y_decl = [k, n]
    y_lead = f"[1, {batch}, {y_decl[0]}, {y_decl[1]}]" if rank4 else f"[{batch}, {y_decl[0]}, {y_decl[1]}]"
    out_lead = f"[1, {batch}, {rows}, {n}]" if rank4 else f"[{batch}, {rows}, {n}]"
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3520.4.1"}}, {{"coremlc-version", "3520.5.1"}}}})]
{{
    func main<ios18>(tensor<fp16, {x_lead}> x) {{
        tensor<fp16, {y_lead}> w = const()[name = string("w"), val = tensor<fp16, {y_lead}>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
        tensor<fp16, {out_lead}> product = matmul(transpose_x = bool(false), transpose_y = bool({str(ty).lower()}), x = x, y = w)[name = string("product")];
    }} -> (product);
}}
'''


SHAPES = [
    # Envelope-fitting sanity probes first
    (1, 256, 256, True),
    (1, 256, 512, True),
    (2, 256, 256, False),
    (8, 256, 256, False),
    # Qwen shapes (likely all rejected)
    (1, 2048, 2048, True),
    (1, 6144, 2048, True),
    (1, 16, 2048, True),
    (1, 512, 2048, True),
    (1, 4096, 2048, True),
    (1, 2048, 6144, True),
    (1, 1024, 1024, True),
    (1, 2048, 4096, True),
    (1, 2048, 1024, True),
    (1, 1024, 2048, True),
]
BATCHED = [
    (1, 128, 128, 16, False, False),
    (1, 128, 128, 16, False, True),
]


def run_one(host, scratch_dir, mil, weight_bytes, mask):
    # Create scratch dir on remote
    cmd = (f"rm -rf {scratch_dir} && mkdir -p {scratch_dir}/cache")
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                   capture_output=True, text=True, timeout=30, check=True)
    # Stage locally
    stage = Path(tempfile.mkdtemp(prefix="qwenmint-stage-"))
    (stage / "model.mil").write_text(mil)
    (stage / "weights.bin").write_bytes(b"\x00" * weight_bytes)
    subprocess.run(["scp", "-q", str(stage / "model.mil"),
                    f"{host}:{scratch_dir}/model.mil"], check=True, timeout=30)
    subprocess.run(["scp", "-q", str(stage / "weights.bin"),
                    f"{host}:{scratch_dir}/weights.bin"], check=True, timeout=120)
    # Build the python invocation that imports the dylib and runs compile
    # We can't pass complex args, so write a helper script
    return mil, weight_bytes, mask, scratch_dir


def build_runner_py(inputs, outputs, mask, scratch_dir):
    in_names = list(inputs.keys())
    in_shapes = list(inputs.values())
    out_names = list(outputs.keys())
    out_shapes = list(outputs.values())
    py = f'''
import sys, os, math, ctypes
lib = ctypes.cdll.LoadLibrary('/Users/joshuawarren/src/ane-af-split-wt/aneforge/_lib/libane_e5rt_dispatch.dylib')
fn = lib.ane_e5rt_program_compile
fn.restype = ctypes.c_void_p
fn.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint64,
               ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t,
               ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t]
in_names = {in_names!r}
in_shapes = {in_shapes!r}
out_names = {out_names!r}
out_shapes = {out_shapes!r}
in_n = (ctypes.c_char_p * len(in_names))(*[n.encode() for n in in_names])
in_s = (ctypes.c_size_t * len(in_names))(*[ctypes.c_size_t(math.prod(s) * 2) for s in in_shapes])
out_n = (ctypes.c_char_p * len(out_names))(*[n.encode() for n in out_names])
out_s = (ctypes.c_size_t * len(out_names))(*[ctypes.c_size_t(math.prod(s) * 2) for s in out_shapes])
os.makedirs('/tmp/qwenmint-cache', exist_ok=True)
h = fn('{scratch_dir}/model.mil'.encode(), '/tmp/qwenmint-cache'.encode(), ctypes.c_uint64({mask}),
       in_n, in_s, ctypes.c_size_t(len(in_names)),
       out_n, out_s, ctypes.c_size_t(len(out_names)))
print('HANDLE', 'YES' if h else 'NO', hex(h) if h else 0)
'''
    return py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mask", type=int, default=4)
    args = ap.parse_args()
    remote_root = f"/tmp/qwenmint-dylib-{int(time.time())}"
    records = []
    for shape in SHAPES:
        rows, K, N, ty = shape
        weight_bytes = K * N * 2 + 64
        mil = matvec_mil(rows, K, N, ty)
        case = f"qwenmv_m{rows}_k{K}_n{N}_ty{int(ty)}"
        scratch_dir = f"{remote_root}/{case}"
        run_one(args.host, scratch_dir, mil, weight_bytes, args.mask)
        inputs = {"x": (rows, K)}
        outputs = {"product": (rows, N)}
        py = build_runner_py(inputs, outputs, args.mask, scratch_dir)
        runner_local = Path(tempfile.mktemp(suffix=".py", prefix="qwenmint-"))
        runner_local.write_text(py)
        subprocess.run(["scp", "-q", str(runner_local),
                        f"{args.host}:{scratch_dir}/run.py"], check=True, timeout=30)
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                            f"cd {scratch_dir} && /opt/homebrew/bin/python3.13 run.py 2>&1"],
                           capture_output=True, text=True, timeout=180)
        out = r.stdout.strip()
        ok = "HANDLE YES" in out
        records.append({"case": case, "shape": {"rows": rows, "K": K, "N": N, "ty": ty},
                        "compiled": ok, "out": out[-300:]})
        print(f"{case}: compiled={ok}", flush=True)
    for shape in BATCHED:
        rows, K, N, batch, rank4, ty = shape
        weight_bytes = K * N * 2 + 64
        mil = batched_mil(rows, K, N, batch, rank4, ty)
        case = f"qwenbm_m{rows}_K{K}_n{N}_b{batch}_r4{rank4}_ty{int(ty)}"
        scratch_dir = f"{remote_root}/{case}"
        run_one(args.host, scratch_dir, mil, weight_bytes, args.mask)
        x_lead = (1, batch, rows, K) if rank4 else (batch, rows, K)
        out_lead = (1, batch, rows, N) if rank4 else (batch, rows, N)
        inputs = {"x": x_lead}
        outputs = {"product": out_lead}
        py = build_runner_py(inputs, outputs, args.mask, scratch_dir)
        runner_local = Path(tempfile.mktemp(suffix=".py", prefix="qwenmint-"))
        runner_local.write_text(py)
        subprocess.run(["scp", "-q", str(runner_local),
                        f"{args.host}:{scratch_dir}/run.py"], check=True, timeout=30)
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                            f"cd {scratch_dir} && /opt/homebrew/bin/python3.13 run.py 2>&1"],
                           capture_output=True, text=True, timeout=180)
        out = r.stdout.strip()
        ok = "HANDLE YES" in out
        records.append({"case": case, "shape": {"rows": rows, "K": K, "N": N,
                                                "batch": batch, "rank4": rank4, "ty": ty},
                        "compiled": ok, "out": out[-300:]})
        print(f"{case}: compiled={ok}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=1, default=str)
    print(f"\nWROTE {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()