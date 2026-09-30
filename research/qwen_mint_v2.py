#!/usr/bin/env python3
"""Mint Qwen matvec/batched-matmul shapes using the CORRECT blob format
that ANECCompile accepts (mint_oracles.blob structure).

Key insight from reading mint_oracles.py: the weights.bin must be a
128-byte sub-header followed by the payload, with magic=1, version=2,
0xDEADBEEF at byte 64, payload_length at byte 68, payload_offset=128.
Without this format, ANECCompile returns
"MILFramework error: Cannot retrieve file blob properties."

Run on macstudio (light, niced, scratch dir).
"""
import argparse
import ctypes
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Mirror of mint_oracles.blob (must match exactly)
def blob(payload: bytes) -> bytes:
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IQQ", result, 64, 0xDEADBEEF, len(payload), 128)
    result[128:] = payload
    return bytes(result)


def aligned_blob(payload: bytes) -> bytes:
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<I", result, 64, 0xDEADBEEF)
    struct.pack_into("<Q", result, 72, len(payload))
    struct.pack_into("<Q", result, 80, 128)
    result[128:] = payload
    return bytes(result)


def matvec_mil(rows, k, n, ty):
    # When transpose_y=true the BLOBFILE shape is [K, N] (matmul transposes
    # it to [N, K] internally). When transpose_y=false, BLOBFILE shape is
    # already [N, K] — no transposition.
    y_decl = [k, n] if ty else [n, k]
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3520.4.1"}}, {{"coremlc-version", "3520.5.1"}}}})]
{{
    func main<ios18>(tensor<fp16, [{rows}, {k}]> x) {{
        tensor<fp16, [{y_decl[0]}, {y_decl[1]}]> w = const()[name = string("w"), val = tensor<fp16, [{y_decl[0]}, {y_decl[1]}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
        tensor<fp16, [{rows}, {n}]> product = matmul(transpose_x = bool(false), transpose_y = bool({str(ty).lower()}), x = x, y = w)[name = string("product")];
    }} -> (product);
}}
'''


def batched_mil_const_y(rows, k, n, batch, rank4):
    """Const-y batched matmul: y is the BLOBFILE constant, x is runtime.
    BLOBFILE shape is [N, K] since weight lives as packed rank-2 fp16.
    """
    x_lead = f"[1, {batch}, {rows}, {k}]" if rank4 else f"[{batch}, {rows}, {k}]"
    y_lead = f"[1, {batch}, {n}, {k}]" if rank4 else f"[{batch}, {n}, {k}]"
    out_lead = f"[1, {batch}, {rows}, {n}]" if rank4 else f"[{batch}, {rows}, {n}]"
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3520.4.1"}}, {{"coremlc-version", "3520.5.1"}}}})]
{{
    func main<ios18>(tensor<fp16, {x_lead}> x) {{
        tensor<fp16, {y_lead}> w = const()[name = string("w"), val = tensor<fp16, {y_lead}>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
        tensor<fp16, {out_lead}> product = matmul(transpose_x = bool(false), transpose_y = bool(true), x = x, y = w)[name = string("product")];
    }} -> (product);
}}
'''


def batched_mil_runtime_y(rows, k, n, batch, rank4):
    """Runtime-y batched matmul matching H14IslandTemplates.inc:818."""
    x_lead = f"[1, {batch}, {rows}, {k}]" if rank4 else f"[{batch}, {rows}, {k}]"
    y_lead = f"[1, {batch}, {batch}, {k}, {n}]" if rank4 else f"[{batch}, {batch}, {k}, {n}]"
    out_lead = f"[1, {batch}, {rows}, {n}]" if rank4 else f"[{batch}, {rows}, {n}]"
    return f'''program(1.3)
[buildInfo = dict<string, string>({{{{"coremlc-component-MIL", "3520.4.1"}}, {{"coremlc-version", "3520.5.1"}}}})]
{{
    func main<ios18>(tensor<fp16, {x_lead}> x, tensor<fp16, {y_lead}> y) {{
        tensor<fp16, {out_lead}> product = matmul(transpose_x = bool(false), transpose_y = bool(false), x = x, y = y)[name = string("product")];
    }} -> (product);
}}
'''


SHAPES = [
    # matvec const-y probes (from the Qwen census)
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
    # matvec that fits envelope (sanity)
    (1, 256, 256, True),
    (1, 256, 512, True),
    (1, 512, 256, True),
    (1, 512, 512, True),
    (1, 1024, 256, True),
    (1, 256, 1024, True),
]
BATCHED_CONST_Y = [
    # Qwen DeltaNet state shape with const y
    (1, 128, 128, 16, False),
    (1, 128, 128, 16, True),
]
BATCHED_RUNTIME_Y = [
    # Qwen DeltaNet state shape with runtime y
    (1, 128, 128, 16, False),
    (1, 128, 128, 16, True),
]


def run_one(host, scratch_dir, mil, weights_bytes):
    cmd = (f"rm -rf {scratch_dir} && mkdir -p {scratch_dir}")
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                   capture_output=True, text=True, timeout=30, check=True)
    stage = Path(tempfile.mkdtemp(prefix="qwenmint-stage-"))
    (stage / "model.mil").write_text(mil)
    (stage / "weights.bin").write_bytes(weights_bytes)
    subprocess.run(["scp", "-q", str(stage / "model.mil"),
                    f"{host}:{scratch_dir}/model.mil"], check=True, timeout=30)
    subprocess.run(["scp", "-q", str(stage / "weights.bin"),
                    f"{host}:{scratch_dir}/weights.bin"], check=True, timeout=120)
    cmd = (f"/tmp/h13-oracle/bin/ane-compile-hwx {scratch_dir} {scratch_dir}/out h14 "
           f">{scratch_dir}/compile.log 2>&1; echo EXIT=$?; cat {scratch_dir}/compile.log")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                       capture_output=True, text=True, timeout=180)
    log = r.stdout
    # check hwx
    check = subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                            f"ls -la {scratch_dir}/out/model.hwx 2>/dev/null"],
                           capture_output=True, text=True, timeout=10)
    has_hwx = bool(check.stdout.strip()) and "No such file" not in check.stdout
    return log, has_hwx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    remote_root = f"/tmp/qwenmint-v2-{int(time.time())}"
    records = []

    # Matvec const-y probes
    for shape in SHAPES:
        rows, K, N, ty = shape
        # Weight shape when ty=true: [N, K] (post-transpose matmul sees [K, N])
        # When ty=false: [K, N]
        weight_shape = [N, K] if ty else [K, N]
        payload = b"\x00" * (weight_shape[0] * weight_shape[1] * 2)
        weights_bytes = blob(payload)
        mil = matvec_mil(rows, K, N, ty)
        case = f"qwenmv_m{rows}_k{K}_n{N}_ty{int(ty)}"
        scratch_dir = f"{remote_root}/{case}"
        log, has_hwx = run_one(args.host, scratch_dir, mil, weights_bytes)
        ok = has_hwx and "callback_status=0" in log
        records.append({"case": case, "shape": {"rows": rows, "K": K, "N": N, "ty": ty},
                        "compiled": ok, "has_hwx": has_hwx, "log": log[-400:]})
        print(f"{case}: compiled={ok}", flush=True)

    # Batched const-y
    for shape in BATCHED_CONST_Y:
        rows, K, N, batch, rank4 = shape
        # const-y: weight as [N, K] per head, broadcast across batch
        payload = b"\x00" * (N * K * 2)
        weights_bytes = blob(payload)
        mil = batched_mil_const_y(rows, K, N, batch, rank4)
        case = f"qwenbm_const_m{rows}_K{K}_n{N}_b{batch}_r4{rank4}"
        scratch_dir = f"{remote_root}/{case}"
        log, has_hwx = run_one(args.host, scratch_dir, mil, weights_bytes)
        ok = has_hwx and "callback_status=0" in log
        records.append({"case": case, "shape": {"rows": rows, "K": K, "N": N,
                                                "batch": batch, "rank4": rank4, "const_y": True},
                        "compiled": ok, "has_hwx": has_hwx, "log": log[-400:]})
        print(f"{case}: compiled={ok}", flush=True)

    # Batched runtime-y (sanity)
    for shape in BATCHED_RUNTIME_Y:
        rows, K, N, batch, rank4 = shape
        # no payload
        weights_bytes = blob(b"")
        mil = batched_mil_runtime_y(rows, K, N, batch, rank4)
        case = f"qwenbm_rt_m{rows}_K{K}_n{N}_b{batch}_r4{rank4}"
        scratch_dir = f"{remote_root}/{case}"
        log, has_hwx = run_one(args.host, scratch_dir, mil, weights_bytes)
        ok = has_hwx and "callback_status=0" in log
        records.append({"case": case, "shape": {"rows": rows, "K": K, "N": N,
                                                "batch": batch, "rank4": rank4, "const_y": False},
                        "compiled": ok, "has_hwx": has_hwx, "log": log[-400:]})
        print(f"{case}: compiled={ok}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(records, open(args.out, "w"), indent=1, default=str)
    n_ok = sum(1 for r in records if r["compiled"])
    print(f"\nWROTE {len(records)} (compiled={n_ok}) to {args.out}")


if __name__ == "__main__":
    main()