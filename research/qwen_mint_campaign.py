#!/usr/bin/env python3
"""Mint Qwen3.8-2B matvec/batched-matmul shapes on macstudio.

For each (rows, K, N, transpose_y) tuple from the census, write a MIL,
compile via /tmp/h13-oracle/bin/ane-compile-hwx h14, decode the result
with mint_oracles.parse_hwx, record constant-section layout. Output a
JSON campaign record compatible with the H14 oracle format.

Light, niced, scratch dir only.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def matvec_mil(rows: int, k: int, n: int, ty: bool) -> str:
    """A single matvec, opaque, no extra ops.

    rows: x.shape[0]  (after transpose_x, so x is [rows, K])
    K: reduction
    N: columns
    ty: transpose_y flag (constant y is [K, N] if ty else [N, K])
    """
    y_shape = [k, n] if ty else [n, k]
    return (
        "program(1.3)\n"
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, '
        '{"coremlc-version", "3520.5.1"}})]\n'
        "{\n"
        f"    func main<ios18>(tensor<fp16, [{rows}, {k}]> x) {{\n"
        f"        bool f = const()[name = string(\"f\"), val = bool(false)];\n"
        f"        bool ty = const()[name = string(\"ty\"), val = bool({str(ty).lower()})];\n"
        f"        tensor<fp16, [{y_shape[0]}, {y_shape[1]}]> w = const()"
        f"[name = string(\"w\"), val = tensor<fp16, [{y_shape[0]}, {y_shape[1]}]>"
        f"(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))];\n"
        f"        tensor<fp16, [{rows}, {n}]> product = matmul(transpose_x = f, "
        f"transpose_y = ty, x = x, y = w)[name = string(\"product\")];\n"
        f"    }} -> (product);\n"
        "}}\n"
    )


def batched_mil(rows: int, k: int, n: int, batch: int, rank4: bool, ty: bool) -> str:
    """A rank-3 / rank-4 batched matmul with constant y."""
    x_lead = f"[1, {batch}, {rows}, {k}]" if rank4 else f"[{batch}, {rows}, {k}]"
    y_shape = [k, n] if ty else [n, k]
    y_lead = f"[1, {batch}, {y_shape[0]}, {y_shape[1]}]" if rank4 else f"[{batch}, {y_shape[0]}, {y_shape[1]}]"
    out_lead = f"[1, {batch}, {rows}, {n}]" if rank4 else f"[{batch}, {rows}, {n}]"
    return (
        "program(1.3)\n"
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, '
        '{"coremlc-version", "3520.5.1"}})]\n'
        "{\n"
        f"    func main<ios18>(tensor<fp16, {x_lead}> x) {{\n"
        f"        bool f = const()[name = string(\"f\"), val = bool(false)];\n"
        f"        bool ty = const()[name = string(\"ty\"), val = bool({str(ty).lower()})];\n"
        f"        tensor<fp16, {y_lead}> w = const()"
        f"[name = string(\"w\"), val = tensor<fp16, {y_lead}>"
        f"(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))];\n"
        f"        tensor<fp16, {out_lead}> product = matmul(transpose_x = f, "
        f"transpose_y = ty, x = x, y = w)[name = string(\"product\")];\n"
        f"    }} -> (product);\n"
        "}}\n"
    )


def run_one(host: str, scratch_remote: Path, mil: str, weight_bytes: int) -> dict:
    """Compile via Apple ane-compile-hwx on remote; fetch hwx; decode locally."""
    out = scratch_remote / "out"
    cmd = (f"rm -rf {scratch_remote} && mkdir -p {scratch_remote}/cache {out}")
    subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                   capture_output=True, text=True, timeout=30, check=True)
    # Write mil + weights to LOCAL staging, then scp to remote
    stage = Path(tempfile.mkdtemp(prefix=f"qwenmint-stage-"))
    (stage / "model.mil").write_text(mil)
    (stage / "weights.bin").write_bytes(b"\x00" * weight_bytes)
    subprocess.run(["scp", "-q", str(stage / "model.mil"),
                    f"{host}:{scratch_remote}/model.mil"], check=True, timeout=30)
    subprocess.run(["scp", "-q", str(stage / "weights.bin"),
                    f"{host}:{scratch_remote}/weights.bin"], check=True, timeout=120)
    cmd = (f"rm -rf {out}/*; "
           f"/tmp/h13-oracle/bin/ane-compile-hwx {scratch_remote} {out} h14 "
           f">{scratch_remote}/compile.log 2>&1; echo EXIT=$?; "
           f"cat {scratch_remote}/compile.log")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd],
                       capture_output=True, text=True, timeout=180)
    log = r.stdout
    # check if hwx exists on remote
    check = subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                            f"ls -la {out}/model.hwx 2>/dev/null"],
                           capture_output=True, text=True, timeout=10)
    if not check.stdout.strip() or "No such file" in check.stdout:
        return {"compiled": False, "log": log, "exit": r.returncode}
    # fetch hwx locally
    local_scratch = scratch_remote.parent / ("local_" + scratch_remote.name)
    local_scratch.mkdir(parents=True, exist_ok=True)
    local_hwx = local_scratch / "model.hwx"
    subprocess.run(["scp", "-q", f"{host}:{out}/model.hwx", str(local_hwx)],
                   check=True, timeout=30)
    hwx_path = local_hwx
    if not hwx_path.exists() or hwx_path.stat().st_size == 0:
        return {"compiled": False, "log": log, "exit": r.returncode}
    # fetch hwx bytes
    hwx_local = scratch / "model.hwx"
    subprocess.run(["scp", "-q", f"{host}:{out}/model.hwx", str(hwx_local)], check=True)
    data = hwx_local.read_bytes()
    # decode
    sys.path.insert(0, str(Path(__file__).resolve().parent / "mil-hwx-h14-integ-wt/research"))
    try:
        import mint_oracles as om
        result = om.parse_hwx(data, "h14", capture_constant_bytes=True)
        return {
            "compiled": True,
            "hwx_bytes": len(data),
            "constant_section": result.get("constant_section"),
            "program_descriptor": result.get("program_descriptor"),
            "task_descriptors": result.get("task_descriptors"),
            "hwx_sha256": result.get("hwx_sha256"),
            "log_tail": log[-200:],
        }
    except Exception as e:
        import traceback
        return {"compiled": True, "decode_error": str(e), "traceback": traceback.format_exc(),
                "hwx_bytes": len(data), "log_tail": log[-200:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--shapes", default="qwen", help="qwen = use the Qwen census")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Qwen census: rows, K, N, transpose_y, kind)
    # Tuple from census2: (LHS shape, transpose_y)
    #   (1, 2048), True, K=N=2048 → rows=1 K=2048 N=2048 ty=True
    #   (1, 2048), True, K=512 N=2048 → rows=1 K=512 N=2048 ty=True
    #   (1, 6144), True, K=6144 N=2048 → rows=1 K=6144 N=2048 ty=True
    #   (1, 16), True, K=16 N=2048 → rows=1 K=16 N=2048 ty=True (but x is [1,16])
    #   (2, 256), False, K=256 N=256 ty=False (fits envelope)
    #   (8, 256), False, K=256 N=256 ty=False (fits envelope)
    #   (1, 4096), True, K=4096 N=2048
    #   (16, 1, 128) runtime-runtime (matmul inputs come from inputs not const)
    #   (16, 128) const-x (matmul with const x = t13 [16,16], runtime y)
    qwen_matvec = [
        # (rows, K, N, transpose_y)
        (1, 2048, 2048, True),
        (1, 6144, 2048, True),
        (1, 16, 2048, True),
        (1, 512, 2048, True),
        (1, 4096, 2048, True),
        (1, 2048, 6144, True),
        (2, 256, 256, False),
        (8, 256, 256, False),
        (1, 2048, 4096, True),  # extra coverage
        (1, 2048, 1024, True),  # probe envelope edge
        (1, 1024, 2048, True),
    ]
    # Batched matmul candidates from Qwen shapes
    qwen_batched = [
        # (rows, K, N, batch, rank4, ty)
        (1, 128, 128, 16, False, False),  # Qwen DeltaNet state, runtime-y
        # (We can't really test const-y batched here because the compiler routes
        #  const-y to matvec first. Constant y needs the constanty encoding path
        #  which we don't have yet; documented as open.)
    ]
    records = []
    remote_root = Path(f"/tmp/qwenmint-{int(time.time())}")
    for shape in qwen_matvec:
        rows, K, N, ty = shape
        weight_bytes = K * N * 2 + 64  # constant weight + 64-byte blob header
        mil = matvec_mil(rows, K, N, ty)
        case = f"qwenmv_m{rows}_k{K}_n{N}_ty{int(ty)}"
        scratch = remote_root / case
        r = run_one(args.host, scratch, mil, weight_bytes)
        r["case"] = case
        r["shape"] = {"rows": rows, "reduction": K, "columns": N, "transpose_y": ty}
        r["weight_bytes"] = weight_bytes
        records.append(r)
        print(f"{case}: compiled={r.get('compiled')} hwx={r.get('hwx_bytes')} log={(r.get('log_tail') or r.get('log',''))[:80]}", flush=True)
        time.sleep(0.1)
    for shape in qwen_batched:
        rows, K, N, batch, rank4, ty = shape
        weight_bytes = K * N * 2 + 64
        mil = batched_mil(rows, K, N, batch, rank4, ty)
        case = f"qwenbm_m{rows}_K_{K}_n{N}_b{batch}_r4{int(rank4)}_ty{int(ty)}"
        scratch = remote_root / case
        r = run_one(args.host, scratch, mil, weight_bytes)
        r["case"] = case
        r["shape"] = {"rows": rows, "reduction": K, "columns": N,
                      "batch": batch, "rank4": rank4, "transpose_y": ty}
        r["weight_bytes"] = weight_bytes
        records.append(r)
        print(f"{case}: compiled={r.get('compiled')} hwx={r.get('hwx_bytes')} log={(r.get('log_tail') or r.get('log',''))[:80]}", flush=True)
        time.sleep(0.1)

    json.dump(records, open(out, "w"), indent=1, default=str)
    print(f"\nWROTE {len(records)} records to {out}")


if __name__ == "__main__":
    main()