#!/usr/bin/env python3
"""Mint Qwen matvec/batched-matmul shapes via ANEForge's e5rt bridge on
macstudio, instead of ane-compile-hwx. The e5rt path resolves @model_path
weights correctly; ane-compile-hwx refuses fresh MILs with BLOBFILE.

Light, niced, scratch dir on macstudio.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def matvec_mil(rows: int, k: int, n: int, ty: bool) -> str:
    y_shape = [k, n] if ty else [n, k]
    return (
        "program(1.3)\n"
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, '
        '{"coremlc-version", "3520.5.1"}})]\n'
        "{\n"
        f"    func main<ios18>(tensor<fp16, [{rows}, {k}]> x) {{\n"
        f"        tensor<fp16, [{y_shape[0]}, {y_shape[1]}]> w = const()"
        f"[name = string(\"w\"), val = tensor<fp16, [{y_shape[0]}, {y_shape[1]}]>"
        f"(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))];\n"
        f"        tensor<fp16, [{rows}, {n}]> product = matmul(transpose_x = bool(false), "
        f"transpose_y = bool({str(ty).lower()}), x = x, y = w)[name = string(\"product\")];\n"
        f"    }} -> (product);\n"
        "}}\n"
    )


def batched_mil(rows: int, k: int, n: int, batch: int, rank4: bool, ty: bool) -> str:
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
        f"        tensor<fp16, {y_lead}> w = const()"
        f"[name = string(\"w\"), val = tensor<fp16, {y_lead}>"
        f"(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))];\n"
        f"        tensor<fp16, {out_lead}> product = matmul(transpose_x = bool(false), "
        f"transpose_y = bool({str(ty).lower()}), x = x, y = w)[name = string(\"product\")];\n"
        f"    }} -> (product);\n"
        "}}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--dylib", default="/Users/joshuawarren/src/ane-af-split-wt/aneforge/_lib/libane_e5rt_dispatch.dylib")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    remote_root = f"/tmp/qwenmint-e5rt-{int(time.time())}"
    # Make scratch
    subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                    f"rm -rf {remote_root}; mkdir -p {remote_root}"],
                   capture_output=True, text=True, timeout=30, check=True)

    qwen_matvec = [
        (1, 2048, 2048, True),
        (1, 6144, 2048, True),
        (1, 16, 2048, True),
        (1, 512, 2048, True),
        (1, 4096, 2048, True),
        (1, 2048, 6144, True),
        (2, 256, 256, False),
        (8, 256, 256, False),
        (1, 1024, 1024, True),
        (1, 2048, 4096, True),
        (1, 2048, 1024, True),
        (1, 1024, 2048, True),
    ]
    qwen_batched = [
        (1, 128, 128, 16, False, False),
        (1, 128, 128, 16, True, False),
        (1, 128, 128, 16, False, True),
    ]

    records = []
    ANE_MASK = 0x4  # device 1 (ANE #0)

    for shape in qwen_matvec + qwen_batched:
        rows, K, N = shape[:3]
        if len(shape) == 4:
            ty = shape[3]
            is_batched = False
            batch = rank4 = None
        else:
            batch, rank4, ty = shape[3], shape[4], shape[5]
            is_batched = True
        weight_bytes = K * N * 2 + 64
        if is_batched:
            mil = batched_mil(rows, K, N, batch, rank4, ty)
            case = f"qwenbm_m{rows}_K_{K}_n{N}_b{batch}_r4{int(rank4)}_ty{int(ty)}"
        else:
            mil = matvec_mil(rows, K, N, ty)
            case = f"qwenmv_m{rows}_k{K}_n{N}_ty{int(ty)}"
        case_dir = f"{remote_root}/{case}"
        # write mil + weights via ssh
        stage = Path(tempfile.mkdtemp(prefix=f"qwenmint-{case}-"))
        (stage / "model.mil").write_text(mil)
        (stage / "weights.bin").write_bytes(b"\x00" * weight_bytes)
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"mkdir -p {case_dir}/cache"],
                       capture_output=True, text=True, timeout=30, check=True)
        subprocess.run(["scp", "-q", str(stage / "model.mil"),
                        f"{args.host}:{case_dir}/model.mil"], check=True, timeout=30)
        subprocess.run(["scp", "-q", str(stage / "weights.bin"),
                        f"{args.host}:{case_dir}/weights.bin"], check=True, timeout=120)

        # Compile via e5rt
        if is_batched:
            x_lead = f"1,{batch},{rows},{K}" if rank4 else f"{batch},{rows},{K}"
            inputs = {"x": tuple(int(v) for v in x_lead.split(","))}
            outputs = {"product": tuple(int(v) for v in (f"1,{batch},{rows},{N}" if rank4 else f"{batch},{rows},{N}").split(","))}
        else:
            inputs = {"x": (rows, K)}
            outputs = {"product": (rows, N)}
        # write a small python script that calls the dylib
        py = f"""
import ctypes, sys
sys.path.insert(0, '/Users/joshuawarren/src/ane-af-split-wt')
from aneforge._runtime import _lib, _port_bytes, _fp16_bytes
in_names = {list(inputs.keys())!r}; out_names = {list(outputs.keys())!r}
in_shapes = {list(inputs.values())!r}; out_shapes = {list(outputs.values())!r}
in_n = (ctypes.c_char_p * len(in_names))(*[n.encode() for n in in_names])
in_s = (ctypes.c_size_t * len(in_names))(*[_port_bytes(tuple(s), 'fp16') for s in in_shapes])
out_n = (ctypes.c_char_p * len(out_names))(*[n.encode() for n in out_names])
out_s = (ctypes.c_size_t * len(out_names))(*[_fp16_bytes(tuple(s)) for s in out_shapes])
cache = '/tmp/qwenmint-aneforge-cache'
import os; os.makedirs(cache, exist_ok=True)
h = _lib.ane_e5rt_program_compile(
  '{case_dir}/model.mil'.encode(), cache.encode(),
  ctypes.c_uint64({ANE_MASK}),
  in_n, in_s, ctypes.c_size_t(len(in_names)),
  out_n, out_s, ctypes.c_size_t(len(out_names)))
print('HANDLE', 'YES' if h else 'NO')
"""
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                            f"cd /Users/joshuawarren/src/ane-af-split-wt && /opt/homebrew/bin/python3.13 -c \"{py}\""],
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        ok = "HANDLE YES" in out
        records.append({
            "case": case,
            "shape": {"rows": rows, "reduction": K, "columns": N,
                      "transpose_y": ty,
                      "batch": batch, "rank4": rank4, "is_batched": is_batched},
            "weight_bytes": weight_bytes,
            "compiled": ok,
            "log": out[-300:],
        })
        print(f"{case}: compiled={ok}", flush=True)
        time.sleep(0.2)

    json.dump(records, open(args.out, "w"), indent=1, default=str)
    print(f"\nWROTE {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()