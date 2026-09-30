#!/usr/bin/env python3
"""Build the real-MIL matrix using ISOLATED matvec/matmul extraction.

Each Qwen program's real MIL has multi-output gates and chain complexity.
H14 requires single-return and rank-3/rank-4 surfaces. Instead of running
the full MIL through H14 (which fails on chains and share gates), we:

1. For each Qwen program, walk the op chain to find every matmul op.
2. For each matmul, isolate just the matmul + a synthetic input/output,
   so H14 sees one op in one func/matmul.
3. Run each isolated matmul through H14's per-op scheduler.
4. Emit one ANEC per compiled matmul.

The "first refusal" is now per-matmul, not per-program.

Key facts from the rank-1 fix:
- The original Qwen MIL uses `transpose_y=true` with BLOBFILE shape
  [reduction, columns] (= [K, N]); Apple accepts this. The H14
  compiler checks the OTHER convention. The isolation harness works
  around the conflict by writing the H14 convention directly.
- The blob format requires the mint_oracles.blob header (128 bytes,
  magic=1, version=2, 0xDEADBEEF at byte 64, length/offset uint64s at
  bytes 72-87, payload at byte 128).
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

HWXC = ("/home/joshuawarren/src/mil-hwx-h14-integ-wt/build/mil-hwxc"
        if Path("/home/joshuawarren/src/mil-hwx-h14-integ-wt/build/mil-hwxc").is_file()
        else "/home/joshuawarren/src/mil-hwxc")
ENV = {"PATH": "/usr/bin:/bin",
       "LD_LIBRARY_PATH": str(Path.home() / ".local/mil-hwx-gnustep/lib")}

SRC = Path("/var/tmp/qwen-real-mil")
ROOT = Path("/var/tmp/qwen-real-mil-matrix")
MATVEC_OUT = ROOT / "matvec_isolated"
ANEC_DIR = Path("/var/tmp/h14-qwen-real-anec")
MATVEC_OUT.mkdir(parents=True, exist_ok=True)
ANEC_DIR.mkdir(parents=True, exist_ok=True)


def blob_header(payload: bytes) -> bytes:
    """mint_oracles.blob() format."""
    result = bytearray(128 + len(payload))
    struct.pack_into("<II", result, 0, 1, 2)
    struct.pack_into("<IQQ", result, 64, 0xDEADBEEF, len(payload), 128)
    result[128:] = payload
    return bytes(result)


def isolated_matmul_mil(xshape, yblob_shape, transpose_y, output_shape):
    mil = (
        "program(1.3)\n"
        '[buildInfo = dict<string, string>({{"coremlc-component-MIL", "3520.4.1"}, '
        '{"coremlc-version", "3520.5.1"}})]{'
        "\n    func main<ios18>(tensor<fp16, "
        + str(xshape).replace("'", '')
        + "> x) {\n"
        + f"        tensor<fp16, {yblob_shape}> w = const()"
        + f"[name = string(\"w\"), val = tensor<fp16, {yblob_shape}>"
        + f"(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))];\n"
        + f"        tensor<fp16, {output_shape}> product = matmul(transpose_x = bool(false), "
        + f"transpose_y = bool({str(transpose_y).lower()}), x = x, y = w)[name = string(\"product\")];\n"
        + f"    }} -> (product);\n"
        + "}\n"
    )
    return mil


def is_const_blob(text, ssa):
    return bool(re.search(rf'{ssa}\s*=\s*const\([^;]*?BLOBFILE', text, re.DOTALL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "matvec_isolated_results.json"))
    args = ap.parse_args()

    programs_meta = json.load(open(SRC / "manifest.json"))["programs"]

    # Index original MIL text by prog id
    orig_mil = {}
    for n in range(38):
        pname = f"prog_{n:03d}"
        orig_mil[pname] = (SRC / pname / "model.mil").read_text()

    per_program = {}
    for n in range(38):
        pname = f"prog_{n:03d}"
        text = orig_mil[pname]
        meta = programs_meta[n]
        # find all matmuls
        matmul_results = []
        for mm in re.finditer(r'(\w+)\s*=\s*matmul\(\s*transpose_x\s*=\s*bool\((\w+)\)\s*,\s*transpose_y\s*=\s*bool\((\w+)\)\s*,\s*x\s*=\s*(\w+)\s*,\s*y\s*=\s*(\w+)\)', text):
            out = mm.group(1); tx = mm.group(2); ty = mm.group(3)
            xs = mm.group(4); ys = mm.group(5)
            # resolve x, y shapes from upstream declarations
            xm = re.search(rf'tensor<\s*fp16\s*,\s*\[([^\]]+)\]>\s+{xs}\s*=', text)
            ym = re.search(rf'tensor<\s*fp16\s*,\s*\[([^\]]+)\]>\s+{ys}\s*=', text)
            if not xm or not ym: continue
            xshape = [int(v) for v in xm.group(1).split(', ')]
            yshape = [int(v) for v in ym.group(1).split(', ')]
            x_is_const = is_const_blob(text, xs)
            y_is_const = is_const_blob(text, ys)
            rank = len(xshape)
            if rank == 2 and y_is_const:
                # const-y matvec; H14 supports
                K = xshape[-1]; N = yshape[-1]
                if ty == "true":
                    yblob = [K, N]  # pre-transpose
                else:
                    yblob = [N, K]  # no transpose
                output = [xshape[0], N]
                print(f'  candidate: {pname} {out} xshape={xshape} yblob={yblob}')
                mil = isolated_matmul_mil(xshape, yblob, ty == "true", output)
                weight_payload = b"\x00" * (N * K * 2)
                wblob = blob_header(weight_payload)
                case = f"const_y_{out}"
                stage = MATVEC_OUT / pname / case
                stage.mkdir(parents=True, exist_ok=True)
                (stage / "model.mil").write_text(mil)
                (stage / "weights.bin").write_bytes(wblob)
                out_dir = stage / "out"
                out_dir.mkdir(exist_ok=True)
                cmd = [HWXC, "--mil", str(stage / "model.mil"),
                       "--model-root", str(stage), "--output", str(out_dir),
                       "--target", "H14", "--format", "anec", "--schedule", "per-op"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=60, env=ENV)
                anecs = sorted(out_dir.glob("*.anec"))
                ok = r.returncode == 0 and len(anecs) > 0
                if ok:
                    # also copy to /var/tmp/h14-qwen-real-anec/
                    dest = ANEC_DIR / pname / case
                    dest.mkdir(parents=True, exist_ok=True)
                    for a in anecs:
                        (dest / a.name).write_bytes(a.read_bytes())
                matmul_results.append({
                    "out": out, "x": xs, "y": ys, "xshape": xshape, "yshape": yshape,
                    "ty": ty, "kind": "const-y matvec",
                    "compiled": ok, "n_anecs": len(anecs),
                    "refusal": ((r.stderr or r.stdout).strip()[:200] if not ok else None),
                })
        per_program[pname] = {
            "idx": n, "class": "auto", "n_matmuls": len(matmul_results),
            "n_compiled": sum(1 for x in matmul_results if x["compiled"]),
            "results": matmul_results,
        }
    json.dump(per_program, open(args.out, "w"), indent=1, default=str)
    total = sum(p["n_matmuls"] for p in per_program.values())
    compiled = sum(p["n_compiled"] for p in per_program.values())
    print(f"\n{compiled}/{total} matvec ops compiled across {len(per_program)} programs")
    print(f"ANECs emitted to {ANEC_DIR}")


if __name__ == "__main__":
    main()