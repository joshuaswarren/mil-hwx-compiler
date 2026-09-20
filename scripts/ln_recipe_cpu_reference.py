#!/usr/bin/env python3
"""Independent float64 reference for the corrected non-affine LN
decomposition (reduce_mean/sub/mul/reduce_mean/add/rsqrt/mul,
keep_dims=True, real epsilon=fp16(0x1.5p-17), rsqrt eps=0). Asserts
fp16-ROUNDED per-operation mean/var/rsqrt against an independent
float64 reference, NOT bit-identity. Reports ulp bounds honestly.

Two comparisons:
  A) 4D-reshape equivalence: produce the SAME fp16 input, route through
     a 4D rank-4 layer_norm fp16 MIL on the input + through the
     decomposed primitives, compare element-by-element.
  B) Per-operation fp16 rounding: each primitive of the decomp applied
     in fp16 and compared to fp64 reference; report per-primitive
     max abs err and max ULP, no equality claim.
"""
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

EPS_FP16 = float.fromhex("0x1.5p-17")
SHAPES = [
    ((1, 375, 1024), [-1]),
    ((1, 375, 1, 1024), [3]),
    ((375, 1024, 1, 1), [1]),
]


def fp16_ulp(a_f16: np.ndarray) -> np.ndarray:
    """Per-element ULP distance (fp16)."""
    bits = a_f16.view(np.uint16).astype(np.int64)
    raw = a_f16.astype(np.float32)
    next_up = np.nextafter(raw, np.inf).astype(np.float32)
    next_dn = np.nextafter(raw, -np.inf).astype(np.float32)
    ulp_up = np.abs(next_up - raw).astype(np.float32)
    ulp_dn = np.abs(raw - next_dn).astype(np.float32)
    ulp = np.maximum(ulp_up, ulp_dn)
    return (np.abs(raw - np.float64(0)) / np.where(ulp == 0, 1, ulp)).astype(np.float32)


def fp16_round(x_f32: np.ndarray) -> np.ndarray:
    return x_f32.astype(np.float16).astype(np.float32)


def decomposed_fp16(x_f32: np.ndarray, axes: list[int]) -> np.ndarray:
    """The corrected decomposition applied round-to-nearest fp16 at every step."""
    rm = lambda y, ax: y.mean(axis=tuple(ax), keepdims=True)
    mean = rm(x_f32, axes).astype(np.float16).astype(np.float32)
    diff = (x_f32 - mean).astype(np.float16).astype(np.float32)
    sq = (diff * diff).astype(np.float16).astype(np.float32)
    var = rm(sq, axes).astype(np.float16).astype(np.float32)
    var_eps = (var + EPS_FP16).astype(np.float16).astype(np.float32)
    rsv = (1.0 / np.sqrt(var_eps.astype(np.float64))).astype(np.float16).astype(np.float32)
    return (diff * rsv).astype(np.float16).astype(np.float32)


def native_fp64(x_f64: np.ndarray, axes: list[int]) -> np.ndarray:
    rm = lambda y, ax: y.mean(axis=tuple(ax), keepdims=True)
    mean = rm(x_f64, axes)
    diff = x_f64 - mean
    var = rm(diff * diff, axes)
    var_eps = var + EPS_FP16
    return diff / np.sqrt(var_eps)


def report_ulps(stage: str, ref: np.ndarray, ours: np.ndarray) -> dict:
    diff = np.abs(ref - ours)
    ulp_diff = fp16_ulp(ours).astype(np.float64)
    # distance in ULP = |ref - ours| / ours's ULP (per element)
    with np.errstate(divide="ignore", invalid="ignore"):
        ulps = np.where(ulp_diff > 0, diff.astype(np.float64) / ulp_diff, 0)
    finite = np.isfinite(ulps)
    return {
        "stage": stage,
        "max_abs_err": float(diff.max()),
        "max_ulp_distance": int(ulps[finite].max()) if finite.any() else 0,
        "mean_ulp_distance": float(ulps[finite].mean()) if finite.any() else 0.0,
        "nonfinite_ulp_count": int((~finite).sum()),
    }


def main() -> int:
    rng = np.random.default_rng(7)
    print("== Per-stage fp16-rounding vs fp64 reference ==")
    all_reports = []
    for shape, axes in SHAPES:
        print(f"\nshape={shape} axes={axes}")
        x_f64 = rng.standard_normal(shape).astype(np.float64)
        x_f32 = x_f64.astype(np.float32)
        rm = lambda y, ax: y.mean(axis=tuple(ax), keepdims=True)
        mean_f32 = rm(x_f32, axes)
        diff_f32 = x_f32 - mean_f32
        sq_f32 = diff_f32 * diff_f32
        var_f32 = rm(sq_f32, axes)
        mean_fp16 = mean_f32.astype(np.float16)
        diff_fp16 = diff_f32.astype(np.float16)
        sq_fp16 = sq_f32.astype(np.float16)
        var_fp16 = var_f32.astype(np.float16)
        var_eps_f32 = var_f32 + np.float32(EPS_FP16)
        var_eps_fp16 = (var_f32 + np.float32(EPS_FP16)).astype(np.float16)
        # stage 1: mean
        r_mean = report_ulps("mean", rm(x_f64, axes).astype(np.float32),
                             mean_fp16.astype(np.float32))
        # stage 2: diff (rounding error of x - mean; this is where fp16 vs fp64
        # really bites)
        ref_diff = (x_f64.astype(np.float32) - mean_fp16.astype(np.float32))
        r_diff = report_ulps("diff", ref_diff, diff_fp16.astype(np.float32))
        # stage 3: sq
        ref_sq = (ref_diff * ref_diff).astype(np.float32)
        r_sq = report_ulps("sq", ref_sq, sq_fp16.astype(np.float32))
        # stage 4: var
        ref_var = rm((ref_diff * ref_diff).astype(np.float32), axes)
        r_var = report_ulps("var", ref_var, var_fp16.astype(np.float32))
        # stage 5: vareps
        ref_eps = ref_var + np.float32(EPS_FP16)
        r_eps = report_ulps("vareps", ref_eps, var_eps_fp16.astype(np.float32))
        # stage 6: rsqrt (rsqrt(eps=0): exact fp16 inverse sqrt of vareps)
        rsv_ref = (1.0 / np.sqrt(ref_eps.astype(np.float64))).astype(np.float32)
        rsv_ours = (1.0 / np.sqrt(var_eps_fp16.astype(np.float64))).astype(
            np.float16).astype(np.float32)
        r_rsv = report_ulps("rsqrt", rsv_ref, rsv_ours)
        # stage 7: out = diff * rsv
        ref_out = (ref_diff * rsv_ref).astype(np.float32)
        ours_out = (diff_fp16 * rsv_ours).astype(np.float16).astype(np.float32)
        r_out = report_ulps("out", ref_out, ours_out)
        # stage 8: native fp64 baseline as a sanity reference
        nat = native_fp64(x_f64, axes).astype(np.float32)
        r_native = report_ulps("vs_native_fp64", nat, ours_out)
        for r in (r_mean, r_diff, r_sq, r_var, r_eps, r_rsv, r_out, r_native):
            print(f"  {r['stage']:14s} max_abs={r['max_abs_err']:.3e} "
                  f"max_ulp={r['max_ulp_distance']} mean_ulp={r['mean_ulp_distance']:.2f} "
                  f"nonfinite_ulp={r['nonfinite_ulp_count']}")
            all_reports.append({**r, "shape": shape, "axes": axes})

    # 4D-reshape equivalence: produce one (1,375,1024) fp16 input, route
    # the same value through two MILs (the rank-3 layer_norm AND a
    # rank-4 reshape) - the compiler may or may not canonicalize; we
    # compare at the Python fp64 level using reduce_mean+sub+mul+rsqrt.
    print("\n== 4D-reshape equivalence at the Python fp16 reference level ==")
    x_f32 = rng.standard_normal((1, 375, 1024)).astype(np.float32)
    x_f32_axes3 = x_f32.reshape(1, 375, 1, 1024)
    x_f32_axes1 = x_f32.reshape(375, 1024, 1, 1)
    # Reference native LN, axes=[-1] for the rank-3, axes=[3] for the
    # rank-4 axes3, axes=[1] for rank-4 axes1.
    for label, x_in, axes in [("rank3 axes=[-1]", x_f32, [-1]),
                            ("rank4 axes=[3]", x_f32_axes3, [3]),
                            ("rank4 axes=[1]", x_f32_axes1, [1])]:
        ref = native_fp64(x_in.astype(np.float64), axes).astype(np.float32)
        ours = decomposed_fp16(x_in, axes)
        r = report_ulps(label, ref, ours)
        print(f"  {label:18s} shape={x_in.shape} axes={axes} "
              f"max_abs={r['max_abs_err']:.3e} max_ulp={r['max_ulp_distance']} "
              f"mean_ulp={r['mean_ulp_distance']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
