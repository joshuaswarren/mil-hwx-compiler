#!/usr/bin/env python3
"""H13 LUT-unary acceptance check against a dense fp16 reference.

Policy (docs/ane/h13-acceptance.md): every lane must be finite and satisfy
|got - ref| <= --abs-bound. Relative bounds are unsatisfiable near zero
because the LUT quantizes the zero bin (measured 0.0017 at ref == 0).
Prints a JSON summary; exit 1 on any violation.
"""
import json
import math
import struct
import sys


def load(path):
    with open(path, "rb") as handle:
        data = handle.read()
    return struct.unpack(f"<{len(data) // 2}e", data)


def main():
    if len(sys.argv) != 4:
        print("usage: h13_acceptance.py GOT_FP16 REF_FP16 ABS_BOUND", file=sys.stderr)
        return 2
    got, ref = load(sys.argv[1]), load(sys.argv[2])
    bound = float(sys.argv[3])
    if len(got) != len(ref):
        print(f"FAIL length mismatch: {len(got)} vs {len(ref)}", file=sys.stderr)
        return 1
    nonfinite = [index for index, value in enumerate(got) if not math.isfinite(value)]
    errors = [abs(value - expected) for value, expected in zip(got, ref)]
    exact = sum(error == 0.0 for error in errors)
    over = [index for index, error in enumerate(errors) if error > bound]
    print(json.dumps({
        "lanes": len(ref), "exact": exact, "max_abs_error": max(errors),
        "mean_abs_error": sum(errors) / len(errors), "abs_bound": bound,
        "over_bound": len(over), "first_over": over[:8],
        "nonfinite": len(nonfinite), "first_nonfinite": nonfinite[:8],
    }))
    return 1 if over or nonfinite else 0


if __name__ == "__main__":
    sys.exit(main())
