#!/usr/bin/env python3
"""Check finite H13 FP16 results against a dense reference and a pinned control."""
import argparse
import json
import math
import struct
import sys


def load(path):
    with open(path, "rb") as handle:
        data = handle.read()
    if not data or len(data) % 2:
        raise ValueError(f"{path}: expected nonempty whole FP16 lanes")
    return struct.unpack(f"<{len(data) // 2}e", data)


def load_comparison(args):
    got, ref = load(args.got), load(args.ref)
    control = load(args.control) if args.control else None
    if len(got) != len(ref) or (control is not None and len(control) != len(ref)):
        raise ValueError("lane count mismatch")
    if not all(map(math.isfinite, ref)) or (
        control is not None and not all(map(math.isfinite, control))
    ):
        raise ValueError("nonfinite reference or control")
    bound = (max(abs(c - r) for c, r in zip(control, ref))
             if control is not None else args.abs_bound)
    if bound is None or not math.isfinite(bound) or bound < 0:
        raise ValueError("expected a finite, nonnegative absolute bound or --control")
    return got, ref, control, bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("got")
    parser.add_argument("ref")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("abs_bound", nargs="?", type=float)
    mode.add_argument("--control", help="16-dispatch output from the same input as GOT and REF")
    args = parser.parse_args()
    try:
        got, ref, control, bound = load_comparison(args)
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    nonfinite = [index for index, value in enumerate(got) if not math.isfinite(value)]
    errors = [abs(value - expected) for value, expected in zip(got, ref)]
    exact = sum(error == 0.0 for error in errors)
    over = [index for index, error in enumerate(errors) if error > bound]
    report = {
        "lanes": len(ref), "exact": exact,
        "max_abs_error": None if nonfinite else max(errors),
        "mean_abs_error": None if nonfinite else sum(errors) / len(errors),
        "abs_bound": bound, "over_bound": len(over), "first_over": over[:8],
        "nonfinite": len(nonfinite), "first_nonfinite": nonfinite[:8],
    }
    if control is not None:
        report["fused_vs_control_exact"] = sum(a == b for a, b in zip(got, control))
        report["fused_vs_control_max_abs_error"] = (
            None if nonfinite else max(abs(a - b) for a, b in zip(got, control))
        )
    print(json.dumps(report))
    return 1 if over or nonfinite else 0


if __name__ == "__main__":
    sys.exit(main())
