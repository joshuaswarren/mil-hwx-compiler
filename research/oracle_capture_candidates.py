#!/usr/bin/env python3
"""Emit the encoder forms that block full-graph coverage and have no decoded
H13 row — the exact probe MILs for the next studio-host oracle window.

Each candidate mirrors a real Parakeet-encoder statement (spelling, dtypes,
shapes). Apple's pinned `ane-compile-hwx` either decodes it (the compiler
can then land a row + tests, as with every earlier capture round) or
refuses it (the refusal is recorded and the form stays fail-closed).

Run:  python3 research/oracle_capture_candidates.py [output-dir]
Then run the pinned oracle per emitted MIL exactly as in the
broadcast-compare round (receipt 2026-09-20; pinned tool sha256
3d13fc85c2a6baa0b7628f0848269b16fb99f742180f0085f608c371bb9da6e0).

Candidates, in encoder-statement order:
1. logical_and over bool [1, 375, 375] x [1, 375, 375] — the attention-mask
   composition (probe line 163). No decoded form; its refusal is the first
   blocker after the mask-prelude respell.
2. transpose bool [1, 375, 375] perm [0, 2, 1] — the tail-swap view the
   same statement needs. Decoded transpose parity rows are fp16-only.
3. cast fp16 -> bool [1, 1, 375] — enables the all_masked_rows respell
   through the decoded fp16 reduce_min (the int32 reduce_min and int32->
   bool cast are both Apple-refused today).
4. select runtime-a/runtime-b with a broadcast bool cond [1, 1, 375] at
   [1, 1024, 375] — the 24 GLU/input-gate selects' geometry (the decoded
   select rows cover (64,1,1) and (8,375,375) only).
5. reduce_min int32 [1, 1, 375, 375] axes [2] keep_dims false — re-ask
   with the exact real spelling (earlier refusal was recorded at the
   result gate, not the op).
"""
import sys
from pathlib import Path


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


CANDIDATES = [
    ("logical_and_bool_1x375x375",
     program("tensor<bool, [1, 375, 375]> a, tensor<bool, [1, 375, 375]> b",
             ['tensor<bool, [1, 375, 375]> out = logical_and(x = a, y = b)'
              '[name = string("out")];'],
             "out")),
    ("transpose_bool_1x375x375_021",
     program("tensor<bool, [1, 375, 375]> x",
             ['tensor<int32, [3]> perm = const()[name = string("perm"), '
              'val = tensor<int32, [3]>([0, 2, 1])];',
              'tensor<bool, [1, 375, 375]> out = transpose(perm = perm, '
              'x = x)[name = string("out")];'],
             "out")),
    ("cast_f16_to_b_1x1x375",
     program("tensor<fp16, [1, 1, 375]> x",
             ['tensor<string, []> dt = const()[name = string("dt"), '
              'val = tensor<string, []>("bool")];',
              'tensor<bool, [1, 1, 375]> out = cast(dtype = dt, x = x)'
              '[name = string("out")];'],
             "out")),
    ("select_rrb_1x1024x375_bcast_cond",
     program("tensor<fp16, [1, 1024, 375]> a, tensor<fp16, [1, 1024, 375]> b, "
             "tensor<bool, [1, 1, 375]> cond",
             ['tensor<fp16, [1, 1024, 375]> out = select(a = a, b = b, '
              'cond = cond)[name = string("out")];'],
             "out")),
    ("reduce_min_i32_1x1x375x375_axes2",
     program("tensor<int32, [1, 1, 375, 375]> x",
             ['tensor<int32, [1]> axes = const()[name = string("axes"), '
              'val = tensor<int32, [1]>([2])];',
              'tensor<bool, []> keep = const()[name = string("keep"), '
              'val = bool(false)];',
              'tensor<int32, [1, 1, 375]> out = reduce_min(axes = axes, '
              'keep_dims = keep, x = x)[name = string("out")];'],
             "out")),
]


def main() -> int:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else
                  "/tmp/encoder-capture-candidates")
    output.mkdir(parents=True, exist_ok=True)
    for name, mil in CANDIDATES:
        (output / f"{name}.mil").write_text(mil)
        print(f"{name}.mil")
    print(f"{len(CANDIDATES)} candidate MILs in {output}; run the pinned "
          "oracle per file (CPU-only, no OS or baseline changes) and feed "
          "the results back as capture records or recorded refusals.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
