#!/usr/bin/env python3
"""Build an elementwise oracle JSON from the qwenmint mul [16,128,128] HWX
decode and write it into research/oracles/h14/, so generate_h14_templates.py
regenerates the .inc with the new entry."""
import json
import struct
import sys
from pathlib import Path

DECODED = Path('/tmp/_mul_decode.json')

d = json.load(open(DECODED))
# shape: [1, 16, 128, 128] (rank-4 with batch=1 so the encoder CHW = (16,128,128))
# Case name follows the existing envelope naming.
case = "env_bcast_mul_1x16x128x128_runtime_1x16x128x128"
oracle = {
    "case": case,
    "family": "env_broadcast",
    "parameters": {
        "operation": "mul",
        "shape": [1, 16, 128, 128],
        "output_shape": [1, 16, 128, 128],
        "operand": "runtime",
        "operand_shape": [1, 16, 128, 128],
    },
    "hwx_bytes": d["hwx_bytes"],
    "hwx_sha256": d["hwx_sha256"],
    "constant_section": d["constant_section"],
    "program_descriptor": d["program_descriptor"],
    "task_descriptors": d["task_descriptors"],
    "tensor_descriptors": [
        {"binding": 1, "element_code": 5, "shape": [1, 16, 128, 128],
         "strides": [524288, 32768, 256, 2], "total_bytes": 524288},
        {"binding": 1, "element_code": 5, "shape": [1, 16, 128, 128],
         "strides": [524288, 32768, 256, 2], "total_bytes": 524288},
        {"binding": 2, "element_code": 5, "shape": [1, 16, 128, 128],
         "strides": [524288, 32768, 256, 2], "total_bytes": 524288},
    ],
    "error": None,
    "mil": "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3520.4.1\"}, {\"coremlc-version\", \"3520.5.1\"}})]{\n    func main<ios18>(tensor<fp16, [1, 16, 128, 128]> x, tensor<fp16, [1, 16, 128, 128]> y) {\n        tensor<fp16, [1, 16, 128, 128]> z = mul(x = x, y = y)[name = string(\"z\")];\n    } -> (z);\n}\n",
}
out = Path('/home/joshuawarren/src/mil-hwx-h14-integ-wt/research/oracles/h14') / f"{case}.json"
out.write_text(json.dumps(oracle, indent=4, sort_keys=True))
print(f"WROTE {out}")
print(f"  hwx_bytes={d['hwx_bytes']}, tasks={len(d['task_descriptors'])}, "
      f"text_words={d['program_descriptor']['text_words']}, "
      f"const_size={d['constant_section']['size']}, "
      f"const_nonzero={d['constant_section']['nonzero_bytes']}")