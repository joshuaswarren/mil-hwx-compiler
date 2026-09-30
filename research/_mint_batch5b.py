#!/usr/bin/env python3
"""H14Batch5 mints (corrected F1b).

Each F1b mint must have shape == output_shape (the bigger operand), with
x at the bigger shape and y at the smaller, so generate_h14_templates.py
classifies it correctly.
"""
import sys
sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om
from pathlib import Path
import subprocess, hashlib, shutil

ROOT = Path('/tmp/oracle_out_batch5b')
TOOL = Path('/private/tmp/ane-compile-hwx')


def mint(name, parameters, mil, family='env_broadcast'):
    case = {
        "name": name, "family": family,
        "parameters": parameters, "mil": mil,
        "weights": None, "weights_description": {"storage": "none"},
    }
    out = om.run_case(case, 'h14', ROOT, TOOL, 'H14Batch5b-oneoff')
    print(name, '->', out)
    return out


# F1b: per-head broadcast at the head-major surface.
# shape = output_shape = [1,16,1,128], operand_shape = [1,16,1,1]

mint("env_bcast_mul_1x16x1x128_runtime_1x16x1x1",
     {"operation": "mul", "shape": [1, 16, 1, 128],
      "operand_shape": [1, 16, 1, 1], "operand": "runtime",
      "output_shape": [1, 16, 1, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 128]> x, tensor<fp16, [1, 16, 1, 1]> z) {
    tensor<fp16, [1, 16, 1, 128]> y = mul(x = x, y = z)[name = string("y")];
  } -> (y);
}
""", family="env_broadcast")

mint("env_bcast_mul_1x16x1x128_runtime_1x16x128x1",
     {"operation": "mul", "shape": [1, 16, 1, 128],
      "operand_shape": [1, 16, 128, 1], "operand": "runtime",
      "output_shape": [1, 16, 1, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 128]> x, tensor<fp16, [1, 16, 128, 1]> z) {
    tensor<fp16, [1, 16, 1, 128]> y = mul(x = x, y = z)[name = string("y")];
  } -> (y);
}
""", family="env_broadcast")

mint("env_bcast_add_1x16x1x128_runtime_1x16x1x1",
     {"operation": "add", "shape": [1, 16, 1, 128],
      "operand_shape": [1, 16, 1, 1], "operand": "runtime",
      "output_shape": [1, 16, 1, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 128]> x, tensor<fp16, [1, 16, 1, 1]> z) {
    tensor<fp16, [1, 16, 1, 128]> y = add(x = x, y = z)[name = string("y")];
  } -> (y);
}
""", family="env_broadcast")

mint("env_bcast_add_1x16x1x128_runtime_1x16x128x1",
     {"operation": "add", "shape": [1, 16, 1, 128],
      "operand_shape": [1, 16, 128, 1], "operand": "runtime",
      "output_shape": [1, 16, 1, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 128]> x, tensor<fp16, [1, 16, 128, 1]> z) {
    tensor<fp16, [1, 16, 1, 128]> y = add(x = x, y = z)[name = string("y")];
  } -> (y);
}
""", family="env_broadcast")

print("done.")