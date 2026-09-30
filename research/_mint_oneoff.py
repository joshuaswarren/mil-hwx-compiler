#!/usr/bin/env python3
"""Mint the corrected broadcast shapes that the agent needs but that the
shape-ext campaign missed. Each call is a one-off oracle save to the
batch2 JSON tree.
"""
import sys
sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om
import json, subprocess, platform
from pathlib import Path

ROOT = Path('/tmp/oracle_out')

def mint(name, parameters, mil, family='env_broadcast'):
    case = {
        "name": name,
        "family": family,
        "parameters": parameters,
        "mil": mil,
        "weights": None,
        "weights_description": {"storage": "none"},
    }
    out = om.run_case(case, 'h14', ROOT, Path('/private/tmp/ane-compile-hwx'),
                      'H14Batch2-oneoff')
    print(name, '->', out)
    return out


# F1: broadcast at [1, 16, 128, 128] × [1, 16, 1, 1] — Apple accepted.
mint("env_bcast_mul_1x16x128x128_runtime_1x16x1x1",
     {"operation": "mul", "shape": [1, 16, 128, 128],
      "operand_shape": [1, 16, 1, 1], "operand": "runtime",
      "output_shape": [1, 16, 128, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 128, 128]> x, tensor<fp16, [1, 16, 1, 1]> z) {
    tensor<fp16, [1, 16, 128, 128]> y = mul(x = x, y = z)[name = string("y")];
  } -> (y);
}
""")

mint("env_bcast_add_1x16x128x128_runtime_1x16x1x1",
     {"operation": "add", "shape": [1, 16, 128, 128],
      "operand_shape": [1, 16, 1, 1], "operand": "runtime",
      "output_shape": [1, 16, 128, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 128, 128]> x, tensor<fp16, [1, 16, 1, 1]> z) {
    tensor<fp16, [1, 16, 128, 128]> y = add(x = x, y = z)[name = string("y")];
  } -> (y);
}
""")

# F4: softmax attention pipeline at [16, 50] — try rank-2 (Apple accepted
# the bare softmax at [16,50]; the chain may work too).
mint("env_chain_softmax_attn_h16_s50_v2",
     {"shape": [16, 50], "axis": -1, "upstream": "mul+add"},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [16, 50]> x, tensor<fp16, [16, 50]> scale, tensor<fp16, [50]> mask) {
    tensor<fp16, [16, 50]> scaled = mul(x = x, y = scale)[name = string("scaled")];
    tensor<fp16, [16, 50]> masked = add(x = scaled, y = mask)[name = string("masked")];
    tensor<fp16, [16, 50]> y = softmax(axis = 1, x = masked)[name = string("y")];
  } -> (y);
}
""", family="env_chain")

# F2: rms_norm pow-form at [1, 2048, 1, 1] axes=[1] — already in the
# Apple-decoded set (h14norm_reduce_sum_ax1_1x2048x1x1.json).
# Try the chain form (mul + reduce_sum + add + sqrt + real_div + mul +
# real_div) at the channel-flat surface — this is what the chain decoder
# would need to fuse.
mint("env_chain_rms_norm_pow_h1_c2048",
     {"shape": [1, 2048, 1, 1], "axes": [1], "ops": "pow-form"},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 2048, 1, 1]> x) {
    tensor<fp16, [1, 2048, 1, 1]> x2 = mul(x = x, y = x)[name = string("x2")];
    tensor<int32, [1]> ss_axes = const()[name = string("ss_axes"), val = tensor<int32, [1]>([1])];
    bool ss_keep = const()[name = string("ss_keep"), val = bool(true)];
    tensor<fp16, [1, 1, 1, 1]> ss = reduce_sum(axes = ss_axes, keep_dims = ss_keep, x = x2)[name = string("ss")];
    tensor<fp16, [1, 1, 1, 1]> eps = const()[name = string("eps"), val = tensor<fp16, [1, 1, 1, 1]>([5.96e-8])];
    tensor<fp16, [1, 1, 1, 1]> adj = add(x = ss, y = eps)[name = string("adj")];
    tensor<fp16, [1, 1, 1, 1]> rt = sqrt(x = adj)[name = string("rt")];
    tensor<fp16, [1, 1, 1, 1]> k_one = const()[name = string("k_one"), val = tensor<fp16, [1, 1, 1, 1]>([1.0])];
    tensor<fp16, [1, 1, 1, 1]> inv = real_div(x = k_one, y = rt)[name = string("inv")];
    tensor<fp16, [1, 2048, 1, 1]> y = mul(x = x, y = inv)[name = string("y")];
  } -> (y);
}
""", family="env_chain")

print("done.")
