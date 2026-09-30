#!/usr/bin/env python3
"""H14Batch5 mints.

Re-mint F5 (creadout chain), add F1b per-head broadcast shapes, add F4
alternative softmax-attention spellings, and mint the F2-followup
chain -> linear reshape boundary on the Mac side.
"""
import sys
sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om
from pathlib import Path

ROOT = Path('/tmp/oracle_out_batch5')

TOOL = Path('/private/tmp/ane-compile-hwx')

def mint(name, parameters, mil, family='env_chain'):
    case = {
        "name": name,
        "family": family,
        "parameters": parameters,
        "mil": mil,
        "weights": None,
        "weights_description": {"storage": "none"},
    }
    out = om.run_case(case, 'h14', ROOT, TOOL, 'H14Batch5-oneoff')
    print(name, '->', out)
    return out


# ---- F5 re-mint: 3-op sigmoid → mul → add at [1, 16, 128, 1] ----
mint("env_chain_creadout_h16_s128",
     {"ops": "sigmoid+mul+add", "shape": [1, 16, 128, 1]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 128, 1]> o, tensor<fp16, [1, 16, 128, 1]> gate, tensor<fp16, [1, 16, 128, 1]> h_residual) {
    tensor<fp16, [1, 16, 128, 1]> sg = sigmoid(x = gate)[name = string("sg")];
    tensor<fp16, [1, 16, 128, 1]> gated = mul(x = o, y = sg)[name = string("gated")];
    tensor<fp16, [1, 16, 128, 1]> h = add(x = gated, y = h_residual)[name = string("h")];
  } -> (h);
}
""", family="env_chain")

# ---- F1b: per-head broadcast at the head-major surface Apple prefers ----
# Apple accepted the channel-broadcast at [1,16,128,128]x[1,16,1,1] in
# batch2. The B-state programs use a different surface: [1,16,1,128]x[1,16,1,1]
# (per-head scale). Try both mul and add.

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

mint("env_bcast_mul_1x16x1x1_runtime_1x16x1x128",
     {"operation": "mul", "shape": [1, 16, 1, 1],
      "operand_shape": [1, 16, 1, 128], "operand": "runtime",
      "output_shape": [1, 16, 1, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 1]> z, tensor<fp16, [1, 16, 1, 128]> x) {
    tensor<fp16, [1, 16, 1, 128]> y = mul(x = z, y = x)[name = string("y")];
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

mint("env_bcast_add_1x16x1x1_runtime_1x16x1x128",
     {"operation": "add", "shape": [1, 16, 1, 1],
      "operand_shape": [1, 16, 1, 128], "operand": "runtime",
      "output_shape": [1, 16, 1, 128]},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 1]> z, tensor<fp16, [1, 16, 1, 128]> x) {
    tensor<fp16, [1, 16, 1, 128]> y = add(x = z, y = x)[name = string("y")];
  } -> (y);
}
""", family="env_broadcast")

# ---- F4: alternative softmax-attention upstream spellings ----
# Try (a) mul(x, fp16 scale) scalar literal, (b) rank-2 softmax only.

mint("env_chain_softmax_attn_h16_s50_v3",
     {"shape": [16, 50], "axis": -1, "upstream": "mul+add+softmax_rank2"},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [16, 50]> x) {
    tensor<fp16, [1, 1, 1, 1]> scale = const()[name = string("scale"), val = tensor<fp16, [1, 1, 1, 1]>([0.1817])];
    tensor<fp16, [1, 1, 1, 1]> bias = const()[name = string("bias"), val = tensor<fp16, [1, 1, 1, 1]>([0.0])];
    tensor<fp16, [16, 50]> scaled = mul(x = x, y = scale)[name = string("scaled")];
    tensor<fp16, [16, 50]> masked = add(x = scaled, y = bias)[name = string("masked")];
    tensor<fp16, [16, 50]> y = softmax(axis = 1, x = masked)[name = string("y")];
  } -> (y);
}
""", family="env_chain")

# Also try rank-4 [1,16,50,1] with fp16 scale as scalar.
mint("env_chain_softmax_attn_h16_s50_v4",
     {"shape": [1, 16, 50, 1], "axis": -1, "upstream": "mul+add+softmax_rank4"},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 50, 1]> x) {
    tensor<fp16, [1, 1, 1, 1]> scale = const()[name = string("scale"), val = tensor<fp16, [1, 1, 1, 1]>([0.1817])];
    tensor<fp16, [1, 1, 1, 1]> bias = const()[name = string("bias"), val = tensor<fp16, [1, 1, 1, 1]>([0.0])];
    tensor<fp16, [1, 16, 50, 1]> scaled = mul(x = x, y = scale)[name = string("scaled")];
    tensor<fp16, [1, 16, 50, 1]> masked = add(x = scaled, y = bias)[name = string("masked")];
    tensor<fp16, [1, 16, 50, 1]> y = softmax(axis = -1, x = masked)[name = string("y")];
  } -> (y);
}
""", family="env_chain")

# ---- F2-followup: chain output -> linear/matmul [1,2048,1,1] -> [1,2048] -> [1,N] ----
# Try: (a) Apple-compiled with chain output as linear input (no reshape).
#      (b) explicit reshape chain -> [1,2048] before linear.
#      (c) matvec spelling "1x2048x1x1" treated as conv 1x1.

mint("env_chain_linear_1x2048x1x1_to_1x6144",
     {"shape_in": [1, 2048, 1, 1], "weight_shape": [6144, 2048],
      "ops": "rms_norm+linear_no_reshape"},
     # This requires the rms_norm chain + linear in the same program.
     # Apple may fuse chain + linear via L2 output channel into the linear.
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 2048, 1, 1]> x, tensor<fp16, [6144, 2048]> w) {
    tensor<fp16, [1, 1, 1, 1]> amax;
    tensor<fp16, [1, 1, 1, 1]> rcp;
    tensor<fp16, [1, 2048, 1, 1]> n = mul(x = x, y = rcp)[name = string("n")];
    tensor<fp16, [1, 2048, 1, 1]> t = linear(x = n, weight = w)[name = string("t")];
  } -> (t);
}
""", family="env_chain")

mint("env_chain_reshape_1x2048x1x1_to_1x2048",
     {"shape_in": [1, 2048, 1, 1], "shape_out": [1, 2048],
      "ops": "reshape_only"},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 2048, 1, 1]> x) {
    tensor<fp16, [1, 2048]> y = reshape(shape = [1, 2048], x = x)[name = string("y")];
  } -> (y);
}
""", family="env_chain")

mint("env_chain_reshape_then_linear",
     {"shape_in": [1, 2048, 1, 1], "weight_shape": [6144, 2048],
      "ops": "reshape+linear"},
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 2048, 1, 1]> x, tensor<fp16, [6144, 2048]> w) {
    tensor<fp16, [1, 2048]> r = reshape(shape = [1, 2048], x = x)[name = string("r")];
    tensor<fp16, [1, 6144]> t = linear(x = r, weight = w)[name = string("t")];
  } -> (t);
}
""", family="env_chain")

print("done.")