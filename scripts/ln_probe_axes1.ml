program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [375, 1024, 1, 1]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>([1])];
    fp16 epsilon = const()[name = string("epsilon"), val = fp16(0x1.5p-17)];
    tensor<fp16, [375, 1, 1, 1]> mean = reduce_mean(axes = axes, keep_dims = bool(true), x = x)[name = string("mean")];
    tensor<fp16, [375, 1024, 1, 1]> diff = sub(x = x, y = mean)[name = string("diff")];
    tensor<fp16, [375, 1024, 1, 1]> sq = mul(x = diff, y = diff)[name = string("sq")];
    tensor<fp16, [375, 1, 1, 1]> var = reduce_mean(axes = axes, keep_dims = bool(true), x = sq)[name = string("var")];
    tensor<fp16, [375, 1, 1, 1]> vareps = add(x = var, y = epsilon)[name = string("vareps")];
    tensor<fp16, [375, 1, 1, 1]> rsv = rsqrt(x = vareps, epsilon = fp32(0.0))[name = string("rsv")];
    tensor<fp16, [375, 1024, 1, 1]> out = mul(x = diff, y = rsv)[name = string("out")];
  } -> (out);
}
