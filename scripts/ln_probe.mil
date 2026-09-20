program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 375, 1024]> x) {
    tensor<int32, [1]> axes = const()[name = string("axes"), val = tensor<int32, [1]>(-1)];
    fp16 epsilon = const()[name = string("epsilon"), val = fp16(0x1p-17)];
    tensor<fp16, [1, 375, 1024]> y = layer_norm(axes = axes, epsilon = epsilon, x = x)[name = string("y")];
  } -> (y);
}
