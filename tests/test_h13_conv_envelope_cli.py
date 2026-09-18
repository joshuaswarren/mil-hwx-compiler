#!/usr/bin/env python3
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())


def blob(payload):
    header = bytearray(128)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(payload), 128)
    return bytes(header) + payload


def conv_mil(x_shape, w_shape, y_shape, stride, groups, pad, pad_type, bias,
             dilations=None):
    if dilations is None:
        dilations = [1] * len(stride)
    x_t = f"tensor<fp16, {x_shape}>"
    w_t = f"tensor<fp16, {w_shape}>"
    y_t = f"tensor<fp16, {y_shape}>"
    st_t = f"tensor<int32, [{len(stride)}]>"
    pd_t = f"tensor<int32, [{len(pad)}]>"
    dl_t = f"tensor<int32, [{len(dilations)}]>"
    bias_decl = ""
    bias_arg = ""
    if bias:
        b_t = f"tensor<fp16, [{w_shape[0]}]>"
        bias_decl = (
            f'    {b_t} b = const()[name = string("b"), val = {b_t}'
            f'(BLOBFILE(path = string("@model_path/bias.bin"), offset = uint64(64)))];\n')
        bias_arg = "bias = b, "
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_t} x) {{
    string pt = const()[name = string("pt"), val = string("{pad_type}")];
    {st_t} st = const()[name = string("st"), val = {st_t}({stride})];
    {pd_t} pd = const()[name = string("pd"), val = {pd_t}({pad})];
    {dl_t} dl = const()[name = string("dl"), val = {dl_t}({dilations})];
    int32 gp = const()[name = string("gp"), val = int32({groups})];
    {w_t} w = const()[name = string("w"), val = {w_t}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
{bias_decl}    {y_t} y = conv({bias_arg}dilations = dl, groups = gp, pad = pd, pad_type = pt, strides = st, weight = w, x = x)[name = string("y")];
  }} -> (y);
}}
"""


def compile_source(root, name, text, expected_code=None):
    mil = root / f"{name}.mil"
    out = root / name
    mil.write_text(text)
    run = subprocess.run(
        [compiler, "--mil", str(mil), "--model-root", str(root),
         "--target", "H13", "--output", str(out)],
        capture_output=True, text=True, check=False, timeout=30)
    if expected_code is None:
        assert run.returncode == 0, run.stdout + run.stderr
        return json.loads((out / "manifest.json").read_text())
    assert run.returncode == 65, run.stdout + run.stderr
    assert expected_code in run.stderr, run.stderr
    assert not out.exists(), f"failed compilation wrote {out}"
    return None


with tempfile.TemporaryDirectory(prefix="mil-hwx-h13-conv-envelope-") as directory:
    root = Path(directory)
    # Four megabytes: the decoded out-projection row packs a [1024, 1024, 1]
    # weight, the decoded in-projection row packs a [2048, 1024, 1] weight,
    # and the blob loader checks the declared size per case. The
    # bias blob covers the 1024-entry depthwise bias.
    (root / "weights.bin").write_bytes(blob(bytes(2048 * 1024 * 2)))
    (root / "bias.bin").write_bytes(blob(bytes(1024 * 2)))
    hit = compile_source(
        root, "k1-c256-s32",
        conv_mil([1, 256, 32, 32], [256, 256, 1, 1], [1, 256, 32, 32],
                 [1, 1], 1, [0, 0, 0, 0], "same", True))
    assert len(hit["programs"]) == 1
    assert hit["programs"][0]["operation"] == "conv"
    assert hit["programs"][0]["encoder"] == "apple-parity-conv"
    assert hit["programs"][0]["taskDescriptors"] == 1
    # The rank-3 out-projection spell lowers through the W-major surface the
    # encoder's rank-3 tensors bind as: one linked two-task parity program.
    hit = compile_source(
        root, "enc-1d-pw-outproj",
        conv_mil([1, 1024, 375], [1024, 1024, 1], [1, 1024, 375],
                 [1], 1, [0, 0], "valid", False))
    assert hit["programs"][0]["encoder"] == "apple-parity-conv"
    assert hit["programs"][0]["taskDescriptors"] == 2
    # The bias-bearing rank-3 depthwise lowers through the custom-to-same
    # normalization (pad [4, 4] is exactly the same pad of a unit-stride
    # odd kernel): one linked two-task parity program with the slotted
    # 64-lane section.
    hit = compile_source(
        root, "enc-1d-dw-bias",
        conv_mil([1, 1024, 375], [1024, 1, 9], [1, 1024, 375],
                 [1], 1024, [4, 4], "custom", True))
    assert hit["programs"][0]["encoder"] == "apple-parity-conv"
    assert hit["programs"][0]["taskDescriptors"] == 2
    # The rank-3 in-projection spell lowers through the W-major surface the
    # encoder's rank-3 tensors bind as: the decoded colgroup-order row, one
    # linked four-task parity program. The colgroup order past 64 was
    # decoded with a 32-bit-pair payload remint (the uint16 capture aliases
    # colgroups 64 apart).
    hit = compile_source(
        root, "enc-1d-pw-inproj",
        conv_mil([1, 1024, 375], [2048, 1024, 1], [1, 2048, 375],
                 [1], 1, [0, 0], "valid", False))
    assert hit["programs"][0]["encoder"] == "apple-parity-conv"
    assert hit["programs"][0]["taskDescriptors"] == 4
    # The two encoder pointwise bias1 forms ([1,256,750,32] and
    # [1,256,375,16]) lower since the distinct-payload round qualified their
    # rows; the parity suite byte-compiles them.
    for name, mil in (
        ("enc-subsample",
         conv_mil([1, 1, 3000, 128], [256, 1, 3, 3], [1, 256, 1500, 64],
                  [2, 2], 1, [1, 1, 1, 1], "custom", True)),
        ("enc-subsample-dw",
         conv_mil([1, 256, 1500, 64], [256, 1, 3, 3], [1, 256, 750, 32],
                  [2, 2], 256, [1, 1, 1, 1], "custom", True)),
        ("enc-subsample-dw-375",
         conv_mil([1, 256, 750, 32], [256, 1, 3, 3], [1, 256, 375, 16],
                  [2, 2], 256, [1, 1, 1, 1], "custom", True)),
        ("enc-padconv",
         conv_mil([1, 8, 375, 749], [8, 1, 1, 1], [1, 8, 375, 750],
                  [1, 1], 8, [0, 0, 1, 0], "custom", False)),
        ("k3-st2-square",
         conv_mil([1, 64, 16, 16], [64, 64, 3, 3], [1, 64, 8, 8],
                  [2, 2], 1, [0, 0, 0, 0], "same", False)),
    ):
        compile_source(root, name, mil, expected_code="h13.conv-outside-envelope")
print("h13 conv envelope cli: PASS")
