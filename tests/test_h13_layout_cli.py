#!/usr/bin/env python3
"""Stream-2 layout lowering: transpose, slice_by_index, pad, tile.

Covers the three host-side forms the H13 backend can express exactly —
free aliases for row-major-preserving views, offset views for contiguous
slices, and tail-swap transposes folded into a consuming matmul's
transpose flag — plus the exact rejection contracts for the encoder
forms that need a data-movement program the decoded corpus lacks.
The transpose-absorb extension adds the per-class encoder blocker
contracts: fast-axis swaps, multi-consumer views, shape-view composites,
middle-swap matmul feeds, and the fp16-only bool case.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else "build/mil-hwxc").resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")


def header(shape):
    return f"tensor<fp16, [{', '.join(map(str, shape))}]>"


def source(body, inputs, result):
    return f"""program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({inputs}) {{
{body}  }} -> ({result});
}}
"""


def const(name, literal):
    return f"    {literal.split('(')[0]} {name} = const()" \
           f"[name = string(\"{name}\"), val = {literal}];\n"



def tensor_const(name, dtype, values):
    shape = f"[{len(values)}]"
    return const(name, f"tensor<{dtype}, {shape}>([{', '.join(map(str, values))}])")

def bool_const(name, values):
    shape = f"[{len(values)}]"
    body = ", ".join("true" if v else "false" for v in values)
    return const(name, f"tensor<bool, {shape}>([{body}])")

def transpose_source(shape=(1, 8, 1, 16), perm=(0, 2, 1, 3),
                     consumer="relu"):
    normalized = tuple(p + len(shape) if p < 0 else p for p in perm)
    result_shape = tuple(shape[p] for p in normalized)
    body = tensor_const("perm", "int32", perm)
    body += f"    {header(result_shape)} t = transpose(perm = perm, x = a)[name = string(\"t\")];\n"
    body += f"    {header(result_shape)} y = {consumer}(x = t)[name = string(\"y\")];\n"
    return source(body, f"{header(shape)} a", "y")


def matmul_source(x_shape=(1, 64, 64), w_shape=(1, 64, 64),
                  result_shape=(1, 64, 64), transpose_x=False,
                  transpose_y=True):
    tx = const("tx", f"bool({'true' if transpose_x else 'false'})")
    ty = const("ty", f"bool({'true' if transpose_y else 'false'})")
    body = tx + ty
    body += (f"    {header(result_shape)} y = matmul("
             f"transpose_x = tx, transpose_y = ty, x = a, y = b)"
             f"[name = string(\"y\")];\n")
    return source(body, f"{header(x_shape)} a, {header(w_shape)} b", "y")


def fold_source(axis="y", x_shape=(1, 64, 64), w_shape=(1, 64, 64),
                result_shape=(1, 64, 64), transpose_x=False,
                transpose_y=False):
    """A tail-swap transpose feeding one matmul operand.

    matmul(x=t, flag=f) with t = tail-swap(v) is matmul(v, flag=!f), so the
    folded form must emit exactly the direct matmul written with the flipped
    flag.
    """
    shape = w_shape if axis == "y" else x_shape
    rank = len(shape)
    perm = tuple(list(range(rank - 2)) + [rank - 1, rank - 2])
    transposed = tuple(shape[p] for p in perm)
    body = tensor_const("perm", "int32", perm)
    body += f"    {header(transposed)} t = transpose(perm = perm, x = {'b' if axis == 'y' else 'a'})[name = string(\"t\")];\n"
    tx = const("tx", f"bool({'true' if transpose_x else 'false'})")
    ty = const("ty", f"bool({'true' if transpose_y else 'false'})")
    body += tx + ty
    if axis == "y":
        body += (f"    {header(result_shape)} y = matmul("
                 f"transpose_x = tx, transpose_y = ty, x = a, y = t)"
                 f"[name = string(\"y\")];\n")
    else:
        body += (f"    {header(result_shape)} y = matmul("
                 f"transpose_x = tx, transpose_y = ty, x = t, y = b)"
                 f"[name = string(\"y\")];\n")
    return source(body, f"{header(x_shape)} a, {header(w_shape)} b", "y")


def slice_source(shape=(1, 128, 1, 1), begin=(0, 64, 0, 0), end=(0, 128, 0, 0),
                 end_mask=None, begin_mask=None, stride=None,
                 result_shape=None, consumer="sigmoid"):
    rank = len(shape)
    if result_shape is None:
        sliced = None
        for index in range(rank):
            lower = begin[index]
            upper = shape[index] if (end_mask and end_mask[index]) else end[index]
            if begin_mask and begin_mask[index]:
                lower = 0
            if lower < 0:
                lower += shape[index]
            if upper < 0:
                upper += shape[index]
            if lower != 0 or upper != shape[index]:
                sliced = (index, upper - lower)
        if sliced is None:
            result_shape = shape
        else:
            index, count = sliced
            rest = list(shape)
            rest[index] = count
            result_shape = tuple(rest)
    body = tensor_const("begin", "int32", begin)
    body += tensor_const("end", "int32", end)
    if end_mask is not None:
        body += bool_const("emask", end_mask)
    if begin_mask is not None:
        body += bool_const("bmask", begin_mask)
    if stride is not None:
        body += tensor_const("stride", "int32", stride)
    arguments = "begin = begin, end = end"
    if end_mask is not None:
        arguments += ", end_mask = emask"
    if begin_mask is not None:
        arguments += ", begin_mask = bmask"
    if stride is not None:
        arguments += ", stride = stride"
    body += f"    {header(result_shape)} s = slice_by_index({arguments}, x = a)[name = string(\"s\")];\n"
    body += f"    {header(result_shape)} y = {consumer}(x = s)[name = string(\"y\")];\n"
    return source(body, f"{header(shape)} a", "y")


def pad_source(shape=(1, 64, 1, 1), amounts=(0, 0, 0, 0, 0, 0, 0, 0),
               result_shape=None, mode="constant"):
    rank = len(shape)
    if result_shape is None:
        result_shape = shape
    body = tensor_const("pad_amounts", "int32", amounts)
    body += f"    {header(result_shape)} p = pad(mode = string(\"{mode}\"), pad = pad_amounts, x = a)[name = string(\"p\")];\n"
    body += f"    {header(result_shape)} y = relu(x = p)[name = string(\"y\")];\n"
    return source(body, f"{header(shape)} a", "y")


def tile_source(shape=(1, 64, 1, 1), reps=(1, 1, 1, 1), result_shape=None):
    if result_shape is None:
        result_shape = shape
    body = tensor_const("reps", "int32", reps)
    body += f"    {header(result_shape)} t = tile(reps = reps, x = a)[name = string(\"t\")];\n"
    body += f"    {header(result_shape)} y = relu(x = t)[name = string(\"y\")];\n"
    return source(body, f"{header(shape)} a", "y")

def blob_const(name, dtype, shape):
    dims = ", ".join(map(str, shape))
    return const(name, f"tensor<{dtype}, [{dims}]>"
                          f"(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"),"
                          f" offset = uint64(0)))")


def encoder_conv_source():
    """The encoder's 24 depthwise-pointwise conv feeders: a rank-3 [0,2,1]
    transpose whose result the conv reads as x."""
    body = tensor_const("perm", "int32", (0, 2, 1))
    body += blob_const("w", "fp16", (2048, 1024, 1))
    body += tensor_const("strides", "int32", (1,))
    body += tensor_const("dilations", "int32", (1,))
    body += tensor_const("pad", "int32", (0, 0))
    body += const("pad_type", "string(\"valid\")")
    body += const("groups", "int32(1)")
    body += ("    tensor<fp16, [1, 1024, 375]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<fp16, [1, 2048, 375]> y = conv(dilations = dilations,"
             " groups = groups, pad = pad, pad_type = pad_type,"
             " strides = strides, weight = w, x = t)"
             "[name = string(\"y\")];\n")
    return source(body, "tensor<fp16, [1, 375, 1024]> a", "y")


def encoder_residual_add_source():
    """The encoder's 24 residual add feeders: a rank-3 [0,2,1] transpose read
    as add's y against a plain-layout x."""
    body = tensor_const("perm", "int32", (0, 2, 1))
    body += ("    tensor<fp16, [1, 375, 1024]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<fp16, [1, 375, 1024]> y = add(x = b, y = t)"
             "[name = string(\"y\")];\n")
    return source(body, "tensor<fp16, [1, 1024, 375]> a, tensor<fp16, [1, 375, 1024]> b", "y")


def encoder_bias_add_source():
    """The encoder's 24 attention bias adds: a rank-4 [0,2,1,3] transpose read
    twice as add's x against [1,8,1,128] constant biases."""
    body = tensor_const("perm", "int32", (0, 2, 1, 3))
    body += blob_const("q", "fp16", (1, 8, 1, 128))
    body += blob_const("v", "fp16", (1, 8, 1, 128))
    body += ("    tensor<fp16, [1, 8, 375, 128]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<fp16, [1, 8, 375, 128]> q1 = add(x = t, y = q)"
             "[name = string(\"q1\")];\n")
    body += ("    tensor<fp16, [1, 8, 375, 128]> y = add(x = t, y = v)"
             "[name = string(\"y\")];\n")
    return source(body, "tensor<fp16, [1, 375, 8, 128]> a", "q1, y")


def encoder_reshape_source(shape=(1, 8, 375, 128), perm=(0, 2, 1, 3),
                           reshaped=(1, 375, 1024)):
    """The encoder's 25 reshape feeders: transpose then a flat-order-preserving
    reshape whose consuming linear reads contiguous rows."""
    result_shape = tuple(shape[p + len(shape) if p < 0 else p] for p in perm)
    body = tensor_const("perm", "int32", perm)
    body += tensor_const("rs", "int32", reshaped)
    body += (f"    {header(result_shape)} t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += (f"    {header(reshaped)} r = reshape(shape = rs, x = t)"
             "[name = string(\"r\")];\n")
    body += (f"    {header(reshaped)} y = relu(x = r)"
             "[name = string(\"y\")];\n")
    return source(body, f"{header(shape)} a", "y")


def encoder_matmul_source():
    """The encoder's 48 attention matmul feeders: a rank-4 [0,2,-3,-1]
    middle-swap whose result feeds matmul's y operand."""
    body = tensor_const("perm", "int32", (0, 2, -3, -1))
    body += const("tx", "bool(false)")
    body += const("ty", "bool(false)")
    body += ("    tensor<fp16, [1, 8, 375, 128]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<fp16, [1, 8, 375, 749]> y = matmul("
             "transpose_x = tx, transpose_y = ty, x = b, y = t)"
             "[name = string(\"y\")];\n")
    return source(body,
                  "tensor<fp16, [1, 375, 8, 128]> a, tensor<fp16, [1, 8, 375, 749]> b",
                  "y")


def encoder_bool_source():
    """The encoder's single bool mask transpose feeding logical_and."""
    body = tensor_const("perm", "int32", (0, 2, 1))
    body += ("    tensor<bool, [1, 375, 375]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<bool, [1, 375, 375]> m = logical_and(x = b, y = t)"
             "[name = string(\"m\")];\n")
    body += ("    tensor<fp16, [1, 375, 375]> y = select(a = c, b = d, cond = m)"
             "[name = string(\"y\")];\n")
    return source(body,
                  "tensor<bool, [1, 375, 375]> a, tensor<bool, [1, 375, 375]> b,"
                  " tensor<fp16, [1, 375, 375]> c, tensor<fp16, [1, 375, 375]> d",
                  "y")




def compile_source(root, name, text, expected_code=None, format="anec",
                   expected_message=None):
    path = root / f"{name}.mil"
    path.write_text(text)
    output = root / name
    result = subprocess.run(
        [compiler, "--mil", str(path), "--model-root", str(root),
         "--output", str(output), "--target", "H13", "--format", format],
        capture_output=True, text=True, timeout=60, check=False)
    if expected_code is None:
        assert result.returncode == 0, result.stderr
        return output
    assert result.returncode == 65, result.stderr
    assert expected_code in result.stderr, result.stderr
    if expected_message:
        assert expected_message in result.stderr, result.stderr
    return None


def validate(root, package):
    result = subprocess.run([sys.executable, inspector, str(package)],
                            capture_output=True, text=True, timeout=30,
                            check=False)
    assert result.returncode == 0, result.stderr


def deterministic(root, name, text, format="anec"):
    first = compile_source(root, f"{name}-a", text, format=format)
    second = compile_source(root, f"{name}-b", text, format=format)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()}
    return first


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)

    # transpose: a row-major-preserving permutation is a free alias.
    unit_move = deterministic(root, "transpose-unit-move",
                              transpose_source())
    manifest = json.loads((unit_move / "manifest.json").read_text())
    assert manifest["tensors"]["t"] == {
        "shape": [1, 1, 8, 16], "logicalBytes": 256,
        "role": "intermediate", "aliasOf": "a"}
    assert manifest["tensors"]["a"]["shape"] == [1, 8, 1, 16]
    assert [program["operation"] for program in manifest["programs"]] == [
        "maximum", "maximum"]
    validate(root, unit_move)

    identity_perm = compile_source(root, "transpose-identity",
                                   transpose_source(perm=(0, 1, 2, 3)))
    identity_manifest = json.loads((identity_perm / "manifest.json").read_text())
    assert identity_manifest["tensors"]["t"]["aliasOf"] == "a"

    # transpose: a tail-swap feeding the y operand folds into transpose_y,
    # byte-exact against the equivalent matmul written directly.
    folded = deterministic(root, "transpose-fold-y", fold_source(axis="y"))
    direct = compile_source(root, "transpose-fold-y-reference",
                            matmul_source(transpose_x=False,
                                          transpose_y=True))
    assert (folded / "program-0.anec").read_bytes() == \
        (direct / "program-0.anec").read_bytes()
    assert json.loads((folded / "manifest.json").read_text()) == \
        json.loads((direct / "manifest.json").read_text())
    validate(root, folded)

    # The same fold on the x operand hands the flipped flag to the verified
    # matmul envelope. No mirrored tx oracle exists in the decoded corpus —
    # every covered tx=true geometry pairs with an uncovered tx=false mirror —
    # so the hand-off is proven by the matmul envelope's own rejection rather
    # than by byte parity.
    compile_source(root, "transpose-fold-x",
                   fold_source(axis="x", x_shape=(64, 128),
                               w_shape=(1, 64, 256),
                               result_shape=(128, 256), transpose_x=False,
                               transpose_y=False),
                   expected_code="h13.matmul-outside-envelope")
    hwx_fold = deterministic(root, "transpose-fold-y-hwx",
                             fold_source(axis="y"), format="hwx")
    hwx_result = subprocess.run(
        [sys.executable,
         str(Path(__file__).resolve().parents[1] / "research" / "inspect_hwx.py"),
         str(hwx_fold / "program-0.hwx")],
        capture_output=True, text=True, timeout=30, check=False)
    assert hwx_result.returncode == 0, hwx_result.stderr
    assert "name=H13" in hwx_result.stdout

    # transpose: the encoder's attention permutation moves a non-unit leading
    # dimension, and no decoded encoder permutes surface strides.
    compile_source(root, "transpose-encoder-perm",
                   transpose_source(shape=(1, 375, 8, 128),
                                    perm=(0, 2, -3, -1)),
                   expected_code="h13.nonfoldable-transpose")
    # A tail-swap feeding anything but a matmul x/y operand cannot fold.
    compile_source(root, "transpose-tail-consumer",
                   transpose_source(shape=(1, 64, 64), perm=(0, 2, 1)),
                   expected_code="h13.nonfoldable-transpose")
    compile_source(root, "transpose-bad-perm",
                   transpose_source(perm=(0, 2, 2, 3)),
                   expected_code="h13.invalid-transpose-parameters")

    # Encoder blocker classification, transpose-absorb stream: every one of
    # the 98 add/conv/reshape-feeding encoder transposes rejects with the
    # exact reason that stops it, and none has a compilable direct path a
    # fold could be byte-compared against.
    #
    # 24 conv feeders: the rank-3 [0,2,1] permutation moves the fastest
    # storage axis, so the conv's row-contiguous read is inexpressible as a
    # surface interpretation at any size.
    compile_source(root, "transpose-encoder-conv",
                   encoder_conv_source(),
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="moves the storage-fastest axis")
    # 24 residual add feeders: same fast-axis class read as add's y operand.
    compile_source(root, "transpose-encoder-residual-add",
                   encoder_residual_add_source(),
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="moves the storage-fastest axis")
    # 24 attention bias adds: the rank-4 [0,2,1,3] permutation keeps the
    # fastest axis but has two consumers, so nothing absorbs it.
    compile_source(root, "transpose-encoder-bias-add",
                   encoder_bias_add_source(),
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="several consumers")
    # 25 reshape feeders: in the encoder these feed one linear and now fold
    # through the channel-plane decomposition (composite suite); a non-linear
    # consumer is a near-miss composite and rejects with that exact reason.
    compile_source(root, "transpose-encoder-reshape",
                   encoder_reshape_source(),
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="only when exactly one linear with a constant weight consumes")
    compile_source(root, "transpose-encoder-reshape-subsampling",
                   encoder_reshape_source(shape=(1, 256, 375, 16),
                                          reshaped=(1, 375, 4096)),
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="only when exactly one linear with a constant weight consumes")
    # 48 matmul feeders: a fast-axis-stable middle swap feeding matmul y;
    # the trailing-swap fold does not apply and no decoded encoder reads a
    # matmul operand with permuted leading strides.
    compile_source(root, "transpose-encoder-matmul-y",
                   encoder_matmul_source(),
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="no decoded encoder permutes surface strides")
    # The single bool mask transpose: the H13 path is fp16-only.
    compile_source(root, "transpose-encoder-bool",
                   encoder_bool_source(),
                   expected_code="h13.invalid-transpose-parameters")

    # Positive control for the composite rule: a layout-preserving
    # permutation feeding a reshape stays the free alias the transpose
    # lowering already emits.
    composite = deterministic(
        root, "transpose-reshape-pure-view",
        encoder_reshape_source(shape=(1, 8, 1, 16), perm=(0, 2, 1, 3),
                               reshaped=(1, 8, 16)))
    composite_manifest = json.loads(
        (composite / "manifest.json").read_text())
    assert composite_manifest["tensors"]["r"]["aliasOf"] == "a"
    validate(root, composite)

    # slice_by_index: one contiguous view with the consumer reading at the
    # slice offset — the same binding ABI the split lowering established.
    sliced = deterministic(root, "slice-offset",
                           slice_source(end_mask=(True, False, True, True)))
    manifest = json.loads((sliced / "manifest.json").read_text())
    consumer_input = manifest["programs"][0]["inputs"][0]
    assert consumer_input["name"] == "a"
    assert consumer_input["slice"] == {
        "tensor": "a", "elementOffset": 64, "elementCount": 64,
        "physicalElements": 64}
    dense = list(range(128))
    offset = consumer_input["slice"]["elementOffset"]
    count = consumer_input["slice"]["elementCount"]
    assert dense[offset:offset + count] == dense[64:128]
    validate(root, sliced)

    # Negative end indices normalize to extent-relative bounds.
    negative = compile_source(root, "slice-negative-end",
                              slice_source(begin=(0, -64, 0, 0),
                                           end=(0, 128, 0, 0),
                                           end_mask=(True, False, True, True)))
    negative_manifest = json.loads((negative / "manifest.json").read_text())
    assert negative_manifest["programs"][0]["inputs"][0]["slice"][
        "elementOffset"] == 64
    assert (negative / "program-0.anec").read_bytes() == \
        (sliced / "program-0.anec").read_bytes()

    # begin_mask pins a dimension's lower bound to zero.
    pinned = compile_source(root, "slice-begin-mask",
                            slice_source(begin=(0, 128, 0, 0),
                                         end_mask=(True, True, True, True),
                                         begin_mask=(True, True, True, True),
                                         consumer="relu"))
    pinned_manifest = json.loads((pinned / "manifest.json").read_text())
    assert pinned_manifest["programs"][0]["inputs"][0]["name"] == "a"
    assert pinned_manifest["tensors"]["s"]["aliasOf"] == "a"
    validate(root, pinned)

    # The encoder's mask slicing interleaves chunks across a non-unit head
    # dimension; one binding slice cannot represent that.
    compile_source(root, "slice-encoder-mask",
                   slice_source(shape=(1, 8, 750, 375),
                                begin=(0, 0, 1, 0),
                                end=(1, 8, 750, 375),
                                end_mask=(True, True, True, True),
                                result_shape=(1, 8, 749, 375),
                                consumer="relu"),
                   expected_code="h13.noncontiguous-slice",
                   expected_message="8 chunks of 280875 elements spaced 281250 apart")
    compile_source(root, "slice-encoder-bias",
                   slice_source(shape=(1, 8, 375, 749),
                                begin=(0, 0, 0, 0),
                                end=(1, 8, 375, 375),
                                end_mask=(True, True, True, False),
                                result_shape=(1, 8, 375, 375),
                                consumer="relu"),
                   expected_code="h13.noncontiguous-slice",
                   expected_message="3000 chunks of 375 elements spaced 749 apart")
    compile_source(root, "slice-strided",
                   slice_source(stride=(1, 1, 1, 2)),
                   expected_code="h13.noncontiguous-slice")
    compile_source(root, "slice-two-axes",
                   slice_source(shape=(1, 128, 1, 2), begin=(0, 64, 0, 1),
                                end=(0, 128, 0, 2),
                                end_mask=(True, False, True, True),
                                result_shape=(1, 64, 1, 1)),
                   expected_code="h13.noncontiguous-slice")

    # pad: an all-zero pad is the identity view.
    padded = deterministic(root, "pad-zero", pad_source())
    manifest = json.loads((padded / "manifest.json").read_text())
    assert manifest["tensors"]["p"]["aliasOf"] == "a"
    validate(root, padded)
    # The encoder pads attention scores by one row; that needs movement.
    compile_source(root, "pad-encoder-scores",
                   pad_source(shape=(1, 1, 749, 375),
                              amounts=(0, 0, 0, 0, 0, 0, 1, 0),
                              result_shape=(1, 1, 750, 375)),
                   expected_code="h13.unsupported-pad",
                   expected_message="pad [0,0,0,0,0,0,1,0]")

    # tile: all-ones repeats are the identity view.
    tiled = deterministic(root, "tile-ones", tile_source())
    manifest = json.loads((tiled / "manifest.json").read_text())
    assert manifest["tensors"]["t"]["aliasOf"] == "a"
    validate(root, tiled)
    compile_source(root, "tile-encoder-mask",
                   tile_source(shape=(1, 1, 375), reps=(1, 375, 1),
                               result_shape=(1, 375, 375)),
                   expected_code="h13.unsupported-tile",
                   expected_message="reps [1,375,1]")

    print("h13 layout cli: PASS")
