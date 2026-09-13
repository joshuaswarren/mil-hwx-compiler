#!/usr/bin/env python3
"""Channel-plane composite lowering: transpose→reshape→linear decomposition.

A [0,2,1,3] transpose of [1,C,P,D] feeding a reshape that merges the
transposed pair into one reduction, consumed by one linear, lowers as C
contiguous [1,P,D] channel-plane matvecs whose partials accumulate with adds
and the expanded bias — zero data movement. Every case is proven byte-equal
to the compilable per-plane direct form (nested axis-1 splits expose each
plane as the offset view the slice lowering already emits), recompile
deterministic, and inspector-validated. Near-miss composites reject with the
exact reason.
"""
import json
import struct
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
    return const(name, f"tensor<{dtype}, [{len(values)}]>([{', '.join(map(str, values))}])")


def fp16_const(name, shape, values):
    body = ", ".join(f"fp16({v})" for v in values)
    dims = ", ".join(map(str, shape))
    return const(name, f"tensor<fp16, [{dims}]>([{body}])")


def blob_const(name, shape, path):
    dims = ", ".join(map(str, shape))
    return const(name, f"tensor<fp16, [{dims}]>"
                        f"(BLOBFILE(path = string(\"{path}\"), offset = uint64(0)))")


def blobfile(values):
    payload = b"".join(struct.pack("<e", v) for v in values)
    return struct.pack("<IxxxxQQ", 0xDEADBEEF, len(payload), 24) + payload


def folded_source(channels, plane_rows, inner, columns, bias=True):
    """The encoder's o-projection form: transpose, merge-reshape, linear."""
    reduction = channels * inner
    body = tensor_const("perm", "int32", (0, 2, 1, 3))
    body += tensor_const("rs", "int32", (1, plane_rows, reduction))
    body += blob_const("w", [columns, reduction], "@model_path/weights/w.bin")
    if bias:
        body += fp16_const("bi", [columns], [0.5] * columns)
    body += (f"    tensor<fp16, [1, {plane_rows}, {channels}, {inner}]> t = "
             f"transpose(perm = perm, x = a)[name = string(\"t\")];\n")
    body += (f"    tensor<fp16, [1, {plane_rows}, {reduction}]> r = "
             f"reshape(shape = rs, x = t)[name = string(\"r\")];\n")
    bias_operand = "bias = bi, " if bias else ""
    body += (f"    tensor<fp16, [1, {plane_rows}, {columns}]> y = "
             f"linear({bias_operand}weight = w, x = r)"
             f"[name = string(\"y\")];\n")
    return source(body, header([1, channels, plane_rows, inner]) + " a", "y")




def direct_source(channels, plane_rows, inner, columns, bias=True):
    """The per-plane equivalent a user writes directly: nested axis-1 splits
    expose each channel plane as a contiguous offset view, one matmul per
    plane with the plane's weight columns, chunked accumulation, bias."""
    reduction = channels * inner
    body = const("tx", "bool(false)") + const("ty", "bool(true)")
    body += const("ax", "tensor<int32, []>(1)")
    body += const("ns", "tensor<int32, []>(2)")
    planes = []
    level = [f"a::{header([1, channels, plane_rows, inner])}"]
    split_index = 0
    extent = channels
    while extent > 1:
        nxt = []
        for item in level:
            name, _ = item.split("::")
            half = extent // 2
            plane_shape = [1, half, plane_rows, inner]
            nxt.append((f"s{split_index}a", f"s{split_index}b",
                        plane_shape, name))
            split_index += 1
        dims = ", ".join(map(str, nxt[0][2]))
        for result_a, result_b, plane_shape, operand in nxt:
            body += (f"    ({header(plane_shape)} {result_a}, "
                     f"{header(plane_shape)} {result_b}) = "
                     f"split(axis = ax, num_splits = ns, x = {operand})"
                     f"[name = string(\"sp\")];\n")
        level = []
        for result_a, result_b, _, _ in nxt:
            level.append(f"{result_a}::{dims}")
            level.append(f"{result_b}::{dims}")
        extent //= 2
        if extent == 1:
            planes = [item.split("::")[0] for item in level]
    for index, plane in enumerate(planes):
        body += tensor_const(f"rr{index}", "int32", (1, plane_rows, inner))
        body += (f"    tensor<fp16, [1, {plane_rows}, {inner}]> p{index} = "
                 f"reshape(shape = rr{index}, x = {plane})"
                 f"[name = string(\"p{index}\")];\n")
        body += blob_const(f"wc{index}", [columns, inner],
                           f"@model_path/weights/w{index}.bin")
        body += (f"    tensor<fp16, [1, {plane_rows}, {columns}]> m{index} = "
                 f"matmul(transpose_x = tx, transpose_y = ty, x = p{index}, "
                 f"y = wc{index})[name = string(\"m{index}\")];\n")
    accumulator = "m0"
    for index in range(1, len(planes)):
        name = "y" if index == len(planes) - 1 and not bias else f"acc{index}"
        body += (f"    tensor<fp16, [1, {plane_rows}, {columns}]> {name} = "
                 f"add(x = {accumulator}, y = m{index})"
                 f"[name = string(\"{name}\")];\n")
        accumulator = name
    if bias:
        body += fp16_const("bx", [1, plane_rows, columns],
                           [0.5] * (plane_rows * columns))
        body += (f"    tensor<fp16, [1, {plane_rows}, {columns}]> y = "
                 f"add(x = {accumulator}, y = bx)[name = string(\"y\")];\n")
    return source(body, header([1, channels, plane_rows, inner]) + " a", "y")


def compile_source(root, name, text, expected_code=None, expected_message=None,
                   format="anec"):
    path = root / f"{name}.mil"
    path.write_text(text)
    output = root / name
    result = subprocess.run(
        [compiler, "--mil", str(path), "--model-root", str(root),
         "--output", str(output), "--target", "H13", "--format", format],
        capture_output=True, text=True, timeout=300, check=False)
    if expected_code is None:
        assert result.returncode == 0, result.stderr
        return output
    assert result.returncode == 65, result.stderr
    assert expected_code in result.stderr, result.stderr
    if expected_message:
        assert expected_message in result.stderr, result.stderr
    return None


def deterministic(root, name, text, format="anec"):
    first = compile_source(root, f"{name}-a", text, format=format)
    second = compile_source(root, f"{name}-b", text, format=format)
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()}
    return first


def validate(root, package):
    result = subprocess.run([sys.executable, inspector, str(package)],
                            capture_output=True, text=True, timeout=60,
                            check=False)
    assert result.returncode == 0, result.stderr


def weight_files(root, channels, plane_rows, inner, columns):
    """The whole [N, K] weight and each plane's [N, D] column slice."""
    reduction = channels * inner
    weights = [float((index % 7) - 3) * 0.25
               for index in range(columns * reduction)]
    (root / "w.bin").write_bytes(blobfile(weights))
    for channel in range(channels):
        chunk = []
        for column in range(columns):
            chunk.extend(weights[column * reduction + channel * inner:
                                 column * reduction + (channel + 1) * inner])
        (root / f"w{channel}.bin").write_bytes(blobfile(chunk))


NAME_LITERALS = {"float16", "input", "output", "intermediate", "constant",
                 "add", "mul", "matmul", "linear", "maximum", "minimum"}

def structural_key(manifest):
    def erase_names(value):
        if isinstance(value, dict):
            return {key: erase_names(item) for key, item in sorted(value.items())
                    if key not in ("name", "tensor", "aliasOf")}
        if isinstance(value, list):
            return [erase_names(item) for item in value]
        if isinstance(value, str):
            return None if value not in NAME_LITERALS and \
                not value.startswith("program-") else value
        return value

    programs = []
    for record in manifest["programs"]:
        entry = erase_names(
            {key: item for key, item in sorted(record.items())
             if key not in ("constantInputs", "file")})
        entry["constantInputValues"] = sorted(
            record["constantInputs"].values())
        programs.append(entry)
    tensor_signatures = sorted(
        (value["shape"], value["logicalBytes"], value["role"])
        for key, value in manifest["tensors"].items()
        if key not in ("a", "y"))
    return {"programs": programs, "tensorSignatures": tensor_signatures,
            "intermediateCount": len(manifest["intermediates"])}

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    weights_root = root / "weights"
    weights_root.mkdir()

    # Byte-equality core: folded composite vs per-plane direct form.
    for channels, plane_rows, inner, columns, bias, tag in (
            (2, 4, 128, 512, True, "plane"),
            (4, 3, 16, 512, True, "subsampling"),
            (2, 2, 128, 512, False, "nobias")):
        weight_files(weights_root, channels, plane_rows, inner, columns)
        folded = deterministic(root, f"composite-fold-{tag}",
                               folded_source(channels, plane_rows, inner,
                                             columns, bias))
        direct = compile_source(root, f"composite-direct-{tag}",
                                direct_source(channels, plane_rows, inner,
                                              columns, bias))
        folded_programs = sorted(folded.glob("program-*.anec"),
                                 key=lambda p: int(p.stem.split("-")[1]))
        direct_programs = sorted(direct.glob("program-*.anec"),
                                 key=lambda p: int(p.stem.split("-")[1]))
        assert len(folded_programs) == len(direct_programs)
        for folded_program, direct_program in zip(folded_programs,
                                                  direct_programs):
            assert folded_program.read_bytes() == direct_program.read_bytes(), \
                f"{tag}: {folded_program.name} differs from the direct form"
        folded_manifest = json.loads((folded / "manifest.json").read_text())
        direct_manifest = json.loads((direct / "manifest.json").read_text())
        assert structural_key(folded_manifest) == structural_key(direct_manifest)
        assert folded_manifest["physicalOutputs"] == \
            direct_manifest["physicalOutputs"]
        assert folded_manifest["logicalResults"] == \
            direct_manifest["logicalResults"]
        validate(root, folded)
        validate(root, direct)

    # The encoder's head count: eight planes read the storage at the exact
    # channel-plane offsets, accumulate in seven chunked adds, and add the
    # expanded bias.
    channels, plane_rows, inner, columns = 8, 2, 128, 512
    weight_files(weights_root, channels, plane_rows, inner, columns)
    encoder = deterministic(root, "composite-encoder-heads",
                            folded_source(channels, plane_rows, inner, columns))
    manifest = json.loads((encoder / "manifest.json").read_text())
    matmuls = [record for record in manifest["programs"]
               if record["operation"] == "matmul"]
    adds = [record for record in manifest["programs"]
            if record["operation"] == "add"]
    assert len(matmuls) == channels * plane_rows
    # The 1024-element accumulation fits the decoded (1024,1,1) parity add,
    # so each accumulation step is one whole-tensor program; the constant
    # bias add cannot take that path and slices into 64-element programs.
    assert len(adds) == (channels - 1) + plane_rows * columns // 64
    whole_adds = [record for record in adds
                  if not record["inputs"][0].get("slice")]
    assert len(whole_adds) == channels - 1
    plane_offsets = sorted(
        record["inputs"][0]["slice"]["elementOffset"]
        for record in matmuls if record["inputs"][0].get("slice"))
    expected = sorted(channel * plane_rows * inner + row * inner
                      for channel in range(channels)
                      for row in range(plane_rows))
    assert plane_offsets == expected
    validate(root, encoder)

    # Near-miss composites reject with the exact reason.
    body = tensor_const("perm", "int32", (0, 2, 1, 3))
    body += tensor_const("rs", "int32", (1, 4, 256))
    body += ("    tensor<fp16, [1, 4, 2, 128]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<fp16, [1, 4, 256]> r = reshape(shape = rs, x = t)"
             "[name = string(\"r\")];\n")
    body += ("    tensor<fp16, [1, 4, 256]> y = relu(x = r)"
             "[name = string(\"y\")];\n")
    compile_source(root, "composite-relu-consumer", source(
        body, header([1, 2, 4, 128]) + " a", "y"),
        expected_code="h13.nonfoldable-transpose",
        expected_message="only when exactly one linear with a constant weight consumes")

    body = tensor_const("perm", "int32", (0, 2, 1, 3))
    body += tensor_const("rs", "int32", (1, 4, 1024))
    body += blob_const("w", [512, 1024], "@model_path/weights/w.bin")
    body += ("    tensor<fp16, [1, 4, 8, 128]> t = transpose(perm = perm, x = a)"
             "[name = string(\"t\")];\n")
    body += ("    tensor<fp16, [1, 4, 1024]> r = reshape(shape = rs, x = t)"
             "[name = string(\"r\")];\n")
    body += ("    tensor<fp16, [1, 4, 512]> y = linear(weight = w, x = r)"
             "[name = string(\"y\")];\n")
    two_consumers = source(
        body.replace("[name = string(\"y\")];",
                     "[name = string(\"y\")];\n"
                     "    tensor<fp16, [1, 4, 512]> z = linear(weight = w, x = r)"
                     "[name = string(\"z\")];"),
        header([1, 8, 4, 128]) + " a", "y, z")
    weight_files(weights_root, 8, 4, 128, 512)
    compile_source(root, "composite-two-linears", two_consumers,
                   expected_code="h13.nonfoldable-transpose",
                   expected_message="channel-plane decomposition only when exactly one linear")

    print("h13 composite cli: PASS")
