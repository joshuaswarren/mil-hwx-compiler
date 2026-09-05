#!/usr/bin/env python3
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else 'build/mil-hwxc').resolve())
inspector = str(Path(__file__).resolve().parents[1] / "research" / "inspect_anec.py")


def inspect(package, *args, success=True):
    result = subprocess.run([sys.executable, inspector, str(package), *map(str, args)],
                            capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == (0 if success else 1), result.stderr
    return result.stdout


def tensor_type(shape):
    return f"tensor<fp16, [{', '.join(map(str, shape))}]>"


def source(op='add', shape=(1, 64, 1, 1), y_shape=None, output_shape=None, extra=''):
    x_type = tensor_type(shape)
    y_type = tensor_type(shape if y_shape is None else y_shape)
    output_type = tensor_type(shape if output_shape is None else output_shape)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_type} a, {y_type} b) {{
    {output_type} y = {op}(x = a, y = b)[name = string("binary")];
    {extra}
  }} -> (y);
}}
'''
def constant_source(op, value, const_first=False, blob=None):
    shape = (1, 64, 1, 1)
    value_type = tensor_type(shape)
    if blob is not None:
        constant_type = value_type
        literal = (f'{value_type}(BLOBFILE(path = string("@model_path/{blob}"), '
                   'offset = uint64(64)))')
    elif isinstance(value, (tuple, list)):
        constant_type = value_type
        literal = f'{value_type}([{", ".join(f"fp16({item})" for item in value)}])'
    else:
        constant_type = 'fp16'
        literal = f'fp16({value})'
    x, y = ('c', 'a') if const_first else ('a', 'c')
    return f'''program(1.3)
 [buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({value_type} a) {{
    {constant_type} c = const()[name = string("c"), val = {literal}];
    {value_type} y = {op}(x = {x}, y = {y})[name = string("binary")];
  }} -> (y);
}}
'''


def matmul_source(reduction, x_shape=None, output_shape=None, transpose_x=False,
                  transpose_y=True, weight_shape=None):
    x_shape = (1, reduction) if x_shape is None else x_shape
    output_shape = (1, 512) if output_shape is None else output_shape
    if weight_shape is None:
        weight_shape = (512, reduction) if transpose_y else (reduction, 512)
    tx = 't' if transpose_x else 'f'
    ty = 't' if transpose_y else 'f'
    x_type = tensor_type(x_shape)
    weight_type = tensor_type(weight_shape)
    output_type = tensor_type(output_shape)
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({x_type} x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    {weight_type} W = const()[name = string("W"), val = {weight_type}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    {output_type} y = matmul(x = x, y = W, transpose_x = {tx}, transpose_y = {ty})[name = string("projection")];
  }} -> (y);
}}
'''


with tempfile.TemporaryDirectory(prefix='mil-hwx-h13-test-') as directory:
    root = Path(directory)
    mil = root / 'model.mil'

    def compile_text(text, name, success=True, diagnostic="h13."):
        mil.write_text(text)
        out = root / name
        run = subprocess.run([compiler, '--mil', str(mil), '--model-root', str(root),
                              '--target', 'H13', '--output', str(out)],
                             capture_output=True, text=True, check=False, timeout=15)
        assert (run.returncode == 0) == success, run.stdout + run.stderr
        if not success:
            assert diagnostic in run.stderr, run.stderr
            assert not out.exists(), f'failed compilation wrote {out}'
        return out

    first = compile_text(source(), 'add')
    repeated = compile_text(source(), 'repeat')
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in repeated.iterdir()}
    manifest = json.loads((first / 'manifest.json').read_text())
    assert manifest['target'] == 'H13'
    assert manifest['artifactFormat'] == 'anec'
    assert manifest["schema"] == "mil-hwxc.h13-anec-package.v1"
    assert manifest['inputs'][0]['name'] == 'a'
    assert manifest['inputs'][1]['name'] == 'b'
    assert manifest['outputs'][0]['name'] == 'y'
    data = (first / 'program-0.anec').read_bytes()
    size, td_size, count, task_size, kernel_size, inputs, outputs = struct.unpack_from('<QIIQQII', data)
    assert len(data) == 4096 + size
    assert (td_size, count, task_size, inputs, outputs) == (0x274, 1, 0x274, 2, 1)
    inspection = json.loads(inspect(first))
    assert inspection["manifest"] == manifest
    assert inspection["bufferAllocation"]["totalBytes"] == 65536
    dense = bytes(range(128))
    raw, padded, unpacked = root / "input.fp16", root / "input.buffer", root / "output.fp16"
    raw.write_bytes(dense)
    inspect(first, "--pack-input", "a", raw, "--output", padded)
    expected = bytearray(16384)
    for channel in range(64):
        expected[channel * 64:channel * 64 + 2] = dense[channel * 2:channel * 2 + 2]
    assert padded.read_bytes() == expected
    inspect(first, "--unpack-output", "y", padded, "--output", unpacked)
    assert unpacked.read_bytes() == dense
    inspect(first, "--pack-input", "a", raw, "--output", padded, success=False)
    assert padded.read_bytes() == expected
    raw.write_bytes(dense[:-1])
    rejected = root / "rejected.buffer"
    inspect(first, "--pack-input", "a", raw, "--output", rejected, success=False)
    assert not rejected.exists()
    inspect(first, "--pack-input", "y", raw, "--output", rejected, success=False)
    manifest["inputs"][0]["index"] = 4
    (repeated / "manifest.json").write_text(json.dumps(manifest))
    inspect(repeated, success=False)
    for op in ('add', 'mul', 'maximum', 'minimum'):
        canonical = first if op == 'add' else compile_text(source(op), f'{op}-canonical')
        canonical_data = (canonical / 'program-0.anec').read_bytes()
        for shape in ((64,), (1, 64), (2, 4, 8)):
            logical = compile_text(source(op, shape), f'{op}-{len(shape)}d')
            assert (logical / 'program-0.anec').read_bytes() == canonical_data
            logical_manifest = json.loads((logical / 'manifest.json').read_text())
            assert logical_manifest['inputs'][0]['shape'] == list(shape)
            assert logical_manifest['inputs'][1]['shape'] == list(shape)
            assert logical_manifest['outputs'][0]['shape'] == list(shape)
    assert (root / 'mul-canonical' / 'program-0.anec').read_bytes() != data
    values = tuple(1.0 + (index % 8) * 0.125 for index in range(64))
    unchanged_constants = b''.join(struct.pack('<e', value) for value in values)
    for op in ('add', 'mul', 'maximum', 'minimum'):
        canonical = first if op == 'add' else root / f'{op}-canonical'
        for const_first in (False, True):
            folded = compile_text(constant_source(op, values, const_first),
                                  f'{op}-constant-{"first" if const_first else "second"}')
            folded_manifest = json.loads((folded / 'manifest.json').read_text())
            assert (folded / 'program-0.anec').read_bytes() == (canonical / 'program-0.anec').read_bytes()
            assert bytes.fromhex(folded_manifest['constantInputs']['c']) == unchanged_constants
            assert folded_manifest['inputs'][0]['name'] == 'a'
            assert folded_manifest['inputs'][1]['binding'] == 'constant'
    sub = compile_text(constant_source('sub', values), 'sub-constant')
    sub_manifest = json.loads((sub / 'manifest.json').read_text())
    negated = b''.join(struct.pack('<e', -value) for value in values)
    assert (sub / 'program-0.anec').read_bytes() == data
    assert sub_manifest['operation'] == 'add'
    assert sub_manifest['inputs'][1]['binding'] == 'constant'
    assert bytes.fromhex(sub_manifest['constantInputs']['c']) == negated

    packed_constant = root / 'constant.buffer'
    inspect(sub, '--pack-constant', 'c', '--output', packed_constant)
    expected_constant = bytearray(16384)
    for channel in range(64):
        expected_constant[channel * 64:channel * 64 + 2] = negated[channel * 2:channel * 2 + 2]
    assert packed_constant.read_bytes() == expected_constant

    quarters = (0.25,) * 64
    div = compile_text(constant_source('real_div', (4.0,) * 64), 'div-four')
    mul_quarter = compile_text(constant_source('mul', quarters), 'mul-quarter')
    assert (div / 'program-0.anec').read_bytes() == (mul_quarter / 'program-0.anec').read_bytes()
    div_manifest = json.loads((div / 'manifest.json').read_text())
    mul_manifest = json.loads((mul_quarter / 'manifest.json').read_text())
    assert div_manifest['operation'] == 'mul'
    assert div_manifest['constantInputs'] == mul_manifest['constantInputs']
    assert bytes.fromhex(div_manifest['constantInputs']['c']) == struct.pack('<e', 0.25) * 64
    compile_text(constant_source('real_div', (3.0,) * 64), 'div-three', False,
                 'h13.inexact-reciprocal')
    extremes = compile_text(constant_source('real_div', (32768.0, 2.0 ** -14) * 32), 'div-extremes')
    extremes_manifest = json.loads((extremes / 'manifest.json').read_text())
    assert bytes.fromhex(extremes_manifest['constantInputs']['c']) == \
        struct.pack('<ee', 2.0 ** -15, 16384.0) * 32
    compile_text(constant_source('real_div', (2.0 ** -15,) * 64), 'div-subnormal', False,
                 'h13.inexact-reciprocal')
    compile_text(constant_source('sub', values, const_first=True), 'constant-minus-input',
                 False, 'h13.nonfoldable-binary')
    compile_text(constant_source('real_div', values, const_first=True),
                 'constant-divided-by-input', False, 'h13.nonfoldable-binary')
    compile_text(constant_source('add', 1.0), 'scalar-add', False,
                 'h13.invalid-constant-input')

    scalar_mul = compile_text(constant_source('mul', -1.0), 'scalar-mul')
    tensor_mul = compile_text(constant_source('mul', (-1.0,) * 64), 'tensor-mul')
    assert (scalar_mul / 'program-0.anec').read_bytes() == (tensor_mul / 'program-0.anec').read_bytes()
    scalar_manifest = json.loads((scalar_mul / 'manifest.json').read_text())
    tensor_manifest = json.loads((tensor_mul / 'manifest.json').read_text())
    assert scalar_manifest['constantInputs'] == tensor_manifest['constantInputs']
    assert scalar_manifest['inputs'][1]['shape'] == [1, 64, 1, 1]

    blob = bytearray(128 + len(values) * 2)
    struct.pack_into('<IIQQ', blob, 64, 0xDEADBEEF, 1, len(values) * 2, 128)
    blob[128:] = b''.join(struct.pack('<e', value) for value in values)
    (root / 'constant.bin').write_bytes(blob)
    blob_sub = compile_text(constant_source('sub', None, blob='constant.bin'), 'blob-sub')
    blob_manifest = json.loads((blob_sub / 'manifest.json').read_text())
    assert (blob_sub / 'program-0.anec').read_bytes() == data
    assert bytes.fromhex(blob_manifest['constantInputs']['c']) == negated

    invalid_manifest = json.loads((blob_sub / 'manifest.json').read_text())
    struct.pack_into('<H', blob, 128, 0x7C00)
    (root / 'constant.bin').write_bytes(blob)
    compile_text(constant_source('sub', None, blob='constant.bin'),
                 'nonfinite-sub-constant', False, 'h13.nonfinite-constant')
    invalid_manifest['constantInputs']['c'] = invalid_manifest['constantInputs']['c'][:-2]
    invalid = compile_text(constant_source('sub', values), 'invalid-constant-manifest')
    (invalid / 'manifest.json').write_text(json.dumps(invalid_manifest))
    inspect(invalid, success=False)
    compile_text(source(shape=(0, 64)), 'zero-binary-shape', False)
    compile_text(source(shape=(1, 64), y_shape=(64,), output_shape=(1, 64)),
                 'broadcast-binary-shape', False)
    compile_text(source().replace("program(1.3)", "@program(1.3)"),
                 "lexical-error", False, "mil.lex.")
    (root / "empty-output").mkdir()
    compile_text(source(), "empty-output")
    compile_text(source(shape=(1, 128, 1, 1)), 'unsupported-shape', False)
    compile_text(source('sub'), 'two-input-sub', False, 'h13.nonfoldable-binary')
    compile_text(source('real_div'), 'two-input-real-div', False,
                 'h13.nonfoldable-binary')
    compile_text(source(extra='tensor<fp16, [1, 64, 1, 1]> z = relu(x = y)[name = string("z")];'),
                 'multiple-ops', False)
    compile_text(source().replace('x = a, y = b', 'x = a, y = a'), 'unused-input', False)
    preserved = {p.name: p.read_bytes() for p in first.iterdir()}
    mil.write_text(source())
    blocked = subprocess.run([compiler, "--mil", str(mil), "--model-root", str(root),
                              "--target", "H13", "--output", str(first)],
                             capture_output=True, text=True, timeout=15, check=False)
    assert blocked.returncode == 74, blocked.stderr
    assert preserved == {p.name: p.read_bytes() for p in first.iterdir()}

    for reduction in (256, 512):
        weights = bytearray(128 + 512 * reduction * 2)
        struct.pack_into('<IIQQ', weights, 64, 0xDEADBEEF, 1, 512 * reduction * 2, 128)
        for row in range(512):
            struct.pack_into(f"<{reduction}H", weights, 128 + row * reduction * 2,
                             *((row * 17 + column * 31) & 0x3fff for column in range(reduction)))
        struct.pack_into('<H', weights, 128, 0x3C00)
        struct.pack_into("<H", weights, 128 + (7 * reduction + 13) * 2, 0x4200)
        (root / 'weights.bin').write_bytes(weights)
        matmul = matmul_source(reduction)
        projection = compile_text(matmul, f'projection-{reduction}')
        payload = (projection / 'program-0.anec').read_bytes()
        assert struct.unpack_from('<H', payload, 4096 + 0x280)[0] == 0x3C00
        record = json.loads((projection / 'manifest.json').read_text())
        assert record['inputs'][0]['shape'] == [1, reduction]
        assert record['outputs'][0]['shape'] == [1, 512]
        assert struct.unpack_from('<Q', payload, 24)[0] == record['constantBytes']
        assert record["constantBytes"] == 0x80000
        assert struct.unpack_from("<I", payload, 40)[0] == 33
        inspection = json.loads(inspect(projection))
        assert inspection["manifest"] == record
        assert inspection["bufferAllocation"]["totalBytes"] == 573440 + reduction * 64
        unrelated = matmul.replace(
            f"val = tensor<fp16, [512, {reduction}]>",
            "val = fp16(1.0), debug_payload = tensor<fp16, [512, " + str(reduction) + "]>")
        compile_text(unrelated, f"wrong-constant-{reduction}", False)
        metadata = matmul.replace(
            "val = tensor<",
            'debug_payload = BLOBFILE(path = string("@model_path/not-a-weight.bin"), offset = uint64(64)), val = tensor<')
        unchanged = compile_text(metadata, f"metadata-{reduction}")
        assert (unchanged / "program-0.anec").read_bytes() == payload
        for case, x_shape, output_shape, transpose_x in (
                ('vector', (reduction,), (512,), False),
                ('singleton-batches', (1, 1, reduction), (1, 1, 512), False),
                ('transposed', (reduction, 1), (1, 512), True),
                ('transposed-batches', (1, 1, reduction, 1), (1, 1, 1, 512), True)):
            logical = compile_text(
                matmul_source(reduction, x_shape, output_shape, transpose_x),
                f'{case}-{reduction}')
            assert (logical / 'program-0.anec').read_bytes() == payload
            logical_manifest = json.loads((logical / 'manifest.json').read_text())
            assert logical_manifest['inputs'][0]['shape'] == list(x_shape)
            assert logical_manifest['outputs'][0]['shape'] == list(output_shape)
        compile_text(matmul_source(reduction, (reduction,), (512,), True),
                     f'rank-one-transpose-{reduction}', False)
        compile_text(matmul_source(reduction, (2, reduction), (2, 512)),
                     f'multirow-{reduction}', False)
        compile_text(matmul_source(reduction, (reduction, 2), (2, 512), True),
                     f'transposed-multirow-{reduction}', False)
        compile_text(matmul_source(reduction, (2, 1, reduction), (2, 1, 512)),
                     f'broadcast-batch-{reduction}', False)
        compile_text(matmul_source(reduction, (0, 1, reduction), (0, 1, 512)),
                     f'zero-batch-{reduction}', False)
        compile_text(matmul_source(reduction, weight_shape=(1, 512, reduction)),
                     f'rank-three-weight-{reduction}', False)
        normal_weights = bytearray(weights)
        for row in range(512):
            for column in range(reduction):
                old = 128 + (row * reduction + column) * 2
                new = 128 + (column * 512 + row) * 2
                normal_weights[new:new + 2] = weights[old:old + 2]
        (root / "weights.bin").write_bytes(normal_weights)
        normal = matmul.replace(f"[512, {reduction}]", f"[{reduction}, 512]")
        normal = normal.replace("transpose_y = t", "transpose_y = f")
        equivalent = compile_text(normal, f"normal-weights-{reduction}")
        assert (equivalent / "program-0.anec").read_bytes() == payload
        weights[128:130] = struct.pack('<H', 0x4000)
        (root / 'weights.bin').write_bytes(weights)
        changed = compile_text(matmul, f'changed-weights-{reduction}')
        assert struct.unpack_from('<H', (changed / 'program-0.anec').read_bytes(),
                                  4096 + 0x280)[0] == 0x4000

print('H13 MIL-to-ANEC CLI: PASS (device-free)')
