#!/usr/bin/env python3
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

compiler = str(Path(sys.argv[1] if len(sys.argv) > 1 else 'build/mil-hwxc').resolve())


def source(op='add', channels=64, extra=''):
    shape = f'tensor<fp16, [1, {channels}, 1, 1]>'
    return f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>({shape} a, {shape} b) {{
    {shape} y = {op}(x = a, y = b)[name = string("binary")];
    {extra}
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
    multiply = compile_text(source('mul'), 'mul')
    assert (multiply / 'program-0.anec').read_bytes() != data
    compile_text(source().replace("program(1.3)", "@program(1.3)"),
                 "lexical-error", False, "mil.lex.")
    (root / "empty-output").mkdir()
    compile_text(source(), "empty-output")
    compile_text(source(channels=128), 'unsupported-shape', False)
    compile_text(source('sub'), 'unsupported-op', False)
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
        struct.pack_into('<H', weights, 128, 0x3C00)
        (root / 'weights.bin').write_bytes(weights)
        matmul = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {reduction}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [512, {reduction}]> W = const()[name = string("W"), val = tensor<fp16, [512, {reduction}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 512]> y = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("projection")];
  }} -> (y);
}}
'''
        projection = compile_text(matmul, f'projection-{reduction}')
        payload = (projection / 'program-0.anec').read_bytes()
        assert struct.unpack_from('<H', payload, 4096 + 0x280)[0] == 0x3C00
        record = json.loads((projection / 'manifest.json').read_text())
        assert record['inputs'][0]['shape'] == [1, reduction]
        assert record['outputs'][0]['shape'] == [1, 512]
        assert struct.unpack_from('<Q', payload, 24)[0] == record['constantBytes']
        assert record["constantBytes"] == 0x80000
        assert struct.unpack_from("<I", payload, 40)[0] == 33
        unrelated = matmul.replace(
            f"val = tensor<fp16, [512, {reduction}]>",
            "val = fp16(1.0), debug_payload = tensor<fp16, [512, " + str(reduction) + "]>")
        compile_text(unrelated, f"wrong-constant-{reduction}", False)
        metadata = matmul.replace(
            "val = tensor<",
            'debug_payload = BLOBFILE(path = string("@model_path/not-a-weight.bin"), offset = uint64(64)), val = tensor<')
        unchanged = compile_text(metadata, f"metadata-{reduction}")
        assert (unchanged / "program-0.anec").read_bytes() == payload
        compile_text(matmul.replace('transpose_y = t', 'transpose_y = f'),
                     f'wrong-transpose-{reduction}', False)
        weights[128:130] = struct.pack('<H', 0x4000)
        (root / 'weights.bin').write_bytes(weights)
        changed = compile_text(matmul, f'changed-weights-{reduction}')
        assert struct.unpack_from('<H', (changed / 'program-0.anec').read_bytes(),
                                  4096 + 0x280)[0] == 0x4000

print('H13 MIL-to-ANEC CLI: PASS (device-free)')
