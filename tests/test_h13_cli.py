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
    assert json.loads(inspect(first))["manifest"] == manifest
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
        for row in range(512):
            struct.pack_into(f"<{reduction}H", weights, 128 + row * reduction * 2,
                             *((row * 17 + column * 31) & 0x3fff for column in range(reduction)))
        struct.pack_into('<H', weights, 128, 0x3C00)
        struct.pack_into("<H", weights, 128 + (7 * reduction + 13) * 2, 0x4200)
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
        assert json.loads(inspect(projection))["manifest"] == record
        unrelated = matmul.replace(
            f"val = tensor<fp16, [512, {reduction}]>",
            "val = fp16(1.0), debug_payload = tensor<fp16, [512, " + str(reduction) + "]>")
        compile_text(unrelated, f"wrong-constant-{reduction}", False)
        metadata = matmul.replace(
            "val = tensor<",
            'debug_payload = BLOBFILE(path = string("@model_path/not-a-weight.bin"), offset = uint64(64)), val = tensor<')
        unchanged = compile_text(metadata, f"metadata-{reduction}")
        assert (unchanged / "program-0.anec").read_bytes() == payload
        compile_text(matmul.replace("transpose_x = f", "transpose_x = t"),
                     f"wrong-input-transpose-{reduction}", False)
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
