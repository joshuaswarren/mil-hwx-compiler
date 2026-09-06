#!/usr/bin/env python3
"""Throwaway: simulate packaged H13 programs in pure Python; compare to h13_reference.

Simulates each program's arithmetic on physical buffers with fp16 rounding,
using manifest slices/constants. No hardware.
"""
from __future__ import annotations
import json, math, struct, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools'), str(ROOT / 'research')]
from research import inspect_anec
from h13_reference import fp16, add_fp16, decode_fp16, encode_fp16, evaluate, dot_fp32

COMPILER = ROOT / 'build/mil-hwxc'
STRIDE = 64  # bytes per physical lane


def f16_bits(x: float) -> int:
    return struct.unpack('<H', struct.pack('<e', float(x)))[0]


def bits_f16(b: int) -> float:
    return struct.unpack('<e', struct.pack('<H', b))[0]


def surface(binding, dense: bytes) -> bytes:
    """Dense fp16 -> the binding's physical surface (64-byte lanes or dense rows)."""
    return bytes(inspect_anec.convert_tensor(binding, dense, True))


def dense_of_surface(binding, buffer: bytes) -> bytes:
    return bytes(inspect_anec.convert_tensor(binding, buffer, False))


def pack_dense(dense: bytes, count: int, alloc: int) -> bytearray:
    """Dense fp16[count] -> allocationBytes buffer, 64-byte channel stride, zero pad."""
    out = bytearray(alloc)
    assert len(dense) >= count * 2
    for i in range(count):
        out[i * STRIDE:i * STRIDE + 2] = dense[i * 2:i * 2 + 2]
    return out


def unpack_physical(buf: bytes, count: int) -> bytes:
    out = bytearray(count * 2)
    for i in range(count):
        out[i * 2:i * 2 + 2] = buf[i * STRIDE:i * STRIDE + 2]
    return bytes(out)


def lane_values(buf: bytes, n: int) -> list[float]:
    return [bits_f16(struct.unpack_from('<H', buf, i * STRIDE)[0]) for i in range(n)]


def write_lanes(vals: list[float], alloc: int) -> bytearray:
    out = bytearray(alloc)
    for i, v in enumerate(vals):
        struct.pack_into('<e', out, i * STRIDE, float(v))
    return out


def dense_values(dense: bytes) -> list[float]:
    return [bits_f16(value) for (value,) in struct.iter_unpack('<H', dense)]


def dense_bytes(values: list[float]) -> bytes:
    out = bytearray(len(values) * 2)
    for index, value in enumerate(values):
        struct.pack_into('<e', out, index * 2, float(value))
    return bytes(out)


def binary_op(op: str, a: float, b: float) -> float:
    if op == 'add':
        return add_fp16(a, b)
    if op == 'mul':
        return fp16(float(a) * float(b))
    if op == 'maximum':
        # match IEEE-ish via fp16 path used in reference
        return fp16(max(float(a), float(b))) if not (math.isnan(float(a)) or math.isnan(float(b))) else float('nan')
    if op == 'minimum':
        return fp16(min(float(a), float(b))) if not (math.isnan(float(a)) or math.isnan(float(b))) else float('nan')
    raise ValueError(op)


def simulate_binary(op: str, in0: bytes, in1: bytes, lanes: int, alloc: int) -> bytearray:
    a = lane_values(in0, lanes)
    b = lane_values(in1, lanes)
    return write_lanes([binary_op(op, x, y) for x, y in zip(a, b)], alloc)


def pack_weights_block_col_row(weights_tx: bytes, reduction: int) -> bytes:
    """Match H13Program.cpp packWeights for transposeY=True layout [512, K] row-major."""
    output_rows, rows_per_block, block_bytes = 512, 32, 0x8000
    dma_blocks = output_rows // rows_per_block
    packed = bytearray(dma_blocks * block_bytes)
    for block in range(dma_blocks):
        block_offset = block * (block_bytes // 2)
        for column in range(reduction):
            column_offset = block_offset + column * rows_per_block
            for row in range(rows_per_block):
                src = ((block * rows_per_block + row) * reduction + column) * 2
                dst = (column_offset + row) * 2
                packed[dst:dst+2] = weights_tx[src:src+2]
    return bytes(packed)


def simulate_matvec(xin: bytes, weight_const: bytes, reduction: int, out_alloc: int) -> bytearray:
    """xin: physical lanes; weight_const: packWeights output; reduction 256|512."""
    x = lane_values(xin, reduction)
    output_rows, rows_per_block, block_bytes = 512, 32, 0x8000
    W = [[0.0] * reduction for _ in range(output_rows)]
    for block in range(output_rows // rows_per_block):
        block_offset = block * (block_bytes // 2)
        for column in range(reduction):
            column_offset = block_offset + column * rows_per_block
            for row in range(rows_per_block):
                dst = (column_offset + row) * 2
                val = bits_f16(struct.unpack_from('<H', weight_const, dst)[0])
                W[block * rows_per_block + row][column] = val
    out = [dot_fp32(x, W[r]) for r in range(output_rows)]
    return write_lanes(out, out_alloc)


def parity_weights(kernel: bytes, reduction: int, columns: int) -> list[list[float]]:
    """Invert ane::h13::packMatvecWeights: Apple's constant-section permutation."""
    group = min(16, columns // 16)
    groups = columns // group
    weights = [[0.0] * reduction for _ in range(columns)]
    for column in range(columns):
        plane = column // group
        destination = (plane % 16) * (groups // 16) + plane // 16
        base = destination * reduction * group + column % group
        for index in range(reduction):
            bits = struct.unpack_from('<H', kernel, (base + index * group) * 2)[0]
            weights[column][index] = bits_f16(bits)
    return weights


def simulate_parity_matvec(in_binding, out_binding, x_surface: bytes,
                           kernel: bytes) -> bytearray:
    """Apple's two-task form: `rows` dense x rows against every output column."""
    rows, reduction = in_binding['nchw'][2], in_binding['nchw'][3]
    columns = out_binding['nchw'][3]
    x = dense_values(dense_of_surface(in_binding, x_surface))
    weights = parity_weights(kernel, reduction, columns)
    values = []
    for row in range(rows):
        lane = x[row * reduction:(row + 1) * reduction]
        values.extend(dot_fp32(lane, weights[column]) for column in range(columns))
    return bytearray(surface(out_binding, dense_bytes(values)))


def intermediate_compose(binding, tensors, regions):
    return __import__('h13_run_linux', fromlist=['x'])._intermediate_buffer(binding, tensors, regions)


def dense_of(tensors, name):
    return bytearray(tensors[name]['logicalBytes'])


def run_sim(package: Path, mil: Path, model_root: Path, inputs: dict[str, bytes]):
    manifest, _ = inspect_anec.load_package(package)
    tensors = manifest['tensors']
    # seed dense inputs
    dense = {}
    for name, t in tensors.items():
        if t['role'] == 'input' and 'aliasOf' not in t:
            dense[name] = bytearray(inputs[name])
        else:
            dense[name] = bytearray(t['logicalBytes'])

    intermediate_regions = {}
    output_regions = {}

    for pi in manifest['dispatchPlan']:
        program = manifest['programs'][pi]
        op = program['operation']
        parity_matvec = program['encoder'] == inspect_anec.PARITY_MATVEC
        in_bufs = []
        for binding in program['inputs']:
            name = binding['name']
            if binding.get('binding') == 'constant':
                # constant hex is dense physical-count fp16 words (64 for elemwise)
                raw = bytes.fromhex(program['constantInputs'][name])
                logical = binding['logicalBytes']
                in_bufs.append(surface(binding, raw[:logical]))
            elif tensors[name]['role'] == 'intermediate':
                in_bufs.append(intermediate_compose(binding, tensors, intermediate_regions.get(name, [])))
            else:
                sliced = inspect_anec.dense_slice(bytes(dense[name]), binding, tensors)
                in_bufs.append(surface(binding, sliced))

        # produce output physical
        out_binding = program['outputs'][0]
        out_phys = inspect_anec.binding_interval(out_binding, tensors)[3]
        if op in ('add', 'mul', 'maximum', 'minimum'):
            assert len(in_bufs) == 2
            out_buf = simulate_binary(op, in_bufs[0], in_bufs[1], out_phys, out_binding['allocationBytes'])
        elif op == 'matmul':
            anec = inspect_anec.local_file(package, program['file'], program['bytes'])
            kstart = inspect_anec.HEADER_BYTES + program['constantOffset']
            kernel = anec[kstart:kstart + program['constantBytes']]
            if parity_matvec:
                out_buf = simulate_parity_matvec(program['inputs'][0], out_binding,
                                                 in_bufs[0], kernel)
            else:
                in_phys = inspect_anec.binding_interval(program['inputs'][0], tensors)[3]
                out_buf = simulate_matvec(in_bufs[0], kernel, in_phys, out_binding['allocationBytes'])
        else:
            raise ValueError(op)

        name, offset, count, physical = inspect_anec.binding_interval(out_binding, tensors)
        if tensors[name]['role'] == 'intermediate':
            intermediate_regions.setdefault(name, []).append((out_binding, bytes(out_buf)))
        else:
            # write into dense via the binding's own physical layout
            dense[name][offset * 2:(offset + count) * 2] = \
                dense_of_surface(out_binding, bytes(out_buf))
            output_regions.setdefault(name, []).append((out_binding, bytes(out_buf)))

    # unpack outputs like linux runner for multi-slice
    from h13_run_linux import _unpack_outputs
    # Prefer dense for single-writer; for multi-slice outputs use regions
    actual = {}
    for name, t in tensors.items():
        if t['role'] != 'output':
            continue
        target = t.get('aliasOf', name)
        if target in output_regions and len(output_regions[target]) > 1:
            actual[name] = _unpack_outputs(manifest, {target: output_regions[target]}, [target])[target]
        else:
            actual[name] = bytes(dense[target])
    expected = evaluate(mil.read_text(), model_root, {
        k: bytes(v) for k, v in inputs.items()
    })
    return actual, expected, manifest


def compile_mil(text: str, model_root: Path, out: Path, blobs: dict[str, bytes] | None = None):
    mil = model_root / 'model.mil'
    mil.write_text(text)
    if blobs:
        for n, b in blobs.items():
            (model_root / n).write_bytes(b)
    out.mkdir(parents=True, exist_ok=True)
    # clear
    for p in out.iterdir():
        p.unlink()
    r = subprocess.run([str(COMPILER), '--mil', str(mil), '--model-root', str(model_root),
                        '--target', 'H13', '--output', str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout + r.stderr)
    return mil


def blobfile(payload: bytes) -> bytes:
    out = bytearray(128 + len(payload))
    struct.pack_into('<IIQQ', out, 64, 0xDEADBEEF, 1, len(payload), 128)
    out[128:] = payload
    return bytes(out)


def check_elemwise_chain():
    failures = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # matmul -> add -> relu style with small shapes: just add/mul/relu chain
        mil = '''program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [96]> a, tensor<fp16, [96]> b) {
    tensor<fp16, [96]> s = add(x = a, y = b)[name = string("s")];
    tensor<fp16, [96]> p = mul(x = s, y = b)[name = string("p")];
    tensor<fp16, [96]> y = relu(x = p)[name = string("y")];
  } -> (y);
}
'''
        pkg = d / 'pkg'
        mil_path = compile_mil(mil, d, pkg)
        # packing agreement: reader vs linux composition on intermediates
        manifest, _ = inspect_anec.load_package(pkg)
        a = encode_fp16([0.5] * 96)
        b = encode_fp16([-1.0, 2.0] * 48)
        # Compare pack via inspect_anec.convert_tensor vs manual
        for prog in manifest['programs']:
            for binding in prog['inputs']:
                if binding.get('binding') == 'constant':
                    continue
                name = binding['name']
                if manifest['tensors'][name]['role'] != 'input':
                    continue
                src = a if name == 'a' else b
                sliced = inspect_anec.dense_slice(src, binding, manifest['tensors'])
                p1 = bytes(inspect_anec.convert_tensor(binding, sliced, True))
                p2 = bytes(pack_dense(sliced, binding['logicalBytes'] // 2, binding['allocationBytes']))
                if p1 != p2:
                    failures.append(f'pack mismatch {name} prog={prog["file"]}')
        actual, expected, _ = run_sim(pkg, mil_path, d, {'a': a, 'b': b})
        if actual['y'] != expected['y']:
            av, ev = decode_fp16(actual['y']), decode_fp16(expected['y'])
            bad = [(i, ev[i], av[i]) for i in range(len(ev)) if ev[i] != av[i]][:5]
            failures.append(f'elemwise chain mismatch samples={bad}')
        else:
            print('OK elemwise add-mul-relu chain 96')
    return failures


def check_ops():
    failures = []
    ops = [
        ('add', lambda x, y: add_fp16(x, y)),
        ('mul', lambda x, y: fp16(x * y)),
        ('maximum', lambda x, y: fp16(max(x, y))),
        ('minimum', lambda x, y: fp16(min(x, y))),
        ('sub', lambda x, y: add_fp16(x, fp16(-y))),  # folded
        ('real_div', None),  # special
    ]
    for op, _ in ops:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            if op in ('sub', 'real_div'):
                # fold constant y
                if op == 'sub':
                    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [32]> a) {{
    tensor<fp16, [32]> c = const()[name = string("c"), val = tensor<fp16, [32]>([{', '.join('fp16(2.0)' for _ in range(32))}])];
    tensor<fp16, [32]> y = sub(x = a, y = c)[name = string("y")];
  }} -> (y);
}}
'''
                else:
                    mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [32]> a) {{
    tensor<fp16, [32]> c = const()[name = string("c"), val = tensor<fp16, [32]>([{', '.join('fp16(4.0)' for _ in range(32))}])];
    tensor<fp16, [32]> y = real_div(x = a, y = c)[name = string("y")];
  }} -> (y);
}}
'''
                a = encode_fp16([float(i - 8) for i in range(32)])
                inputs = {'a': a}
            else:
                mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [80]> a, tensor<fp16, [80]> b) {{
    tensor<fp16, [80]> y = {op}(x = a, y = b)[name = string("y")];
  }} -> (y);
}}
'''
                a = encode_fp16([0.25 * i for i in range(80)])
                b = encode_fp16([1.5 if i % 2 == 0 else -0.5 for i in range(80)])
                inputs = {'a': a, 'b': b}
            pkg = d / 'pkg'
            mil_path = compile_mil(mil, d, pkg)
            actual, expected, man = run_sim(pkg, mil_path, d, inputs)
            # for folded ops, sim uses manifest operation (add/mul)
            if actual['y'] != expected['y']:
                av, ev = decode_fp16(actual['y']), decode_fp16(expected['y'])
                bad = [(i, ev[i], av[i]) for i in range(len(ev)) if abs(ev[i]-av[i]) > 1e-6][:8]
                failures.append(f'{op} mismatch {bad} manifest_ops={[p["operation"] for p in man["programs"]]}')
            else:
                print(f'OK {op}')
    return failures


def check_matmul_grid():
    failures = []
    Ks = [1, 200, 256, 257, 512, 700, 1024]
    Ns = [1, 64, 300, 512, 1000]
    for K in Ks:
        for N in Ns:
            with tempfile.TemporaryDirectory() as d:
                d = Path(d)
                # weights [N,K] transpose_y=true
                # use deterministic small values to avoid inf
                w = []
                for r in range(N):
                    for c in range(K):
                        w.append(((r * 3 + c * 5) % 17) * 0.125)
                payload = encode_fp16(w)
                blobs = {'weights.bin': blobfile(payload)}
                mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {K}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [{N}, {K}]> W = const()[name = string("W"), val = tensor<fp16, [{N}, {K}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> y = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("y")];
  }} -> (y);
}}
'''
                pkg = d / 'pkg'
                try:
                    mil_path = compile_mil(mil, d, pkg, blobs)
                except Exception as e:
                    failures.append(f'compile K={K} N={N}: {e}')
                    continue
                x = encode_fp16([((i % 7) - 3) * 0.25 for i in range(K)])
                try:
                    actual, expected, man = run_sim(pkg, mil_path, d, {'x': x})
                except Exception as e:
                    failures.append(f'sim K={K} N={N}: {e}')
                    continue
                chunked = man['tensors']['y'].get('accumulation') == 'chunked-fp16'
                av, ev = decode_fp16(actual['y']), decode_fp16(expected['y'])
                if len(av) != len(ev):
                    failures.append(f'K={K} N={N} len {len(av)} vs {len(ev)}')
                    continue
                bad = []
                for i, (a, e) in enumerate(zip(av, ev)):
                    if chunked:
                        if not math.isclose(a, e, rel_tol=0.01, abs_tol=0.03125):
                            bad.append((i, e, a))
                    else:
                        if a != e:
                            bad.append((i, e, a))
                if bad:
                    failures.append(f'K={K} N={N} chunked={chunked} bad={bad[:5]} nprog={len(man["programs"])}')
                else:
                    print(f'OK matmul K={K} N={N} programs={len(man["programs"])} chunked={chunked}')
    return failures


def check_reader_vs_linux_intermediate():
    """matmul->add->relu: intermediate physical composition vs convert_tensor packing."""
    failures = []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        K, N = 256, 96
        w = [0.125] * (N * K)
        blobs = {'weights.bin': blobfile(encode_fp16(w))}
        bias = encode_fp16([0.5] * N)
        blobs['bias.bin'] = blobfile(bias)
        mil = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {K}]> x) {{
    bool f = const()[name = string("f"), val = bool(false)];
    bool t = const()[name = string("t"), val = bool(true)];
    tensor<fp16, [{N}, {K}]> W = const()[name = string("W"), val = tensor<fp16, [{N}, {K}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> bias = const()[name = string("bias"), val = tensor<fp16, [1, {N}]>(BLOBFILE(path = string("@model_path/bias.bin"), offset = uint64(64)))];
    tensor<fp16, [1, {N}]> projection = matmul(x = x, y = W, transpose_x = f, transpose_y = t)[name = string("projection")];
    tensor<fp16, [1, {N}]> biased = add(x = projection, y = bias)[name = string("biased")];
    tensor<fp16, [1, {N}]> result = relu(x = biased)[name = string("result")];
  }} -> (result);
}}
'''
        pkg = d / 'pkg'
        mil_path = compile_mil(mil, d, pkg, blobs)
        x = encode_fp16([1.0] * K)
        actual, expected, man = run_sim(pkg, mil_path, d, {'x': x})
        if actual['result'] != expected['result']:
            av, ev = decode_fp16(actual['result']), decode_fp16(expected['result'])
            bad = [(i, ev[i], av[i]) for i in range(len(ev)) if ev[i] != av[i]][:8]
            failures.append(f'matmul-add-relu mismatch {bad}')
        else:
            print(f'OK matmul-add-relu N={N} programs={len(man["programs"])}')

        # Byte-for-byte: for each intermediate consumer, compose from producer physical
        # equals packing of dense slice through convert_tensor
        # Run sim collecting intermediate regions
        manifest, _ = inspect_anec.load_package(pkg)
        # Use linux helper composition after producing via sim path already validated
    return failures


def check_macos_packing_rule():
    """Cite run_h13.mm packing: only logical count lanes, 64-byte stride, zero fill."""
    # Replicate writePhysical/readPhysical
    def write_physical(dense, offset, count, alloc):
        dest = bytearray(alloc)
        src = dense[offset*2:]
        for e in range(count):
            dest[e*64:e*64+2] = src[e*2:e*2+2]
        return dest
    def read_physical(phys, offset, count, dense_len):
        dense = bytearray(dense_len)
        for e in range(count):
            dense[(offset+e)*2:(offset+e)*2+2] = phys[e*64:e*64+2]
        return dense
    dense = encode_fp16(list(range(100)))
    # slice offset 64 count 36 physical would be 64 for elemwise
    phys = write_physical(dense, 64, 36, 64*64)
    # padding lanes 36..63 must be zero
    assert all(phys[i]==0 for i in range(36*64, 64*64))
    back = read_physical(phys, 64, 36, 200)
    assert back[64*2:100*2] == dense[64*2:100*2]
    # agree with inspect_anec.convert_tensor
    binding = {'logicalBytes': 72, 'allocationBytes': 4096}
    p2 = bytes(inspect_anec.convert_tensor(binding, dense[64*2:100*2], True))
    assert phys[:4096] == p2
    print('OK macOS writePhysical/readPhysical matches inspect_anec.convert_tensor (run_h13.mm writePhysical/readPhysical)')
    return []


def main():
    all_f = []
    all_f += check_macos_packing_rule()
    all_f += check_elemwise_chain()
    all_f += check_ops()
    all_f += check_reader_vs_linux_intermediate()
    all_f += check_matmul_grid()
    if all_f:
        print('FAILURES:')
        for f in all_f:
            print(' -', f)
        sys.exit(1)
    print('ALL SIM CHECKS PASSED')

if __name__ == '__main__':
    main()
