#!/usr/bin/env python3
"""Inspect compiler ANEC packages and pack/unpack fp16 bytes without a device."""

import argparse
import json
import math
import struct
from pathlib import Path

HEADER = struct.Struct('<QIIQQII32I192Q')
HEADER_BYTES = 0x1000
TASK_BYTES = 0x274
CONSTANT_OFFSET = 0x280
TILE_BYTES = 0x4000


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_file(directory, name, limit):
    require(isinstance(name, str) and name not in ('', '.', '..') and
            Path(name).name == name, 'package filename must be a basename')
    path = (directory / name).resolve()
    require(path.parent == directory and path.is_file(),
            'package file must be a regular file within the package')
    require(path.stat().st_size <= limit, 'package file exceeds format limits')
    return path.read_bytes()


def check_binding(binding, index, shape, tiles, layouts):
    require(isinstance(binding, dict), 'binding must be an object')
    require(isinstance(binding.get('name'), str) and binding['name'],
            'binding needs a non-empty name')
    require(binding.get('dtype') == 'float16', 'binding must use float16')
    require(binding.get('shape') == shape and
            all(type(n) is int for n in binding['shape']), 'incorrect logical shape')
    require(type(binding.get('index')) is int and binding['index'] == index,
            'binding index does not match ANEC channel')
    channels = math.prod(shape)
    layout = [1, channels, 1, 1, 64, 64]
    require(binding.get('nchw') == layout and
            all(type(n) is int for n in binding['nchw']), 'incorrect physical layout')
    require(list(layouts[index * 6:(index + 1) * 6]) == layout,
            'manifest layout differs from ANEC header')
    require(type(binding.get('logicalBytes')) is int and
            binding['logicalBytes'] == channels * 2, 'incorrect logical byte count')
    allocation = TILE_BYTES if channels <= 256 else 2 * TILE_BYTES
    require(type(binding.get('allocationBytes')) is int and
            binding['allocationBytes'] == allocation, 'incorrect allocation size')
    require(tiles[index] * TILE_BYTES == allocation,
            'allocation differs from ANEC header')


def load_package(directory):
    directory = Path(directory).resolve()
    manifest = json.loads(local_file(directory, 'manifest.json', 65536))
    require(isinstance(manifest, dict), 'manifest must be an object')
    require(manifest.get('schema') == 'mil-hwxc.h13-anec-package.v1',
            'unsupported package schema')
    require(manifest.get('target') == 'H13' and manifest.get('artifactFormat') == 'anec',
            'package must target H13 ANEC')
    operation = manifest.get('operation')
    require(operation in ('add', 'mul', 'maximum', 'minimum', 'matmul'),
            'unsupported operation')
    inputs, outputs = manifest.get('inputs'), manifest.get('outputs')
    require(isinstance(inputs, list) and isinstance(outputs, list),
            'input and output bindings must be arrays')
    matmul = operation == 'matmul'
    require(len(inputs) == (1 if matmul else 2) and len(outputs) == 1,
            'incorrect input/output count')
    constants = 0x80000 if matmul else 0
    for key, expected in (('bytes', HEADER_BYTES + CONSTANT_OFFSET + constants),
                          ('constantBytes', constants), ('constantOffset', CONSTANT_OFFSET),
                          ('taskDescriptors', 1)):
        require(type(manifest.get(key)) is int and manifest[key] == expected,
                f'incorrect {key}')
    data = local_file(directory, manifest.get('file'), manifest['bytes'])
    require(len(data) == manifest['bytes'], 'ANEC file length differs from manifest')
    fields = HEADER.unpack_from(data)
    size, td_size, td_count, task_size, kernel_size, source_count, dest_count = fields[:7]
    require((size, td_size, td_count, task_size, kernel_size, source_count, dest_count) ==
            (CONSTANT_OFFSET + constants, TASK_BYTES, 1, TASK_BYTES, constants,
             len(inputs), len(outputs)), 'ANEC header differs from manifest')
    require(not any(data[HEADER.size:HEADER_BYTES]) and
            not any(data[HEADER_BYTES + TASK_BYTES:HEADER_BYTES + CONSTANT_OFFSET]),
            'nonzero reserved padding')
    tiles, layouts = fields[7:39], fields[39:]
    require(tiles[0] == (size + TILE_BYTES - 1) // TILE_BYTES,
            'incorrect command allocation')
    occupied = {4, *range(5, 5 + len(inputs))}
    for index in range(32):
        if index not in occupied:
            require(not any(layouts[index * 6:(index + 1) * 6]),
                    'unexpected tensor channel')
            if index != 0:
                require(tiles[index] == 0, 'unexpected channel allocation')
    if matmul:
        require(isinstance(inputs[0], dict) and inputs[0].get('shape') in
                ([1, 256], [1, 512]), 'unsupported matvec input shape')
        input_shape, output_shape = inputs[0]['shape'], [1, 512]
    else:
        input_shape = output_shape = [1, 64, 1, 1]
    for index, binding in enumerate(inputs, start=5):
        check_binding(binding, index, input_shape, tiles, layouts)
    check_binding(outputs[0], 4, output_shape, tiles, layouts)
    names = [binding['name'] for binding in inputs + outputs]
    require(len(set(names)) == len(names), 'binding names must be unique')
    return manifest


def convert_tensor(manifest, name, data, pack):
    bindings = manifest['inputs'] if pack else manifest['outputs']
    binding = next((item for item in bindings if item['name'] == name), None)
    if binding is None:
        raise ValueError("no such input" if pack else "no such output")
    logical, physical = binding['logicalBytes'], binding['allocationBytes']
    require(len(data) == (logical if pack else physical), 'incorrect tensor byte count')
    result = bytearray(physical if pack else logical)
    for channel in range(logical // 2):
        dense, padded = channel * 2, channel * 64
        if pack:
            result[padded:padded + 2] = data[dense:dense + 2]
        else:
            result[dense:dense + 2] = data[padded:padded + 2]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('package', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--pack-input', nargs=2, metavar=('NAME', 'RAW_FP16'))
    mode.add_argument('--unpack-output', nargs=2, metavar=('NAME', 'PHYSICAL_BUFFER'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    conversion = args.pack_input or args.unpack_output
    if bool(conversion) != bool(args.output):
        parser.error('a conversion and --output must be supplied together')
    try:
        manifest = load_package(args.package)
        if conversion:
            name, source = conversion
            data = convert_tensor(manifest, name, Path(source).read_bytes(),
                                  args.pack_input is not None)
            with args.output.open('xb') as destination:
                destination.write(data)
            print(f'wrote {len(data)} bytes to {args.output}; no device execution')
        else:
            print(json.dumps({'validation': 'container and binding consistency only',
                              'manifest': manifest}, indent=2, sort_keys=True))
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f'ANEC package error: {error}\n')


if __name__ == '__main__':
    main()
