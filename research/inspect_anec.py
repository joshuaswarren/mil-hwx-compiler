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
PROGRAM_FIELDS = (
    'file', 'bytes', 'taskDescriptors', 'operation', 'inputs', 'constantInputs',
    'outputs', 'constantOffset', 'constantBytes')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def local_file(directory, name, limit):
    require(isinstance(name, str) and name not in ('', '.', '..') and
            Path(name).name == name, 'package filename must be a basename')
    require(type(limit) is int and 0 <= limit, 'package file limit must be nonnegative')
    path = (directory / name).resolve()
    require(path.parent == directory and path.is_file(),
            'package file must be a regular file within the package')
    require(path.stat().st_size <= limit, 'package file exceeds format limits')
    return path.read_bytes()


def check_binding(binding, index, channels, tiles, layouts):
    require(isinstance(binding, dict), 'binding must be an object')
    require(isinstance(binding.get('name'), str) and binding['name'],
            'binding needs a non-empty name')
    require(binding.get('dtype') == 'float16', 'binding must use float16')
    shape = binding.get('shape')
    require(isinstance(shape, list) and shape and
            all(type(n) is int and 0 < n <= channels for n in shape),
            'logical shape must have positive static dimensions')
    require(math.prod(shape) == channels, 'incorrect logical element count')
    require(type(binding.get('index')) is int and binding['index'] == index,
            'binding index does not match ANEC channel')
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
    require(binding.get('role') in (None, 'intermediate'),
            'unsupported binding role')


def validate_program(directory, program):
    require(isinstance(program, dict), 'program must be an object')
    operation = program.get('operation')
    require(operation in ('add', 'mul', 'maximum', 'minimum', 'matmul'),
            'unsupported operation')
    inputs, outputs = program.get('inputs'), program.get('outputs')
    require(isinstance(inputs, list) and isinstance(outputs, list),
            'input and output bindings must be arrays')
    constant_inputs = program.get('constantInputs')
    require(isinstance(constant_inputs, dict), 'constantInputs must be an object')
    matmul = operation == 'matmul'
    require(len(inputs) == (1 if matmul else 2) and len(outputs) == 1,
            'incorrect input/output count')
    constants = 0x80000 if matmul else 0
    for key, expected in (('bytes', HEADER_BYTES + CONSTANT_OFFSET + constants),
                          ('constantBytes', constants), ('constantOffset', CONSTANT_OFFSET),
                          ('taskDescriptors', 1)):
        require(type(program.get(key)) is int and program[key] == expected,
                f'incorrect {key}')
    data = local_file(directory, program.get('file'), program['bytes'])
    require(len(data) == program['bytes'], 'ANEC file length differs from manifest')
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
        require(isinstance(inputs[0], dict) and
                inputs[0].get('logicalBytes') in (512, 1024),
                'unsupported matvec input size')
        input_channels, output_channels = inputs[0]['logicalBytes'] // 2, 512
    else:
        input_channels = output_channels = 64
    for index, item in enumerate(inputs, start=5):
        check_binding(item, index, input_channels, tiles, layouts)
    check_binding(outputs[0], 4, output_channels, tiles, layouts)
    if not matmul:
        require(inputs[0]['shape'] == inputs[1]['shape'] == outputs[0]['shape'],
                'binary operation shapes must match')
    names = [item['name'] for item in inputs + outputs]
    require(len(set(names)) == len(names), 'program binding names must be unique')
    marked_constants = {item['name'] for item in inputs
                        if item.get('binding') == 'constant'}
    require(all(item.get('binding') in (None, 'constant') for item in inputs),
            'unsupported input binding kind')
    require(not any(item.get('binding') for item in outputs),
            'output bindings cannot be constants')
    require(set(constant_inputs) == marked_constants,
            'constantInputs must match constant bindings')
    for item in inputs:
        if item['name'] not in marked_constants:
            continue
        encoded = constant_inputs[item['name']]
        require(isinstance(encoded, str) and len(encoded) == item['logicalBytes'] * 2 and
                all(character in '0123456789abcdefABCDEF' for character in encoded),
                'constant input must be dense hexadecimal fp16 bytes')
    command_bytes = ((program['bytes'] - HEADER_BYTES + TILE_BYTES - 1)
                     // TILE_BYTES) * TILE_BYTES
    input_bytes = sum(item['allocationBytes'] for item in inputs)
    output_bytes = sum(item['allocationBytes'] for item in outputs)
    return {
        'file': program['file'], 'commandAndConstantsBytes': command_bytes,
        'inputBytes': input_bytes, 'outputBytes': output_bytes,
        'totalBytes': command_bytes + input_bytes + output_bytes,
    }


def binding_layout(item):
    return tuple(item[key] if not isinstance(item[key], list) else tuple(item[key])
                 for key in ('dtype', 'shape', 'logicalBytes', 'nchw', 'allocationBytes'))


def load_package(directory):
    directory = Path(directory).resolve()
    manifest = json.loads(local_file(directory, 'manifest.json', 65536))
    require(isinstance(manifest, dict), 'manifest must be an object')
    require(manifest.get('schema') == 'mil-hwxc.h13-anec-package.v1',
            'unsupported package schema')
    require(manifest.get('target') == 'H13' and manifest.get('artifactFormat') == 'anec',
            'package must target H13 ANEC')
    programs = manifest.get('programs')
    dispatch = manifest.get('dispatchPlan')
    intermediates = manifest.get('intermediates')
    require(isinstance(programs, list) and programs, 'programs must be a non-empty array')
    require(isinstance(dispatch, list) and
            all(type(index) is int for index in dispatch) and
            sorted(dispatch) == list(range(len(programs))),
            'dispatchPlan must contain every program index exactly once')
    require(isinstance(intermediates, list) and
            all(isinstance(name, str) and name for name in intermediates) and
            len(set(intermediates)) == len(intermediates),
            'intermediates must contain unique non-empty names')
    files = [program.get('file') for program in programs if isinstance(program, dict)]
    require(len(files) == len(programs) and len(set(files)) == len(files),
            'program files must be unique')
    if len(programs) == 1:
        for key in PROGRAM_FIELDS:
            require(manifest.get(key) == programs[0].get(key),
                    f'legacy top-level {key} must match the sole program')
    else:
        require(not any(key in manifest for key in PROGRAM_FIELDS),
                'multi-program packages must omit legacy top-level program fields')

    allocations = [validate_program(directory, program) for program in programs]
    intermediate_set = set(intermediates)
    producers = {}
    consumers = {}
    for program_index, program in enumerate(programs):
        for item in program['outputs']:
            name = item['name']
            if name in intermediate_set:
                require(item.get('role') == 'intermediate',
                        'intermediate output must have intermediate role')
                require(name not in producers, 'intermediate must be produced once')
                producers[name] = (program_index, item)
            else:
                require(item.get('role') is None,
                        'non-intermediate output cannot have intermediate role')
        for item in program['inputs']:
            name = item['name']
            if name in intermediate_set:
                require(item.get('role') == 'intermediate' and
                        item.get('binding') is None,
                        'intermediate input must have intermediate role')
                require(name not in consumers, 'intermediate must be consumed once')
                consumers[name] = (program_index, item)
            else:
                require(item.get('role') is None,
                        'non-intermediate input cannot have intermediate role')
    require(set(producers) == intermediate_set and set(consumers) == intermediate_set,
            'every intermediate must be produced and consumed exactly once')
    require(len(intermediates) == len(programs) - 1,
            'a straight-line package needs one intermediate per program edge')
    positions = {program_index: position for position, program_index in enumerate(dispatch)}
    edges = set()
    for name in intermediates:
        producer_index, produced = producers[name]
        consumer_index, consumed = consumers[name]
        require(binding_layout(produced) == binding_layout(consumed),
                'intermediate physical layouts must match')
        require(positions[producer_index] < positions[consumer_index],
                'dispatchPlan violates an intermediate dependency')
        edges.add((producer_index, consumer_index))
    require(edges == set(zip(dispatch, dispatch[1:])),  # noqa: RUF007 - macOS ships Python 3.9
            'intermediates must connect adjacent dispatched programs')
    return manifest, allocations


def find_bindings(manifest, name, direction, constants=False):
    matches = []
    for program in manifest['programs']:
        for item in program[direction]:
            if item['name'] != name:
                continue
            if constants != (item.get('binding') == 'constant'):
                continue
            matches.append((program, item))
    require(matches, 'no such constant input' if constants else
            ('no such input' if direction == 'inputs' else 'no such output'))
    expected = binding_layout(matches[0][1])
    require(all(binding_layout(item) == expected for _, item in matches),
            'binding name has inconsistent layouts across programs')
    return matches


def convert_tensor(binding, data, pack):
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
    mode.add_argument('--pack-constant', metavar='NAME')
    mode.add_argument('--unpack-output', nargs=2, metavar=('NAME', 'PHYSICAL_BUFFER'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    conversion = args.pack_input or args.pack_constant or args.unpack_output
    if bool(conversion) != bool(args.output):
        parser.error('a conversion and --output must be supplied together')
    try:
        manifest, allocations = load_package(args.package)
        if conversion:
            if args.pack_constant:
                name = args.pack_constant
                matches = find_bindings(manifest, name, 'inputs', constants=True)
                encoded = {program['constantInputs'][name] for program, _ in matches}
                require(len(encoded) == 1,
                        'constant name has inconsistent payloads across programs')
                source = bytes.fromhex(encoded.pop())
                data = convert_tensor(matches[0][1], source, True)
            else:
                name, source = conversion
                direction = 'inputs' if args.pack_input else 'outputs'
                matches = find_bindings(manifest, name, direction)
                data = convert_tensor(matches[0][1], Path(source).read_bytes(),
                                      args.pack_input is not None)
            with args.output.open('xb') as destination:
                destination.write(data)
            print(f'wrote {len(data)} bytes to {args.output}; no device execution')
        else:
            command_bytes = sum(item['commandAndConstantsBytes'] for item in allocations)
            input_bindings = {}
            output_bindings = {}
            for program in manifest['programs']:
                for item in program['inputs']:
                    if item['name'] not in set(manifest['intermediates']):
                        input_bindings.setdefault(item['name'], item)
                for item in program['outputs']:
                    output_bindings.setdefault(item['name'], item)
            input_bytes = sum(item['allocationBytes'] for item in input_bindings.values())
            output_bytes = sum(item['allocationBytes'] for item in output_bindings.values())
            print(json.dumps({
                'validation': 'container, binding, and dispatch consistency only',
                'manifest': manifest,
                'bufferAllocation': {
                    'programs': allocations,
                    'commandAndConstantsBytes': command_bytes,
                    'inputBytes': input_bytes, 'outputBytes': output_bytes,
                    'totalBytes': command_bytes + input_bytes + output_bytes,
                    'scope': 'encoded buffers only; shared names counted once; excludes driver and runtime overhead',
                },
            }, indent=2, sort_keys=True))
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f'ANEC package error: {error}\n')


if __name__ == '__main__':
    main()
