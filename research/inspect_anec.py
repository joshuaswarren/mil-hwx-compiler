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
PARITY_MATVEC = 'apple-parity-matvec'
PROGRAM_FIELDS = (
    'file', 'bytes', 'taskDescriptors', 'encoder', 'operation', 'inputs',
    'constantInputs', 'outputs', 'constantOffset', 'constantBytes')


def physical_layout(rows, width):
    """Apple's matvec surface: `rows` dense fp16 rows of `width` elements."""
    row = width * 2
    return [1, 1, rows, width, row * rows, row]


def channel_layout(elements):
    """The elementwise surface: one 64-byte physical lane per element."""
    return [1, elements, 1, 1, 64, 64]


def dense_layout(binding):
    """The declared dense-row layout of one parity matvec surface."""
    nchw = binding.get('nchw') if isinstance(binding, dict) else None
    require(isinstance(nchw, list) and len(nchw) == 6 and
            all(type(value) is int for value in nchw) and
            nchw[2] > 0 and nchw[3] > 0 and
            nchw == physical_layout(nchw[2], nchw[3]),
            'parity matvec surfaces must be dense fp16 rows')
    return nchw


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


def check_tensor(name, tensor):
    require(isinstance(name, str) and name, 'tensor needs a non-empty name')
    fields = set(tensor) if isinstance(tensor, dict) else set()
    require(fields in ({'shape', 'logicalBytes', 'role'},
                       {'shape', 'logicalBytes', 'role', 'aliasOf'},
                       {'shape', 'logicalBytes', 'role', 'accumulation'}),
            'tensor record has incorrect fields')
    require('accumulation' not in tensor or
            tensor['accumulation'] == 'chunked-fp16',
            'unsupported tensor accumulation')
    shape = tensor.get('shape')
    require(isinstance(shape, list) and shape and
            all(type(n) is int and n > 0 for n in shape),
            'tensor shape must have positive static dimensions')
    require(type(tensor.get('logicalBytes')) is int and
            tensor['logicalBytes'] == math.prod(shape) * 2,
            'tensor logical byte count does not match its shape')
    require(tensor.get('role') in ('input', 'output', 'intermediate', 'constant'),
            'unsupported tensor role')
    require('aliasOf' not in tensor or
            isinstance(tensor['aliasOf'], str) and tensor['aliasOf'],
            'tensor aliasOf must be a non-empty name')


def binding_interval(binding, tensors):
    slice_record = binding.get('slice')
    if slice_record is None:
        tensor_name = binding['name']
        require(tensor_name in tensors, 'binding references an unknown tensor')
        tensor = tensors[tensor_name]
        require('aliasOf' not in tensor,
                'bindings must reference an underlying tensor')
        require(binding['shape'] == tensor['shape'] and
                binding['logicalBytes'] == tensor['logicalBytes'],
                'whole-tensor binding differs from its tensor')
        count = tensor['logicalBytes'] // 2
        return tensor_name, 0, count, count
    fields = set(slice_record) if isinstance(slice_record, dict) else set()
    require(fields in ({'tensor', 'elementOffset', 'elementCount'},
                       {'tensor', 'elementOffset', 'elementCount',
                        'physicalElements'}), 'slice has incorrect fields')
    tensor_name = slice_record.get('tensor')
    offset, count = slice_record.get('elementOffset'), slice_record.get('elementCount')
    physical = slice_record.get('physicalElements', count)
    require(tensor_name == binding['name'] and tensor_name in tensors,
            'slice references an unknown or mismatched tensor')
    require('aliasOf' not in tensors[tensor_name],
            'bindings must reference an underlying tensor')
    require(type(offset) is int and type(count) is int and
            type(physical) is int and offset >= 0 and count > 0 and physical >= count,
            'slice offsets, counts, and physical elements must be valid integers')
    elements = tensors[tensor_name]['logicalBytes'] // 2
    require(count == binding['logicalBytes'] // 2 and offset <= elements - count,
            'slice exceeds its tensor or differs from its binding')
    return tensor_name, offset, count, physical


def exact_tiling(bindings, tensors, message, allow_repeats=False):
    intervals = [binding_interval(item, tensors)[1:3] for _, item in bindings]
    if allow_repeats:
        intervals = set(intervals)
    intervals = sorted(intervals)
    cursor = 0
    for offset, count in intervals:
        require(offset == cursor, message)
        cursor += count
    require(bindings and
            cursor == tensors[binding_interval(bindings[0][1], tensors)[0]]['logicalBytes'] // 2,
            message)


def check_binding(binding, index, physical_elements, tiles, layouts, tensors,
                  layout=None):
    require(isinstance(binding, dict), 'binding must be an object')
    require(isinstance(binding.get('name'), str) and binding['name'],
            'binding needs a non-empty name')
    require(binding.get('dtype') == 'float16', 'binding must use float16')
    shape = binding.get('shape')
    require(isinstance(shape, list) and shape and
            all(type(n) is int and 0 < n <= physical_elements for n in shape),
            'logical shape must have positive static dimensions within its layout')
    logical_elements = math.prod(shape)
    require(type(binding.get('index')) is int and binding['index'] == index,
            'binding index does not match ANEC channel')
    layout = layout or channel_layout(physical_elements)
    require(binding.get('nchw') == layout and
            all(type(n) is int for n in binding['nchw']), 'incorrect physical layout')
    require(list(layouts[index * 6:(index + 1) * 6]) == layout,
            'manifest layout differs from ANEC header')
    require(type(binding.get('logicalBytes')) is int and
            binding['logicalBytes'] == logical_elements * 2,
            'incorrect logical byte count')
    span = layout[0] * layout[1] * layout[4]
    allocation = -(-span // TILE_BYTES) * TILE_BYTES
    require(type(binding.get('allocationBytes')) is int and
            binding['allocationBytes'] == allocation, 'incorrect allocation size')
    require(tiles[index] * TILE_BYTES == allocation,
            'allocation differs from ANEC header')
    require(binding.get('role') in (None, 'intermediate'),
            'unsupported binding role')
    _, _, count, physical = binding_interval(binding, tensors)
    require(count == logical_elements and physical == physical_elements,
            'slice physical elements differ from its binding layout')


def validate_program(directory, program, tensors):
    require(isinstance(program, dict), 'program must be an object')
    operation = program.get('operation')
    require(operation in ('add', 'mul', 'maximum', 'minimum', 'sub', 'real_div',
                          'matmul', 'abs', 'exp', 'gelu', 'leaky_relu', 'relu',
                          'rsqrt', 'sigmoid', 'silu', 'sqrt', 'tanh'),
            'unsupported operation')
    inputs, outputs = program.get('inputs'), program.get('outputs')
    require(isinstance(inputs, list) and isinstance(outputs, list),
            'input and output bindings must be arrays')
    constant_inputs = program.get('constantInputs')
    require(isinstance(constant_inputs, dict), 'constantInputs must be an object')
    matmul = operation == 'matmul'
    require(1 <= len(inputs) <= 2 and len(outputs) == 1,
            'incorrect input/output count')
    for key in ('bytes', 'constantBytes', 'constantOffset', 'taskDescriptors'):
        require(type(program.get(key)) is int and program[key] > 0 or
                (key == 'constantBytes' and program.get(key) == 0),
                f'incorrect {key}')
    constants, offset = program['constantBytes'], program['constantOffset']
    require(program['bytes'] == HEADER_BYTES + offset + constants,
            'incorrect bytes')
    data = local_file(directory, program.get('file'), program['bytes'])
    require(len(data) == program['bytes'], 'ANEC file length differs from manifest')
    fields = HEADER.unpack_from(data)
    size, td_size, td_count, task_size, kernel_size, source_count, dest_count = fields[:7]
    require((size, td_count, kernel_size, source_count, dest_count) ==
            (offset + constants, program['taskDescriptors'], constants,
             len(inputs), len(outputs)), 'ANEC header differs from manifest')
    require(td_size % 4 == 0 and 0 < td_size <= task_size <= offset and
            offset == (task_size + 0x3f) & ~0x3f,
            'ANEC task stream does not fit its constant offset')
    require(not any(data[HEADER.size:HEADER_BYTES]) and
            not any(data[HEADER_BYTES + task_size:HEADER_BYTES + offset]),
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
    parity = program.get('encoder') == PARITY_MATVEC
    input_layout = output_layout = None
    if matmul and parity:
        input_layout, output_layout = (dense_layout(item)
                                       for item in (inputs[0], outputs[0]))
        input_channels = input_layout[2] * input_layout[3]
        output_channels = output_layout[2] * output_layout[3]
    elif matmul:
        require(isinstance(inputs[0], dict), 'unsupported matvec input size')
        input_channels = binding_interval(inputs[0], tensors)[3]
        output_channels = binding_interval(outputs[0], tensors)[3]
        require(input_channels in (256, 512) and output_channels == 512,
                'unsupported matvec physical size')
    else:
        input_channels = binding_interval(inputs[0], tensors)[3]
        output_channels = binding_interval(outputs[0], tensors)[3]
    for index, item in enumerate(inputs, start=5):
        check_binding(item, index, input_channels, tiles, layouts, tensors,
                      input_layout)
    check_binding(outputs[0], 4, output_channels, tiles, layouts, tensors,
                  output_layout)
    if not matmul:
        output_is_returned_alias = any(
            tensor.get('aliasOf') == outputs[0]['name'] and
            tensor['role'] == 'output' and tensor['shape'] == outputs[0]['shape']
            for tensor in tensors.values())
        require(all(item['shape'] == inputs[0]['shape'] for item in inputs) and
                (outputs[0]['shape'] == inputs[0]['shape'] or
                 output_is_returned_alias and
                 math.prod(outputs[0]['shape']) == math.prod(inputs[0]['shape'])),
                'elementwise operation shapes must match')
    require(outputs[0]['name'] not in {item['name'] for item in inputs},
            'program output must differ from its inputs')
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
        physical_elements = binding_interval(item, tensors)[3]
        require(isinstance(encoded, str) and len(encoded) == physical_elements * 4 and
                all(character in '0123456789abcdefABCDEF' for character in encoded),
                'constant input must be physical hexadecimal fp16 bytes')
        payload = bytes.fromhex(encoded)
        require(not any(payload[item['logicalBytes']:]),
                'constant input padding must be zero')
    command_bytes = ((program['bytes'] - HEADER_BYTES + TILE_BYTES - 1)
                     // TILE_BYTES) * TILE_BYTES
    input_bytes = sum(item['allocationBytes'] for item in inputs)
    output_bytes = sum(item['allocationBytes'] for item in outputs)
    return {
        'file': program['file'], 'commandAndConstantsBytes': command_bytes,
        'inputBytes': input_bytes, 'outputBytes': output_bytes,
        'totalBytes': command_bytes + input_bytes + output_bytes,
    }


def load_package(directory):
    directory = Path(directory).resolve()
    manifest = json.loads(local_file(directory, 'manifest.json', 64 << 20))
    require(isinstance(manifest, dict), 'manifest must be an object')
    require(manifest.get('schema') == 'mil-hwxc.h13-anec-package.v1',
            'unsupported package schema')
    require(manifest.get('target') == 'H13' and manifest.get('artifactFormat') == 'anec',
            'package must target H13 ANEC')
    programs = manifest.get('programs')
    dispatch = manifest.get('dispatchPlan')
    intermediates = manifest.get('intermediates')
    tensors = manifest.get('tensors')
    require(isinstance(programs, list) and programs, 'programs must be a non-empty array')
    require(isinstance(dispatch, list) and
            all(type(index) is int for index in dispatch) and
            sorted(dispatch) == list(range(len(programs))),
            'dispatchPlan must contain every program index exactly once')
    require(isinstance(intermediates, list) and
            all(isinstance(name, str) and name for name in intermediates) and
            len(set(intermediates)) == len(intermediates),
            'intermediates must contain unique non-empty names')
    require(isinstance(tensors, dict) and tensors, 'tensors must be a non-empty object')
    for name, tensor in tensors.items():
        check_tensor(name, tensor)
    alias_names = {name for name, tensor in tensors.items() if 'aliasOf' in tensor}
    for name in alias_names:
        target = tensors[name]['aliasOf']
        require(target != name and target in tensors and 'aliasOf' not in tensors[target],
                'tensor aliasOf must name an underlying tensor')
        require(tensors[name]['logicalBytes'] == tensors[target]['logicalBytes'],
                'tensor alias element count differs from its underlying tensor')
    intermediate_set = {name for name, tensor in tensors.items()
                        if tensor['role'] == 'intermediate'}
    require(set(intermediates) == intermediate_set,
            'intermediates must match tensors with intermediate role')
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

    allocations = [validate_program(directory, program, tensors) for program in programs]
    storage_intermediates = [name for name in intermediates if name not in alias_names]
    producers = {name: [] for name in storage_intermediates}
    consumers = {name: [] for name in storage_intermediates}
    output_tensors = {name: [] for name, tensor in tensors.items()
                      if tensor['role'] == 'output' and name not in alias_names}
    referenced = set()
    for program_index, program in enumerate(programs):
        for item in program['outputs']:
            name, _, _, _ = binding_interval(item, tensors)
            referenced.add(name)
            role = tensors[name]['role']
            if role == 'intermediate':
                require(item.get('role') == 'intermediate',
                        'intermediate output must have intermediate role')
                producers[name].append((program_index, item))
            else:
                require(role == 'output' and item.get('role') is None,
                        'non-intermediate output must have output tensor role')
                output_tensors[name].append((program_index, item))
        for item in program['inputs']:
            name, _, _, _ = binding_interval(item, tensors)
            referenced.add(name)
            role = tensors[name]['role']
            if role == 'intermediate':
                require(item.get('role') == 'intermediate' and
                        item.get('binding') is None,
                        'intermediate input must have intermediate role')
                consumers[name].append((program_index, item))
            elif role == 'constant':
                require(item.get('role') is None and item.get('binding') == 'constant',
                        'constant tensor input must have constant binding')
            else:
                require(role == 'input' and item.get('role') is None and
                        item.get('binding') is None,
                        'runtime input must have input tensor role')
    require(all(name in referenced or tensor['role'] == 'constant' or
                name in alias_names for name, tensor in tensors.items()),
            'only constants and aliases may be unreferenced by program bindings')
    positions = {program_index: position for position, program_index in enumerate(dispatch)}
    for name in storage_intermediates:
        exact_tiling(producers[name], tensors,
                     'intermediate producer slices must exactly tile the tensor')
        exact_tiling(consumers[name], tensors,
                     'intermediate consumer slices must exactly tile the tensor', True)
        produced_ranges = sorted(
            (offset, offset + physical, program_index)
            for program_index, item in producers[name]
            for _, offset, _, physical in [binding_interval(item, tensors)])
        previous_end = 0
        for start, end, _ in produced_ranges:
            require(start >= previous_end,
                    'intermediate producer physical writes overlap')
            previous_end = end
        for consumer_index, consumed in consumers[name]:
            _, consumed_offset, _, consumed_physical = binding_interval(consumed, tensors)
            consumed_end = consumed_offset + consumed_physical
            cursor = consumed_offset
            for produced_offset, produced_end, producer_index in produced_ranges:
                if produced_end <= cursor:
                    continue
                if produced_offset > cursor:
                    break
                if produced_offset < consumed_end and consumed_offset < produced_end:
                    require(positions[producer_index] < positions[consumer_index],
                            'dispatchPlan violates an intermediate dependency')
                cursor = max(cursor, produced_end)
                if cursor >= consumed_end:
                    break
            require(cursor >= consumed_end,
                    'intermediate consumer physical range exceeds producer writes')
    for name, bindings in output_tensors.items():
        exact_tiling(bindings, tensors, 'output slices must exactly tile the tensor')
    return manifest, allocations


def find_bindings(manifest, name, direction, constants=False):
    matches = []
    for program_index, program in enumerate(manifest['programs']):
        for item in program[direction]:
            tensor_name, _, _, _ = binding_interval(item, manifest['tensors'])
            if tensor_name != name:
                continue
            if constants != (item.get('binding') == 'constant'):
                continue
            matches.append((program_index, program, item))
    require(matches, 'no such constant input' if constants else
            ('no such input' if direction == 'inputs' else 'no such output'))
    return matches


def convert_tensor(binding, data, pack):
    """Packs dense fp16 into, or reads it back from, the physical surface.

    The elementwise surface holds one element per 64-byte lane; the parity
    matvec surface holds dense rows, so the row stride comes from the binding.
    """
    logical, physical = binding['logicalBytes'], binding['allocationBytes']
    require(len(data) == (logical if pack else physical), 'incorrect tensor byte count')
    nchw = binding.get('nchw')
    width, row = (nchw[3], nchw[5]) if nchw else (1, 64)
    result = bytearray(physical if pack else logical)
    for element in range(logical // 2):
        dense = element * 2
        offset = (element // width) * row + (element % width) * 2
        if pack:
            result[offset:offset + 2] = data[dense:dense + 2]
        else:
            result[dense:dense + 2] = data[offset:offset + 2]
    return result


def dense_slice(data, binding, tensors):
    name, offset, count, _ = binding_interval(binding, tensors)
    require(len(data) == tensors[name]['logicalBytes'], 'incorrect dense tensor byte count')
    return data[offset * 2:(offset + count) * 2]


def binding_buffer_name(program_index, binding):
    name = binding['name']
    require(Path(name).name == name and name not in ('.', '..'),
            'binding name cannot form a safe buffer filename')
    return f'program-{program_index}.{name}.buffer'


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
                encoded = {program['constantInputs'][item['name']]
                           for _, program, item in matches}
                require(len(encoded) == 1,
                        'constant name has inconsistent payloads across programs')
                source = bytes.fromhex(encoded.pop())
                item = matches[0][2]
                data = convert_tensor(item, source[:item['logicalBytes']], True)
                with args.output.open('xb') as destination:
                    destination.write(data)
                written = len(data)
            elif args.pack_input:
                name, source_path = args.pack_input
                matches = find_bindings(manifest, name, 'inputs')
                source = Path(source_path).read_bytes()
                buffers = [(program_index, item,
                            convert_tensor(item, dense_slice(source, item, manifest['tensors']), True))
                           for program_index, _, item in matches]
                if len(buffers) == 1:
                    with args.output.open('xb') as destination:
                        destination.write(buffers[0][2])
                else:
                    args.output.mkdir()
                    for program_index, item, data in buffers:
                        (args.output / binding_buffer_name(program_index, item)).write_bytes(data)
                written = sum(len(data) for _, _, data in buffers)
            else:
                name, source_path = args.unpack_output
                matches = find_bindings(manifest, name, 'outputs')
                if len(matches) == 1:
                    item = matches[0][2]
                    data = convert_tensor(item, Path(source_path).read_bytes(), False)
                else:
                    exact_tiling([(program_index, item) for program_index, _, item in matches],
                                 manifest['tensors'],
                                 'output slices must exactly tile the tensor')
                    data = bytearray(manifest['tensors'][name]['logicalBytes'])
                    source_directory = Path(source_path).resolve()
                    require(source_directory.is_dir(),
                            'multi-binding output source must be a directory')
                    for program_index, _, item in matches:
                        physical = local_file(source_directory,
                                              binding_buffer_name(program_index, item),
                                              item['allocationBytes'])
                        dense = convert_tensor(item, physical, False)
                        _, offset, count, _ = binding_interval(item, manifest['tensors'])
                        data[offset * 2:(offset + count) * 2] = dense
                with args.output.open('xb') as destination:
                    destination.write(data)
                written = len(data)
            print(f'wrote {written} bytes to {args.output}; no device execution')
        else:
            command_bytes = sum(item['commandAndConstantsBytes'] for item in allocations)
            input_bindings = {}
            output_bindings = {}
            for program in manifest['programs']:
                for item in program['inputs']:
                    name, offset, count, physical = binding_interval(
                        item, manifest['tensors'])
                    if manifest['tensors'][name]['role'] != 'intermediate':
                        input_bindings.setdefault((name, offset, count, physical), item)
                for item in program['outputs']:
                    name, offset, count, physical = binding_interval(
                        item, manifest['tensors'])
                    output_bindings.setdefault((name, offset, count, physical), item)
            input_bytes = sum(item['allocationBytes'] for item in input_bindings.values())
            output_bytes = sum(item['allocationBytes'] for item in output_bindings.values())
            print(json.dumps({
                'validation': 'container, tensor, binding, slice, and dispatch consistency only',
                'manifest': manifest,
                'bufferAllocation': {
                    'programs': allocations,
                    'commandAndConstantsBytes': command_bytes,
                    'inputBytes': input_bytes, 'outputBytes': output_bytes,
                    'totalBytes': command_bytes + input_bytes + output_bytes,
                    'scope': 'encoded buffers only; shared tensor slices counted once; excludes driver and runtime overhead',
                },
            }, indent=2, sort_keys=True))
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f'ANEC package error: {error}\n')


if __name__ == '__main__':
    main()
