#!/usr/bin/env python3
"""Validate, dispatch, and numerically check an H13 ANEC package with libane."""

import argparse
import ctypes
import json
import math
import os
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from research import inspect_anec
from h13_reference import decode_fp16, evaluate

CHUNKED_ATOL = 0.02
CHUNKED_RTOL = 0.02
CHECKOUT_LIBANE = Path.home() / "src/omarchy-ane/bindings/python/dylib/libane_python.so"
DEFAULT_LIBANE = CHECKOUT_LIBANE if CHECKOUT_LIBANE.is_file() else Path("/usr/lib/libane_python.so")


def chunked_close(reference, actual):
    """The documented chunked-fp16 envelope: |device - reference| <= 0.02 + 0.02 * |reference|."""
    return math.isfinite(actual) and abs(actual - reference) <= CHUNKED_ATOL + CHUNKED_RTOL * abs(reference)


def compare_fp16(actual, expected, chunked=False):
    if not chunked:
        if actual != expected:
            raise ValueError("device output differs from the exact fp16 reference")
        return
    actual_values, expected_values = decode_fp16(actual), decode_fp16(expected)
    if len(actual_values) != len(expected_values):
        raise ValueError("device output byte count differs from the reference")
    differences = [(index, wanted, got) for index, (wanted, got) in
                   enumerate(zip(expected_values, actual_values))
                   if not chunked_close(wanted, got)]
    if differences:
        index, wanted, got = differences[0]
        raise ValueError(
            f"chunked matmul output differs at element {index}: "
            f"reference={wanted} device={got}; tolerance is "
            f"atol={CHUNKED_ATOL}, rtol={CHUNKED_RTOL}")


class LibANEAdapter:
    """All calls into libane's Python binding shared library live here."""

    def __init__(self, library, device=0):
        self.device = device
        self.lib = ctypes.CDLL(os.fspath(library), use_errno=True)
        pointer = ctypes.c_void_p
        self.lib.pyane_init.restype = pointer
        self.lib.pyane_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self.lib.pyane_free.restype = ctypes.c_int
        self.lib.pyane_free.argtypes = [pointer]
        self.lib.__ane_src_size.restype = ctypes.c_uint64
        self.lib.__ane_src_size.argtypes = [pointer, ctypes.c_uint32]
        self.lib.__ane_dst_size.restype = ctypes.c_uint64
        self.lib.__ane_dst_size.argtypes = [pointer, ctypes.c_uint32]
        self.lib.__ane_send.argtypes = [pointer, pointer, ctypes.c_uint32]
        self.lib.__ane_read.argtypes = [pointer, pointer, ctypes.c_uint32]
        self.lib.ane_bind_kernel.restype = ctypes.c_int
        self.lib.ane_bind_kernel.argtypes = [pointer, pointer, ctypes.c_uint64]
        self.lib.ane_exec.restype = ctypes.c_int
        self.lib.ane_exec.argtypes = [pointer]
        self.lib.ane_exec_loop.restype = ctypes.c_int
        self.lib.ane_exec_loop.argtypes = [pointer, ctypes.c_uint32,
                                           ctypes.c_uint32, ctypes.c_uint32]

    @staticmethod
    def _failure(name, result):
        number = ctypes.get_errno()
        detail = os.strerror(number) if number else "no errno"
        return RuntimeError(f"{name} failed: {result}, errno={number} ({detail})")

    def execute(self, anec, kernel, inputs, output_sizes):
        with tempfile.NamedTemporaryFile(prefix="h13-program-", suffix=".anec") as file:
            file.write(anec)
            file.flush()
            handle = self.lib.pyane_init(os.fsencode(file.name), self.device)
        if not handle:
            raise self._failure("pyane_init", handle)
        try:
            if kernel:
                source = ctypes.create_string_buffer(kernel, len(kernel))
                result = self.lib.ane_bind_kernel(handle, source, len(kernel))
                if result:
                    raise self._failure("ane_bind_kernel", result)
            input_buffers = []
            for index, data in enumerate(inputs):
                size = self.lib.__ane_src_size(handle, index)
                if size != len(data):
                    raise RuntimeError(
                        f"libane input {index} size {size} differs from manifest {len(data)}")
                buffer = ctypes.create_string_buffer(data, len(data))
                input_buffers.append(buffer)
                self.lib.__ane_send(handle, buffer, index)
            for index, expected in enumerate(output_sizes):
                size = self.lib.__ane_dst_size(handle, index)
                if size != expected:
                    raise RuntimeError(
                        f"libane output {index} size {size} differs from manifest {expected}")
            result = self.lib.ane_exec(handle)
            if result:
                raise self._failure("ane_exec", result)
            outputs = []
            for index, size in enumerate(output_sizes):
                buffer = ctypes.create_string_buffer(size)
                self.lib.__ane_read(handle, buffer, index)
                outputs.append(buffer.raw)
            return outputs
        finally:
            result = self.lib.pyane_free(handle)
            if result:
                raise self._failure("pyane_free", result)

    def execute_loop(self, handle, iterations, state_input, state_output):
        result = self.lib.ane_exec_loop(handle, iterations, state_input, state_output)
        if result:
            raise self._failure("ane_exec_loop", result)


def _paths(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("bindings must use NAME=PATH")
        name, path = item.split("=", 1)
        if not name or name in result:
            raise ValueError("binding names must be unique and non-empty")
        result[name] = Path(path)
    return result


def _logical_outputs(tensors):
    aliases = {tensor["aliasOf"] for tensor in tensors.values()
               if tensor["role"] == "output" and "aliasOf" in tensor}
    return {name for name, tensor in tensors.items()
            if tensor["role"] == "output" and
            ("aliasOf" in tensor or name not in aliases)}


def _binding_record(binding, tensors, source=None):
    name, offset, count, physical = inspect_anec.binding_interval(binding, tensors)
    result = {
        "tensor": name,
        "channel": binding["index"],
        "elementOffset": offset,
        "elementCount": count,
        "physicalElements": physical,
        "allocationBytes": binding["allocationBytes"],
    }
    if source:
        result["source"] = source
    return result


def build_plan(manifest, reference_outputs):
    tensors = manifest["tensors"]
    programs = []
    for index in manifest["dispatchPlan"]:
        program = manifest["programs"][index]
        inputs = []
        for binding in program["inputs"]:
            name = binding["name"]
            if binding.get("binding") == "constant":
                source = "constantInputs"
            else:
                source = tensors[name]["role"]
            inputs.append(_binding_record(binding, tensors, source))
        programs.append({
            "index": index,
            "file": program["file"],
            "operation": program["operation"],
            "inputs": inputs,
            "outputs": [_binding_record(item, tensors) for item in program["outputs"]],
        })
    return {
        "schema": "mil-hwxc.h13-linux-plan.v1",
        "deviceCalls": False,
        "programs": programs,
        "referenceOutputs": {name: len(data) for name, data in sorted(reference_outputs.items())},
    }


def _runtime_buffer(dense, binding, tensors):
    sliced = inspect_anec.dense_slice(dense, binding, tensors)
    return bytes(inspect_anec.convert_tensor(binding, sliced, True))


def _constant_buffer(program, binding):
    dense = bytes.fromhex(program["constantInputs"][binding["name"]])
    return bytes(inspect_anec.convert_tensor(
        binding, dense[:binding["logicalBytes"]], True))


def _intermediate_buffer(binding, tensors, regions):
    """Builds one consumer surface from its producers' surfaces.

    Producers and consumers can use different physical layouts -- a parity
    matvec writes dense rows while an elementwise program reads 64-byte lanes
    -- so composition goes through dense fp16.
    """
    _, offset, count, _ = inspect_anec.binding_interval(binding, tensors)
    dense = bytearray(count * 2)
    covered = [False] * count
    for produced_binding, data in regions:
        _, produced_offset, produced_count, _ = \
            inspect_anec.binding_interval(produced_binding, tensors)
        produced = inspect_anec.convert_tensor(produced_binding, data, False)
        start = max(offset, produced_offset)
        end = min(offset + count, produced_offset + produced_count)
        for element in range(start, end):
            source = (element - produced_offset) * 2
            destination = (element - offset) * 2
            dense[destination:destination + 2] = produced[source:source + 2]
            covered[element - offset] = True
    if not all(covered):
        raise ValueError(f"intermediate {binding['name']} lacks a produced logical range")
    return bytes(inspect_anec.convert_tensor(binding, bytes(dense), True))


def _unpack_outputs(manifest, regions, names):
    tensors = manifest["tensors"]
    dense = {}
    for name in names:
        target = tensors[name].get("aliasOf", name)
        target_tensor = tensors[target]
        result = bytearray(target_tensor["logicalBytes"])
        written = [False] * (target_tensor["logicalBytes"] // 2)
        for binding, data in regions.get(target, []):
            _, offset, count, _ = inspect_anec.binding_interval(binding, tensors)
            values = inspect_anec.convert_tensor(binding, data, False)
            result[offset * 2:(offset + count) * 2] = values
            written[offset:offset + count] = [True] * count
        if not all(written):
            raise ValueError(f"output {name} lacks a produced logical range")
        dense[name] = bytes(result)
    return dense


def run_package(package, mil, model_root, inputs, adapter):
    manifest, _ = inspect_anec.load_package(package)
    tensors = manifest["tensors"]
    input_names = {name for name, tensor in tensors.items()
                   if tensor["role"] == "input" and "aliasOf" not in tensor}
    output_names = _logical_outputs(tensors)
    if set(inputs) != input_names:
        raise ValueError(f"inputs must be exactly {', '.join(sorted(input_names))}")
    dense_inputs = {name: path.read_bytes() for name, path in inputs.items()}
    for name, data in dense_inputs.items():
        if len(data) != tensors[name]["logicalBytes"]:
            raise ValueError(f"input {name} byte count differs from its manifest tensor")
    expected = evaluate(Path(mil).read_text(), model_root, dense_inputs)
    if set(expected) != output_names:
        raise ValueError("MIL returns differ from manifest output tensors")
    if adapter is None:
        return manifest, expected, build_plan(manifest, expected)

    intermediate_regions = {}
    output_regions = {}
    package = Path(package).resolve()
    for program_index in manifest["dispatchPlan"]:
        program = manifest["programs"][program_index]
        input_buffers = []
        for binding in program["inputs"]:
            name = binding["name"]
            if binding.get("binding") == "constant":
                data = _constant_buffer(program, binding)
            elif tensors[name]["role"] == "intermediate":
                data = _intermediate_buffer(binding, tensors, intermediate_regions.get(name, []))
            else:
                data = _runtime_buffer(dense_inputs[name], binding, tensors)
            input_buffers.append(data)
        anec = inspect_anec.local_file(package, program["file"], program["bytes"])
        kernel_start = inspect_anec.HEADER_BYTES + program["constantOffset"]
        kernel = anec[kernel_start:kernel_start + program["constantBytes"]]
        outputs = adapter.execute(
            anec, kernel, input_buffers,
            [binding["allocationBytes"] for binding in program["outputs"]])
        for binding, data in zip(program["outputs"], outputs):
            name, _, _, _ = inspect_anec.binding_interval(binding, tensors)
            if tensors[name]["role"] == "intermediate":
                intermediate_regions.setdefault(name, []).append((binding, data))
            else:
                output_regions.setdefault(name, []).append((binding, data))
    actual = _unpack_outputs(manifest, output_regions, output_names)
    for name in output_names:
        target = tensors[name].get("aliasOf", name)
        compare_fp16(actual[name], expected[name],
                     tensors[target].get("accumulation") == "chunked-fp16")
    return manifest, actual, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--mil", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--input", action="append", default=[], metavar="NAME=DENSE_FP16")
    parser.add_argument("--output", action="append", default=[], metavar="NAME=DENSE_FP16")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--libane-library", type=Path, default=DEFAULT_LIBANE)
    args = parser.parse_args()
    try:
        inputs, outputs = _paths(args.input), _paths(args.output)
        adapter = None if args.dry_run else LibANEAdapter(args.libane_library, args.device)
        manifest, result, plan = run_package(
            args.package, args.mil, args.model_root, inputs, adapter)
        expected_outputs = _logical_outputs(manifest["tensors"])
        if set(outputs) != expected_outputs:
            raise ValueError(f"outputs must be exactly {', '.join(sorted(expected_outputs))}")
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return
        for name, path in outputs.items():
            path.write_bytes(result[name])
            print(f"PASS {name}: {len(result[name])} fp16 bytes match the reference; wrote {path}")
    except (OSError, ValueError, RuntimeError, struct.error) as error:
        parser.exit(1, f"H13 Linux runner error: {error}\n")


if __name__ == "__main__":
    main()
