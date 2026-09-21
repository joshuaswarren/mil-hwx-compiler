#!/usr/bin/env python3
"""Validate, dispatch, and numerically check an H13 ANEC package with libane."""

import argparse
import ctypes
import json
import math
import os
import platform
import statistics
import struct
import sys
import tempfile
import time
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
    actual_values, expected_values = decode_fp16(actual), decode_fp16(expected)
    if not chunked:
        if actual_values != expected_values:
            raise ValueError("device output differs from the exact fp16 reference")
        return
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
        try:
            # getattr: `self.lib.__ane_src_size` would name-mangle inside
            # this class body and look up a symbol libane never exports.
            self._init = getattr(self.lib, "pyane_init")
            self._init.restype = pointer
            self._init.argtypes = [ctypes.c_char_p, ctypes.c_int]
            self._free = getattr(self.lib, "pyane_free")
            self._free.restype = ctypes.c_int
            self._free.argtypes = [pointer]
            self._src_size = getattr(self.lib, "__ane_src_size")
            self._src_size.restype = ctypes.c_uint64
            self._src_size.argtypes = [pointer, ctypes.c_uint32]
            self._dst_size = getattr(self.lib, "__ane_dst_size")
            self._dst_size.restype = ctypes.c_uint64
            self._dst_size.argtypes = [pointer, ctypes.c_uint32]
            self._send = getattr(self.lib, "__ane_send")
            self._send.restype = None  # void
            self._send.argtypes = [pointer, pointer, ctypes.c_uint32]
            self._read = getattr(self.lib, "__ane_read")
            self._read.restype = None  # void
            self._read.argtypes = [pointer, pointer, ctypes.c_uint32]
            self._exec = getattr(self.lib, "ane_exec")
            self._exec.restype = ctypes.c_int
            self._exec.argtypes = [pointer]
        except AttributeError as error:
            raise RuntimeError(
                f"{library} does not export the libane ABI: {error}") from error

    @staticmethod
    def _failure(name, result):
        number = ctypes.get_errno()
        detail = os.strerror(number) if number else "no errno"
        return RuntimeError(f"{name} failed: {result}, errno={number} ({detail})")

    def execute(self, anec, inputs, output_sizes):
        program = PreparedProgram(self, anec, inputs, output_sizes)
        try:
            return program.run_once()
        finally:
            _close_all([program], sys.exc_info()[1])

    @staticmethod
    def call_sequence(program):
        """The exact order the runner issues libane calls for one program, so a
        driver owner can review the submission before any hardware run. Size
        validation is part of the prepared phase, before the first transfer."""
        calls = [f"pyane_init(\"{program['file']}\", device)"]
        for index, binding in enumerate(program["inputs"]):
            calls.append(f"__ane_src_size(handle, {index}) == "
                         f"{binding['allocationBytes']}  # {binding['name']}")
        for index, binding in enumerate(program["outputs"]):
            calls.append(f"__ane_dst_size(handle, {index}) == "
                         f"{binding['allocationBytes']}  # {binding['name']}")
        for index, binding in enumerate(program["inputs"]):
            calls.append(f"__ane_send(handle, buffer, {index})")
        calls.append("ane_exec(handle)")
        for index in range(len(program["outputs"])):
            calls.append(f"__ane_read(handle, buffer, {index})")
        calls.append("pyane_free(handle)")
        return calls


class PreparedProgram:
    """An open libane handle with validated sizes and prepared buffers.

    Everything that answers questions about the package (`pyane_init`,
    `__ane_src_size`, `__ane_dst_size`) and every allocation happens in
    `__init__`: a size disagreement fails closed before the first
    `__ane_send`. The instance is reusable -- benchmark iterations repeat only
    transfers, submission, and readback on the same handle and buffers, so
    initialization is reported as setup and never hidden inside a timed window.
    """

    def __init__(self, adapter, anec, inputs, output_sizes):
        self.adapter = adapter
        self.handle = None
        with tempfile.NamedTemporaryFile(prefix="h13-program-", suffix=".anec") as file:
            file.write(anec)
            file.flush()
            handle = adapter._init(os.fsencode(file.name), adapter.device)
        if not handle:
            raise adapter._failure("pyane_init", 0)
        self.handle = handle
        try:
            self.input_buffers = []
            for index, data in enumerate(inputs):
                size = adapter._src_size(self.handle, index)
                if size != len(data):
                    raise RuntimeError(
                        f"libane input {index} size {size} differs from manifest {len(data)}")
                self.input_buffers.append(ctypes.create_string_buffer(data, len(data)))
            self.output_buffers = []
            for index, expected in enumerate(output_sizes):
                size = adapter._dst_size(self.handle, index)
                if size != expected:
                    raise RuntimeError(
                        f"libane output {index} size {size} differs from manifest {expected}")
                self.output_buffers.append(ctypes.create_string_buffer(size))
        except BaseException:
            _close_all([self], sys.exc_info()[1])
            raise

    def refill(self, inputs):
        """Refresh the prepared input buffers with newly composed surfaces."""
        if len(inputs) != len(self.input_buffers):
            raise RuntimeError(
                f"program takes {len(self.input_buffers)} input surfaces, got {len(inputs)}")
        for index, (buffer, data) in enumerate(zip(self.input_buffers, inputs)):
            if len(data) != len(buffer):
                raise RuntimeError(
                    f"input {index} surface is {len(data)} bytes, "
                    f"prepared buffer is {len(buffer)}")
            ctypes.memmove(buffer, data, len(data))

    def transfer_in(self):
        for index, buffer in enumerate(self.input_buffers):
            self.adapter._send(self.handle, buffer, index)

    def submit(self):
        result = self.adapter._exec(self.handle)
        if result:
            raise self.adapter._failure("ane_exec", result)

    def transfer_out(self):
        outputs = []
        for index, buffer in enumerate(self.output_buffers):
            self.adapter._read(self.handle, buffer, index)
            outputs.append(buffer.raw)
        return outputs

    def run_once(self):
        """One full execution: transfers, submission, readback."""
        self.transfer_in()
        self.submit()
        return self.transfer_out()

    def close(self):
        handle, self.handle = self.handle, None
        if not handle:
            return
        result = self.adapter._free(handle)
        if result:
            raise self.adapter._failure("pyane_free", result)



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
    """One plan entry. `allocationBytes` is the surface's whole span, which is
    what libane sizes a buffer from; a batched surface's tensor descriptor
    declares only one batch element."""
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


def reference_criteria(manifest, reference_outputs):
    """Per-output pass criterion, as published in the dry-run plan and honored
    by every correctness check before benchmark timings are reported."""
    tensors = manifest["tensors"]
    chunked = chunked_tensors(manifest)

    def criterion(name):
        target = tensors[name].get("aliasOf", name)
        if tensors[target].get("dtype") == "bool":
            # Bool outputs are exact 0/1 bytes no matter what produced
            # them: there is no rounding to envelope, so even a tensor
            # downstream of a chunked-fp16 source compares exactly.
            return "exact bool bytes"
        if target in chunked:
            return (f"|device - reference| <= {CHUNKED_ATOL} + "
                    f"{CHUNKED_RTOL} * |reference|")
        return "exact fp16 values; signed zeros equal, NaNs rejected"

    return {name: criterion(name) for name in sorted(reference_outputs)}


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
            "encoder": program["encoder"],
            "taskDescriptors": program["taskDescriptors"],
            "inputs": inputs,
            "outputs": [_binding_record(item, tensors) for item in program["outputs"]],
            "libaneCalls": LibANEAdapter.call_sequence(program),
        })
    plan = {
        "schema": "mil-hwxc.h13-linux-plan.v1",
        "deviceCalls": False,
        "programs": programs,
        "referenceOutputs": {name: len(data)
                             for name, data in sorted(reference_outputs.items())},
        "referenceCriteria": reference_criteria(manifest, reference_outputs),
    }
    if manifest.get("schedule") == "chain":
        plan["schedule"] = "chain"
        plan["tasks"] = manifest["tasks"]
        plan["scratch"] = manifest["scratch"]
    return plan


def _runtime_buffer(dense, binding, tensors):
    sliced = inspect_anec.dense_slice(dense, binding, tensors)
    buffer = bytearray(inspect_anec.convert_tensor(binding, sliced, True))
    if binding.get("broadcast"):
        _broadcast_fill(buffer, binding)
    return bytes(buffer)


def _broadcast_fill(buffer: bytearray, binding) -> None:
    """A declared broadcast binding carries a smaller source operand; the
    decoded row reads the surface full-width, so the host repeats the
    source value across every lane before dispatch (exact: a copy, no
    arithmetic). The source must be one element."""
    source = binding["logicalBytes"]
    if source not in (1, 2):
        raise ValueError("declared broadcast source must be one element")
    element = bytes(buffer[:source])
    width, row = binding["nchw"][3], binding["nchw"][5]
    element_size = source
    total = 0
    offset = 0
    while offset + element_size <= len(buffer):
        buffer[offset:offset + element_size] = element
        total += 1
        offset = (total // width) * row + (total % width) * element_size


def _constant_buffer(program, binding):
    dense = bytes.fromhex(program["constantInputs"][binding["name"]])
    return bytes(inspect_anec.convert_tensor(
        binding, dense[:binding["logicalBytes"]], True))


def _intermediate_buffer(binding, tensors, regions):
    """Builds one consumer surface from its producers' surfaces.

    Producers and consumers can use different physical layouts -- a parity
    matvec writes dense rows while an elementwise program reads 64-byte lanes
    -- so composition goes through dense values at the intermediate's own
    element size (one byte bool, two fp16; a producer of a different dtype
    has no meaning and is refused).
    """
    _, offset, count, _ = inspect_anec.binding_interval(binding, tensors)
    element = _element_bytes(tensors[binding["name"]])
    dense = bytearray(count * element)
    covered = [False] * count
    for produced_binding, data in regions:
        produced_name, produced_offset, produced_count, _ = \
            inspect_anec.binding_interval(produced_binding, tensors)
        produced_element = _element_bytes(tensors[produced_name])
        if produced_element != element:
            raise ValueError(
                f"intermediate {binding['name']} producer {produced_name} "
                "carries a different element size")
        produced = inspect_anec.convert_tensor(produced_binding, data, False)
        start = max(offset, produced_offset)
        end = min(offset + count, produced_offset + produced_count)
        for element_index in range(start, end):
            source = (element_index - produced_offset) * element
            destination = (element_index - offset) * element
            dense[destination:destination + element] = \
                produced[source:source + element]
            covered[element_index - offset] = True
    if not all(covered):
        raise ValueError(f"intermediate {binding['name']} lacks a produced logical range")
    return bytes(inspect_anec.convert_tensor(binding, bytes(dense), True))


def _element_bytes(tensor):
    """Bytes per element for a manifest tensor record: one for a bool
    surface, two for fp16."""
    return 1 if tensor.get("dtype") == "bool" else 2


def _element_label(tensor):
    """Human label for a manifest tensor's element type, used in the
    success line and the published pass criteria."""
    return "bool" if tensor.get("dtype") == "bool" else "fp16"


def _unpack_outputs(manifest, regions, names):
    tensors = manifest["tensors"]
    dense = {}
    for name in names:
        target = tensors[name].get("aliasOf", name)
        target_tensor = tensors[target]
        element = _element_bytes(target_tensor)
        result = bytearray(target_tensor["logicalBytes"])
        written = [False] * (target_tensor["logicalBytes"] // element)
        for binding, data in regions.get(target, []):
            _, offset, count, _ = inspect_anec.binding_interval(binding, tensors)
            values = inspect_anec.convert_tensor(binding, data, False)
            result[offset * element:(offset + count) * element] = values
            written[offset:offset + count] = [True] * count
        if not all(written):
            raise ValueError(f"output {name} lacks a produced logical range")
        dense[name] = bytes(result)
    return dense


def chunked_tensors(manifest):
    """Tensors whose device values carry chunked-fp16 rounding: the chunked
    reductions themselves and everything computed from one. A residual add
    downstream of a 768-element reduction inherits that extra rounding per
    512-element chunk, so comparing it for exact equality would report a
    device failure for arithmetic the compiler chose on purpose."""
    tensors = manifest["tensors"]
    chunked = {name for name, tensor in tensors.items()
               if tensor.get("accumulation") == "chunked-fp16"}
    for index in manifest["dispatchPlan"]:
        program = manifest["programs"][index]
        sources = {binding["name"] for binding in program["inputs"]}
        if sources & chunked:
            chunked.update(binding["name"] for binding in program["outputs"])
    return chunked


def _package_contract(package, mil, model_root, inputs, outputs):
    """Load the package and settle the whole request before any device work:
    requested input names, requested output names, input byte counts, and the
    MIL's returns must all agree with the manifest."""
    manifest, _ = inspect_anec.load_package(package)
    tensors = manifest["tensors"]
    input_names = {name for name, tensor in tensors.items()
                   if tensor["role"] == "input" and "aliasOf" not in tensor}
    output_names = _logical_outputs(tensors)
    if set(inputs) != input_names:
        raise ValueError(f"inputs must be exactly {', '.join(sorted(input_names))}")
    if set(outputs) != output_names:
        raise ValueError(f"outputs must be exactly {', '.join(sorted(output_names))}")
    dense_inputs = {name: path.read_bytes() for name, path in inputs.items()}
    for name, data in dense_inputs.items():
        if len(data) != tensors[name]["logicalBytes"]:
            raise ValueError(f"input {name} byte count differs from its manifest tensor")
    expected = evaluate(Path(mil).read_text(), model_root, dense_inputs)
    if set(expected) != output_names:
        raise ValueError("MIL returns differ from manifest output tensors")
    return manifest, tensors, output_names, dense_inputs, expected


def _program_inputs(manifest, tensors, dense_inputs, intermediate_regions,
                    program_index):
    """Compose one program's input surfaces from the request and the
    intermediates produced so far."""
    program = manifest["programs"][program_index]
    input_buffers = []
    for binding in program["inputs"]:
        name = binding["name"]
        if binding.get("binding") == "constant":
            data = _constant_buffer(program, binding)
        elif tensors[name]["role"] == "intermediate":
            data = _intermediate_buffer(binding, tensors, intermediate_regions.get(name, []))
            if binding.get("broadcast"):
                filled = bytearray(data)
                _broadcast_fill(filled, binding)
                data = bytes(filled)
        else:
            data = _runtime_buffer(dense_inputs[name], binding, tensors)
        input_buffers.append(data)
    return program, input_buffers


def _collect_outputs(tensors, program, outputs, intermediate_regions, output_regions):
    for binding, data in zip(program["outputs"], outputs):
        name, _, _, _ = inspect_anec.binding_interval(binding, tensors)
        if tensors[name]["role"] == "intermediate":
            intermediate_regions.setdefault(name, []).append((binding, data))
        else:
            output_regions.setdefault(name, []).append((binding, data))


def _qualification_deadline(adapter, deadline_seconds, now):
    if deadline_seconds is None:
        if adapter is not None:
            raise ValueError(
                "device execution requires --deadline-seconds with a positive finite value")
        return None
    if not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
        raise ValueError("--deadline-seconds must be a positive finite number")
    return now() + deadline_seconds


def _check_deadline(deadline, deadline_seconds, program_index, now):
    if deadline is not None and now() >= deadline:
        raise ValueError(
            f"whole-run qualification deadline of {deadline_seconds}s exceeded "
            f"before program {program_index}; no next-program work was allocated "
            "or submitted. This deadline cannot interrupt or recover an in-flight "
            "kernel submission")


def _host_boundary_convert(convert, raw):
    """The host half of a manifest-declared boundary conversion: widen the
    read-back source surface to the logical output dtype. fp16 -> fp32 and
    bool 0/1 -> int32 are lossless, so the comparison after conversion is
    exact."""
    if convert == "fp32":
        halves = struct.unpack(f"<{len(raw) // 2}e", raw)
        return struct.pack(f"<{len(halves)}f", *halves)
    if convert == "int32":
        flags = struct.unpack(f"<{len(raw)}B", raw)
        return struct.pack(f"<{len(flags)}i", *flags)
    raise ValueError(f"unsupported host boundary conversion {convert!r}")


def envelope_tensors(manifest):
    """Tensors computed downstream of a statistics row (layer_norm): the
    row's fp16 statistics rounding propagates elementwise, so these compare
    against the pre-declared fp16 envelope instead of bit-exactness."""
    tensors = manifest["tensors"]
    return {name for name, tensor in tensors.items()
            if tensor.get("comparison") == "fp16-envelope"
            or tensors.get(tensor.get("aliasOf", name), {}).get("comparison")
            == "fp16-envelope"}


def _check_outputs(manifest, tensors, output_names, actual, expected,
                   failure_directory=None):
    chunked = chunked_tensors(manifest)
    envelope = envelope_tensors(manifest)
    for name in output_names:
        target = tensors[name].get("aliasOf", name)
        convert = tensors[name].get("hostConvert")
        if convert:
            converted = _host_boundary_convert(convert, actual[name])
            if converted != expected[name]:
                raise ValueError(
                    f"{name}: converted device output differs from the exact "
                    f"reference ({convert} boundary conversion)")
            actual[name] = converted
            continue
        if name in envelope or target in envelope:
            dv = struct.unpack(f"<{len(actual[name]) // 2}e", actual[name])
            rv = struct.unpack(f"<{len(expected[name]) // 2}e", expected[name])
            if any(abs(a - b) > 0.02 + 0.02 * abs(b) for a, b in zip(dv, rv)):
                raise ValueError(
                    f"{name}: device output exceeds the fp16 envelope "
                    f"(0.02 + 0.02 * |reference|)")
            continue
        if tensors[target].get("dtype") == "bool":
            if actual[name] != expected[name]:
                raise ValueError(
                    f"{name}: device bool output differs from the exact "
                    f"reference")
            continue
        try:
            compare_fp16(actual[name], expected[name], target in chunked)
        except ValueError as error:
            if failure_directory is None:
                raise
            failed = failure_directory / f"{name}.failed.fp16"
            failed.write_bytes(actual[name])
            raise ValueError(f"{error}; preserved {failed}") from None


def run_package(package, mil, model_root, inputs, outputs, adapter,
                deadline_seconds=None, now=time.monotonic):
    deadline = _qualification_deadline(adapter, deadline_seconds, now)
    manifest, tensors, output_names, dense_inputs, expected = _package_contract(
        package, mil, model_root, inputs, outputs)
    if adapter is None:
        return manifest, expected, build_plan(manifest, expected)

    package = Path(package).resolve()
    intermediate_regions = {}
    output_regions = {}
    for program_index in manifest["dispatchPlan"]:
        _check_deadline(deadline, deadline_seconds, program_index, now)
        program, input_buffers = _program_inputs(
            manifest, tensors, dense_inputs, intermediate_regions, program_index)
        anec = inspect_anec.local_file(package, program["file"], program["bytes"])
        data = adapter.execute(
            anec, input_buffers,
            [binding["allocationBytes"] for binding in program["outputs"]])
        _collect_outputs(tensors, program, data, intermediate_regions, output_regions)
    actual = _unpack_outputs(manifest, output_regions, output_names)
    _check_outputs(manifest, tensors, output_names, actual, expected, package)
    return manifest, actual, None


def _timing_summary(samples):
    return {
        "minSeconds": min(samples),
        "medianSeconds": statistics.median(samples),
        "meanSeconds": statistics.fmean(samples),
        "maxSeconds": max(samples),
        "samples": samples,
    }


def benchmark_package(package, mil, model_root, inputs, outputs, adapter, warmup,
                      iterations, deadline_seconds=None, now=time.monotonic):
    """Correctness-gated native timing on prepared programs.

    Reference evaluation happens once before any timing, and every program is
    prepared (pyane_init, size validation, buffer allocation) before the first
    measured iteration -- even with warmup 0 -- so initialization lands in
    setupSeconds and never inside a timed window. Prepared programs stay open
    for the whole run; measured iterations repeat only transfers, submission,
    and readback on refilled buffers. Every iteration -- warmup included --
    is validated against the reference after its timed window, the report is
    only returned when every output of every iteration meets the declared
    criteria, and every handle is closed on every exit path.
    """
    manifest, tensors, output_names, dense_inputs, expected = _package_contract(
        package, mil, model_root, inputs, outputs)
    deadline = _qualification_deadline(adapter, deadline_seconds, now)
    package = Path(package).resolve()
    records = {}
    package_samples = []
    final_actual = None
    try:
        for program_index in manifest["dispatchPlan"]:
            _check_deadline(deadline, deadline_seconds, program_index, now)
            program = manifest["programs"][program_index]
            placeholders = []
            for binding in program["inputs"]:
                if binding.get("binding") == "constant":
                    placeholders.append(_constant_buffer(program, binding))
                else:
                    placeholders.append(bytes(binding["allocationBytes"]))
            anec = inspect_anec.local_file(package, program["file"], program["bytes"])
            sizes = [binding["allocationBytes"] for binding in program["outputs"]]
            setup_started = time.perf_counter()
            prepared = PreparedProgram(adapter, anec, placeholders, sizes)
            setup = time.perf_counter() - setup_started
            records[program_index] = {
                "program": prepared,
                "index": program_index,
                "file": program["file"],
                "operation": program["operation"],
                "encoder": program["encoder"],
                "setupSeconds": setup,
            }
        for iteration in range(warmup + iterations):
            intermediate_regions = {}
            output_regions = {}
            package_started = None
            package_finished = None
            for program_index in manifest["dispatchPlan"]:
                _check_deadline(deadline, deadline_seconds, program_index, now)
                record = records[program_index]
                program, input_buffers = _program_inputs(
                    manifest, tensors, dense_inputs, intermediate_regions,
                    program_index)
                record["program"].refill(input_buffers)
                started = time.perf_counter()
                record["program"].transfer_in()
                sent = time.perf_counter()
                record["program"].submit()
                executed = time.perf_counter()
                data = record["program"].transfer_out()
                finished = time.perf_counter()
                if package_started is None:
                    package_started = started
                package_finished = finished
                _collect_outputs(tensors, program, data,
                                 intermediate_regions, output_regions)
                if iteration >= warmup:
                    record.setdefault("transferSamples", []).append(sent - started)
                    record.setdefault("submitSamples", []).append(executed - sent)
                    record.setdefault("readbackSamples", []).append(finished - executed)
                    record.setdefault("totalSamples", []).append(finished - started)
            if iteration >= warmup:
                package_samples.append(package_finished - package_started)
            actual = _unpack_outputs(manifest, output_regions, output_names)
            _check_outputs(manifest, tensors, output_names, actual, expected)
            final_actual = actual
        report = {
            "schema": "mil-hwxc.h13-linux-benchmark.v1",
            "package": str(package),
            "warmup": warmup,
            "iterations": iterations,
            "clock": "time.perf_counter",
            "host": {"kernel": platform.release(), "machine": platform.machine()},
            "timingScope": ("Wall time around libane calls via time.perf_counter. "
                            "Every program is prepared (pyane_init, size validation, "
                            "buffer allocation) before the first measured iteration and "
                            "reported as setupSeconds, so no timed window contains "
                            "initialization. A program's totalSeconds spans that "
                            "program's first transfer through its last readback. "
                            "endToEnd spans the first program's first transfer through "
                            "the last program's last readback per iteration and "
                            "therefore includes Python intermediate composition "
                            "between programs. Final output unpacking and correctness "
                            "checks are outside every timed window. Reference "
                            "evaluation is also outside every timed window."),
            "referenceCriteria": reference_criteria(manifest, expected),
            "programs": [],
        }
        for program_index in manifest["dispatchPlan"]:
            record = records[program_index]
            report["programs"].append({
                "index": record["index"],
                "file": record["file"],
                "operation": record["operation"],
                "encoder": record["encoder"],
                "setupSeconds": record["setupSeconds"],
                "warmup": warmup,
                "iterations": iterations,
                "transferSeconds": _timing_summary(record["transferSamples"]),
                "submitSeconds": _timing_summary(record["submitSamples"]),
                "readbackSeconds": _timing_summary(record["readbackSamples"]),
                "totalSeconds": _timing_summary(record["totalSamples"]),
            })
        report["endToEnd"] = _timing_summary(package_samples)
        report["endToEnd"]["note"] = ("First program's first transfer through the last "
                                      "program's last readback per iteration; includes "
                                      "Python composition between programs, never "
                                      "program setup.")
        return manifest, final_actual, report
    finally:
        _close_all([record["program"] for record in records.values()],
                   sys.exc_info()[1])


def _close_all(programs, primary):
    """Close every prepared handle. The first cleanup failure is raised only
    after every handle has been closed: the primary propagates with the
    cleanup as its cause, or the cleanup propagates alone."""
    first_cleanup = None
    for program in programs:
        try:
            program.close()
        except BaseException as cleanup_error:
            if first_cleanup is None:
                first_cleanup = cleanup_error
    if first_cleanup is not None:
        if primary is None:
            raise first_cleanup
        raise primary from first_cleanup


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
    parser.add_argument("--deadline-seconds", type=float, default=None,
                        help="required for device execution: positive finite "
                             "whole-device-phase budget, separate from the kernel "
                             "per-submit timeout; checked before each program and "
                             "cannot interrupt an in-flight submission")
    parser.add_argument("--warmup", type=int, default=None, metavar="N",
                        help="validated, untimed warmup executions before a "
                             "benchmark (requires --benchmark-json)")
    parser.add_argument("--iterations", type=int, default=None, metavar="N",
                        help="timed benchmark iterations (requires --benchmark-json)")
    parser.add_argument("--benchmark-json", type=Path, metavar="PATH",
                        help="write a correctness-gated native benchmark report; "
                             "refused with --dry-run")
    args = parser.parse_args()
    benchmarking = args.benchmark_json is not None
    if args.deadline_seconds is not None and (
            not math.isfinite(args.deadline_seconds) or args.deadline_seconds <= 0):
        parser.error("--deadline-seconds must be a positive finite number")
    if args.dry_run and (benchmarking or args.warmup is not None
                         or args.iterations is not None):
        parser.error("--dry-run never reports native timings; drop "
                     "--benchmark-json/--warmup/--iterations")
    if not benchmarking and (args.warmup is not None or args.iterations is not None):
        parser.error("--warmup/--iterations require --benchmark-json")
    warmup = 3 if args.warmup is None else args.warmup
    iterations = 10 if args.iterations is None else args.iterations
    if warmup < 0:
        parser.error("--warmup must be >= 0")
    if iterations < 1:
        parser.error("--iterations must be >= 1")
    if not args.dry_run and args.deadline_seconds is None:
        parser.error(
            "device execution requires --deadline-seconds with a positive finite value")
    try:
        inputs, outputs = _paths(args.input), _paths(args.output)
        adapter = None if args.dry_run else LibANEAdapter(args.libane_library, args.device)
        if benchmarking:
            manifest, result, report = benchmark_package(
                args.package, args.mil, args.model_root, inputs, outputs, adapter,
                warmup, iterations, deadline_seconds=args.deadline_seconds)
        else:
            manifest, result, plan = run_package(
                args.package, args.mil, args.model_root, inputs, outputs, adapter,
                deadline_seconds=args.deadline_seconds)
        if args.dry_run:
            print(json.dumps(plan, indent=2, sort_keys=True))
            return
        tensors = manifest["tensors"]
        for name, path in outputs.items():
            label = _element_label(tensors[name])
            path.write_bytes(result[name])
            print(f"PASS {name}: {len(result[name])} {label} bytes match the reference; wrote {path}")
        if benchmarking:
            args.benchmark_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
            for program in report["programs"]:
                print(f"BENCH {program['file']}: setup {program['setupSeconds']:.6f}s, "
                      f"submit median {program['submitSeconds']['medianSeconds']:.6f}s "
                      f"over {program['iterations']} iterations")
            print(f"WROTE {args.benchmark_json}")
    except (OSError, ValueError, RuntimeError, struct.error) as error:
        parser.exit(1, f"H13 Linux runner error: {error}\n")


if __name__ == "__main__":
    main()
