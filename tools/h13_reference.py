#!/usr/bin/env python3
"""Numpy-free fp16 reference interpreter for the MIL subset accepted by H13.

Matmul products accumulate in float32 and round once to fp16. H13 reductions
larger than 512 elements use chunked-fp16 partial sums, so device results may
have one extra fp16 rounding per chunk.
"""

import argparse
import ast
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path


_TOKEN = re.compile(
    r"\s+|//[^\n]*|/\*.*?\*/|->|"
    r'"(?:\\.|[^"\\])*"|'
    r"(?:0[xX][0-9a-fA-F]+|[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)|"
    r"[A-Za-z_$][A-Za-z0-9_.$]*|[][(){}<>,;=]",
    re.DOTALL,
)


@dataclass(frozen=True)
class MILType:
    dtype: str
    shape: tuple = ()


@dataclass(frozen=True)
class Call:
    name: str
    call_type: MILType
    positional: tuple
    named: dict


@dataclass(frozen=True)
class Ref:
    name: str


@dataclass(frozen=True)
class Tensor:
    dtype: str
    shape: tuple
    values: tuple


@dataclass(frozen=True)
class Operation:
    result_type: MILType
    result: str
    name: str
    arguments: dict
    attributes: dict


def fp16(value):
    value = float(value)
    try:
        return struct.unpack("<e", struct.pack("<e", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def fp32(value):
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def add_fp16(left, right):
    return fp16(float(left) + float(right))


def dot_fp32(left, right):
    if len(left) != len(right):
        raise ValueError("matmul reduction dimensions differ")
    total = 0.0
    for x, y in zip(left, right):
        total = fp32(total + fp32(float(x) * float(y)))
    return fp16(total)


def encode_fp16(items):
    return b"".join(struct.pack("<e", value) for value in items)


def decode_fp16(data):
    if len(data) % 2:
        raise ValueError("fp16 data must have an even byte count")
    return tuple(value[0] for value in struct.iter_unpack("<e", data))


def _tokenize(text):
    tokens = []
    offset = 0
    while offset < len(text):
        match = _TOKEN.match(text, offset)
        if not match:
            raise ValueError(f"unsupported MIL token at byte {offset}")
        token = match.group(0)
        offset = match.end()
        if not token.isspace() and not token.startswith(("//", "/*")):
            tokens.append(token)
    return tokens


class Parser:
    def __init__(self, text):
        self.tokens = _tokenize(text)
        self.index = 0

    def peek(self, token=None):
        if self.index >= len(self.tokens):
            return False if token is not None else None
        return self.tokens[self.index] == token if token is not None else self.tokens[self.index]

    def take(self, token=None):
        value = self.peek()
        if value is None or token is not None and value != token:
            raise ValueError(f"expected {token or 'token'}, got {value or 'end of input'}")
        self.index += 1
        return value

    def parse_type(self):
        name = self.take()
        if not re.match(r"[A-Za-z_$]", name):
            raise ValueError(f"expected MIL type, got {name}")
        dtype = name
        shape = ()
        if self.peek("<"):
            self.take("<")
            depth = 1
            inner = []
            while depth:
                token = self.take()
                if token == "<":
                    depth += 1
                elif token == ">":
                    depth -= 1
                    if not depth:
                        break
                inner.append(token)
            if name == "tensor":
                if not inner:
                    raise ValueError("tensor type lacks an element type")
                dtype = inner[0]
                try:
                    opening = inner.index("[")
                    closing = len(inner) - 1 - inner[::-1].index("]")
                except ValueError as error:
                    raise ValueError("tensor type lacks a shape") from error
                shape = tuple(int(item, 0) for item in inner[opening + 1:closing] if item != ",")
        return MILType(dtype, shape)

    def parse_arguments(self, closing):
        positional = []
        named = {}
        while not self.peek(closing):
            if (self.index + 1 < len(self.tokens) and
                    re.match(r"[A-Za-z_$]", self.tokens[self.index]) and
                    self.tokens[self.index + 1] == "="):
                name = self.take()
                self.take("=")
                named[name] = self.parse_expression()
            else:
                positional.append(self.parse_expression())
            if not self.peek(","):
                break
            self.take(",")
        self.take(closing)
        return tuple(positional), named

    def parse_expression(self):
        if self.peek("["):
            self.take("[")
            positional, named = self.parse_arguments("]")
            if named:
                raise ValueError("MIL list cannot contain named values")
            return list(positional)
        token = self.peek()
        if token is None:
            raise ValueError("unexpected end of MIL expression")
        if token.startswith('"'):
            self.take()
            return ast.literal_eval(token)
        if re.fullmatch(r"0[xX][0-9a-fA-F]+", token):
            self.take()
            return int(token, 0)
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token):
            self.take()
            return float(token) if any(character in token for character in ".eE") else int(token)
        if token in ("true", "false"):
            self.take()
            return token == "true"
        expression_type = self.parse_type()
        if not self.peek("("):
            if expression_type.shape or expression_type.dtype != token:
                raise ValueError("a constructed MIL type requires a value")
            return Ref(token)
        self.take("(")
        positional, named = self.parse_arguments(")")
        return Call(token, expression_type, positional, named)

    def parse_operation(self):
        result_type = self.parse_type()
        result = self.take()
        self.take("=")
        name = self.take()
        self.take("(")
        _, arguments = self.parse_arguments(")")
        attributes = {}
        if self.peek("["):
            self.take("[")
            _, attributes = self.parse_arguments("]")
        self.take(";")
        return Operation(result_type, result, name, arguments, attributes)

    def parse(self):
        try:
            self.index = self.tokens.index("func")
        except ValueError as error:
            raise ValueError("MIL program has no function") from error
        self.take("func")
        self.take()
        self.take("<")
        while not self.peek(">"):
            self.take()
        self.take(">")
        self.take("(")
        parameters = []
        while not self.peek(")"):
            parameters.append((self.parse_type(), self.take()))
            if not self.peek(","):
                break
            self.take(",")
        self.take(")")
        self.take("{")
        operations = []
        while not self.peek("}"):
            operations.append(self.parse_operation())
        self.take("}")
        self.take("->")
        self.take("(")
        returns = []
        while not self.peek(")"):
            returns.append(self.take())
            if not self.peek(","):
                break
            self.take(",")
        self.take(")")
        self.take(";")
        return parameters, operations, returns


def _tensor(value, expected_type=None):
    if isinstance(value, Tensor):
        result = value
    elif isinstance(value, (bytes, bytearray, memoryview)):
        if expected_type is None or expected_type.dtype != "fp16":
            raise ValueError("raw input requires an fp16 tensor type")
        result = Tensor("fp16", expected_type.shape, decode_fp16(bytes(value)))
    elif isinstance(value, (list, tuple)):
        if expected_type is None:
            raise ValueError("input sequences require a tensor type")
        converted = tuple(fp16(item) for item in value) if expected_type.dtype == "fp16" else tuple(value)
        result = Tensor(expected_type.dtype, expected_type.shape, converted)
    else:
        raise ValueError("tensor value must be dense bytes or a sequence")
    if expected_type is not None:
        if result.dtype != expected_type.dtype or result.shape != expected_type.shape:
            raise ValueError("tensor value does not match its MIL type")
        if len(result.values) != math.prod(expected_type.shape):
            raise ValueError("tensor byte count does not match its MIL shape")
    return result


def _flatten(items):
    result = []
    for item in items:
        if isinstance(item, list):
            result.extend(_flatten(item))
        elif isinstance(item, Tensor):
            result.extend(item.values)
        else:
            result.append(item)
    return result


def _blob(call, result_type, model_root):
    path = _resolve(call.named.get("path"), {}, model_root)
    offset = _resolve(call.named.get("offset"), {}, model_root)
    if not isinstance(path, str) or not path.startswith("@model_path/") or not isinstance(offset, int):
        raise ValueError("BLOBFILE requires @model_path and an integer offset")
    root = Path(model_root).resolve()
    candidate = (root / path[len("@model_path/"):]).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError("BLOBFILE path escapes model root") from error
    data = candidate.read_bytes()
    if offset < 0 or offset + 24 > len(data):
        raise ValueError("BLOBFILE chunk header exceeds the file")
    magic, _, payload_length, payload_offset = struct.unpack_from("<IIQQ", data, offset)
    expected = math.prod(result_type.shape) * (2 if result_type.dtype == "fp16" else 4)
    if magic != 0xDEADBEEF or payload_length < expected or payload_offset + expected > len(data):
        raise ValueError("BLOBFILE subheader has invalid magic, size, or payload offset")
    payload = data[payload_offset:payload_offset + expected]
    if result_type.dtype == "fp16":
        values = decode_fp16(payload)
    elif result_type.dtype in ("int32", "uint32"):
        code = "i" if result_type.dtype == "int32" else "I"
        values = struct.unpack(f"<{math.prod(result_type.shape)}{code}", payload)
    else:
        raise ValueError(f"unsupported BLOBFILE element type {result_type.dtype}")
    return Tensor(result_type.dtype, result_type.shape, tuple(values))


def _resolve(expression, environment, model_root):
    if isinstance(expression, Ref):
        if expression.name not in environment:
            raise ValueError(f"unknown MIL value {expression.name}")
        return environment[expression.name]
    if isinstance(expression, list):
        return [_resolve(item, environment, model_root) for item in expression]
    if not isinstance(expression, Call):
        return expression
    arguments = [_resolve(item, environment, model_root) for item in expression.positional]
    if expression.name == "BLOBFILE":
        return expression
    if expression.name == "string":
        return str(arguments[0])
    if expression.name in ("int32", "uint32", "uint64"):
        return int(arguments[0])
    if expression.name == "bool":
        return bool(arguments[0])
    if expression.name == "fp16":
        return fp16(arguments[0])
    if expression.name == "fp32":
        return fp32(arguments[0])
    if expression.name == "tensor":
        if len(expression.positional) != 1:
            raise ValueError("tensor literal requires one payload")
        payload = expression.positional[0]
        if isinstance(payload, Call) and payload.name == "BLOBFILE":
            return _blob(payload, expression.call_type, model_root)
        resolved = _flatten([_resolve(payload, environment, model_root)])
        if len(resolved) != math.prod(expression.call_type.shape):
            raise ValueError("tensor literal element count does not match its shape")
        if expression.call_type.dtype == "fp16":
            resolved = [fp16(item) for item in resolved]
        elif expression.call_type.dtype.startswith(("int", "uint")):
            resolved = [int(item) for item in resolved]
        else:
            raise ValueError(f"unsupported tensor element type {expression.call_type.dtype}")
        return Tensor(expression.call_type.dtype, expression.call_type.shape, tuple(resolved))
    raise ValueError(f"unsupported MIL constructor {expression.name}")


def _broadcast_shape(left, right):
    if not left:
        return tuple(right)
    if not right:
        return tuple(left)
    result = []
    for a, b in zip(reversed(left), reversed(right)):
        if a != b and a != 1 and b != 1:
            raise ValueError("elementwise tensor shapes do not broadcast")
        result.append(max(a, b))
    result.extend(reversed(left[:-len(right)] if len(left) > len(right) else right[:-len(left)]))
    return tuple(reversed(result))


def _broadcast_values(value, shape):
    if not isinstance(value, Tensor):
        return [value] * math.prod(shape)
    if value.shape == shape:
        return list(value.values)
    padded = (1,) * (len(shape) - len(value.shape)) + value.shape
    if any(source not in (1, target) for source, target in zip(padded, shape)):
        raise ValueError("elementwise tensor shapes do not broadcast")
    source_strides = []
    stride = 1
    for dimension in reversed(padded):
        source_strides.append(stride)
        stride *= dimension
    source_strides.reverse()
    result = []
    for flat in range(math.prod(shape)):
        remainder = flat
        source = 0
        target_stride = math.prod(shape)
        for dimension, source_dimension, source_stride in zip(shape, padded, source_strides):
            target_stride //= dimension
            coordinate, remainder = divmod(remainder, target_stride)
            if source_dimension != 1:
                source += coordinate * source_stride
        result.append(value.values[source])
    return result


def _binary(name, left, right, result_type):
    left_shape = left.shape if isinstance(left, Tensor) else ()
    right_shape = right.shape if isinstance(right, Tensor) else ()
    shape = _broadcast_shape(left_shape, right_shape)
    if shape != result_type.shape:
        raise ValueError(f"{name} result shape differs from its declaration")
    operations = {
        "add": lambda a, b: a + b,
        "mul": lambda a, b: a * b,
        "maximum": max,
        "minimum": min,
        "sub": lambda a, b: a - b,
        "real_div": lambda a, b: a / b,
    }
    values = [fp16(operations[name](a, b))
              for a, b in zip(_broadcast_values(left, shape), _broadcast_values(right, shape))]
    return Tensor("fp16", shape, tuple(values))


def _matmul(left, right, transpose_x, transpose_y, result_type):
    left = _tensor(left)
    right = _tensor(right)
    if left.dtype != "fp16" or right.dtype != "fp16" or len(right.shape) != 2:
        raise ValueError("matmul requires fp16 tensors and rank-2 y")
    if len(left.shape) == 1:
        if transpose_x:
            raise ValueError("transpose_x is invalid for rank-one x")
        leading, matrix_rows, reduction = (), 1, left.shape[0]
        expected_shape = (right.shape[0] if transpose_y else right.shape[1],)
    elif transpose_x:
        leading = left.shape[:-2]
        matrix_rows, reduction = left.shape[-1], left.shape[-2]
        expected_shape = leading + (matrix_rows, right.shape[0] if transpose_y else right.shape[1])
    else:
        leading, matrix_rows, reduction = left.shape[:-1], 1, left.shape[-1]
        expected_shape = leading + (right.shape[0] if transpose_y else right.shape[1],)
    columns = right.shape[0] if transpose_y else right.shape[1]
    weight_reduction = right.shape[1] if transpose_y else right.shape[0]
    if reduction != weight_reduction or result_type.shape != expected_shape:
        raise ValueError("matmul shapes differ from its declaration")
    row_count = math.prod(leading) * matrix_rows
    result = []
    for row in range(row_count):
        if transpose_x:
            batch, local_row = divmod(row, matrix_rows)
            base = batch * reduction * matrix_rows
            lhs = tuple(left.values[base + index * matrix_rows + local_row]
                        for index in range(reduction))
        else:
            lhs = left.values[row * reduction:(row + 1) * reduction]
        for column in range(columns):
            if transpose_y:
                rhs = right.values[column * reduction:(column + 1) * reduction]
            else:
                rhs = tuple(right.values[index * columns + column] for index in range(reduction))
            result.append(dot_fp32(lhs, rhs))
    return Tensor("fp16", result_type.shape, tuple(result))


def _shape_op(name, value, parameter, result_type):
    value = _tensor(value)
    requested = (tuple(index for index, dimension in enumerate(value.shape) if dimension == 1)
                 if name == "squeeze" and parameter is None else
                 tuple(int(item) for item in _tensor(parameter).values))
    if name == "reshape":
        if requested != result_type.shape:
            raise ValueError("reshape shape differs from its declaration")
    elif name == "squeeze":
        axes = tuple(index if index >= 0 else index + len(value.shape) for index in requested)
        expected = tuple(dimension for index, dimension in enumerate(value.shape) if index not in axes)
        if any(value.shape[index] != 1 for index in axes) or expected != result_type.shape:
            raise ValueError("squeeze axes differ from its declaration")
    else:
        rank = len(value.shape) + len(requested)
        axes = tuple(index if index >= 0 else index + rank for index in requested)
        shape = list(value.shape)
        for index in sorted(axes):
            shape.insert(index, 1)
        if tuple(shape) != result_type.shape:
            raise ValueError("expand_dims axes differ from its declaration")
    if math.prod(value.shape) != math.prod(result_type.shape):
        raise ValueError(f"{name} changes the element count")
    return Tensor(value.dtype, result_type.shape, value.values)


def _execute(operation, environment, model_root):
    arguments = {name: _resolve(value, environment, model_root)
                 for name, value in operation.arguments.items()}
    attributes = {name: _resolve(value, environment, model_root)
                  for name, value in operation.attributes.items()}
    name = operation.name
    if name == "const":
        if "val" not in attributes:
            raise ValueError("const lacks val")
        value = attributes["val"]
        if operation.result_type.shape:
            return _tensor(value, operation.result_type)
        return value
    if name in ("add", "mul", "maximum", "minimum", "sub", "real_div"):
        return _binary(name, arguments["x"], arguments["y"], operation.result_type)
    if name == "relu":
        return _binary("maximum", arguments["x"], 0.0, operation.result_type)
    if name == "clip":
        lower = _binary("maximum", arguments["x"], arguments["alpha"], operation.result_type)
        return _binary("minimum", lower, arguments["beta"], operation.result_type)
    if name == "matmul":
        return _matmul(arguments["x"], arguments["y"], bool(arguments.get("transpose_x", False)),
                       bool(arguments.get("transpose_y", False)), operation.result_type)
    if name == "linear":
        projected_type = MILType("fp16", operation.result_type.shape)
        projected = _matmul(arguments["x"], arguments["weight"], False, True, projected_type)
        return projected if "bias" not in arguments else _binary(
            "add", projected, arguments["bias"], operation.result_type)
    if name in ("reshape", "squeeze", "expand_dims"):
        parameter = arguments.get("shape" if name == "reshape" else "axes")
        return _shape_op(name, arguments["x"], parameter, operation.result_type)
    raise ValueError(f"unsupported H13 MIL operation {name}")


def evaluate(mil_text, model_root, inputs):
    """Evaluate one MIL function and return its named outputs as dense fp16 bytes."""
    parameters, operations, returns = Parser(mil_text).parse()
    expected_names = {name for _, name in parameters}
    if set(inputs) != expected_names:
        raise ValueError(f"inputs must be exactly {', '.join(sorted(expected_names))}")
    environment = {name: _tensor(inputs[name], value_type)
                   for value_type, name in parameters}
    for operation in operations:
        if operation.result in environment:
            raise ValueError(f"duplicate MIL value {operation.result}")
        environment[operation.result] = _execute(operation, environment, Path(model_root))
    outputs = {}
    for name in returns:
        value = environment.get(name)
        if not isinstance(value, Tensor) or value.dtype != "fp16":
            raise ValueError("H13 reference outputs must be fp16 tensors")
        outputs[name] = encode_fp16(value.values)
    return outputs


def _bindings(items):
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError("bindings must use NAME=PATH")
        name, path = item.split("=", 1)
        if not name or name in result:
            raise ValueError("binding names must be unique and non-empty")
        result[name] = Path(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mil", type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--input", action="append", default=[], metavar="NAME=DENSE_FP16")
    parser.add_argument("--output", action="append", default=[], metavar="NAME=DENSE_FP16")
    args = parser.parse_args()
    try:
        input_paths = _bindings(args.input)
        output_paths = _bindings(args.output)
        outputs = evaluate(args.mil.read_text(), args.model_root,
                           {name: path.read_bytes() for name, path in input_paths.items()})
        if set(output_paths) != set(outputs):
            raise ValueError(f"outputs must be exactly {', '.join(sorted(outputs))}")
        for name, path in output_paths.items():
            path.write_bytes(outputs[name])
            print(f"wrote {len(outputs[name])} fp16 bytes to {path}")
    except (OSError, ValueError, struct.error) as error:
        parser.exit(1, f"H13 reference error: {error}\n")


if __name__ == "__main__":
    main()
