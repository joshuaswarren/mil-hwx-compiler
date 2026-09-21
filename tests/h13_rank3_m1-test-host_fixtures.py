#!/usr/bin/env python3
"""Generate the four rank3 device-qualification fixture packages and
their raw input buffers on m1-test-host.

Fixtures:
 A. GLU select promoted: rank-0 fp16 -inf fill (var_13 @33558272) via the
    runtime-a select row at [1, 1024, 375] with a broadcast bool cond.
 B. select rrb runtime-a full: same geometry, all three operands as
    runtime inputs (the captured row byte-matches this form).
 C. cast fp16->bool at [1, 1, 375]: exact zero/nonzero-to-bool mapping.
 D. bool tail-swap transpose at [1, 375, 375]: permutation correctness.

Each fixture writes a MIL, weights.bin (sparse where needed), the named
input raw-fp16 / bool files, and the expected reference outputs so the
hardware runner has everything to execute and compare.
"""
import struct
from pathlib import Path

FIXTURES = Path("/var/tmp/m1-test-host-rank3-fixtures")


def fp16(x):
    return struct.pack("<e", float(x))


def write_sparse(path, records):
    size = max(off for off, _ in records) + 64 + max(len(b) for _, b in records)
    data = bytearray(size)
    for off, blob in records:
        data[off:off + len(blob)] = blob
    path.write_bytes(bytes(data))


def cond_bytes(length):
    return bytes((i % 3 == 0) for i in range(length))


def b_bytes(width, height, batch):
    return b"".join(struct.pack("<e", (i * 17 + j * 13 + batch * 5) % 19)
                    for batch in range(batch)
                    for j in range(height) for i in range(width))


def a_bytes(width, height, batch):
    return b"".join(struct.pack("<e", (i + j * 3 + batch * 7) % 11)
                    for batch in range(batch) for j in range(height)
                    for i in range(width))


def mask_bytes(width, height):
    # Asymmetric under transpose (coefficients 7 != 11): a symmetric mask
    # would equal its own tail-swap and could not detect a transpose that
    # never permutes.
    return bytes(((i * 7 + j * 11) % 5 == 0)
                 for j in range(height) for i in range(width))


def write_mil(path, body, result):
    path.write_text(
        "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
        "  func main<ios18>(" + ", ".join(body["args"]) + ") {\n"
        + "\n".join("    " + s for s in body["stmts"])
        + "\n  } -> (" + result + ");\n}\n")


def fixture_A(root):
    d = root / "A_glu_select_promoted"
    d.mkdir(parents=True, exist_ok=True)
    one_neg_inf = struct.pack("<e", float("-inf"))
    write_sparse(d / "weights.bin", [(33558272,
        struct.pack("<IIQQ", 0xDEADBEEF, 1, 2, 33558336)),
        (33558336, one_neg_inf)])
    write_mil(d / "model.mil", {
        "args": ["tensor<fp16, [1, 1024, 375]> b",
                 "tensor<bool, [1, 1, 375]> all_masked_rows_1"],
        "stmts": [
            "tensor<fp16, []> var_13_to_fp16 = const()[name = string(\"var_13_to_fp16\"), "
            "val = tensor<fp16, []>(BLOBFILE(path = string(\"@model_path/weights.bin\"), "
            "offset = uint64(33558272)))];",
            "tensor<fp16, [1, 1024, 375]> input_41_cast_fp16 = select("
            "a = var_13_to_fp16, b = b, cond = all_masked_rows_1)"
            "[name = string(\"out\")];"]},
        "input_41_cast_fp16")
    (d / "b.fp16").write_bytes(b_bytes(375, 1024, 1))
    (d / "cond.bool").write_bytes(cond_bytes(375))


def fixture_B(root):
    d = root / "B_select_runtime_a"
    d.mkdir(parents=True, exist_ok=True)
    write_mil(d / "model.mil", {
        "args": ["tensor<fp16, [1, 1024, 375]> a",
                 "tensor<fp16, [1, 1024, 375]> b",
                 "tensor<bool, [1, 1, 375]> cond"],
        "stmts": [
            "tensor<fp16, [1, 1024, 375]> out = select(a = a, b = b, "
            "cond = cond)[name = string(\"out\")];"]},
        "out")
    (d / "a.fp16").write_bytes(a_bytes(375, 1024, 1))
    (d / "b.fp16").write_bytes(b_bytes(375, 1024, 1))
    (d / "cond.bool").write_bytes(cond_bytes(375))


def fixture_C(root):
    d = root / "C_cast_f16_to_bool"
    d.mkdir(parents=True, exist_ok=True)
    write_mil(d / "model.mil", {
        "args": ["tensor<fp16, [1, 1, 375]> x"],
        "stmts": [
            "tensor<string, []> dt = const()[name = string(\"dt\"), "
            "val = tensor<string, []>(\"bool\")];",
            "tensor<bool, [1, 1, 375]> out = cast(dtype = dt, x = x)"
            "[name = string(\"out\")];"]},
        "out")
    (d / "x.fp16").write_bytes(b"".join(fp16(0.0 if i % 4 == 0 else float(i + 1))
                                     for i in range(375)))


def fixture_D(root):
    d = root / "D_bool_tail_swap_transpose"
    d.mkdir(parents=True, exist_ok=True)
    write_mil(d / "model.mil", {
        "args": ["tensor<bool, [1, 375, 375]> attention_mask_3"],
        "stmts": [
            "tensor<int32, [3]> perm = const()[name = string(\"perm\"), "
            "val = tensor<int32, [3]>([0, 2, 1])];",
            "tensor<bool, [1, 375, 375]> out = transpose(perm = perm, "
            "x = attention_mask_3)[name = string(\"out\")];"]},
        "out")
    (d / "attention_mask_3.bool").write_bytes(mask_bytes(375, 375))


def main():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    fixture_A(FIXTURES)
    fixture_B(FIXTURES)
    fixture_C(FIXTURES)
    fixture_D(FIXTURES)
    for stem in ("A_glu_select_promoted", "B_select_runtime_a",
                 "C_cast_f16_to_bool", "D_bool_tail_swap_transpose"):
        d = FIXTURES / stem
        files = sorted(p.name + " " + str((d / p.name).stat().st_size)
                      for p in d.iterdir())
        print(f"{stem}: {len(files)} files")
        for f in files:
            print(" ", f)


if __name__ == "__main__":
    main()
