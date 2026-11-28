"""Pin the full H18G ANEP that ran the 256-wide matvec on M6."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

PACK_SHA = "32e86745d8ab452939efa72f47aaaeeda18a158b4aa49bb9022c6c651c0971cb"
MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 256]> x) {
    bool f = const()[name = string("f"), val = bool(false)];
    tensor<fp16, [256, 256]> w = const()[name = string("w"), val = tensor<fp16, [256, 256]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 256]> y = matmul(transpose_x = f, transpose_y = f, x = x, y = w)[name = string("y")];
  } -> (y);
}
"""


def write_inputs(root):
    (root / "model.mil").write_text(MIL)
    weights = struct.pack("<65536e", *(
        (2 * (k % 4) + 8 * (n % 8) - 31) / 32
        for k in range(256) for n in range(256)
    ))
    header = bytearray(128)
    struct.pack_into("<II", header, 0, 1, 2)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(weights), 128)
    (root / "weights.bin").write_bytes(header + weights)
    x = [((37 * i + 11) % 257 - 128) / 128 for i in range(256)]
    x[24] = 1 / 128
    (root / "x.fp16").write_bytes(struct.pack("<256e", *x))


def main():
    compiler = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_inputs(root)
        output = root / "out"
        args = [compiler, "--target", "H18G", "--mil", root / "model.mil",
                "--model-root", root, "--output", output, "--input", root / "x.fp16"]
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stderr
        image = (output / "program-0.anep").read_bytes()
        assert hashlib.sha256(image).hexdigest() == PACK_SHA
        manifest = json.loads((output / "manifest.json").read_bytes())
        assert manifest["target"] == "H18G" and manifest["soc"] == "T8152"
        assert manifest["runProg"] == {
            "bufs": 0x30000000, "text": 0x30008000, "text_const": 0x280,
            "kern": 0x30010000, "kern_slot": 1,
        }
        (root / "model.mil").write_text(MIL.replace("transpose_y = f", "transpose_y = true"))
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        assert result.returncode != 0 and "h18g.unsupported-program" in result.stderr, result.stderr
        check_input_emission(compiler, root, image)
    print(f"H18G ANEP matches device-proven SHA-256 {PACK_SHA}")


def h18g_compile(compiler, root, name, input_path=None, target="H18G"):
    args = [compiler, "--target", target, "--mil", root / "model.mil",
            "--model-root", root, "--output", root / name]
    if input_path is not None:
        args += ["--input", input_path]
    return subprocess.run(args, capture_output=True, text=True, check=False)


def input_surface_offset(pack):
    # The io records follow the six section records at offset 0x20; the
    # inputBuffer (io index 1, record index 7) names its surface offset in
    # record field 4 (16 bytes into the 32-byte record).
    return struct.unpack_from("<I", pack, 0x20 + 7 * 0x20 + 16)[0]


def check_input_emission(compiler, root, image):
    # main() left model.mil in the transpose_y refusal form; restore it.
    (root / "model.mil").write_text(MIL)
    # The --input fp16 vector is embedded verbatim at the input surface.
    offset = input_surface_offset(image)
    assert image[offset:offset + 512] == (root / "x.fp16").read_bytes()

    # A second input produces a pack that differs only inside the input
    # surface (the surface is zero-padded to 0x4000 bytes).
    other = [((53 * i + 7) % 257 - 128) / 128 for i in range(256)]
    (root / "y.fp16").write_bytes(struct.pack("<256e", *other))
    assert h18g_compile(compiler, root, "out-b", root / "y.fp16").returncode == 0
    image_b = (root / "out-b" / "program-0.anep").read_bytes()
    assert input_surface_offset(image_b) == offset
    assert image_b[offset:offset + 512] == (root / "y.fp16").read_bytes()
    assert (image[:offset] + image[offset + 0x4000:]) == \
           (image_b[:offset] + image_b[offset + 0x4000:])

    # --input is required for H18G, rejected elsewhere, and checked.
    result = h18g_compile(compiler, root, "out-c")
    assert result.returncode == 64 and "required for H18G" in result.stderr, result.stderr
    result = h18g_compile(compiler, root, "out-d", root / "x.fp16", "H14")
    assert result.returncode == 64 and "only accepted for H18G" in result.stderr, result.stderr
    result = h18g_compile(compiler, root, "out-e", root / "missing.bin")
    assert result.returncode == 66 and "cannot read input file" in result.stderr, result.stderr
    (root / "short.bin").write_bytes(struct.pack("<128e", *other[:128]))
    result = h18g_compile(compiler, root, "out-f", root / "short.bin")
    assert result.returncode != 0 and "h18g.invalid-input" in result.stderr, result.stderr


if __name__ == "__main__":
    main()
