"""The G709 fixture: H18G without an embedded input. Buffer 5 is declared with a
zeroed data offset and size so the driver maps a writable activation window; the
embedded --input pack stays as the byte-level control."""
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

N = K = 512
TILE_COLUMNS = 16
SURFACE_BYTES = 0x4000
DMA_TILES = N // TILE_COLUMNS
STREAM_BYTES = 16 + 44 * 4 + 28 * 4 + 2 * DMA_TILES * 4 + 39 * 4

MIL = """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 512]> x) {
    bool f = const()[name = string("f"), val = bool(false)];
    tensor<fp16, [512, 512]> w = const()[name = string("w"), val = tensor<fp16, [512, 512]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64(64)))];
    tensor<fp16, [1, 512]> y = matmul(transpose_x = f, transpose_y = f, x = x, y = w)[name = string("y")];
  } -> (y);
}
"""


def write_inputs(root):
    (root / "model.mil").write_text(MIL)
    weights = struct.pack("<262144e", *(
        (2 * (k % 4) + 8 * (n % 8) - 31) / 32
        for k in range(K) for n in range(N)
    ))
    header = bytearray(128)
    struct.pack_into("<II", header, 0, 1, 2)
    struct.pack_into("<IIQQ", header, 64, 0xDEADBEEF, 1, len(weights), 128)
    (root / "weights.bin").write_bytes(header + weights)
    x = [((61 * i + 17) % 511 - 255) / 256 for i in range(K)]
    x[40] = 1 / 256
    (root / "x.fp16").write_bytes(struct.pack(f"<{K}e", *x))


def run(compiler, args):
    return subprocess.run([compiler, *args], capture_output=True, text=True, check=False)


def records(image):
    out = []
    for i in range(8):  # 6 section records, then the two IO records
        out.append(struct.unpack_from("<I4xQQ", image, 0x20 + i * 0x20))
    return out


def main():
    compiler = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_inputs(root)
        args = ["--target", "H18G", "--mil", root / "model.mil", "--model-root", root,
                "--output", root / "io"]

        # The IO pack: no --input, no input payload.
        result = run(compiler, args)
        assert result.returncode == 0, result.stderr
        image = (root / "io" / "program-0.anep").read_bytes()
        secs = {sid: (size, offset) for sid, size, offset in records(image)[:6]}
        input_bytes = K * 2
        pack_bytes_io = secs[7][1] + ((0x28 + 15) & ~15)
        assert len(image) == pack_bytes_io, (len(image), pack_bytes_io)
        assert secs[2][0] == SURFACE_BYTES + N * K * 2 and secs[3][0] == STREAM_BYTES

        out_rec = struct.unpack_from("<10I", image, 0x20 * 7)
        in_rec = struct.unpack_from("<10I", image, 0x20 * 8)
        assert out_rec[:3] == (4, 1, SURFACE_BYTES)
        assert in_rec[0] == 5 and in_rec[1] == 0 and in_rec[2] == SURFACE_BYTES
        assert in_rec[3] == 0 and in_rec[4] == 0 and in_rec[5] == 0 and in_rec[6] == 0, in_rec
        assert struct.unpack_from("<Q", image, 16)[0] == len(image), "the header total must match the io pack length"

        manifest = json.loads((root / "io" / "manifest.json").read_bytes())
        assert manifest["buffers"] == {
            "inputs": [{"index": 5, "size": input_bytes}],
            "outputs": [{"index": 4, "size": input_bytes}],
        }
        assert manifest["runProg"]["text_const"] == 0x300

        # The sections themselves stay the embedded layout: weights, stream words.
        weights = (root / "weights.bin").read_bytes()[128:]
        tiled = image[secs[2][1] + SURFACE_BYTES:secs[2][1] + SURFACE_BYTES + N * K * 2]
        untiled = bytearray(N * K * 2)
        at = 0
        for n in range(0, N, TILE_COLUMNS):
            for k in range(K):
                untiled[(k * N + n) * 2:(k * N + n) * 2 + TILE_COLUMNS * 2] = tiled[at:at + TILE_COLUMNS * 2]
                at += TILE_COLUMNS * 2
        assert bytes(untiled) == weights
        stream = image[secs[3][1]:secs[3][1] + STREAM_BYTES]
        td0 = struct.unpack_from("<44I", stream, 16)
        assert all(word == 512 for word in td0[30:34])

        # The embedded control: the same fixture with --input keeps its payload.
        (root / "control").mkdir()
        result = run(compiler, args + ["--output", root / "control", "--input", root / "x.fp16"])
        assert result.returncode == 0, result.stderr
        control = (root / "control" / "program-0.anep").read_bytes()
        assert len(control) == pack_bytes_io + SURFACE_BYTES
        ctrl_secs = {sid: (size, offset) for sid, size, offset in records(control)[:6]}
        assert ctrl_secs[2] == secs[2] and ctrl_secs[3] == secs[3]
        ctrl_in = struct.unpack_from("<10I", control, 0x20 * 8)
        assert ctrl_in[4] == secs[7][1] + ((0x28 + 15) & ~15) and ctrl_in[6] == SURFACE_BYTES
        assert control[ctrl_in[4]:ctrl_in[4] + K * 2] == (root / "x.fp16").read_bytes()
        assert "buffers" not in json.loads((root / "control" / "manifest.json").read_bytes())

        # CPU gold for the full 512-wide output.
        x = struct.unpack(f"<{K}e", (root / "x.fp16").read_bytes())
        w = struct.unpack(f"<{N * K}e", weights)
        gold = struct.pack(f"<{N}e", *(sum(x[k] * w[k * N + n] for k in range(K)) for n in range(N)))
        (root / "gold_y512.fp16").write_bytes(gold)
        print(f"io pack ok: {len(image)} bytes, sha256 {hashlib.sha256(image).hexdigest()}; "
              f"control sha256 {hashlib.sha256(control).hexdigest()}")
    print("H18G io-input fixture stages clean with CPU gold")


if __name__ == "__main__":
    main()
