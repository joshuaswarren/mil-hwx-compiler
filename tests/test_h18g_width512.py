"""Stage the 512-wide H18G fixture: the compiler emits it, the pack carries the
width in the six proven stream slots, the tiles decode back to the weights, and
the CPU gold comes from the same fp16 dot products the M6 scorer uses."""
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
SECTION2_BYTES = SURFACE_BYTES + N * K * 2

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

MIL_256_BY_512 = MIL.replace("[1, 512]> x", "[1, 256]> x") \
    .replace("[512, 512]> w", "[256, 512]> w").replace("[1, 512]> y", "[1, 256]> y")


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
    result = subprocess.run([compiler, *args], capture_output=True, text=True, check=False)
    return result


def main():
    compiler = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_inputs(root)
        output = root / "out"
        args = ["--target", "H18G", "--mil", root / "model.mil",
                "--model-root", root, "--output", output, "--input", root / "x.fp16"]
        result = run(compiler, args)
        assert result.returncode == 0, result.stderr
        image = (output / "program-0.anep").read_bytes()
        magic, version, nsec, nio = struct.unpack_from("<IIII", image, 0)
        assert (magic, version) == (0x50454e41, 1)
        assert (nsec, nio) == (6, 2)
        assert struct.unpack_from("<Q", image, 16)[0] == len(image)

        sections = {}
        for i in range(nsec):
            sid, size, offset = struct.unpack_from("<I4xQQ", image, 0x20 + i * 0x20)
            sections[sid] = (size, offset)
        assert sections[2][0] == SECTION2_BYTES
        assert sections[3][0] == STREAM_BYTES

        stream = image[sections[3][1]:sections[3][1] + sections[3][0]]
        body = stream[16:]
        td0 = struct.unpack_from("<44I", body, 0)
        assert all(word == 512 for word in td0[30:34]), [hex(w) for w in td0[30:34]]
        prefix_end = 44 * 4 + 28 * 4
        tiles = struct.unpack_from("<I", body, 44 * 4 + 28 * 4)[0]
        assert tiles == ((DMA_TILES - 1) << 16 | 0x1559), hex(tiles)
        tail_at = 44 * 4 + 28 * 4 + 2 * DMA_TILES * 4
        tail = struct.unpack_from("<39I", body, tail_at)
        assert tail[3] == 512 and tail[7] == 512

        tile_bytes = N * TILE_COLUMNS * 2
        expected_offsets = [tile * tile_bytes for tile in range(1, DMA_TILES)]
        got_offsets = list(struct.unpack_from(f"<{DMA_TILES - 1}I", body, 44 * 4 + 28 * 4 + 4))
        assert got_offsets == expected_offsets

        weights = (root / "weights.bin").read_bytes()[128:]
        tiled = image[sections[2][1] + SURFACE_BYTES:sections[2][1] + SURFACE_BYTES + N * K * 2]
        untiled = bytearray(N * K * 2)
        at = 0
        for n in range(0, N, TILE_COLUMNS):
            for k in range(K):
                untiled[(k * N + n) * 2:(k * N + n) * 2 + TILE_COLUMNS * 2] = tiled[at:at + TILE_COLUMNS * 2]
                at += TILE_COLUMNS * 2
        assert bytes(untiled) == weights

        rec7 = struct.unpack_from("<I4xQQ", image, 0x20 + 5 * 0x20)
        assert rec7[0] == 7
        input_offset = rec7[2] + ((rec7[1] + 15) & ~15)
        assert (input_offset + K * 2) <= len(image)
        assert image[input_offset:input_offset + K * 2] == (root / "x.fp16").read_bytes()

        x = struct.unpack(f"<{K}e", (root / "x.fp16").read_bytes())
        w = struct.unpack(f"<{N * K}e", weights)
        reference = struct.pack(f"<{N}e", *(sum(x[k] * w[k * N + n] for k in range(K)) for n in range(N)))
        (root / "gold_y512.fp16").write_bytes(reference)

        manifest = json.loads((output / "manifest.json").read_bytes())
        assert manifest["runProg"] == {
            "bufs": 0x30000000, "text": 0x30008000, "text_const": 0x280,
            "kern": 0x30010000, "kern_slot": 1,
        }
        print(f"512 pack ok: {len(image)} bytes, sha256 {hashlib.sha256(image).hexdigest()}")

        # The ambiguous split stays refused until the M6 proves which axis the six words carry.
        (root / "model-256x512.mil").write_text(MIL_256_BY_512)
        (root / "x256.fp16").write_bytes(struct.pack("<256e", *(i / 512 for i in range(256))))
        weights_256x512 = struct.pack("<131072e", *(w[k * N + n] for k in range(256) for n in range(N)))
        header2 = bytearray(128)
        struct.pack_into("<II", header2, 0, 1, 2)
        struct.pack_into("<IIQQ", header2, 64, 0xDEADBEEF, 1, len(weights_256x512), 128)
        (root / "weights256x512.bin").write_bytes(header2 + weights_256x512)
        result = run(compiler, ["--target", "H18G", "--mil", root / "model-256x512.mil",
                                "--model-root", root, "--output", root / "out2",
                                "--input", root / "x256.fp16"])
        assert result.returncode != 0 and "h18g.unsupported-program" in result.stderr, result.stderr

        result = run(compiler, ["--target", "H18G", "--mil", root / "model.mil",
                                "--model-root", root, "--output", root / "out3",
                                "--input", root / "weights256x512.bin"])
        assert result.returncode != 0 and "h18g.invalid-input" in result.stderr, result.stderr
    print("H18G 512 fixture stages clean with CPU gold")


if __name__ == "__main__":
    main()
