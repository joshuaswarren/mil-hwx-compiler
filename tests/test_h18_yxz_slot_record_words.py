#!/usr/bin/env python3
"""M6 hardware falsifier (runs 2026-11-19, receipts flagswap/addyxz3, 64/64 x20 exact):
the H18 add y,x->z task-stream slot-record words at +0x100/+0x118 must carry the
template values 0x22809346/0x23809444. The name-sort surface renumber shifts them
to 0x23809346/0x24009444 and the M6 DART rejects every output write (SID 0 STE
invalid, 10 faults/boot, buffer 4 all zero). Apple's oracle JSON does not record
these words, so this gate coexists with the surface-slot parity test."""
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from timeouts import scaled_timeout  # noqa: E402

COMPILER = Path(sys.argv[1]).resolve()
ORACLE_PATH = ROOT / "research/oracles/h18/binary_add_names_yx_z.json"
WANT = {0x100: 0x22809346, 0x118: 0x23809444}


def text_section(hwx: bytes) -> bytes:
    off = 0x20
    ncmds = struct.unpack_from("<I", hwx, 16)[0]
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", hwx, off)
        if cmd == 0x19:
            nsects = struct.unpack_from("<I", hwx, off + 64)[0]
            for i in range(nsects):
                at = off + 72 + i * 80
                seg = hwx[at + 16 : at + 32].split(b"\0", 1)[0]
                sect = hwx[at : at + 16].split(b"\0", 1)[0]
                if seg == b"__TEXT" and sect == b"__text":
                    vsize = struct.unpack_from("<I", hwx, at + 40)[0]
                    foff = struct.unpack_from("<I", hwx, at + 48)[0]
                    return hwx[foff : foff + vsize]
        off += size
    return b""


def main():
    oracle = json.loads(ORACLE_PATH.read_text())
    with tempfile.TemporaryDirectory(prefix="h18-yxz-words-") as directory:
        root = Path(directory)
        path = root / "binary_add_names_yx_z.mil"
        path.write_text(oracle["mil"])
        out = root / "out"
        r = subprocess.run(
            [str(COMPILER), "--mil", str(path), "--model-root", str(root),
             "--target", "H18", "--output", str(out)],
            capture_output=True, text=True, timeout=scaled_timeout(60), check=False)
        assert r.returncode == 0, r.stdout + r.stderr
        manifest = json.loads((out / "manifest.json").read_text())
        hwx = (out / manifest["programs"][0]["file"]).read_bytes()
        stream = text_section(hwx)
        assert len(stream) == 316, f"task stream is {len(stream)} B, want 316"
        bad = {}
        for off, want in WANT.items():
            got = struct.unpack_from("<I", stream, off)[0]
            if got != want:
                bad[hex(off)] = f"got {got:#010x}, want {want:#010x}"
        assert not bad, f"slot-record words differ: {bad}"
    print("PASS binary_add_names_yx_z slot-record words match M6-proven template order")


if __name__ == "__main__":
    main()
