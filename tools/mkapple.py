#!/usr/bin/env python3
"""Splice H18 HWX __TEXT.__text into v2 ANEP descriptor section (316 B)."""
import hashlib, struct, sys
from pathlib import Path

def text_section(hwx: bytes) -> bytes:
    off = 0x20
    ncmds = struct.unpack_from("<I", hwx, 16)[0]
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<2I", hwx, off)
        if cmd == 0x19:
            nsects = struct.unpack_from("<I", hwx, off + 64)[0]
            for i in range(nsects):
                at = off + 72 + i * 80
                sect = hwx[at : at + 16].split(b"\0", 1)[0]
                seg = hwx[at + 16 : at + 32].split(b"\0", 1)[0]
                if seg == b"__TEXT" and sect == b"__text":
                    vsize = struct.unpack_from("<I", hwx, at + 40)[0]
                    foff = struct.unpack_from("<I", hwx, at + 48)[0]
                    return hwx[foff : foff + vsize]
        off += size
    return b""

def splice(template: bytes, stream: bytes) -> bytes:
    magic, ver, nsec, _, _, _ = struct.unpack_from("<IIIIQQ", template, 0)
    if magic != 0x50454E41 or ver != 1:
        sys.exit("template is not ANEP v1")
    for idx in range(nsec):
        sid, _, sz, do, _ = struct.unpack_from("<IIQQQ", template, 0x20 + idx * 0x20)
        if sid != 3:
            continue
        if sz != 316 or len(stream) != 316:
            sys.exit(f"descriptor must be 316 B (template {sz}, hwx {len(stream)})")
        out = bytearray(template)
        out[do : do + 316] = stream
        return bytes(out)
    sys.exit("template has no section id 3")

def main() -> None:
    if len(sys.argv) != 5 or sys.argv[3] != "-o":
        sys.exit("usage: mkapple.py TEMPLATE.bin HWX -o OUT.bin")
    tmpl = Path(sys.argv[1]).read_bytes()
    stream = text_section(Path(sys.argv[2]).read_bytes())
    out = splice(tmpl, stream)
    Path(sys.argv[4]).write_bytes(out)
    print("descriptor_sha256", hashlib.sha256(stream).hexdigest())
    print("prog_sha256", hashlib.sha256(out).hexdigest())

if __name__ == "__main__":
    main()
