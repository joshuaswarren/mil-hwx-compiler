"""Extract the task stream bytes from an H14 HWX and emit a C++ template
entry to insert into H14ElementwiseTemplates.inc.

The HWX is a Mach-O container. The first task descriptor lives in the
__TEXT,__h14 section. We extract it as a uint32[] array.

The last 4 uint32s of the task stream are the "programRecordCount,
unresolvedDescriptorWord" pair — actually the task stream layout is:
[2-task-header words] + [descriptor] + [padding]
The decode_hwx task_descriptors[0].size_bytes = 248 (this mul) tells us
the actual text size.
"""
import argparse
import json
import struct
import sys
import subprocess
from pathlib import Path

# u32 elements = bytes // 4. task_descriptor size_bytes is the descriptor
# body size; the text array includes 4-byte header + descriptor bytes.

def parse_macho(data):
    """Parse Mach-O __TEXT,__h14 section + constants section."""
    if data[:4] != b'\xce\xfa\xef\xbe':  # 0xBEEFFACE, LE (HWX, not Mach-O feedface)
        raise ValueError(f"not a HWX; magic = {data[:4].hex()}")
    fmt = "<"
    # header: 0x20 bytes for 64-bit
    ncmds = struct.unpack_from(fmt + "I", data, 16)[0]
    # walk load commands
    off = 32
    sections = {}
    text_section = None
    const_section = None
    for _ in range(ncmds):
        cmd, cmdsize = struct.unpack_from(fmt + "II", data, off)
        if cmd == 0x19:  # LC_SEGMENT_64
            segname = data[off+8:off+24].split(b'\0', 1)[0].decode()
            nsects = struct.unpack_from(fmt + "I", data, off+64)[0]
            sec_off = off + 72
            for i in range(nsects):
                secname = data[sec_off:sec_off+16].split(b'\0', 1)[0].decode()
                _sect_addr, sect_size, sect_file_off = struct.unpack_from(fmt + "QQI", data, sec_off+32)
                sections[f'{segname},{secname}'] = (sect_file_off, sect_size)
                if segname == '__TEXT' and secname == '__text':
                    text_section = data[sect_file_off:sect_file_off+sect_size]
                if segname == '__FVMLIB' and secname == '__const':
                    const_section = data[sect_file_off:sect_file_off+sect_size]
                sec_off += 80  # sizeof section_64
        off += cmdsize
    return text_section, const_section, sections


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hwx")
    ap.add_argument("--case", required=True, help="name for the template")
    ap.add_argument("--kind", required=True, choices=["elementwise"])
    args = ap.parse_args()
    data = Path(args.hwx).read_bytes()
    text_section, _, _ = parse_macho(data)
    if not text_section:
        raise ValueError("no __TEXT,__text section")
    # task_words is in program_descriptor. The task stream in __TEXT/__text
    # is `task_words` uint32s. Read it via parse_hwx for accuracy.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import mint_oracles as om
    decoded = om.parse_hwx(data, 'h14', capture_constant_bytes=False)
    pd = decoded.get('program_descriptor', {})
    task_words = pd.get('text_words')
    task_count = pd.get('task_count')
    print(f"// {args.case}: task_words={task_words} task_count={task_count}", file=sys.stderr)
    # task stream is at the __TEXT/__text section, size = task_words * 4
    # Read the first task_words uint32s (LE)
    raw = text_section[:task_words * 4]
    words = list(struct.unpack("<%dI" % task_words, raw))
    name = f"kH14Text{args.case}"
    print(f"static constexpr std::uint32_t {name}[] = {{")
    per_line = 8
    for i in range(0, len(words), per_line):
        chunk = words[i:i+per_line]
        line = ", ".join(f"0x{w:08x}" for w in chunk)
        end = "," if (i + per_line) < len(words) else ""
        print(f"    {line}{end}")
    print("};")
    # also print the constants — here it's zeros so no need to embed
    # but report the metadata
    print(f"// taskCount={task_count} unresolvedDescriptorWord=0x{pd.get('unresolvedDescriptorWord', 0):x} programRecordCount={pd.get('programRecordCount', 0)}", file=sys.stderr)


if __name__ == "__main__":
    main()