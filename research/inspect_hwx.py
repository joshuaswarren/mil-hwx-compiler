#!/usr/bin/env python3
"""Print the segment and section table of an H16G HWX object."""

import struct
import sys


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def main(path: str) -> None:
    data = open(path, "rb").read()
    if len(data) < 32:
        raise ValueError("file is shorter than a Mach-O header")
    _, _, _, _, command_count, _, _, _ = struct.unpack_from("<8I", data, 0)
    cursor = 32
    for command_index in range(command_count):
        command, command_size = struct.unpack_from("<2I", data, cursor)
        kind = struct.unpack_from("<I", data, cursor + 8)[0] \
            if command_size >= 12 else None
        print(
            f"command[{command_index}] offset=0x{cursor:x} "
            f"cmd=0x{command:x} size=0x{command_size:x} "
            f"kind={kind if kind is not None else '-'}"
        )
        if command == 0x40 and command_size == 0x20:
            address = struct.unpack_from("<Q", data, cursor + 0x10)[0]
            name = cstring(data[cursor + 0x18:cursor + 0x20])
            print(f"  buffer_reference address=0x{address:x} name={name!r}")
        if command == 4 and kind == 4 and command_size >= 0x898:
            record_count = struct.unpack_from("<I", data, cursor + 0x860)[0]
            format_code = struct.unpack_from("<I", data, cursor + 0x890)[0]
            task_count = struct.unpack_from("<I", data, cursor + 0x830)[0]
            addresses = [
                struct.unpack_from("<Q", data, cursor + offset)[0]
                for offset in (0x10, 0x20, 0x40, 0x70, 0x80, 0x90)
            ]
            print(
                "  program_descriptor "
                f"tasks={task_count} records={record_count} "
                f"format=0x{format_code:x} "
                "text=0x%x text_const=0x%x scratch=0x%x "
                "slot70=0x%x slot80=0x%x slot90=0x%x" % tuple(addresses)
            )
        if command == 4 and kind == 3 and command_size >= 0x80:
            binding_index = struct.unpack_from("<I", data, cursor + 0x14)[0]
            element_code = struct.unpack_from("<I", data, cursor + 0x24)[0]
            shape = struct.unpack_from("<4I", data, cursor + 0x28)
            strides = struct.unpack_from("<4Q", data, cursor + 0x50)
            total = struct.unpack_from("<Q", data, cursor + 0x70)[0]
            print(
                "  tensor_descriptor "
                f"binding={binding_index} element={element_code} "
                f"shape={shape} strides={strides} total={total}"
            )
        if command == 0x19:
            fields = struct.unpack_from("<2I16s4Q4I", data, cursor)
            segment = cstring(fields[2])
            section_count = fields[-2]
            section_cursor = cursor + 72
            for _ in range(section_count):
                section_fields = struct.unpack_from(
                    "<16s16s2Q8I", data, section_cursor)
                print(
                    f"{segment}/{cstring(section_fields[0])} "
                    f"addr=0x{section_fields[2]:x} "
                    f"size=0x{section_fields[3]:x} "
                    f"offset=0x{section_fields[4]:x} "
                    f"reloff=0x{section_fields[6]:x} "
                    f"nreloc={section_fields[7]}"
                )
                relocation_cursor = section_fields[6]
                for relocation_index in range(section_fields[7]):
                    address, info = struct.unpack_from(
                        "<2I", data, relocation_cursor + relocation_index * 8)
                    print(
                        f"  relocation[{relocation_index}] "
                        f"address=0x{address:x} info=0x{info:08x}"
                    )
                section_cursor += 80
        cursor += command_size


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} FILE.hwx")
    main(sys.argv[1])
