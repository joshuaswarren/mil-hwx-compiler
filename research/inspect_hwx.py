#!/usr/bin/env python3
"""Inspect and validate H13, H14, or H16G HWX object structure."""
from __future__ import annotations

import struct
import sys

HWX_MAGIC = 0xBEEFFACE
ARCHITECTURES = {
    5: ("H14", 11),
    4: ("H13", 7),
    7: ("H16G", 17),
}
H14_BLOCKS = (
    ("common", 0x0000, 19),
    ("l2", 0x0500, 25),
    ("pe", 0x0900, 5),
    ("ne", 0x0D00, 5),
    ("tile_dma_src", 0x1100, 53),
    ("tile_dma_dst", 0x1500, 9),
    ("kernel_dma_src", 0x1900, 70),
)


def cstring(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("ascii", errors="replace")


def unpack_from(fmt: str, data: bytes, offset: int, label: str):
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise ValueError(f"{label} is truncated at 0x{offset:x}")
    return struct.unpack_from(fmt, data, offset)


def require_region(offset: int, size: int, limit: int, label: str) -> None:
    if offset < 0 or size < 0 or offset > limit or size > limit - offset:
        raise ValueError(
            f"{label} range 0x{offset:x}+0x{size:x} exceeds file size 0x{limit:x}")


def h14_block(word_address: int) -> tuple[str, int, int] | None:
    for block in H14_BLOCKS:
        _, start, count = block
        if start // 4 <= word_address < start // 4 + count:
            return block
    return None


def validate_h14_addresses(
        task_index: int, record_index: int, addresses: list[int],
        observed: dict[str, set[int]]) -> None:
    blocks = [h14_block(address) for address in addresses]
    first_block = blocks[0] if blocks else None
    if first_block is None or any(block != first_block for block in blocks):
        first = addresses[0] * 4
        end = (addresses[-1] + 1) * 4
        raise ValueError(
            f"H14 task[{task_index}] record[{record_index}] register range "
            f"0x{first:04x}-0x{end:04x} is outside source-backed H14 blocks")
    observed[first_block[0]].update(addresses)


def inspect_h14_tasks(section_data: bytes) -> int:
    observed = {name: set() for name, _, _ in H14_BLOCKS}
    offset = 0
    task_index = 0
    while offset < len(section_data):
        remaining = section_data[offset:]
        if not any(remaining):
            break
        if len(remaining) < 40:
            raise ValueError(
                f"H14 task[{task_index}] header is truncated in __TEXT/__text")
        task_words = unpack_from(
            "<H", section_data, offset + 2, "H14 task size")[0] & 0x7FF
        if task_words == 0:
            offset += 16
            continue
        if task_words < 10:
            raise ValueError(
                f"H14 task[{task_index}] has invalid size {task_words} words")
        task_bytes = task_words * 4
        if task_bytes > len(section_data) - offset:
            raise ValueError(
                f"H14 task[{task_index}] declares {task_words} words beyond "
                "__TEXT/__text")
        words = unpack_from(
            f"<{task_words}I", section_data, offset, f"H14 task[{task_index}]")
        word_index = 8
        record_index = 0
        while word_index < task_words:
            header = words[word_index]
            word_index += 1
            base = header & 0x7FFF
            if header >> 31:
                mask = (header >> 15) & 0xFFFF
                addresses = [base] + [
                    base + bit + 1 for bit in range(16) if mask & (1 << bit)]
            else:
                count = ((header >> 15) & 0x3F) + 1
                addresses = list(range(base, base + count))
            if len(addresses) > task_words - word_index:
                raise ValueError(
                    f"H14 task[{task_index}] record[{record_index}] values are "
                    "truncated")
            validate_h14_addresses(task_index, record_index, addresses, observed)
            word_index += len(addresses)
            record_index += 1
        print(
            f"  h14_task[{task_index}] offset=0x{offset:x} "
            f"size_words={task_words} records={record_index}")
        task_end = offset + task_bytes
        next_offset = offset + ((task_bytes + 15) & ~15)
        padding_end = min(next_offset, len(section_data))
        if any(section_data[task_end:padding_end]):
            raise ValueError(f"H14 task[{task_index}] has nonzero alignment padding")
        if next_offset > len(section_data) and task_end != len(section_data):
            raise ValueError(f"H14 task[{task_index}] alignment padding is truncated")
        offset = next_offset
        task_index += 1
    for name, start, count in H14_BLOCKS:
        print(
            f"  h14_block name={name} range=0x{start:04x}-"
            f"0x{start + count * 4:04x} source_words={count} "
            f"observed_words={len(observed[name])}")
    print(f"  h14_tasks count={task_index}")
    return task_index


def inspect_segment(data: bytes, cursor: int, command_size: int, subtype: int) -> None:
    if command_size < 72:
        raise ValueError(f"LC_SEGMENT_64 at 0x{cursor:x} is shorter than 72 bytes")
    fields = unpack_from(
        "<2I16s4Q4I", data, cursor, f"LC_SEGMENT_64 at 0x{cursor:x}")
    segment = cstring(fields[2])
    segment_file_size = fields[6]
    section_count = fields[-2]
    section_bytes = section_count * 80
    if section_bytes > command_size - 72:
        raise ValueError(
            f"LC_SEGMENT_64 at 0x{cursor:x} declares {section_count} sections "
            f"outside command size 0x{command_size:x}")
    section_cursor = cursor + 72
    for section_index in range(section_count):
        section_fields = unpack_from(
            "<16s16s2Q8I", data, section_cursor,
            f"section[{section_index}] in {segment}")
        section = cstring(section_fields[0])
        size = section_fields[3]
        offset = section_fields[4]
        relocation_offset = section_fields[6]
        relocation_count = section_fields[7]
        if segment_file_size:
            require_region(offset, size, len(data), f"{segment}/{section}")
        require_region(
            relocation_offset, relocation_count * 8, len(data),
            f"{segment}/{section} relocation table")
        print(
            f"{segment}/{section} addr=0x{section_fields[2]:x} "
            f"size=0x{size:x} offset=0x{offset:x} "
            f"reloff=0x{relocation_offset:x} nreloc={relocation_count}")
        for relocation_index in range(relocation_count):
            address, info = unpack_from(
                "<2I", data, relocation_offset + relocation_index * 8,
                f"relocation[{relocation_index}] in {segment}/{section}")
            print(
                f"  relocation[{relocation_index}] "
                f"address=0x{address:x} info=0x{info:08x}")
        if subtype == 5 and segment == "__TEXT" and section in ("__text", "__TEXT"):
            inspect_h14_tasks(data[offset:offset + size])
        section_cursor += 80


def h13_anec(data: bytes) -> tuple[bytes, dict[str, int]]:
    _, _, _, _, command_count, command_bytes, _, _ = unpack_from(
        "<8I", data, 0, "Mach-O header")
    cursor = 32
    command_end = cursor + command_bytes
    sections: dict[tuple[str, str], dict[str, int]] = {}
    surfaces: list[tuple[str, int]] = []
    tensors: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    for command_index in range(command_count):
        command, command_size = unpack_from(
            "<2I", data, cursor, f"load command[{command_index}]")
        if command_size < 8 or command_size > command_end - cursor:
            raise ValueError(f"invalid H13 load command[{command_index}]")
        kind = unpack_from("<I", data, cursor + 8, "load command kind")[0] \
            if command_size >= 12 else None
        if command == 0x19:
            segment_fields = unpack_from(
                "<2I16s4Q4I", data, cursor, "H13 segment")
            segment = cstring(segment_fields[2])
            section_cursor = cursor + 72
            for section_index in range(segment_fields[-2]):
                fields = unpack_from(
                    "<16s16s2Q8I", data, section_cursor, "H13 section")
                section = cstring(fields[0])
                sections[(segment, section)] = {
                    "address": fields[2], "size": fields[3],
                    "offset": fields[4], "allocation": segment_fields[4],
                }
                if segment == "__FVMLIB":
                    surfaces.append((section, segment_fields[4]))
                section_cursor += 80
        elif command == 4 and kind == 3 and command_size >= 0x80:
            tensors.append((
                unpack_from("<4I", data, cursor + 0x28, "H13 tensor shape"),
                unpack_from("<4Q", data, cursor + 0x50, "H13 tensor strides"),
                unpack_from("<Q", data, cursor + 0x70, "H13 tensor total")[0],
                unpack_from("<I", data, cursor + 0x24, "H13 tensor element")[0],
            ))
        cursor += command_size
    if cursor != command_end:
        raise ValueError("H13 load command table size does not match its header")
    text = sections.get(("__TEXT", "__text"))
    constants = sections.get(("__TEXT", "__const"))
    if not text or not constants or text["size"] != 0x274:
        raise ValueError("H13 HWX requires a 0x274-byte __TEXT/__text task")
    if constants["offset"] != text["offset"] + 0x280 or \
            constants["address"] != text["address"] + 0x280:
        raise ValueError("H13 __TEXT/__const must begin at task content offset 0x280")
    if len(tensors) != len(surfaces) or len(tensors) not in (2, 3):
        raise ValueError("H13 HWX requires one or two inputs followed by one output")
    if any(name != "__const" for name, _ in surfaces[:-1]) or \
            surfaces[-1][0] != "__data":
        raise ValueError("H13 FVMLIB surfaces must list inputs before the output")
    for index, ((shape, strides, total, element_code), (_, allocation)) in enumerate(
            zip(tensors, surfaces)):
        batch, plane, row, element = strides
        if element_code != 5 or element != 2 or batch != shape[1] * plane or \
                total != shape[0] * batch or total > allocation:
            raise ValueError(f"H13 tensor descriptor[{index}] has an invalid layout")
    content_size = 0x280 + constants["size"]
    content = data[text["offset"]:text["offset"] + content_size]
    if len(content) != content_size:
        raise ValueError("H13 task and constant content is truncated")
    header = bytearray(0x1000)
    struct.pack_into("<QIIQQII", header, 0, content_size, 0x274, 1,
                     0x274, constants["size"], len(tensors) - 1, 1)
    tiles = [0] * 32
    tiles[0] = (content_size + 0x3fff) // 0x4000
    tiles[4] = (surfaces[-1][1] + 0x3fff) // 0x4000
    for index, (_, allocation) in enumerate(surfaces[:-1], 5):
        tiles[index] = (allocation + 0x3fff) // 0x4000
    struct.pack_into("<32I", header, 40, *tiles)
    layouts = [0] * (32 * 6)
    ordered = [(4, tensors[-1]), *
               ((index + 5, tensor) for index, tensor in enumerate(tensors[:-1]))]
    for channel, (shape, strides, _, _) in ordered:
        layouts[channel * 6:channel * 6 + 6] = [
            *shape, strides[1], strides[2]]
    struct.pack_into("<192Q", header, 0xa8, *layouts)
    return bytes(header) + content, {
        "contentOffset": text["offset"], "contentBytes": content_size,
        "taskBytes": text["size"], "constantOffset": 0x280,
        "constantBytes": constants["size"],
        "inputs": len(tensors) - 1, "outputs": 1,
    }


def main(path: str, extract_path: str | None = None) -> None:
    with open(path, "rb") as stream:
        data = stream.read()
    header = unpack_from("<8I", data, 0, "Mach-O header")
    magic, _, subtype, _, command_count, command_bytes, _, _ = header
    if magic != HWX_MAGIC:
        raise ValueError(
            f"invalid HWX magic 0x{magic:08x}; expected 0x{HWX_MAGIC:08x}")
    architecture = ARCHITECTURES.get(subtype)
    if architecture:
        print(
            f"architecture subtype=0x{subtype:04x} "
            f"name={architecture[0]} isa={architecture[1]}")
    else:
        print(f"architecture subtype=0x{subtype:04x} name=unknown isa=unknown")
    require_region(32, command_bytes, len(data), "load command table")
    command_end = 32 + command_bytes
    cursor = 32
    for command_index in range(command_count):
        command, command_size = unpack_from(
            "<2I", data, cursor, f"load command[{command_index}]")
        if command_size < 8:
            raise ValueError(
                f"load command[{command_index}] has invalid size {command_size}")
        if command_size > command_end - cursor:
            raise ValueError(
                f"load command[{command_index}] exceeds declared command table")
        kind = unpack_from("<I", data, cursor + 8, "load command kind")[0] \
            if command_size >= 12 else None
        print(
            f"command[{command_index}] offset=0x{cursor:x} "
            f"cmd=0x{command:x} size=0x{command_size:x} "
            f"kind={kind if kind is not None else '-'}")
        if command == 0x40 and command_size == 0x20:
            address = unpack_from(
                "<Q", data, cursor + 0x10, "buffer reference address")[0]
            name = cstring(data[cursor + 0x18:cursor + 0x20])
            print(f"  buffer_reference address=0x{address:x} name={name!r}")
        if command == 4 and kind == 4 and command_size >= 0x898:
            record_count = unpack_from(
                "<I", data, cursor + 0x860, "program record count")[0]
            format_code = unpack_from(
                "<I", data, cursor + 0x890, "program format code")[0]
            task_count = unpack_from(
                "<I", data, cursor + 0x830, "program task count")[0]
            addresses = [
                unpack_from("<Q", data, cursor + offset, "program address")[0]
                for offset in (0x10, 0x20, 0x40, 0x70, 0x80, 0x90)
            ]
            print(
                "  program_descriptor "
                f"tasks={task_count} records={record_count} "
                f"format=0x{format_code:x} "
                "text=0x%x text_const=0x%x scratch=0x%x "
                "slot70=0x%x slot80=0x%x slot90=0x%x" % tuple(addresses))
        if command == 4 and kind == 1 and command_size >= 0x880:
            code = unpack_from("<I", data, cursor + 0x0c, "H13 program code")[0]
            text_address, constant_address = unpack_from(
                "<2Q", data, cursor + 0x10, "H13 text addresses")
            resources = unpack_from(
                "<5Q", data, cursor + 0x30, "H13 resource addresses")
            task_words = unpack_from(
                "<I", data, cursor + 0x818, "H13 task words")[0]
            task_count = unpack_from(
                "<I", data, cursor + 0x81c, "H13 task count")[0]
            max_binding = unpack_from(
                "<I", data, cursor + 0x824, "H13 max binding")[0]
            name_offset = unpack_from(
                "<I", data, cursor + 0x83c, "H13 name offset")[0]
            field_offsets = (0x850, 0x854, 0x858, 0x85c, 0x860, 0x868, 0x86c)
            fields = tuple(
                unpack_from("<I", data, cursor + offset, "H13 trailing field")[0]
                for offset in field_offsets)
            name = cstring(data[cursor + 0x878:cursor + command_size])
            print(
                "  h13_program_descriptor "
                f"code=0x{code:x} text=0x{text_address:x} "
                f"text_const=0x{constant_address:x} resources={resources} "
                f"task_words_minus_one={task_words} task_count={task_count} "
                f"max_binding={max_binding} name_offset=0x{name_offset:x} "
                f"field850=0x{fields[0]:x} field854=0x{fields[1]:x} "
                f"field858={fields[2]} field85c={fields[3]} "
                f"field860={fields[4]} field868={fields[5]} "
                f"field86c={fields[6]} name={name!r}")
        if command == 4 and kind == 3 and command_size >= 0x80:
            binding_index = unpack_from(
                "<I", data, cursor + 0x14, "tensor binding index")[0]
            element_code = unpack_from(
                "<I", data, cursor + 0x24, "tensor element code")[0]
            shape = unpack_from("<4I", data, cursor + 0x28, "tensor shape")
            strides = unpack_from("<4Q", data, cursor + 0x50, "tensor strides")
            total = unpack_from("<Q", data, cursor + 0x70, "tensor total")[0]
            print(
                "  tensor_descriptor "
                f"binding={binding_index} element={element_code} "
                f"shape={shape} strides={strides} total={total}")
        if command == 0x19:
            inspect_segment(data, cursor, command_size, subtype)
        cursor += command_size
    if cursor != command_end:
        raise ValueError(
            f"load commands consume 0x{cursor - 32:x} bytes; "
            f"header declares 0x{command_bytes:x}")
    if subtype == 4:
        anec, summary = h13_anec(data)
        print(
            "h13_anec_content " + " ".join(
                f"{key}={hex(value)}" for key, value in summary.items()))
        if extract_path:
            with open(extract_path, "wb") as stream:
                stream.write(anec)
            print(f"extracted_anec path={extract_path!r} bytes=0x{len(anec):x}")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        path, extract_path = sys.argv[1], None
    elif len(sys.argv) == 4 and sys.argv[1] == "--extract-anec":
        path, extract_path = sys.argv[2], sys.argv[3]
    else:
        raise SystemExit(
            f"usage: {sys.argv[0]} [--extract-anec FILE.hwx OUTPUT.anec] FILE.hwx")
    try:
        main(path, extract_path)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
