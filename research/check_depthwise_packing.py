#!/usr/bin/env python3
"""Compare a depthwise packing hypothesis with an Apple-produced KERN section."""

import struct
import sys


def pack(raw: bytes, channels: int) -> bytes:
    channels_per_packet = channels // 16
    packet_bytes = channels + 64
    output = bytearray(16 * packet_bytes)
    for packet in range(16):
        cursor = packet * packet_bytes
        for lane in range(channels_per_packet):
            channel = packet + lane * 16
            output[cursor : cursor + 18] = raw[channel * 18 : channel * 18 + 18]
            cursor += 18
    return bytes(output)


def main(hwx_path: str, weight_path: str, channels: int) -> None:
    image = open(hwx_path, "rb").read()
    blob = open(weight_path, "rb").read()
    expected = image[0xC000 : 0xC000 + 16 * (channels + 64)]
    actual = pack(blob[128 : 128 + channels * 18], channels)
    differences = [index for index, pair in enumerate(zip(actual, expected))
                   if pair[0] != pair[1]]
    print(f"channels={channels} differing_bytes={len(differences)}")
    for index in differences[:16]:
        print(
            f"  byte=0x{index:x} packet={index // (channels + 64)} "
            f"within=0x{index % (channels + 64):x} "
            f"got=0x{actual[index]:02x} want=0x{expected[index]:02x}"
        )
    for packet in range(16):
        start = packet * (channels + 64)
        actual_values = struct.unpack_from(
            f"<{(channels + 64) // 2}e", actual, start)
        expected_values = struct.unpack_from(
            f"<{(channels + 64) // 2}e", expected, start)
        if actual_values != expected_values:
            print(f"first differing packet={packet}")
            print("  got ", actual_values[: min(36, len(actual_values))])
            print("  want", expected_values[: min(36, len(expected_values))])
            break


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(f"usage: {sys.argv[0]} FILE.hwx WEIGHT.bin CHANNELS")
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
