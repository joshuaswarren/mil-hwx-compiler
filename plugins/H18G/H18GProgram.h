#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ane::h18g {

inline constexpr std::size_t width = 256;
inline constexpr std::size_t vectorBytes = width * 2;
inline constexpr std::size_t weightBytes = width * vectorBytes;
inline constexpr std::size_t surfaceBytes = 0x4000;
inline constexpr std::size_t tileColumns = 16;
inline constexpr std::size_t tileBytes = width * tileColumns * 2;
inline constexpr std::uint32_t inputBuffer = 5;
inline constexpr std::uint32_t outputBuffer = 4;
inline constexpr std::uint32_t kernelBuffer = 2;
inline constexpr std::uint32_t taskCount = 2;
inline constexpr std::uint32_t bufs = 0x30000000;
inline constexpr std::uint32_t text = bufs + 2 * surfaceBytes;
inline constexpr std::uint32_t textConst = 0x280;
inline constexpr std::uint32_t kern = text + 2 * surfaceBytes;
inline constexpr std::uint32_t kernSlot = 1;

namespace detail {

// H18G moves the two PE writes to 0x4908 and 0x4914, in both tasks.
inline constexpr std::uint32_t peScatter = 0x80020000 | (0x4908 / 4);
inline constexpr std::uint32_t td0[] = {
    0x002c0000, 0x00000000, 0x00000068, 0x00000000, 0x00000068, 0x00000000,
    0x00000000, 0x00000001, 0x00050001, 0x00008002, 0x00000001, 0x00000001,
    0x89b08006, 0x00000001, 0x00000001, 0x00014000, 0x00000001, 0x10200000,
    0x00000000, 0x00200000, 0x80011340, 0x00000021, 0x000000ce, 0x0000135c,
    0x00003211, 0x00009041, 0x00500172, 0x00500130, 0xd0079044, 0x00001000,
    width, width, width, width, 0x00500170, 0x00000010,
    peScatter, 0x00103c0c, 0x00003c00, 0x23809346, 0x00000000, 0x00000000,
    0x22001344, 0x00000000,
};
inline constexpr std::uint32_t td1Prefix[] = {
    0x00630001, 0x00000002, 0x04000000, 0x00000000, 0x00fff860, 0x00000000,
    0x00000000, 0x00000000, 0x00050009, 0xffc01540, 0x000202c0, 0x00010021,
    0x00010021, 0x00010021, 0x00010021, 0x00010021, 0x00010021, 0x00010021,
    0x00010021, 0x00010021, 0x00031551, 0x00010021, 0x00010021, 0x00010021,
    0x00010021, 0x00010021, 0x00010021, 0x00010021,
};
inline constexpr std::uint32_t td1Tail[] = {
    0x00010001, 0x00000001, 0x00000001, width, 0x93418005, 0x00000001,
    0x00000001, width, 0x00000001, 0x10211104, 0x00000004, 0x00200000,
    0x80041041, 0x00000120, 0x00000010, 0x00009052, 0x0000014a, 0x00001000,
    peScatter, 0x08103c00, 0x00003c00, 0xc0001440, 0x01010021, 0xf0003211,
    0x83119640, 0x00000012, 0x00a000a0, 0x000100e0, 0x00000000, 0x007f0000,
    0x20010701, 0x24009544, 0x00000000, 0x00000000, 0x23009444, 0x00000000,
    0x00000000, 0x22001442, 0x00000000,
};

inline constexpr std::size_t ioCount = 2;
inline constexpr std::size_t recordBytes = 0x20;
inline constexpr std::size_t dmaTiles = width / tileColumns;
inline constexpr std::size_t streamBytes = 16 + sizeof(td0) + sizeof(td1Prefix) +
    2 * dmaTiles * sizeof(std::uint32_t) + sizeof(td1Tail);
inline constexpr std::array<std::uint32_t, 6> sectionIds{1, 2, 3, 4, 5, 7};
inline constexpr std::array<std::size_t, sectionIds.size()> sectionBytes{
    0x208 + ioCount * 0x30, surfaceBytes + weightBytes, streamBytes, 0x410, 0x38, 0x28,
};
constexpr std::size_t aligned(std::size_t size) { return (size + 15) & ~std::size_t{15}; }
constexpr auto sectionOffsets() {
    std::array<std::size_t, sectionIds.size() + 1> offsets{};
    offsets[0] = (1 + sectionIds.size() + ioCount) * recordBytes;
    for (std::size_t i = 0; i < sectionIds.size(); ++i)
        offsets[i + 1] = offsets[i] + aligned(sectionBytes[i]);
    return offsets;
}
inline constexpr auto offsets = sectionOffsets();

inline void put32(std::uint8_t *p, std::uint32_t word) {
    for (unsigned i = 0; i < 4; ++i)
        p[i] = static_cast<std::uint8_t>(word >> (i * 8));
}

template <std::size_t N>
inline std::uint8_t *writeWords(std::uint8_t *p, const std::uint32_t (&words)[N]) {
    for (auto word : words) {
        put32(p, word);
        p += 4;
    }
    return p;
}

} // namespace detail

inline constexpr std::size_t packBytes = detail::offsets.back() + surfaceBytes;

// Destination is a zero-initialized ANEP buffer; input and weights are little-endian fp16.
inline void encode(std::uint8_t *out, const std::uint8_t *input, const std::uint8_t *weights) {
    using namespace detail;
    writeWords(out, {0x50454e41, 1, sectionIds.size(), ioCount, packBytes});
    auto *record = out + recordBytes;
    for (std::size_t i = 0; i < sectionIds.size(); ++i, record += recordBytes) {
        put32(record, sectionIds[i]);
        put32(record + 8, sectionBytes[i]);
        put32(record + 16, offsets[i]);
    }
    writeWords(record, {outputBuffer, 1, surfaceBytes});
    writeWords(record + recordBytes, {inputBuffer, 0, surfaceBytes, 0, offsets.back(), 0, surfaceBytes});
    std::memcpy(out + offsets.back(), input, vectorBytes);

    auto *generic = out + offsets[0];
    writeWords(generic, {1, 0x10});
    put32(generic + 0x204, ioCount);
    writeWords(generic + 0x208, {1, inputBuffer, 0, 0, 1, 0, 0, 0, surfaceBytes, 0, 0xffff});
    writeWords(generic + 0x238, {1, outputBuffer, 1, 0, 2, 0, 0, 0, surfaceBytes, 0, 0xffff});

    auto *coefficients = out + offsets[1] + surfaceBytes;
    for (std::size_t n = 0; n < width; n += tileColumns) {
        for (std::size_t k = 0; k < width; ++k) {
            std::memcpy(coefficients, weights + (k * width + n) * 2, tileColumns * 2);
            coefficients += tileColumns * 2;
        }
    }

    auto *stream = writeWords(out + offsets[2] + 16, td0);
    stream = writeWords(stream, td1Prefix);
    put32(stream, (dmaTiles - 1) << 16 | 0x1559);
    for (std::size_t tile = 1; tile < dmaTiles; ++tile)
        put32(stream + tile * 4, tile * tileBytes);
    stream += dmaTiles * 4;
    for (std::size_t tile = 0; tile < dmaTiles; ++tile)
        put32(stream + tile * 4, tileBytes);
    writeWords(stream + dmaTiles * 4, td1Tail);

    writeWords(out + offsets[3], {1, 0, 0, 3, 6, outputBuffer, 7, inputBuffer, 8, kernelBuffer});
    writeWords(out + offsets[4], {1, 0, 0x18, 0, 0x20, 0, 1, 3, 0, 0, 1, 0, 0xffffffff, outputBuffer});
    writeWords(out + offsets[5], {1, 0, 0, taskCount, 0, 0, 0, 0, streamBytes});
}

} // namespace ane::h18g
