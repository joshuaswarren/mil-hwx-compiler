#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace ane::h18g {

// The device-proven 256 shape keeps its own constants; wider shapes derive what is
// proven (sizes, tile lists, the six stream words that carry the width) and keep
// every other task-stream word at its 256-proven value until the M6 proves it there.
inline constexpr std::size_t width = 256;
inline constexpr std::size_t tileColumns = 16;
inline constexpr std::uint32_t inputBuffer = 5;
inline constexpr std::uint32_t outputBuffer = 4;
inline constexpr std::uint32_t kernelBuffer = 2;
inline constexpr std::uint32_t taskCount = 2;
inline constexpr std::uint32_t bufs = 0x30000000;
inline constexpr std::size_t surfaceBytes = 0x4000;
inline constexpr std::uint32_t text = bufs + 2 * surfaceBytes;
inline constexpr std::uint32_t textConstFloor = 0x280;
inline constexpr std::uint32_t kern = text + 2 * surfaceBytes;
inline constexpr std::uint32_t kernSlot = 1;

namespace detail {

// H18G moves the two PE writes to 0x4908 and 0x4914, in both tasks.
inline constexpr std::uint32_t peScatter = 0x80020000 | (0x4908 / 4);

inline void put32(std::uint8_t *p, std::uint32_t word) {
    for (unsigned i = 0; i < 4; ++i)
        p[i] = static_cast<std::uint8_t>(word >> (i * 8));
}

template <typename Words>
inline std::uint8_t *writeWords(std::uint8_t *p, const Words &words) {
    for (std::uint32_t word : words) {
        put32(p, word);
        p += 4;
    }
    return p;
}

template <std::size_t N>
inline constexpr std::array<std::uint32_t, 44> td0 = {
    0x002c0000, 0x00000000, 0x00000068, 0x00000000, 0x00000068, 0x00000000,
    0x00000000, 0x00000001, 0x00050001, 0x00008002, 0x00000001, 0x00000001,
    0x89b08006, 0x00000001, 0x00000001, 0x00014000, 0x00000001, 0x10200000,
    0x00000000, 0x00200000, 0x80011340, 0x00000021, 0x000000ce, 0x0000135c,
    0x00003211, 0x00009041, 0x00500172, 0x00500130, 0xd0079044, 0x00001000,
    N, N, N, N, 0x00500170, 0x00000010,
    peScatter, 0x00103c0c, 0x00003c00, 0x23809346, 0x00000000, 0x00000000,
    0x22001344, 0x00000000,
};

template <std::size_t N>
inline constexpr std::array<std::uint32_t, 28> td1Prefix = [] {
    std::array<std::uint32_t, 28> words{
        0x00630001, 0x00000002, 0x04000000, 0x00000000, 0x00fff860, 0x00000000,
        0x00000000, 0x00000000, 0x00050009, 0xffc01540, 0x000202c0, 0x00010021,
        0x00010021, 0x00010021, 0x00010021, 0x00010021, 0x00010021, 0x00010021,
        0x00010021, 0x00010021, 0x00031551, 0x00010021, 0x00010021, 0x00010021,
        0x00010021, 0x00010021, 0x00010021, 0x00010021,
    };
    // Word 0 counts task 1's words (prefix + both tile lists + tail); the firmware
    // asserts on it (CAneProgramManager.cpp 254), proven by the M6 runs at both widths.
    constexpr std::size_t count = std::size(words) + 2 * (N / tileColumns) + 39;
    words[0] = static_cast<std::uint32_t>(count << 16 | 0x0001);
    return words;
}();

template <std::size_t N>
inline constexpr std::array<std::uint32_t, 39> td1Tail = {
    0x00010001, 0x00000001, 0x00000001, N, 0x93418005, 0x00000001,
    0x00000001, N, 0x00000001, 0x10211104, 0x00000004, 0x00200000,
    0x80041041, 0x00000120, 0x00000010, 0x00009052, 0x0000014a, 0x00001000,
    peScatter, 0x08103c00, 0x00003c00, 0xc0001440, 0x01010021, 0xf0003211,
    0x83119640, 0x00000012, 0x00a000a0, 0x000100e0, 0x00000000, 0x007f0000,
    0x20010701, 0x24009544, 0x00000000, 0x00000000, 0x23009444, 0x00000000,
    0x00000000, 0x22001442, 0x00000000,
};

inline constexpr std::size_t ioCount = 2;
inline constexpr std::size_t recordBytes = 0x20;

} // namespace detail

// One weighted matvec per pack: [1,K] fp16 input times a constant [K,N] fp16 weight,
// both transpose flags false. Only K == N shapes are staged: the six stream width
// words carry one width each and the axis assignment behind them is proven only
// where K and N were equal on the M6. Every task-stream word that does not carry
// the width stays at its 256-proven value.
template <std::size_t K, std::size_t N>
struct H18GMatvec {
    static_assert(K == N, "the stream width words are proven only for K == N");
    static_assert(K % tileColumns == 0, "the coefficient tiling runs on 16-wide column tiles");

    static constexpr std::size_t vectorBytes = K * 2;
    static constexpr std::size_t weightBytes = N * K * 2;
    static constexpr std::size_t tileBytes = N * tileColumns * 2;
    static constexpr std::size_t dmaTiles = N / tileColumns;
    static constexpr std::size_t streamBytes = 16 + std::size(detail::td0<N>) * 4 +
        std::size(detail::td1Prefix<N>) * 4 + 2 * dmaTiles * 4 + std::size(detail::td1Tail<N>) * 4;
    // The loader refuses a stream longer than the const window, so the window grows
    // with the stream (0x40 granularity, 0x280 floor): 640 at 256, 768 at 512.
    static constexpr std::uint32_t textConst = static_cast<std::uint32_t>(
        std::max<std::size_t>(textConstFloor, (streamBytes + 0x3F) & ~std::size_t{0x3F}));
    static constexpr std::array<std::uint32_t, 6> sectionIds{1, 2, 3, 4, 5, 7};
    static constexpr std::array<std::size_t, sectionIds.size()> sectionBytes{
        0x208 + detail::ioCount * 0x30, surfaceBytes + weightBytes, streamBytes, 0x410, 0x38, 0x28,
    };

    static constexpr std::size_t aligned(std::size_t size) { return (size + 15) & ~std::size_t{15}; }
    static constexpr auto sectionOffsets() {
        std::array<std::size_t, sectionIds.size() + 1> offsets{};
        offsets[0] = (1 + sectionIds.size() + detail::ioCount) * detail::recordBytes;
        for (std::size_t i = 0; i < sectionIds.size(); ++i)
            offsets[i + 1] = offsets[i] + aligned(sectionBytes[i]);
        return offsets;
    }
    static constexpr auto offsets = sectionOffsets();
    static constexpr std::size_t packBytes = offsets.back() + surfaceBytes;
    // The IO-input pack ends after the last section: buffer 5 is a window, not a payload.
    static constexpr std::size_t packBytesIO = offsets.back();

    // Destination is a zero-initialized ANEP buffer; input and weights are little-endian fp16.
    static void encode(std::uint8_t *out, const std::uint8_t *input, const std::uint8_t *weights) {
        encodeImpl(out, weights, input);
    }

    // The G709 variant: the pack carries no input at all. Buffer 5 stays declared with its
    // 16 KiB surface, but its data offset and size read zero, so the driver maps a writable
    // activation window instead of finding x baked into the image. Everything else, including
    // the task stream and the coefficient section, is byte-for-byte the embedded layout.
    static void encodeIO(std::uint8_t *out, const std::uint8_t *weights) {
        encodeImpl(out, weights, nullptr);
    }

    static void encodeImpl(std::uint8_t *out, const std::uint8_t *weights, const std::uint8_t *input) {
        using namespace detail;
        writeWords(out, std::array<std::uint32_t, 5>{0x50454e41, 1, sectionIds.size(), ioCount,
                                                     static_cast<std::uint32_t>(input ? packBytes : packBytesIO)});
        auto *record = out + recordBytes;
        for (std::size_t i = 0; i < sectionIds.size(); ++i, record += recordBytes) {
            put32(record, sectionIds[i]);
            put32(record + 8, static_cast<std::uint32_t>(sectionBytes[i]));
            put32(record + 16, static_cast<std::uint32_t>(offsets[i]));
        }
        writeWords(record, std::array<std::uint32_t, 3>{outputBuffer, 1, surfaceBytes});
        if (input) {
            writeWords(record + recordBytes, std::array<std::uint32_t, 7>{
                inputBuffer, 0, surfaceBytes, 0, static_cast<std::uint32_t>(offsets.back()), 0, surfaceBytes});
            std::memcpy(out + offsets.back(), input, vectorBytes);
        } else {
            writeWords(record + recordBytes, std::array<std::uint32_t, 7>{
                inputBuffer, 0, surfaceBytes, 0, 0, 0, 0});
        }

        auto *generic = out + offsets[0];
        writeWords(generic, std::array<std::uint32_t, 2>{1, 0x10});
        put32(generic + 0x204, ioCount);
        writeWords(generic + 0x208, std::array<std::uint32_t, 11>{
            1, inputBuffer, 0, 0, 1, 0, 0, 0, surfaceBytes, 0, 0xffff});
        writeWords(generic + 0x238, std::array<std::uint32_t, 11>{
            1, outputBuffer, 1, 0, 2, 0, 0, 0, surfaceBytes, 0, 0xffff});

        auto *coefficients = out + offsets[1] + surfaceBytes;
        for (std::size_t n = 0; n < N; n += tileColumns) {
            for (std::size_t k = 0; k < K; ++k) {
                std::memcpy(coefficients, weights + (k * N + n) * 2, tileColumns * 2);
                coefficients += tileColumns * 2;
            }
        }

        auto *stream = writeWords(out + offsets[2] + 16, td0<N>);
        stream = writeWords(stream, td1Prefix<N>);
        put32(stream, static_cast<std::uint32_t>((dmaTiles - 1) << 16 | 0x1559));
        for (std::size_t tile = 1; tile < dmaTiles; ++tile)
            put32(stream + tile * 4, static_cast<std::uint32_t>(tile * tileBytes));
        stream += dmaTiles * 4;
        for (std::size_t tile = 0; tile < dmaTiles; ++tile)
            put32(stream + tile * 4, static_cast<std::uint32_t>(tileBytes));
        writeWords(stream + dmaTiles * 4, td1Tail<N>);

        writeWords(out + offsets[3], std::array<std::uint32_t, 10>{
            1, 0, 0, 3, 6, outputBuffer, 7, inputBuffer, 8, kernelBuffer});
        writeWords(out + offsets[4], std::array<std::uint32_t, 14>{
            1, 0, 0x18, 0, 0x20, 0, 1, 3, 0, 0, 1, 0, 0xffffffff, outputBuffer});
        writeWords(out + offsets[5], std::array<std::uint32_t, 9>{
            1, 0, 0, taskCount, 0, 0, 0, 0, streamBytes});
    }
};

// The device-proven 256 pack keeps its global names: tests and the compiler pin it.
using Proven = H18GMatvec<width, width>;
inline constexpr std::size_t vectorBytes = Proven::vectorBytes;
inline constexpr std::size_t weightBytes = Proven::weightBytes;
inline constexpr std::size_t packBytes = Proven::packBytes;

} // namespace ane::h18g
