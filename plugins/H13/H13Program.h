#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ane::h13 {

constexpr std::size_t taskBytes = 0x274;
constexpr std::size_t tileBytes = 0x4000;
constexpr std::size_t constantOffset = 0x280;

enum class BinaryOperation { Add, Multiply, Maximum, Minimum };

struct TensorLayout {
    std::uint32_t index;
    std::array<std::uint64_t, 6> nchw;
    std::uint64_t allocationBytes;
};

struct Program {
    std::vector<std::uint8_t> task;
    std::vector<std::uint8_t> constants;
    std::vector<TensorLayout> inputs;
    TensorLayout output;
};

Program encodeBinary(BinaryOperation operation);
// Weights are canonical row-major little-endian fp16 bytes.
Program encodeMatvec(std::uint32_t reduction,
                     const std::uint8_t *weights,
                     std::size_t weightBytes);
std::vector<std::uint8_t> encodeANEC(const Program &program);

}
