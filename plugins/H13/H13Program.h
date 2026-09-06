#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ane::h13 {

constexpr std::size_t taskBytes = 0x274;
constexpr std::size_t tileBytes = 0x4000;
constexpr std::size_t constantOffset = 0x280;

enum class BinaryOperation { Add, Multiply, Maximum, Minimum, Subtract, RealDivide };
enum class UnaryOperation {
    Absolute,
    Exponential,
    Gelu,
    LeakyRelu,
    Relu,
    ReciprocalSquareRoot,
    Sigmoid,
    Silu,
    SquareRoot,
    Tanh,
};

struct ElementwiseShape {
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
};

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
    std::size_t firstTaskBytes = taskBytes;
    std::uint32_t taskCount = 1;
    std::size_t constantOffsetBytes = constantOffset;
    /// Task-descriptor byte offsets holding kernel-table addends the HWX
    /// writer must relocate; empty when the task carries absolute addresses.
    std::vector<std::size_t> kernelRelocations;
};

Program encodeBinary(BinaryOperation operation);
bool supportsElementwise(BinaryOperation operation, ElementwiseShape shape,
                         bool scalarConstant = false);
bool supportsElementwise(UnaryOperation operation, ElementwiseShape shape);
Program encodeElementwise(BinaryOperation operation, ElementwiseShape shape,
                          bool scalarConstant = false,
                          std::uint16_t scalarBits = 0x3800);
Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape);
// Weights are row-major little-endian fp16; transposeY selects [512,K] instead of [K,512].
Program encodeMatvec(std::uint32_t reduction,
                     const std::uint8_t *weights,
                     std::size_t weightBytes, bool transposeY);
std::vector<std::uint8_t> encodeANEC(const Program &program);

}
