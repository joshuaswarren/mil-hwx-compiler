#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ane::h14 {

/// Surface allocations and ANEC content are counted in 16 KiB tiles.
constexpr std::size_t tileBytes = 0x4000;
/// H14 tasks are 16-byte aligned inside __TEXT/__text.
constexpr std::size_t taskAlignment = 16;
/// Eight header words precede the first register record of every H14 task.
constexpr std::size_t taskHeaderWords = 8;

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
    /// Complete __TEXT/__text content: the zero-size prefix frame followed by
    /// 16-byte aligned tasks, with the final task left unpadded.
    std::vector<std::uint8_t> taskStream;
    std::vector<std::uint8_t> constants;
    std::vector<TensorLayout> inputs;
    TensorLayout output;
    std::size_t firstTaskBytes = 0;
    std::uint32_t taskCount = 1;
    std::size_t constantOffsetBytes = 0;
    /// Decoded H14 program-descriptor words the campaign resolves no formula
    /// for: the record count at command offset 0x860 and the word at 0x880.
    /// Parity carries the oracle values.
    std::uint32_t programRecordCount = 0;
    std::uint32_t unresolvedDescriptorWord = 0;
};

bool supportsElementwise(BinaryOperation operation, ElementwiseShape shape,
                         bool scalarConstant = false);
bool supportsElementwise(UnaryOperation operation, ElementwiseShape shape);
Program encodeElementwise(BinaryOperation operation, ElementwiseShape shape,
                          bool scalarConstant = false,
                          std::uint16_t scalarBits = 0x3800);
Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape);
/// Byte lengths of the tasks a task stream carries, in stream order.
std::vector<std::size_t> taskSizes(const std::vector<std::uint8_t> &stream);
std::vector<std::uint8_t> encodeANEC(const Program &program);

}
