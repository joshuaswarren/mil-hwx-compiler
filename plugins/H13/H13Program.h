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

enum class NormOperation { Softmax, LayerNorm, ReduceSum, ReduceMax, ReduceMean };

struct ElementwiseShape {
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
};

/// One decoded Apple matmul geometry: `rows` logical x rows, a `reduction`
/// long inner product, and `columns` output columns.
struct MatvecShape {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
};

/// One decoded Apple normalization or reduction geometry: the input and output
/// CHW surfaces, which NCHW axes the operation reduces (bit `i` for axis `i`),
/// and whether the MIL result keeps the reduced axes.
struct NormShape {
    ElementwiseShape input;
    ElementwiseShape output;
    std::uint32_t axisMask;
    bool keepDims;
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
    /// The zero-filled allocation Apple places below every surface; the HWX
    /// writer emits it as __DATA/__bss and every surface address shifts by it.
    std::uint64_t scratchAllocationBytes = 0;
    /// Apple's matvec objects lay the output surface out before the input.
    bool outputSurfaceFirst = false;
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
/// True when the decoded Apple corpus covers this geometry as one program.
bool supportsMatvecParity(MatvecShape shape);
/// Encodes Apple's two-task matvec form. `weights` is the [columns, reduction]
/// row-major fp16 constant, exactly the bytes the MIL blob resolves to.
Program encodeMatvecParity(MatvecShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes);
/// Apple's constant-section permutation for a [columns, reduction] fp16 weight.
std::vector<std::uint8_t> packMatvecWeights(MatvecShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes);
/// True when the decoded Apple corpus covers this softmax, layer_norm, or
/// reduction geometry as one multi-task program.
bool supportsNormParity(NormOperation operation, NormShape shape);
/// Encodes Apple's own task stream for the geometry, with the LUT constant
/// section the decoded oracle carries.
Program encodeNormParity(NormOperation operation, NormShape shape);
std::vector<std::uint8_t> encodeANEC(const Program &program);

}
