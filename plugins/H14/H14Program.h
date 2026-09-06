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
    /// Apple emits a different program per gelu mode.
    GeluSigmoidApproximation,
    GeluTanhApproximation,
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
    /// Command offset 0x858, which Apple sets to the input surface byte count
    /// for the three-task layer_norm-over-every-axis form on a multi-channel
    /// surface and leaves zero for every other decoded H14 program.
    std::uint32_t scratchDescriptorWord = 0;
};

bool supportsElementwise(BinaryOperation operation, ElementwiseShape shape,
                         ElementwiseShape operand, bool scalarConstant = false);
bool supportsElementwise(UnaryOperation operation, ElementwiseShape shape);
/// Encodes Apple's program for `operation` over a `shape` result surface whose
/// second runtime operand covers `operand`, which the decoded broadcast forms
/// let differ from `shape`.
Program encodeElementwise(BinaryOperation operation, ElementwiseShape shape,
                          ElementwiseShape operand, bool scalarConstant = false,
                          std::uint16_t scalarBits = 0x3800);
Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape);
/// True when the decoded Apple corpus covers this matmul geometry as one
/// two-task H14 program.
bool supportsMatvecParity(MatvecShape shape);
/// Encodes Apple's two-task H14 matvec form. `weights` is the
/// [columns, reduction] row-major fp16 constant, exactly the bytes the MIL
/// blob reference resolves to.
Program encodeMatvecParity(MatvecShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes);
/// Apple's constant-section permutation for a [columns, reduction] fp16
/// weight, proven byte-for-byte by research/mint_h14_matvec_probes.py.
std::vector<std::uint8_t> packMatvecWeights(MatvecShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes);
/// True when the decoded Apple corpus covers this softmax, layer_norm, or
/// reduction geometry as one H14 program.
bool supportsNormParity(NormOperation operation, NormShape shape);
/// Encodes Apple's own H14 task stream for the geometry, with the LUT constant
/// section the decoded oracle carries.
Program encodeNormParity(NormOperation operation, NormShape shape);
/// Byte lengths of the tasks a task stream carries, in stream order.
std::vector<std::size_t> taskSizes(const std::vector<std::uint8_t> &stream);
std::vector<std::uint8_t> encodeANEC(const Program &program);

}
