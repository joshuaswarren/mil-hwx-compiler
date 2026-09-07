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

/// One decoded Apple elementwise surface with its batch. Apple records the
/// batch in the tensor descriptor's shape while its declared size covers a
/// single batch element, so the batch never enters the row or plane stride.
struct BatchedShape {
    std::uint32_t batch;
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
};

/// How a binary operation's second operand reaches the program: another
/// runtime surface, an inline fp16 scalar, or a per-channel constant Apple
/// folds into the bias and scale blocks of the constant section.
enum class BroadcastOperand : std::uint8_t { Runtime, Scalar, Constant };

/// One decoded Apple broadcast geometry. `y` is zero for a scalar operand;
/// otherwise it is the second operand's NCHW surface, which may differ from
/// `x` in any axis Apple broadcasts.
struct BroadcastShape {
    BatchedShape x;
    BatchedShape y;
};

/// One decoded Apple matmul geometry: `rows` logical x rows, a `reduction`
/// long inner product, `columns` output columns, the two MIL transpose flags,
/// and whether the second operand is a runtime surface instead of a constant.
struct MatmulShape {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    bool transposeX = false;
    bool transposeY = true;
    bool runtimeWeight = false;
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

/// One decoded Apple convolution geometry: the MIL kernel extent, stride and
/// group count, whether a bias is folded into the constant section, and the
/// input and output CHW surfaces. The padding follows from those, so a
/// `pad_type` of `same` and `valid` share a shape whenever they agree on the
/// output surface.
struct ConvShape {
    std::uint32_t kernel;
    std::uint32_t stride;
    std::uint32_t groups;
    bool bias;
    ElementwiseShape input;
    ElementwiseShape output;
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
    /// Where the output surface sits in Apple's surface order: `inputs.size()`
    /// for last, 0 for a matmul, which lays the output out first, 1 for a
    /// broadcast, which puts it between the two operands.
    std::size_t outputBindingIndex = static_cast<std::size_t>(-1);
    /// Descriptor channels in output, input0, input1 order, before ANEC rebinding.
    std::array<std::uint32_t, 3> taskSurfaceChannels{4, 5, 6};
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
bool supportsMatmulParity(MatmulShape shape);
/// Encodes Apple's own matmul task stream for the geometry. `weights` is the
/// [columns, reduction] row-major fp16 constant, exactly the bytes the MIL
/// blob resolves to, and must be null when the second operand is runtime.
Program encodeMatmulParity(MatmulShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes);
/// Apple's constant-section permutation for a [columns, reduction] fp16
/// weight, whose row-group size depends on `rows` and `reduction`.
std::vector<std::uint8_t> packMatvecWeights(MatmulShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes);
/// True when the decoded Apple corpus covers this broadcast as one program.
bool supportsBroadcast(BinaryOperation operation, BroadcastOperand operand,
                       BroadcastShape shape);
/// Encodes Apple's broadcast task stream. `constant` holds one fp16 value per
/// channel for `BroadcastOperand::Constant` and is null otherwise;
/// `scalarBits` carries the inline fp16 operand for `BroadcastOperand::Scalar`.
Program encodeBroadcast(BinaryOperation operation, BroadcastOperand operand,
                        BroadcastShape shape,
                        const std::uint8_t *constant = nullptr,
                        std::size_t constantBytes = 0,
                        std::uint16_t scalarBits = 0x3800);
/// True when the decoded Apple corpus covers this softmax, layer_norm, or
/// reduction geometry as one multi-task program.
bool supportsNormParity(NormOperation operation, NormShape shape);
/// Encodes Apple's own task stream for the geometry, with the LUT constant
/// section the decoded oracle carries.
Program encodeNormParity(NormOperation operation, NormShape shape);
/// True when the decoded Apple corpus covers this convolution as one program.
bool supportsConvParity(ConvShape shape);
/// Apple's constant-section layout for a convolution weight. `weights` is the
/// MIL `[Cout, Cin / groups, kh, kw]` row-major fp16 constant, exactly the
/// bytes the blob resolves to, and `bias` holds one fp16 value per output
/// channel or is null.
std::vector<std::uint8_t> packConvWeights(ConvShape shape,
                                          const std::uint8_t *weights,
                                          std::size_t weightBytes,
                                          const std::uint8_t *bias = nullptr,
                                          std::size_t biasBytes = 0);
/// Encodes Apple's own convolution task stream for the geometry.
Program encodeConvParity(ConvShape shape, const std::uint8_t *weights,
                         std::size_t weightBytes,
                         const std::uint8_t *bias = nullptr,
                         std::size_t biasBytes = 0);
/// A decoded matmul or elementwise epilogue Apple folds into the producer's
/// own task instead of emitting the consumer as a separate program.
enum class PostOperation : std::uint8_t { Relu };

/// Fuses a decoded post-operation into an Apple-parity matmul program
/// (two tasks, constant weight). Verified byte-exact against the
/// `chain_pair_mm_relu` oracles: the compute task's NE word `0x0c804`
/// carries the clamp bit `0x00010000` and no other word moves.
/// Throws unless the program is exactly the decoded no-post-op form.
void fuseMatmulPostOperation(Program &program, PostOperation operation);

/// Fuses a relu into a one-task Apple elementwise program. Verified byte-exact
/// against `chain_add_relu_c512` on both targets: the PE word `0x08800`
/// (H14 `0x900`) carries the clamp bit `0x20` and no other word moves.
/// Throws unless the PE word is the decoded no-post-op form `0x00080000`.
void fuseElementwisePostOperation(Program &program, PostOperation operation);

/// Relinks already-routed programs into one task stream. Chains are accepted
/// only through the fusion path above, so this composes a single program
/// unchanged and refuses a multi-program list: relinking standalone task
/// streams keeps every task's own surface routing, and the decoded corpus
/// provides no task-DMA addressing for declared intermediate surfaces (reads
/// are wired to the input channels, writes to the output channel or L2), so
/// any multi-program relink would emit tasks that never exchange data.
Program composePrograms(const std::vector<Program> &programs);
std::vector<std::uint8_t> encodeANEC(const Program &program);

}
