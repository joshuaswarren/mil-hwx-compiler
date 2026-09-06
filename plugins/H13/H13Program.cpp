#include "H13Program.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <stdexcept>

namespace ane::h13 {
namespace {

// Field values follow allbilly/ane e159e2d examples/elementwise.py and
// examples/gemm.py. Constants use the aligned 0x280 KDMA-base representation.
namespace reg {
constexpr std::size_t taskWord0 = 0x00;
constexpr std::size_t executionCycles = 0x08;
constexpr std::size_t debugLogEvents = 0x10;
constexpr std::size_t flags = 0x18;
constexpr std::size_t baseEnable = 0x20;
constexpr std::size_t taskWord9 = 0x24;
constexpr std::size_t kernelDMA = 0x28;
constexpr std::size_t firmwareDMA = 0x2c;
constexpr std::size_t commonStream = 0x124;
constexpr std::size_t inputDimensions = 0x128;
constexpr std::size_t commonPad0 = 0x12c;
constexpr std::size_t channelConfig = 0x130;
constexpr std::size_t inputChannels = 0x134;
constexpr std::size_t outputChannels = 0x138;
constexpr std::size_t outputDimensions = 0x13c;
constexpr std::size_t commonPad1 = 0x140;
constexpr std::size_t convolutionConfig = 0x144;
constexpr std::size_t commonPad2 = 0x148;
constexpr std::size_t groupConvolutionConfig = 0x14c;
constexpr std::size_t tileConfig = 0x150;
constexpr std::size_t commonPad3 = 0x154;
constexpr std::size_t pipelineConfig = 0x15c;
constexpr std::size_t taskInfo = 0x160;
constexpr std::size_t sourceStream = 0x168;
constexpr std::size_t sourceDMAConfig = 0x16c;
constexpr std::size_t sourceDMAPad0 = 0x170;
constexpr std::size_t sourceRowStride = 0x178;
constexpr std::size_t sourcePlaneStride = 0x17c;
constexpr std::size_t sourceDepthStride = 0x180;
constexpr std::size_t sourcePad2 = 0x18c;
constexpr std::size_t sourcePad3 = 0x190;
constexpr std::size_t sourcePad4 = 0x194;
constexpr std::size_t sourceFormat = 0x1a4;
constexpr std::size_t sourcePad8 = 0x1a8;
constexpr std::size_t sourcePadStream = 0x1ac;
constexpr std::size_t l2Stream = 0x1dc;
constexpr std::size_t l2SourceConfig = 0x1e4;
constexpr std::size_t l2SourceChannelStride = 0x1ec;
constexpr std::size_t l2SourceRowStride = 0x1f0;
constexpr std::size_t l2Pad0 = 0x1f4;
constexpr std::size_t l2Pad1 = 0x1f8;
constexpr std::size_t l2Pad2 = 0x1fc;
constexpr std::size_t l2Pad3 = 0x200;
constexpr std::size_t l2Pad4 = 0x204;
constexpr std::size_t l2Pad5 = 0x208;
constexpr std::size_t l2Pad6 = 0x20c;
constexpr std::size_t l2ResultConfig = 0x210;
constexpr std::size_t l2ResultBase = 0x214;
constexpr std::size_t convolutionResultChannelStride = 0x218;
constexpr std::size_t convolutionResultRowStride = 0x21c;
constexpr std::size_t l2ResultPad0 = 0x220;
constexpr std::size_t l2ResultPad1 = 0x224;
constexpr std::size_t processingElementStream = 0x228;
constexpr std::size_t processingElementConfig = 0x22c;
constexpr std::size_t biasScale = 0x230;
constexpr std::size_t preScale = 0x234;
constexpr std::size_t finalScale = 0x238;
constexpr std::size_t neuralEngineStream = 0x23c;
constexpr std::size_t kernelConfig = 0x240;
constexpr std::size_t multiplyAccumulateConfig = 0x244;
constexpr std::size_t postScale = 0x250;
constexpr std::size_t destinationStream = 0x254;
constexpr std::size_t destinationDMAConfig = 0x258;
constexpr std::size_t destinationRowStride = 0x260;
constexpr std::size_t destinationPlaneStride = 0x264;
constexpr std::size_t destinationDepthStride = 0x268;
constexpr std::size_t destinationFormat = 0x270;
} // namespace reg

constexpr std::size_t outputRows = 512;
constexpr std::size_t rowsPerBlock = 32;
constexpr std::size_t blockBytes = 0x8000;
constexpr std::size_t blockHalfwords = blockBytes / 2;
constexpr std::size_t dmaBlocks = outputRows / rowsPerBlock;
constexpr std::uint32_t halfOne = 0x3c00;

constexpr std::uint32_t streamHeader(std::uint32_t address,
                                     std::uint32_t words) {
    return ((words - 1) << 26) | address;
}

void putLE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * i));
}

void putBE32(std::vector<std::uint8_t> &bytes, std::size_t offset,
             std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::uint8_t>(value >> (8 * (3 - i)));
}

void putTaskHeader(std::vector<std::uint8_t> &task, std::uint32_t baseEnable,
                   std::uint32_t word9) {
    putLE32(task, reg::taskWord0, (0x40u << 16) | (1u << 25));
    putLE32(task, reg::executionCycles, 1058);
    putLE32(task, reg::debugLogEvents, 0x00fff86a);
    putLE32(task, reg::flags, (38u << 10) | (3u << 28));
    putLE32(task, reg::baseEnable, baseEnable);
    putLE32(task, reg::taskWord9, word9);
    putLE32(task, reg::kernelDMA, streamHeader(0x1f800, 62));
}

void putStreamHeaders(std::vector<std::uint8_t> &task) {
    putLE32(task, reg::commonStream, streamHeader(0x00000, 16));
    putLE32(task, reg::sourceStream, streamHeader(0x13800, 28));
    putLE32(task, reg::l2Stream, streamHeader(0x04800, 18));
    putLE32(task, reg::processingElementStream, streamHeader(0x08800, 4));
    putLE32(task, reg::neuralEngineStream, streamHeader(0x0c800, 5));
    putLE32(task, reg::destinationStream, streamHeader(0x17800, 7));
}

void putFirmwareDMA(std::vector<std::uint8_t> &task) {
    std::size_t offset = reg::firmwareDMA;
    putBE32(task, offset, 0x40000000);
    offset += 8;
    for (std::size_t i = 0; i != dmaBlocks; ++i, offset += 4)
        putBE32(task, offset, 0x81000000);
    for (std::uint32_t i = 0; i != dmaBlocks; ++i, offset += 4)
        putLE32(task, offset, i * blockBytes);
    for (std::size_t i = 0; i != dmaBlocks; ++i, offset += 4)
        putLE32(task, offset, blockBytes);
    for (std::size_t i = 0; i != 4; ++i, offset += 4)
        putBE32(task, offset, 0x80000000);
}

TensorLayout tensor(std::uint32_t index, std::uint64_t channels,
                    std::uint64_t allocationBytes) {
    return {index, {1, channels, 1, 1, 64, 64}, allocationBytes};
}

std::vector<std::uint8_t> binaryTask(std::uint32_t operation) {
    std::vector<std::uint8_t> task(taskBytes, 0);
    putTaskHeader(task,
                  6u | (1u << 5) | (5u << 6) | (1u << 11) | (4u << 12) |
                      (1u << 17),
                  0);
    putStreamHeaders(task);

    putLE32(task, reg::inputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad0, 1);
    putLE32(task, reg::channelConfig, 2u | (2u << 2) | (2u << 4));
    putLE32(task, reg::inputChannels, 64);
    putLE32(task, reg::outputChannels, 64);
    putLE32(task, reg::outputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad1, 1);
    putLE32(task, reg::convolutionConfig,
            1u | (1u << 5) | (1u << 13) | (1u << 15) | (1u << 28) |
                (1u << 30));
    putLE32(task, reg::commonPad2, 0x2041);
    putLE32(task, reg::groupConvolutionConfig, 1u | (1u << 16));
    putLE32(task, reg::tileConfig, 1);
    putLE32(task, reg::commonPad3, 4);
    putLE32(task, reg::pipelineConfig, 3u | (6u << 3));

    putLE32(task, reg::sourceDMAConfig,
            1u | (8u << 4) | (8u << 8) | (3u << 12) | (3u << 16));
    putLE32(task, reg::sourceDMAPad0, 0x33880);
    putLE32(task, reg::sourceRowStride, 0x40);
    putLE32(task, reg::sourcePlaneStride, 0x40);
    putLE32(task, reg::sourceDepthStride, 0x1000);
    putLE32(task, reg::sourcePad2, 0x40);
    putLE32(task, reg::sourcePad3, 0x40);
    putLE32(task, reg::sourcePad4, 0x1000);
    putLE32(task, reg::sourceFormat,
            1u | (3u << 4) | (2u << 12) | (1u << 24));
    putLE32(task, reg::sourcePad8, 0x2030);

    putLE32(task, reg::l2SourceConfig,
            2u | (1u << 4) | (1u << 5) | (1u << 6) | (1u << 8) |
                (1u << 20) | (1u << 22) | (1u << 24));
    putLE32(task, reg::l2SourceChannelStride, 0x10);
    putLE32(task, reg::l2SourceRowStride, 0x420);
    putLE32(task, reg::l2Pad0, 0x400);
    putLE32(task, reg::l2Pad1, 0x400);
    putLE32(task, reg::l2Pad2, 0x440);
    putLE32(task, reg::l2Pad3, 0x10);
    putLE32(task, reg::l2Pad4, 0x420);
    putLE32(task, reg::l2Pad5, 0x400);
    putLE32(task, reg::l2Pad6, 0x400);
    putLE32(task, reg::l2ResultConfig,
            2u | (2u << 2) | (1u << 4) | (1u << 5) | (1u << 6) |
                (1u << 8) | (1u << 20) | (1u << 22));
    putLE32(task, reg::l2ResultBase, 0x860);

    putLE32(task, reg::processingElementConfig,
            (2u << 18) | (operation << 2));
    putLE32(task, reg::biasScale, halfOne << 16);
    putLE32(task, reg::preScale, halfOne << 16);
    putLE32(task, reg::finalScale, 0x3f800000);
    if (operation == 1)
        putLE32(task, reg::multiplyAccumulateConfig, 0x30);

    putLE32(task, reg::destinationDMAConfig,
            1u | (12u << 4) | (1u << 26));
    putLE32(task, reg::destinationRowStride, 0x40);
    putLE32(task, reg::destinationPlaneStride, 0x40);
    putLE32(task, reg::destinationDepthStride, 0x1000);
    putLE32(task, reg::destinationFormat,
            1u | (3u << 4) | (2u << 12) | (1u << 24));
    return task;
}

std::vector<std::uint8_t> matvecTask() {
    std::vector<std::uint8_t> task(taskBytes, 0);
    putTaskHeader(task,
                  5u | (1u << 5) | (36u << 12) | (1u << 24) |
                      (1u << 26),
                  0x21);
    putFirmwareDMA(task);
    putStreamHeaders(task);

    putLE32(task, reg::inputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad0, 1);
    putLE32(task, reg::channelConfig, 2u | (2u << 4));
    putLE32(task, reg::inputChannels, 512);
    putLE32(task, reg::outputChannels, 512);
    putLE32(task, reg::outputDimensions, (1u << 16) | 1u);
    putLE32(task, reg::commonPad1, 1);
    putLE32(task, reg::convolutionConfig, 0x5000b421);
    putLE32(task, reg::commonPad2, 0x2041);
    putLE32(task, reg::groupConvolutionConfig, 0x00010001);
    putLE32(task, reg::tileConfig, 1);
    putLE32(task, reg::pipelineConfig, 0x00244405);
    putLE32(task, reg::taskInfo, 1u << 20);

    putLE32(task, reg::sourceDMAConfig,
            1u | (8u << 4) | (8u << 8) | (3u << 12) | (3u << 16));
    putLE32(task, reg::sourceDMAPad0, 0x8880);
    putLE32(task, reg::sourceRowStride, 0x40);
    putLE32(task, reg::sourcePlaneStride, 0x40);
    putLE32(task, reg::sourceDepthStride, 0x8000);
    putLE32(task, reg::sourceFormat,
            1u | (3u << 4) | (2u << 12) | (1u << 24));
    putLE32(task, reg::sourcePadStream, 0x100);

    putLE32(task, reg::l2SourceConfig, 0x00500172);
    putLE32(task, reg::l2SourceChannelStride, 0x10);
    putLE32(task, reg::l2SourceRowStride, 0x2030);
    putLE32(task, reg::l2Pad0, 0x2000);
    putLE32(task, reg::l2Pad1, 0x2000);
    putLE32(task, reg::l2ResultConfig, 0x00500172);
    putLE32(task, reg::l2ResultBase, 0x2030);
    putLE32(task, reg::convolutionResultChannelStride, 0x10);
    putLE32(task, reg::convolutionResultRowStride, 0x2020);
    putLE32(task, reg::l2ResultPad0, 0x2000);
    putLE32(task, reg::l2ResultPad1, 0x2000);

    putLE32(task, reg::kernelConfig, 0x82);
    putLE32(task, reg::multiplyAccumulateConfig, 0x00101c00);
    putLE32(task, reg::postScale, halfOne);

    putLE32(task, reg::destinationDMAConfig, 1u | (12u << 4));
    putLE32(task, reg::destinationRowStride, 0x40);
    putLE32(task, reg::destinationPlaneStride, 0x40);
    putLE32(task, reg::destinationDepthStride, 0x8000);
    putLE32(task, reg::destinationFormat,
            1u | (3u << 4) | (2u << 12) | (3u << 20) | (1u << 24));
    return task;
}

std::vector<std::uint8_t> packWeights(std::uint32_t reduction,
                                      const std::uint8_t *weights, bool transposeY) {
    std::vector<std::uint8_t> packed(dmaBlocks * blockBytes, 0);
    const std::size_t outputStride = transposeY ? reduction : 1;
    const std::size_t inputStride = transposeY ? 1 : outputRows;
    for (std::size_t block = 0; block != dmaBlocks; ++block) {
        const auto blockOffset = block * blockHalfwords;
        for (std::size_t column = 0; column != reduction; ++column) {
            const auto columnOffset = blockOffset + column * rowsPerBlock;
            for (std::size_t row = 0; row != rowsPerBlock; ++row) {
                const auto source = ((block * rowsPerBlock + row) * outputStride + column * inputStride) * 2;
                const auto destination = (columnOffset + row) * 2;
                packed[destination] = weights[source];
                packed[destination + 1] = weights[source + 1];
            }
        }
    }
    return packed;
}

enum class ElementwiseKind : std::uint8_t { BinaryRuntime, BinaryScalar, Unary };

struct OracleTaskTemplate {
    ElementwiseKind kind;
    std::uint8_t operation;
    ElementwiseShape shape;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantBytes;
};

struct OracleMatvecTemplate {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::uint64_t scratchAllocationBytes;
};

/// Which LUT layout a decoded normalization program's constant section holds.
/// The generator derives this from the recorded section hash, so the encoder
/// rebuilds the same bytes instead of guessing them from the operation.
enum class NormConstants : std::uint8_t {
    Zero,
    Exponential,
    ExponentialReciprocal,
};

struct OracleNormTemplate {
    NormOperation operation;
    ElementwiseShape input;
    ElementwiseShape output;
    std::uint32_t axisMask;
    bool keepDims;
    NormConstants constants;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

/// One decoded Apple convolution program, keyed by everything the compiler
/// knows before it picks one: the kernel, stride, group count, whether a bias
/// is folded in, and the two CHW surfaces. The padding follows from those, so
/// a 1x1 convolution's `same` and `valid` spellings share a template.
struct OracleConvTemplate {
    std::uint32_t kernel;
    std::uint32_t stride;
    std::uint32_t groups;
    bool bias;
    ElementwiseShape input;
    ElementwiseShape output;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

/// One decoded Apple matmul program, keyed by everything the compiler knows
/// before it picks a program: the geometry, both transpose flags, and whether
/// the second operand is runtime.
struct OracleMatmulTemplate {
    std::uint32_t rows;
    std::uint32_t reduction;
    std::uint32_t columns;
    bool transposeX;
    bool transposeY;
    bool runtimeWeight;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

struct OracleBroadcastTemplate {
    std::uint8_t operation;
    BroadcastOperand operand;
    BatchedShape x;
    BatchedShape y;
    const std::uint32_t *words;
    std::size_t wordCount;
    std::size_t firstTaskBytes;
    std::uint32_t taskCount;
    std::size_t constantOffsetBytes;
    std::size_t constantBytes;
    std::uint64_t scratchAllocationBytes;
};

#include "H13ElementwiseTemplates.inc"
#include "H13ElementwiseConstants.inc"
#include "H13MatvecTemplates.inc"
#include "H13NormTemplates.inc"
#include "H13EnvelopeTemplates.inc"
#include "H13ConvTemplates.inc"

bool sameShape(ElementwiseShape left, ElementwiseShape right) {
    return left.channels == right.channels && left.height == right.height &&
           left.width == right.width;
}

const OracleTaskTemplate *elementwiseTemplate(ElementwiseKind kind,
                                               std::uint8_t operation,
                                               ElementwiseShape shape) {
    for (const auto &candidate : kElementwiseTasks)
        if (candidate.kind == kind && candidate.operation == operation &&
            sameShape(candidate.shape, shape)) return &candidate;
    return nullptr;
}

/// The decoded matmul program for a geometry: the envelope tables first, which
/// carry both transpose flags, then the first campaign's `transpose_y=true`
/// constant-weight table, which only ever compiled untransposed x.
const OracleMatmulTemplate *matmulTemplate(MatmulShape shape) {
    for (const auto &candidate : kMatmulEnvelopeTasks)
        if (candidate.rows == shape.rows &&
            candidate.reduction == shape.reduction &&
            candidate.columns == shape.columns &&
            candidate.transposeX == shape.transposeX &&
            candidate.transposeY == shape.transposeY &&
            candidate.runtimeWeight == shape.runtimeWeight) return &candidate;
    return nullptr;
}

const OracleMatvecTemplate *matvecTemplate(MatmulShape shape) {
    if (shape.transposeX || !shape.transposeY || shape.runtimeWeight)
        return nullptr;
    for (const auto &candidate : kMatvecTasks)
        if (candidate.rows == shape.rows &&
            candidate.reduction == shape.reduction &&
            candidate.columns == shape.columns) return &candidate;
    return nullptr;
}

bool sameShape(BatchedShape left, BatchedShape right) {
    return left.batch == right.batch && left.channels == right.channels &&
           left.height == right.height && left.width == right.width;
}

const OracleBroadcastTemplate *broadcastTemplate(BinaryOperation operation,
                                                 BroadcastOperand operand,
                                                 BroadcastShape shape) {
    for (const auto &candidate : kBroadcastTasks)
        if (candidate.operation == static_cast<std::uint8_t>(operation) &&
            candidate.operand == operand && sameShape(candidate.x, shape.x) &&
            (operand == BroadcastOperand::Scalar ||
             sameShape(candidate.y, shape.y))) return &candidate;
    return nullptr;
}

const OracleNormTemplate *normTemplate(NormOperation operation,
                                       NormShape shape) {
    for (const auto &candidate : kNormTasks)
        if (candidate.operation == operation &&
            candidate.axisMask == shape.axisMask &&
            candidate.keepDims == shape.keepDims &&
            sameShape(candidate.input, shape.input) &&
            sameShape(candidate.output, shape.output)) return &candidate;
    return nullptr;
}

std::vector<std::uint8_t> taskBytesFor(const std::uint32_t *words,
                                       std::size_t wordCount) {
    std::vector<std::uint8_t> bytes(wordCount * sizeof(std::uint32_t));
    for (std::size_t index = 0; index != wordCount; ++index) {
        const auto value = words[index];
        for (std::size_t byte = 0; byte != sizeof(value); ++byte)
            bytes[index * sizeof(value) + byte] =
                static_cast<std::uint8_t>(value >> (byte * 8));
    }
    return bytes;
}

std::uint64_t alignTile(std::uint64_t value) {
    return (value + tileBytes - 1) & ~(static_cast<std::uint64_t>(tileBytes) - 1);
}

/// Apple's elementwise surface: 64-byte padded rows, and an allocation that
/// covers every batch element even though the descriptor's declared size
/// covers only one.
TensorLayout elementwiseTensor(std::uint32_t index, BatchedShape shape) {
    const std::uint64_t row = std::max<std::uint64_t>(64, shape.width * 2);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t element = plane * shape.channels;
    return {index,
            {shape.batch, shape.channels, shape.height, shape.width, plane, row},
            alignTile(element * shape.batch)};
}

TensorLayout elementwiseTensor(std::uint32_t index, ElementwiseShape shape) {
    return elementwiseTensor(index, BatchedShape{1, shape.channels,
                                                 shape.height, shape.width});
}

/// Apple's matvec surface: one dense [1, 1, rows, width] fp16 plane whose row
/// stride is the logical row, padded up to the 64-byte floor every H13
/// surface uses.
TensorLayout matvecTensor(std::uint32_t index, std::uint32_t rows,
                          std::uint32_t width) {
    const std::uint64_t row = std::max<std::uint64_t>(64, width * 2);
    const std::uint64_t plane = row * rows;
    return {index, {1, 1, rows, width, plane, row}, alignTile(plane)};
}

/// Writes a decoded LUT block, little-endian, at a byte offset in a section.
template <std::size_t N>
void putWords(std::vector<std::uint8_t> &bytes, std::size_t offset,
              const std::uint32_t (&words)[N]) {
    if (offset + N * sizeof(std::uint32_t) > bytes.size())
        throw std::logic_error("H13 constant table exceeds its decoded section");
    for (std::size_t index = 0; index != N; ++index)
        for (std::size_t byte = 0; byte != sizeof(std::uint32_t); ++byte)
            bytes[offset + index * sizeof(std::uint32_t) + byte] =
                static_cast<std::uint8_t>(words[index] >> (byte * 8));
}

template <std::size_t N>
std::vector<std::uint8_t> paddedConstants(const std::uint32_t (&words)[N],
                                          std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    putWords(bytes, 0, words);
    return bytes;
}

void putHalf(std::vector<std::uint8_t> &bytes, std::size_t index,
             std::uint16_t value) {
    bytes.at(index * 2) = static_cast<std::uint8_t>(value);
    bytes.at(index * 2 + 1) = static_cast<std::uint8_t>(value >> 8);
}

std::vector<std::uint8_t> leakyReluConstants(std::size_t size) {
    std::vector<std::uint8_t> bytes(size, 0);
    putHalf(bytes, 0, 0xfc00); putHalf(bytes, 1, 0x7c00);
    putHalf(bytes, 2, 0xfc00); putHalf(bytes, 3, 0x7c00);
    putHalf(bytes, 37, 0x3000); putHalf(bytes, 39, 0x3c00);
    putHalf(bytes, 41, 0x0040); putHalf(bytes, 42, 0x0001);
    return bytes;
}

/// The exact fp16 reciprocal of a power-of-two divisor, the only real_div
/// divisor class H13 lowers without rounding.
std::uint16_t exactReciprocal(std::uint16_t bits) {
    const std::uint16_t exponent = (bits >> 10) & 0x1f;
    if (!exponent || exponent == 0x1f || (bits & 0x03ff))
        throw std::invalid_argument("H13 real_div requires a power-of-two fp16 divisor");
    const std::uint16_t sign = bits & 0x8000;
    return exponent == 30 ? static_cast<std::uint16_t>(sign | 0x0200)
                          : static_cast<std::uint16_t>(sign | ((30 - exponent) << 10));
}

std::vector<std::uint8_t> scalarConstants(BinaryOperation operation,
                                          std::uint16_t value,
                                          std::size_t size,
                                          std::size_t elements) {
    std::vector<std::uint8_t> bytes(size, 0);
    if (operation == BinaryOperation::Maximum) {
        for (std::size_t index = 0; index != 37; ++index) putHalf(bytes, index, value);
        putHalf(bytes, 1, 0x7c00); putHalf(bytes, 3, 0x7c00);
        putHalf(bytes, 37, 0x3c00); putHalf(bytes, 39, 0x3c00);
        putHalf(bytes, 41, 0x0040); putHalf(bytes, 42, 0x0001);
    } else if (operation == BinaryOperation::Minimum) {
        putHalf(bytes, 0, 0xfc00); putHalf(bytes, 1, value);
        putHalf(bytes, 2, 0xfc00); putHalf(bytes, 3, value);
        putHalf(bytes, 37, 0x3c00); putHalf(bytes, 39, 0x3c00);
        putHalf(bytes, 41, 0x0040); putHalf(bytes, 42, 0x0001);
    } else if (operation == BinaryOperation::RealDivide) {
        const auto reciprocal = exactReciprocal(value);
        for (std::size_t index = 0; index != elements; ++index)
            putHalf(bytes, elements + index, reciprocal);
    }
    return bytes;
}

std::vector<std::uint8_t> unaryConstants(UnaryOperation operation,
                                         std::size_t size) {
    switch (operation) {
    case UnaryOperation::Absolute:
    case UnaryOperation::Relu:
        return std::vector<std::uint8_t>(size, 0);
    case UnaryOperation::Exponential:
        return paddedConstants(kExpKERNWords, size);
    case UnaryOperation::Gelu:
        return paddedConstants(kGeluKERNWords, size);
    case UnaryOperation::LeakyRelu:
        return leakyReluConstants(size);
    case UnaryOperation::ReciprocalSquareRoot:
        return paddedConstants(kRsqrtKERNWords, size);
    case UnaryOperation::Sigmoid:
        return paddedConstants(kSigmoidKERNWords, size);
    case UnaryOperation::Silu:
        return paddedConstants(kSiluKERNWords, size);
    case UnaryOperation::SquareRoot:
        return paddedConstants(kSqrtKERNWords, size);
    case UnaryOperation::Tanh:
        return paddedConstants(kTanhKERNWords, size);
    }
    throw std::invalid_argument("unsupported H13 unary operation");
}

/// Apple's softmax sections hold the exponential table at offset 0 and, when
/// the reduced axis is not the last one, the reciprocal table in the final
/// 128 bytes; layer_norm and every reduction leave the section zero.
std::vector<std::uint8_t> normConstants(NormConstants kind, std::size_t size) {
    if (kind == NormConstants::Zero)
        return std::vector<std::uint8_t>(size, 0);
    auto bytes = paddedConstants(kExpKERNWords, size);
    if (kind == NormConstants::ExponentialReciprocal)
        putWords(bytes, size - sizeof(kRecipKERNWords), kRecipKERNWords);
    return bytes;
}

/// Apple's per-channel section: one fp16 bias per channel, then one fp16
/// scale per channel. `add` stores the constant as the bias and scales by
/// 1.0; `mul` leaves the bias zero and stores the constant as the scale.
std::vector<std::uint8_t> perChannelConstants(BinaryOperation operation,
                                              std::uint32_t channels,
                                              const std::uint8_t *constant,
                                              std::size_t size) {
    const std::size_t block = static_cast<std::size_t>(channels) * 2;
    if (size != block * 2)
        throw std::logic_error(
            "H13 per-channel section is not a bias and a scale block");
    std::vector<std::uint8_t> bytes(size, 0);
    if (operation == BinaryOperation::Add) {
        std::memcpy(bytes.data(), constant, block);
        for (std::uint32_t index = 0; index != channels; ++index)
            putHalf(bytes, channels + index, halfOne);
    } else if (operation == BinaryOperation::Multiply) {
        std::memcpy(bytes.data() + block, constant, block);
    } else {
        throw std::invalid_argument(
            "H13 per-channel constants cover add and mul");
    }
    return bytes;
}

/// The broadcast result surface: every axis is the larger of the two
/// operands', and a scalar or per-channel constant leaves `x` unchanged.
BatchedShape broadcastOutput(BroadcastOperand operand, BroadcastShape shape) {
    if (operand != BroadcastOperand::Runtime) return shape.x;
    return {std::max(shape.x.batch, shape.y.batch),
            std::max(shape.x.channels, shape.y.channels),
            std::max(shape.x.height, shape.y.height),
            std::max(shape.x.width, shape.y.width)};
}

Program oracleProgram(const OracleTaskTemplate &source,
                      std::vector<std::uint8_t> constants,
                      std::size_t inputCount) {
    std::vector<TensorLayout> inputs;
    inputs.reserve(inputCount);
    for (std::size_t index = 0; index != inputCount; ++index)
        inputs.push_back(elementwiseTensor(static_cast<std::uint32_t>(5 + index),
                                           source.shape));
    const auto constantOffsetBytes =
        (source.wordCount * sizeof(std::uint32_t) + 0x3f) & ~std::size_t(0x3f);
    return {taskBytesFor(source.words, source.wordCount), std::move(constants),
            std::move(inputs),
            elementwiseTensor(4, source.shape), source.firstTaskBytes,
            source.taskCount, constantOffsetBytes, {}};
}

} // namespace

Program encodeBinary(BinaryOperation operation) {
    std::uint32_t operationCode;
    switch (operation) {
    case BinaryOperation::Add:
        operationCode = 0;
        break;
    case BinaryOperation::Multiply:
        operationCode = 1;
        break;
    case BinaryOperation::Maximum:
        operationCode = 2;
        break;
    case BinaryOperation::Minimum:
        operationCode = 3;
        break;
    default:
        throw std::invalid_argument("unsupported H13 binary operation");
    }
    return {binaryTask(operationCode),
            {},
            {tensor(5, 64, tileBytes), tensor(6, 64, tileBytes)},
            tensor(4, 64, tileBytes), taskBytes, 1, constantOffset, {}};
}

bool supportsElementwise(BinaryOperation operation, ElementwiseShape shape,
                         bool scalarConstant) {
    const auto kind = scalarConstant ? ElementwiseKind::BinaryScalar
                                     : ElementwiseKind::BinaryRuntime;
    return elementwiseTemplate(kind, static_cast<std::uint8_t>(operation), shape);
}

bool supportsElementwise(UnaryOperation operation, ElementwiseShape shape) {
    return elementwiseTemplate(ElementwiseKind::Unary,
                               static_cast<std::uint8_t>(operation), shape);
}
Program encodeElementwise(BinaryOperation operation, ElementwiseShape shape,
                          bool scalarConstant, std::uint16_t scalarBits) {
    const auto kind = scalarConstant ? ElementwiseKind::BinaryScalar
                                     : ElementwiseKind::BinaryRuntime;
    const auto *source = elementwiseTemplate(
        kind, static_cast<std::uint8_t>(operation), shape);
    if (!source)
        throw std::invalid_argument("H13 binary operation is outside the decoded parity envelope");
    if (scalarConstant && scalarBits != 0x3800)
        throw std::invalid_argument("H13 scalar operation requires the decoded fp16 0.5 operand");
    const auto constants = scalarConstant
        ? scalarConstants(operation, scalarBits, source->constantBytes,
                          static_cast<std::size_t>(shape.channels) * shape.height *
                              shape.width)
        : std::vector<std::uint8_t>(source->constantBytes, 0);
    return oracleProgram(*source, constants, scalarConstant ? 1 : 2);
}
Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape) {
    const auto *source = elementwiseTemplate(
        ElementwiseKind::Unary, static_cast<std::uint8_t>(operation), shape);
    if (!source)
        throw std::invalid_argument("H13 unary operation is outside the decoded parity envelope");
    return oracleProgram(*source, unaryConstants(operation, source->constantBytes), 1);
}

Program encodeMatvec(std::uint32_t reduction, const std::uint8_t *weights,
                     std::size_t weightBytes, bool transposeY) {
    if (reduction != 256 && reduction != 512)
        throw std::invalid_argument("H13 matvec reduction must be 256 or 512");
    if (weightBytes != outputRows * reduction * 2)
        throw std::invalid_argument("H13 matvec requires 512 * reduction fp16 weights");
    if (!weights)
        throw std::invalid_argument("H13 matvec weights must not be null");
    std::vector<std::size_t> relocations(16);
    for (std::size_t index = 0; index != relocations.size(); ++index)
        relocations[index] = 0x74 + index * sizeof(std::uint32_t);
    return {matvecTask(),
            packWeights(reduction, weights, transposeY),
            {tensor(5, reduction, static_cast<std::uint64_t>(reduction) * 64)},
            tensor(4, outputRows, 0x8000),
            taskBytes,
            1,
            constantOffset,
            std::move(relocations)};
}

bool supportsMatmulParity(MatmulShape shape) {
    return matmulTemplate(shape) || matvecTemplate(shape);
}

std::vector<std::uint8_t> packMatvecWeights(MatmulShape shape,
                                            const std::uint8_t *weights,
                                            std::size_t weightBytes) {
    if (!weights)
        throw std::invalid_argument("H13 matvec weights must not be null");
    if (!shape.reduction || shape.columns < 16 || shape.columns % 16 ||
        (shape.columns > 256 && shape.columns % 256))
        throw std::invalid_argument(
            "H13 matvec packing needs a positive reduction and 16 columns per "
            "row group, 256 per plane group above 256 columns");
    if (weightBytes !=
        static_cast<std::size_t>(shape.reduction) * shape.columns * 2)
        throw std::invalid_argument(
            "H13 matvec requires columns * reduction fp16 weights");
    // Apple interleaves `group` weight rows at halfword granularity, packs
    // each group of rows as one contiguous reduction-major plane, and orders
    // the planes by the low four bits of the group index. The group is 16 rows
    // until the column count falls below 256, and above 128 x rows -- the row
    // count at which Apple starts partitioning the program into
    // 1 + K * N / 2^19 tasks -- it also caps at 32768 / reduction halfwords.
    std::uint32_t group = std::min<std::uint32_t>(16, shape.columns / 16);
    if (shape.rows > 128)
        group = std::min<std::uint32_t>(group, 32768 / shape.reduction);
    group = std::max<std::uint32_t>(1, group);
    const std::uint32_t groups = shape.columns / group;
    std::vector<std::uint8_t> packed(weightBytes);
    for (std::uint32_t column = 0; column != shape.columns; ++column) {
        const std::uint32_t plane = column / group;
        const std::uint32_t destinationPlane =
            (plane % 16) * (groups / 16) + plane / 16;
        std::size_t destination =
            (static_cast<std::size_t>(destinationPlane) * shape.reduction *
                 group + column % group) * 2;
        std::size_t source = static_cast<std::size_t>(column) * shape.reduction * 2;
        for (std::uint32_t index = 0; index != shape.reduction; ++index) {
            packed[destination] = weights[source];
            packed[destination + 1] = weights[source + 1];
            destination += static_cast<std::size_t>(group) * 2;
            source += 2;
        }
    }
    return packed;
}

Program encodeMatmulParity(MatmulShape shape, const std::uint8_t *weights,
                           std::size_t weightBytes) {
    const auto *envelope = matmulTemplate(shape);
    const auto *legacy = envelope ? nullptr : matvecTemplate(shape);
    if (!envelope && !legacy)
        throw std::invalid_argument(
            "H13 matmul geometry is outside the decoded parity envelope");
    Program program;
    // Apple's matmul objects lay the output surface out first, then the second
    // operand, then x, and declare the operands in the opposite order.
    program.outputBindingIndex = 0;
    if (shape.runtimeWeight) {
        if (weights)
            throw std::invalid_argument(
                "H13 runtime-operand matmul takes no constant weight");
        program.constants.assign(envelope->constantBytes, 0);
        program.inputs = {
            shape.transposeY ? matvecTensor(5, shape.columns, shape.reduction)
                             : matvecTensor(5, shape.reduction, shape.columns),
            shape.transposeX ? matvecTensor(6, shape.reduction, shape.rows)
                             : matvecTensor(6, shape.rows, shape.reduction)};
    } else {
        program.constants = packMatvecWeights(shape, weights, weightBytes);
        program.inputs = {
            shape.transposeX ? matvecTensor(5, shape.reduction, shape.rows)
                             : matvecTensor(5, shape.rows, shape.reduction)};
    }
    program.output = matvecTensor(4, shape.rows, shape.columns);
    program.task = envelope
        ? taskBytesFor(envelope->words, envelope->wordCount)
        : taskBytesFor(legacy->words, legacy->wordCount);
    program.firstTaskBytes = envelope ? envelope->firstTaskBytes
                                      : legacy->firstTaskBytes;
    program.taskCount = envelope ? envelope->taskCount : legacy->taskCount;
    program.constantOffsetBytes = envelope ? envelope->constantOffsetBytes
                                           : legacy->constantOffsetBytes;
    program.scratchAllocationBytes = envelope
        ? envelope->scratchAllocationBytes : legacy->scratchAllocationBytes;
    return program;
}

bool supportsBroadcast(BinaryOperation operation, BroadcastOperand operand,
                       BroadcastShape shape) {
    return broadcastTemplate(operation, operand, shape);
}

Program encodeBroadcast(BinaryOperation operation, BroadcastOperand operand,
                        BroadcastShape shape, const std::uint8_t *constant,
                        std::size_t constantBytes, std::uint16_t scalarBits) {
    const auto *source = broadcastTemplate(operation, operand, shape);
    if (!source)
        throw std::invalid_argument(
            "H13 broadcast is outside the decoded parity envelope");
    if (operand == BroadcastOperand::Scalar && scalarBits != 0x3800)
        throw std::invalid_argument(
            "H13 scalar broadcast requires the decoded fp16 0.5 operand");
    Program program;
    program.task = taskBytesFor(source->words, source->wordCount);
    if (operand == BroadcastOperand::Constant) {
        if (!constant ||
            constantBytes != static_cast<std::size_t>(shape.x.channels) * 2)
            throw std::invalid_argument(
                "H13 per-channel broadcast requires one fp16 constant per channel");
        program.constants = perChannelConstants(operation, shape.x.channels,
                                                constant, source->constantBytes);
    } else {
        program.constants.assign(source->constantBytes, 0);
    }
    program.inputs = {elementwiseTensor(5, shape.x)};
    if (operand == BroadcastOperand::Runtime) {
        program.inputs.push_back(elementwiseTensor(6, shape.y));
        // Identical operands keep declaration order; any broadcast puts the
        // output surface between the two operands.
        if (!sameShape(shape.x, shape.y)) program.outputBindingIndex = 1;
    }
    program.output = elementwiseTensor(4, broadcastOutput(operand, shape));
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

bool supportsNormParity(NormOperation operation, NormShape shape) {
    return normTemplate(operation, shape);
}

Program encodeNormParity(NormOperation operation, NormShape shape) {
    const auto *source = normTemplate(operation, shape);
    if (!source)
        throw std::invalid_argument(
            "H13 normalization geometry is outside the decoded parity envelope");
    Program program;
    program.task = taskBytesFor(source->words, source->wordCount);
    program.constants = normConstants(source->constants, source->constantBytes);
    program.inputs = {elementwiseTensor(5, source->input)};
    program.output = elementwiseTensor(4, source->output);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

namespace {

const OracleConvTemplate *convTemplate(ConvShape shape) {
    for (const auto &candidate : kConvTasks)
        if (candidate.kernel == shape.kernel &&
            candidate.stride == shape.stride &&
            candidate.groups == shape.groups && candidate.bias == shape.bias &&
            sameShape(candidate.input, shape.input) &&
            sameShape(candidate.output, shape.output)) return &candidate;
    return nullptr;
}

std::uint64_t alignUp(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

/// Apple's convolution surface. The row is the width padded to 64 bytes, not
/// merely floored at it: a `pad_type="valid"` result 62 columns wide takes a
/// 128-byte row, which is what the decoded oracles record.
TensorLayout convTensor(std::uint32_t index, ElementwiseShape shape) {
    const std::uint64_t row =
        alignUp(static_cast<std::uint64_t>(shape.width) * 2, 64);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t element = plane * shape.channels;
    return {index, {1, shape.channels, shape.height, shape.width, plane, row},
            alignTile(element)};
}

/// How Apple splits the output channels into halfword interleave widths.
///
/// Every plane group holds 16 planes, so the channels are consumed in chunks
/// of `16 * lanes`, where `lanes` is the largest power of two at or below
/// `outputs / 16` and at most the cap: 32 for an input surface of 8 or fewer
/// pixels per side and 16 above it. 768 outputs at cap 32 is the case that
/// proves the split -- 512 channels interleave 32 wide, the remaining 256
/// interleave 16 wide -- and known-weight probes pin the cap's boundary
/// between 8 and 9 pixels.
std::vector<std::uint32_t> laneChunks(std::uint32_t outputs, std::uint32_t cap) {
    std::vector<std::uint32_t> chunks;
    std::uint32_t remaining = outputs;
    while (remaining) {
        std::uint32_t lanes = 1;
        if (remaining >= 16) {
            lanes = cap;
            while (lanes > remaining / 16) lanes /= 2;
        }
        chunks.push_back(lanes);
        remaining -= 16 * lanes;
    }
    return chunks;
}

std::uint32_t laneCap(ElementwiseShape input) {
    return std::max(input.height, input.width) <= 8 ? 32 : 16;
}

void putHalfword(std::vector<std::uint8_t> &bytes, std::size_t offset,
                 const std::uint8_t *source) {
    bytes[offset] = source[0];
    bytes[offset + 1] = source[1];
}

/// Apple's dense convolution section. A plane interleaves `lanes` output
/// channels at halfword granularity over `reduction` rows, preceded by one
/// bias row when the convolution has a bias; 16 plane groups follow, each
/// holding one plane per chunk. A grouped convolution pads every plane to 64
/// bytes and a groups-1 convolution pads only the plane group.
std::vector<std::uint8_t> packConvDense(std::uint32_t reduction,
                                        const std::vector<std::uint32_t> &chunks,
                                        const std::uint8_t *weights,
                                        const std::uint8_t *bias,
                                        bool planePadding) {
    const std::uint64_t rows = reduction + (bias ? 1 : 0);
    std::vector<std::uint64_t> planeBytes;
    planeBytes.reserve(chunks.size());
    for (const auto lanes : chunks) {
        const std::uint64_t span = rows * lanes * 2;
        planeBytes.push_back(planePadding ? alignUp(span, 64) : span);
    }
    std::uint64_t total = 0;
    for (const auto span : planeBytes) total += span;
    const std::uint64_t groupBytes = alignUp(total, 64);
    std::vector<std::uint8_t> packed(groupBytes * 16, 0);
    std::uint32_t start = 0;
    for (std::size_t chunk = 0; chunk != chunks.size(); ++chunk) {
        const std::uint32_t lanes = chunks[chunk];
        std::uint64_t offset = 0;
        for (std::size_t earlier = 0; earlier != chunk; ++earlier)
            offset += planeBytes[earlier];
        for (std::uint32_t group = 0; group != 16; ++group) {
            for (std::uint32_t lane = 0; lane != lanes; ++lane) {
                const std::uint32_t column = start + group * lanes + lane;
                std::size_t cursor = static_cast<std::size_t>(
                    group * groupBytes + offset + lane * 2);
                if (bias) {
                    putHalfword(packed, cursor, bias + column * 2);
                    cursor += lanes * 2;
                }
                const std::uint8_t *source =
                    weights + static_cast<std::size_t>(column) * reduction * 2;
                for (std::uint32_t row = 0; row != reduction; ++row) {
                    putHalfword(packed, cursor, source);
                    cursor += lanes * 2;
                    source += 2;
                }
            }
        }
        start += 16 * lanes;
    }
    return packed;
}

/// Apple's depthwise section: 16 lanes, each holding `outputs / 16` channels
/// back to back with no padding between them and the lane padded to 64 bytes.
/// A bias precedes each channel's taps.
std::vector<std::uint8_t> packConvDepthwise(std::uint32_t taps,
                                            std::uint32_t outputs,
                                            const std::uint8_t *weights,
                                            const std::uint8_t *bias) {
    const std::uint32_t lanes = std::min<std::uint32_t>(16, outputs);
    const std::uint32_t slots = outputs / lanes;
    const std::uint64_t rows = taps + (bias ? 1 : 0);
    const std::uint64_t laneBytes = alignUp(slots * rows * 2, 64);
    std::vector<std::uint8_t> packed(laneBytes * lanes, 0);
    for (std::uint32_t column = 0; column != outputs; ++column) {
        std::size_t cursor = static_cast<std::size_t>(
            (column % lanes) * laneBytes + (column / lanes) * rows * 2);
        if (bias) {
            putHalfword(packed, cursor, bias + column * 2);
            cursor += 2;
        }
        const std::uint8_t *source =
            weights + static_cast<std::size_t>(column) * taps * 2;
        for (std::uint32_t tap = 0; tap != taps; ++tap)
            putHalfword(packed, cursor + tap * 2, source + tap * 2);
    }
    return packed;
}

/// Apple's stride-2 section, which skips zero weights.
///
/// A plane holds the plane's `lanes` bias halfwords when the convolution has
/// a bias, a 16-bit count of the body bytes, then one row per reduction step.
/// A row leads with a zero byte and then carries, per group of eight lanes, a
/// mask byte whose bit `l` marks a lane whose weight is not zero followed by
/// those lanes' halfwords, and closes with `lanes / 2 - 2` zero bytes. Both
/// signed zeros count as zero, so the section's size depends on the values.
std::vector<std::uint8_t> packConvStrided(std::uint32_t reduction,
                                          const std::vector<std::uint32_t> &chunks,
                                          const std::uint8_t *weights,
                                          const std::uint8_t *bias) {
    std::vector<std::uint8_t> packed;
    std::uint32_t start = 0;
    for (const auto lanes : chunks) {
        const std::uint32_t padding = lanes >= 4 ? lanes / 2 - 2 : 0;
        for (std::uint32_t group = 0; group != 16; ++group) {
            const std::uint32_t first = start + group * lanes;
            std::vector<std::uint8_t> body;
            for (std::uint32_t row = 0; row != reduction; ++row) {
                body.push_back(0);
                for (std::uint32_t base = 0; base < lanes; base += 8) {
                    const std::uint32_t end = std::min(base + 8, lanes);
                    std::uint8_t mask = 0;
                    std::vector<std::uint8_t> values;
                    for (std::uint32_t lane = base; lane != end; ++lane) {
                        const std::uint8_t *value = weights +
                            (static_cast<std::size_t>(first + lane) * reduction +
                             row) * 2;
                        if (value[0] || (value[1] & 0x7f)) {
                            mask |= static_cast<std::uint8_t>(1u << (lane - base));
                            values.push_back(value[0]);
                            values.push_back(value[1]);
                        }
                    }
                    body.push_back(mask);
                    body.insert(body.end(), values.begin(), values.end());
                }
                body.insert(body.end(), padding, 0);
            }
            if (body.size() > 0xffff)
                throw std::invalid_argument(
                    "H13 strided convolution plane exceeds a 16-bit body count");
            std::vector<std::uint8_t> plane;
            if (bias)
                plane.insert(plane.end(), bias + first * 2,
                             bias + (first + lanes) * 2);
            plane.push_back(static_cast<std::uint8_t>(body.size()));
            plane.push_back(static_cast<std::uint8_t>(body.size() >> 8));
            plane.insert(plane.end(), body.begin(), body.end());
            plane.resize(alignUp(plane.size(), 64), 0);
            packed.insert(packed.end(), plane.begin(), plane.end());
        }
        start += 16 * lanes;
    }
    return packed;
}

} // namespace

bool supportsConvParity(ConvShape shape) { return convTemplate(shape); }

std::vector<std::uint8_t> packConvWeights(ConvShape shape,
                                          const std::uint8_t *weights,
                                          std::size_t weightBytes,
                                          const std::uint8_t *bias,
                                          std::size_t biasBytes) {
    const std::uint32_t taps = shape.kernel * shape.kernel;
    if (!shape.groups || shape.input.channels % shape.groups ||
        shape.output.channels % shape.groups)
        throw std::invalid_argument(
            "H13 convolution groups must divide both channel counts");
    const std::uint32_t reduction = shape.input.channels / shape.groups * taps;
    if (!weights || weightBytes !=
            static_cast<std::size_t>(shape.output.channels) * reduction * 2)
        throw std::invalid_argument(
            "H13 convolution requires Cout * Cin / groups * kh * kw fp16 weights");
    if (shape.bias != (bias != nullptr) ||
        (bias && biasBytes != static_cast<std::size_t>(shape.output.channels) * 2))
        throw std::invalid_argument(
            "H13 convolution bias must hold one fp16 value per output channel");
    if (shape.groups == shape.input.channels &&
        shape.groups == shape.output.channels)
        return packConvDepthwise(taps, shape.output.channels, weights, bias);
    const auto chunks = laneChunks(shape.output.channels, laneCap(shape.input));
    if (shape.stride > 1) {
        if (shape.groups != 1 || taps != 1 ||
            *std::max_element(chunks.begin(), chunks.end()) > 8)
            throw std::invalid_argument(
                "H13 has no derived strided packing for grouped, multi-tap, or "
                "16-lane convolutions");
        return packConvStrided(reduction, chunks, weights, bias);
    }
    if (shape.groups > 1) {
        const std::uint32_t lanes =
            std::max<std::uint32_t>(1, shape.output.channels / 64);
        return packConvDense(reduction,
                             std::vector<std::uint32_t>(
                                 shape.output.channels / lanes / 16, lanes),
                             weights, bias, true);
    }
    return packConvDense(reduction, chunks, weights, bias, false);
}

Program encodeConvParity(ConvShape shape, const std::uint8_t *weights,
                         std::size_t weightBytes, const std::uint8_t *bias,
                         std::size_t biasBytes) {
    const auto *source = convTemplate(shape);
    if (!source)
        throw std::invalid_argument(
            "H13 convolution geometry is outside the decoded parity envelope");
    Program program;
    program.task = taskBytesFor(source->words, source->wordCount);
    program.constants =
        packConvWeights(shape, weights, weightBytes, bias, biasBytes);
    if (program.constants.size() != source->constantBytes)
        throw std::logic_error(
            "H13 packed convolution section differs from the decoded size");
    program.inputs = {convTensor(5, shape.input)};
    program.output = convTensor(4, shape.output);
    program.firstTaskBytes = source->firstTaskBytes;
    program.taskCount = source->taskCount;
    program.constantOffsetBytes = source->constantOffsetBytes;
    program.scratchAllocationBytes = source->scratchAllocationBytes;
    return program;
}

} // namespace ane::h13
