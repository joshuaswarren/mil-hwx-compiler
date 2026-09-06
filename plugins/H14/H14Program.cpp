#include "H14Program.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace ane::h14 {
namespace {

enum class ElementwiseKind : std::uint8_t { BinaryRuntime, BinaryScalar, Unary };

/// One nonzero fp16 halfword run of a decoded constant section.
struct ConstantRun {
    std::uint32_t index;
    std::uint16_t bits;
    std::uint32_t count;
};

struct OracleTaskTemplate {
    ElementwiseKind kind;
    std::uint8_t operation;
    ElementwiseShape shape;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    const ConstantRun *constants;
    std::size_t constantRuns;
    std::size_t constantBytes;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
};

#include "H14ElementwiseTemplates.inc"

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

std::vector<std::uint8_t> streamBytes(const OracleTaskTemplate &source) {
    std::vector<std::uint8_t> bytes(source.textWords * sizeof(std::uint32_t));
    for (std::size_t index = 0; index != source.textWords; ++index) {
        const auto value = source.text[index];
        for (std::size_t byte = 0; byte != sizeof(value); ++byte)
            bytes[index * sizeof(value) + byte] =
                static_cast<std::uint8_t>(value >> (byte * 8));
    }
    return bytes;
}

std::vector<std::uint8_t> constantBytes(const OracleTaskTemplate &source) {
    std::vector<std::uint8_t> bytes(source.constantBytes, 0);
    for (std::size_t run = 0; run != source.constantRuns; ++run) {
        const auto &entry = source.constants[run];
        for (std::uint32_t offset = 0; offset != entry.count; ++offset) {
            const std::size_t index = (entry.index + offset) * 2;
            if (index + 1 >= bytes.size())
                throw std::logic_error("H14 constant run leaves its section");
            bytes[index] = static_cast<std::uint8_t>(entry.bits);
            bytes[index + 1] = static_cast<std::uint8_t>(entry.bits >> 8);
        }
    }
    return bytes;
}

std::uint64_t alignTile(std::uint64_t value) {
    return (value + tileBytes - 1) & ~(static_cast<std::uint64_t>(tileBytes) - 1);
}

TensorLayout elementwiseTensor(std::uint32_t index, ElementwiseShape shape) {
    const std::uint64_t row = std::max<std::uint64_t>(64, shape.width * 2);
    const std::uint64_t plane = row * shape.height;
    const std::uint64_t bytes = plane * shape.channels;
    return {index, {1, shape.channels, shape.height, shape.width, plane, row},
            alignTile(bytes)};
}

Program oracleProgram(const OracleTaskTemplate &source, std::size_t inputCount) {
    Program program;
    program.taskStream = streamBytes(source);
    program.constants = constantBytes(source);
    program.inputs.reserve(inputCount);
    for (std::size_t index = 0; index != inputCount; ++index)
        program.inputs.push_back(elementwiseTensor(
            static_cast<std::uint32_t>(5 + index), source.shape));
    program.output = elementwiseTensor(4, source.shape);
    const auto sizes = taskSizes(program.taskStream);
    if (sizes.size() != source.taskCount)
        throw std::logic_error("H14 task stream task count differs from the oracle");
    program.firstTaskBytes = sizes.front();
    program.taskCount = source.taskCount;
    program.constantOffsetBytes =
        (program.taskStream.size() + 0x3f) & ~std::size_t(0x3f);
    program.programRecordCount = source.programRecordCount;
    program.unresolvedDescriptorWord = source.unresolvedDescriptorWord;
    return program;
}

std::uint32_t loadLE32(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index != 4; ++index)
        value |= static_cast<std::uint32_t>(bytes[offset + index]) << (index * 8);
    return value;
}

void appendLE(std::vector<std::uint8_t> &bytes, std::uint64_t value,
              std::size_t width) {
    for (std::size_t index = 0; index != width; ++index)
        bytes.push_back(static_cast<std::uint8_t>(value >> (index * 8)));
}

std::uint64_t checkedAdd(std::uint64_t left, std::uint64_t right,
                         const char *what) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
        throw std::invalid_argument(what);
    return left + right;
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right,
                              const char *what) {
    if (left && right > std::numeric_limits<std::uint64_t>::max() / left)
        throw std::invalid_argument(what);
    return left * right;
}

std::uint32_t tileCount(std::uint64_t bytes, const char *what) {
    if (!bytes) throw std::invalid_argument(what);
    const auto tiles = checkedAdd(bytes, tileBytes - 1, what) / tileBytes;
    if (tiles > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(what);
    return static_cast<std::uint32_t>(tiles);
}

void validateTensor(const TensorLayout &tensor, std::uint32_t expectedIndex,
                    const char *kind) {
    if (tensor.index != expectedIndex)
        throw std::invalid_argument("unsupported ANEC channel index");
    const auto &nchw = tensor.nchw;
    if (!nchw[0] || !nchw[1] || !nchw[2] || !nchw[3] || !nchw[4] || !nchw[5])
        throw std::invalid_argument("tensor layout has a zero dimension or stride");
    if ((nchw[5] % 2) || (nchw[4] % nchw[5]) || nchw[5] / 2 < nchw[3])
        throw std::invalid_argument("unsupported tensor tiling layout");
    const auto minimumPlane = checkedMultiply(nchw[2], nchw[5],
                                              "tensor row span overflows");
    if (nchw[4] < minimumPlane)
        throw std::invalid_argument("unsupported tensor plane stride");
    const auto physicalSpan = checkedMultiply(
        checkedMultiply(nchw[0], nchw[1], "tensor channel span overflows"),
        nchw[4], "tensor physical span overflows");
    if (tensor.allocationBytes < physicalSpan)
        throw std::invalid_argument(kind);
}

constexpr std::size_t headerBytes = 0x1000;
constexpr std::size_t headerFieldsBytes = 0x6a8;
constexpr std::size_t channelCount = 32;
constexpr std::size_t tensorFields = 6;

} // namespace

std::vector<std::size_t> taskSizes(const std::vector<std::uint8_t> &stream) {
    if (stream.empty() || stream.size() % sizeof(std::uint32_t))
        throw std::invalid_argument("H14 task stream must be nonempty and word-aligned");
    std::vector<std::size_t> sizes;
    std::size_t offset = 0;
    while (offset != stream.size()) {
        if (stream.size() - offset < sizeof(std::uint32_t))
            throw std::invalid_argument("H14 task stream ends inside a task header");
        const auto words = (loadLE32(stream, offset) >> 16) & 0x7ff;
        if (!words) {
            // A zero-size frame consumes one 16-byte slot, as Apple's prefix does.
            const auto next = std::min(offset + taskAlignment, stream.size());
            for (std::size_t index = offset; index != next; ++index)
                if (stream[index])
                    throw std::invalid_argument("H14 zero-size task frame is nonzero");
            offset = next;
            continue;
        }
        if (words < taskHeaderWords)
            throw std::invalid_argument("H14 task declares fewer words than its header");
        const std::size_t bytes = words * sizeof(std::uint32_t);
        if (bytes > stream.size() - offset)
            throw std::invalid_argument("H14 task extends beyond its task stream");
        sizes.push_back(bytes);
        const auto next = std::min((offset + bytes + taskAlignment - 1) &
                                       ~(taskAlignment - 1), stream.size());
        for (std::size_t index = offset + bytes; index != next; ++index)
            if (stream[index])
                throw std::invalid_argument("H14 task alignment padding is nonzero");
        offset = next;
    }
    if (sizes.empty())
        throw std::invalid_argument("H14 task stream carries no tasks");
    return sizes;
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
        throw std::invalid_argument("H14 binary operation is outside the decoded parity envelope");
    if (scalarConstant && scalarBits != 0x3800)
        throw std::invalid_argument("H14 scalar operation requires the decoded fp16 0.5 operand");
    return oracleProgram(*source, scalarConstant ? 1 : 2);
}

Program encodeElementwise(UnaryOperation operation, ElementwiseShape shape) {
    const auto *source = elementwiseTemplate(
        ElementwiseKind::Unary, static_cast<std::uint8_t>(operation), shape);
    if (!source)
        throw std::invalid_argument("H14 unary operation is outside the decoded parity envelope");
    return oracleProgram(*source, 1);
}

std::vector<std::uint8_t> encodeANEC(const Program &program) {
    const auto sizes = taskSizes(program.taskStream);
    if (sizes.size() != program.taskCount || sizes.front() != program.firstTaskBytes)
        throw std::invalid_argument("H14 ANEC task metadata does not match its stream");
    if (program.constantOffsetBytes < program.taskStream.size() ||
        program.constantOffsetBytes % 0x40)
        throw std::invalid_argument("H14 ANEC constant offset is invalid");
    if (program.inputs.empty() || program.inputs.size() > 2)
        throw std::invalid_argument("H14 ANEC requires one or two input tensors");

    validateTensor(program.output, 4, "output allocation does not cover its physical span");
    for (std::size_t index = 0; index != program.inputs.size(); ++index)
        validateTensor(program.inputs[index], static_cast<std::uint32_t>(5 + index),
                       "input allocation does not cover its physical span");

    const auto constantsSize = static_cast<std::uint64_t>(program.constants.size());
    const auto contentSize = checkedAdd(program.constantOffsetBytes, constantsSize,
                                        "ANEC content size overflows");
    if (contentSize > std::numeric_limits<std::size_t>::max() - headerBytes)
        throw std::invalid_argument("ANEC output size overflows");

    std::array<std::uint32_t, channelCount> tiles{};
    tiles[0] = tileCount(contentSize, "ANEC content tile count overflows");
    tiles[4] = tileCount(program.output.allocationBytes,
                         "output tile count overflows");
    for (std::size_t index = 0; index != program.inputs.size(); ++index)
        tiles[5 + index] = tileCount(program.inputs[index].allocationBytes,
                                     "input tile count overflows");

    std::array<std::uint64_t, channelCount * tensorFields> layouts{};
    std::copy(program.output.nchw.begin(), program.output.nchw.end(),
              layouts.begin() + 4 * tensorFields);
    for (std::size_t index = 0; index != program.inputs.size(); ++index)
        std::copy(program.inputs[index].nchw.begin(),
                  program.inputs[index].nchw.end(),
                  layouts.begin() + (5 + index) * tensorFields);

    std::vector<std::uint8_t> anec;
    anec.reserve(headerBytes + static_cast<std::size_t>(contentSize));
    appendLE(anec, contentSize, 8);
    appendLE(anec, program.firstTaskBytes, 4);
    appendLE(anec, program.taskCount, 4);
    appendLE(anec, program.taskStream.size(), 8);
    appendLE(anec, constantsSize, 8);
    appendLE(anec, program.inputs.size(), 4);
    appendLE(anec, 1, 4);
    for (const auto tile : tiles) appendLE(anec, tile, 4);
    for (const auto field : layouts) appendLE(anec, field, 8);
    if (anec.size() != headerFieldsBytes)
        throw std::logic_error("ANEC header layout changed");
    anec.resize(headerBytes, 0);
    anec.insert(anec.end(), program.taskStream.begin(), program.taskStream.end());
    anec.resize(headerBytes + program.constantOffsetBytes, 0);
    anec.insert(anec.end(), program.constants.begin(), program.constants.end());
    return anec;
}

} // namespace ane::h14
