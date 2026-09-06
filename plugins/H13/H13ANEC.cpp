#include "H13Program.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ane::h13 {
namespace {

constexpr std::size_t headerBytes = 0x1000;
constexpr std::size_t headerFieldsBytes = 0x6a8;
constexpr std::size_t channelCount = 32;
constexpr std::size_t tensorFields = 6;

void appendLE(std::vector<std::uint8_t> &bytes, std::uint64_t value,
              std::size_t width) {
    for (std::size_t i = 0; i != width; ++i)
        bytes.push_back(static_cast<std::uint8_t>(value >> (i * 8)));
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
    if (!bytes)
        throw std::invalid_argument(what);
    const auto rounded = checkedAdd(bytes, tileBytes - 1, what);
    const auto tiles = rounded / tileBytes;
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

void validateProgram(const Program &program) {
    if (program.task.empty() || program.task.size() % sizeof(std::uint32_t))
        throw std::invalid_argument("H13 ANEC requires a nonempty word-aligned task stream");
    if (!program.taskCount)
        throw std::invalid_argument("H13 ANEC requires at least one task");
    if (!program.firstTaskBytes || program.firstTaskBytes % sizeof(std::uint32_t) ||
        program.firstTaskBytes > program.task.size())
        throw std::invalid_argument("H13 ANEC first task size is invalid");
    if (program.constantOffsetBytes < program.task.size() ||
        program.constantOffsetBytes % 0x40)
        throw std::invalid_argument("H13 ANEC constant offset is invalid");
}

} // namespace

std::vector<std::uint8_t> encodeANEC(const Program &program) {
    validateProgram(program);
    if (program.inputs.empty() || program.inputs.size() > 2)
        throw std::invalid_argument("H13 ANEC requires one or two input tensors");

    validateTensor(program.output, 4, "output allocation does not cover its physical span");
    for (std::size_t i = 0; i != program.inputs.size(); ++i)
        validateTensor(program.inputs[i], static_cast<std::uint32_t>(5 + i),
                       "input allocation does not cover its physical span");

    const auto constantsSize = static_cast<std::uint64_t>(program.constants.size());
    const auto contentSize = checkedAdd(program.constantOffsetBytes, constantsSize,
                                        "ANEC content size overflows");
    if (contentSize > std::numeric_limits<std::size_t>::max() - headerBytes)
        throw std::invalid_argument("ANEC output size overflows");
    const auto contentTiles = tileCount(contentSize, "ANEC content tile count overflows");

    std::array<std::uint32_t, channelCount> tiles{};
    tiles[0] = contentTiles;
    tiles[4] = tileCount(program.output.allocationBytes,
                         "output tile count overflows");
    for (std::size_t i = 0; i != program.inputs.size(); ++i)
        tiles[5 + i] = tileCount(program.inputs[i].allocationBytes,
                                 "input tile count overflows");

    std::array<std::uint64_t, channelCount * tensorFields> layouts{};
    std::copy(program.output.nchw.begin(), program.output.nchw.end(),
              layouts.begin() + 4 * tensorFields);
    for (std::size_t i = 0; i != program.inputs.size(); ++i)
        std::copy(program.inputs[i].nchw.begin(), program.inputs[i].nchw.end(),
                  layouts.begin() + (5 + i) * tensorFields);

    std::vector<std::uint8_t> anec;
    anec.reserve(headerBytes + static_cast<std::size_t>(contentSize));
    appendLE(anec, contentSize, 8);
    appendLE(anec, program.firstTaskBytes, 4);
    appendLE(anec, program.taskCount, 4);
    appendLE(anec, program.task.size(), 8);
    appendLE(anec, constantsSize, 8);
    appendLE(anec, program.inputs.size(), 4);
    appendLE(anec, 1, 4);
    for (const auto tile : tiles)
        appendLE(anec, tile, 4);
    for (const auto field : layouts)
        appendLE(anec, field, 8);
    if (anec.size() != headerFieldsBytes)
        throw std::logic_error("ANEC header layout changed");
    anec.resize(headerBytes, 0);
    anec.insert(anec.end(), program.task.begin(), program.task.end());
    anec.resize(headerBytes + program.constantOffsetBytes, 0);
    anec.insert(anec.end(), program.constants.begin(), program.constants.end());
    return anec;
}

} // namespace ane::h13
