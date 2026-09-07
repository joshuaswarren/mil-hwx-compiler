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

std::uint32_t loadWord(const std::vector<std::uint8_t> &bytes, std::size_t offset) {
    std::uint32_t value = 0;
    for (std::size_t i = 0; i != 4; ++i)
        value |= static_cast<std::uint32_t>(bytes[offset + i]) << (i * 8);
    return value;
}

void storeWord(std::vector<std::uint8_t> &bytes, std::size_t offset,
               std::uint32_t value) {
    for (std::size_t i = 0; i != 4; ++i)
        bytes[offset + i] = static_cast<std::uint8_t>(value >> (i * 8));
}

void bindTasks(std::vector<std::uint8_t> &anec, const Program &program) {
    auto channels = program.taskSurfaceChannels;
    std::sort(channels.begin(), channels.end());
    if (channels != std::array<std::uint32_t, 3>{4, 5, 6})
        throw std::invalid_argument("H13 task surface channels must be a permutation of 4, 5, 6");
    std::size_t offset = 0, size = program.firstTaskBytes;
    for (std::uint32_t index = 0; index != program.taskCount; ++index) {
        if (offset > program.task.size() || size < 40 ||
            size > program.task.size() - offset)
            throw std::invalid_argument("H13 linked task is truncated");
        const auto base = headerBytes + offset;
        auto selectors = loadWord(anec, base + 32);
        for (unsigned shift : {0u, 6u, 12u}) {
            const auto channel = (selectors >> shift) & 31;
            if (channel < 4) continue;
            const auto role = std::find(program.taskSurfaceChannels.begin(),
                                        program.taskSurfaceChannels.end(), channel);
            if (role == program.taskSurfaceChannels.end())
                throw std::invalid_argument("H13 task selects an undeclared surface channel");
            const auto bound = static_cast<std::uint32_t>(
                4 + (role - program.taskSurfaceChannels.begin()));
            selectors = (selectors & ~(31u << shift)) | (bound << shift);
        }
        storeWord(anec, base + 32, selectors);
        storeWord(anec, base, (loadWord(anec, base) & ~0x00ff0000u) | 0x00400000u);
        const auto next = loadWord(anec, base + 28);
        if (index + 1 == program.taskCount) {
            if (next || offset + size != program.task.size())
                throw std::invalid_argument("H13 final task does not end the stream");
        } else {
            if (next % 4 || next < offset + size || next > program.task.size())
                throw std::invalid_argument("H13 linked task pointer is invalid");
            for (auto padding = offset + size; padding < next; ++padding)
                if (program.task[padding])
                    throw std::invalid_argument("H13 linked task padding is nonzero");
            size = (((loadWord(anec, base + 4) >> 16) & 0x1ff) + 1) * 4;
            offset = next;
        }
    }
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
    bindTasks(anec, program);
    anec.resize(headerBytes + program.constantOffsetBytes, 0);
    anec.insert(anec.end(), program.constants.begin(), program.constants.end());
    return anec;
}

} // namespace ane::h13
