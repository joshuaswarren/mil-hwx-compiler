#include "../plugins/H13/H13Program.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

std::uint64_t le(const std::vector<std::uint8_t> &bytes, std::size_t offset,
                 std::size_t width) {
    std::uint64_t value = 0;
    for (std::size_t i = 0; i != width; ++i)
        value |= static_cast<std::uint64_t>(bytes.at(offset + i)) << (8 * i);
    return value;
}

ane::h13::TensorLayout tensor(std::uint32_t index, std::uint64_t allocation) {
    return {index, {1, 2, 3, 4, 24, 8}, allocation};
}

template <typename Function>
void rejects(Function &&function) {
    try {
        function();
        assert(false);
    } catch (const std::invalid_argument &) {
    }
}

} // namespace

int main() {
    using namespace ane::h13;
    Program program;
    program.task.assign(taskBytes, 0);
    program.constants = {0x11, 0x22, 0x33};
    program.output = tensor(4, 48);
    program.inputs = {tensor(5, 48), tensor(6, 48)};

    const auto anec = encodeANEC(program);
    assert(anec.size() == 0x1000 + 0x280 + 3);
    assert(le(anec, 0, 8) == 0x283);
    assert(le(anec, 8, 4) == taskBytes);
    assert(le(anec, 12, 4) == 1);
    assert(le(anec, 16, 8) == taskBytes);
    assert(le(anec, 24, 8) == 3);
    assert(le(anec, 32, 4) == 2);
    assert(le(anec, 36, 4) == 1);
    assert(le(anec, 40, 4) == 1);
    assert(le(anec, 56, 4) == 1);
    assert(le(anec, 60, 4) == 1);
    assert(le(anec, 64, 4) == 1);
    assert(le(anec, 68, 4) == 0);
    assert(le(anec, 0xa8 + 4 * 48, 8) == 1);
    assert(le(anec, 0xa8 + 5 * 48, 8) == 1);
    assert(le(anec, 0xa8 + 6 * 48, 8) == 1);
    assert(anec.at(0x1000) == 0);
    assert(anec.at(0x1000 + taskBytes) == 0);
    assert(anec.at(0x1000 + 0x280) == 0x11);
    assert(anec.at(0x1000 + 0x282) == 0x33);

    Program invalid = program;
    invalid.task.pop_back();
    rejects([&] { encodeANEC(invalid); });
    invalid = program;
    invalid.inputs[0].index = 6;
    rejects([&] { encodeANEC(invalid); });
    invalid = program;
    invalid.output.allocationBytes = 47;
    rejects([&] { encodeANEC(invalid); });
    invalid = program;
    invalid.output.nchw[5] = 7;
    rejects([&] { encodeANEC(invalid); });
    invalid = program;
    invalid.inputs.clear();
    rejects([&] { encodeANEC(invalid); });
    program.taskSurfaceChannels = {5, 4, 6};
    program.task[32] = 0xa4;
    program.task[33] = 0x59;
    program.task[34] = 0x02;
    const auto routed = encodeANEC(program);
    const auto selectors = le(routed, 0x1020, 4);
    assert((selectors & 31) == 5);
    assert(((selectors >> 6) & 31) == 6);
    assert(((selectors >> 12) & 31) == 4);
    assert(((le(routed, 0x1000, 4) >> 16) & 255) == 64);
    assert(program.task[33] == 0x59);

    program.task.assign(80, 0);
    program.taskCount = 2;
    program.firstTaskBytes = 40;
    program.constantOffsetBytes = 128;
    program.task[6] = 9;
    program.task[28] = 40;
    program.task[32] = 0x24;
    program.task[40] = 1;
    program.task[73] = 0x50;
    program.task[74] = 0x02;
    const auto linked = encodeANEC(program);
    assert((le(linked, 0x1020, 4) & 31) == 5);
    assert(((le(linked, 0x1048, 4) >> 12) & 31) == 4);
    assert(((le(linked, 0x1028, 4) >> 16) & 255) == 64);
    assert((le(linked, 0x1028, 4) & 65535) == 1);
    program.task[28] = 36;
    rejects([&] { encodeANEC(program); });
}
