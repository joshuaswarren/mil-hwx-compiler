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
    program.task.assign(taskBytes, 0xa5);
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
    assert(anec.at(0x1000) == 0xa5);
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
}
