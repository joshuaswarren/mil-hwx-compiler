// H14 F5 chain template.
//
// Generated from the Apple-decoded `env_chain_creadout_h16_s128` oracle
// (research/oracles/batch2/h14/env_chain_creadout_h16_s128.json, 49152 B
// hwx, 3 tasks, 150 text words, 4 fp16 tensors all at [1, 16, 128, 1]).
// Task stream and constant section were captured from the Mac-side mint at
// /tmp/hwx_capture_batch5/env_chain_creadout_h16_s128.hwx on 2026-09-30.
//
// Re-mint on the Mac Studio with `ane-compile-hwx` if the chain
// re-decodes to different words.

struct OracleChainTensor {
    std::uint32_t index;
    std::array<std::uint64_t, 6> nchw;
    std::uint64_t allocationBytes;
    std::uint32_t elementCode;
    std::uint32_t elementBytes;
};

struct OracleChainTemplate {
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
    const std::uint32_t *text;
    std::size_t textWords;
    std::uint32_t taskCount;
    const ConstantRun *constants;
    std::size_t constantRuns;
    std::size_t constantBytes;
    std::uint32_t programRecordCount;
    std::uint32_t unresolvedDescriptorWord;
    std::uint32_t scratchDescriptorWord;
    const OracleChainTensor *inputs;
    std::uint32_t inputCount;
    const OracleChainTensor *output;
};

// env_chain_creadout_h16_s128 (sigmoid -> mul -> add)
// textWords = 150 (three tasks followed by four zero alignment words).
static constexpr std::uint32_t kChainCReadOutText0[] = {
    0x00320000, 0x00000003, 0x00000008, 0x00000000, 0x00fff868, 0x00000000, 0x00000001, 0x00050001,
    0x00000640, 0x00010240, 0x84000679, 0x00000021, 0x00000080, 0xb3280000, 0x00800001, 0x00800001,
    0x00000000, 0x00014001, 0x00000080, 0x00240001, 0x00100000, 0x80310440, 0x00000e21, 0x000000ce,
    0x00000040, 0x00002000, 0x00000454, 0x01002031, 0x9a0f8141, 0x00500172, 0x00008800, 0x00000010,
    0x00000110, 0x00000100, 0x00000100, 0x00500170, 0x00000010, 0x00000110, 0x80048340, 0x00010082,
    0x00121c0c, 0x00003c00, 0x20808680, 0x00000000, 0x00000000, 0x2080067d, 0x00000000, 0x22008444,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00290001, 0x00000003, 0x00000000, 0x00000000, 0x00fff800, 0x00000000, 0x00000003, 0x00000001,
    0xb6080000, 0x00800001, 0x00800001, 0x00000080, 0x00000044, 0x00000031, 0x00000000, 0x80310440,
    0x00010e21, 0x000000ce, 0x00000040, 0x00002000, 0x00000454, 0x01002031, 0x9c6f8141, 0x00000172,
    0x00010800, 0x00000010, 0x00000100, 0x00000100, 0x00000100, 0x00000010, 0x00000110, 0x00008800,
    0x00000010, 0x00000100, 0x00000240, 0x00080004, 0x00000549, 0x0000dead, 0x23808444, 0x00000000,
    0x00000000, 0x00000000, 0x00000000, 0x00000000,
    0x00320002, 0x00000007, 0x00000022, 0x00000000, 0x00fff868, 0x00000000, 0x00000000, 0x00000001,
    0x00000640, 0x00000080, 0xb6080000, 0x00800001, 0x00800001, 0x00000080, 0x00000044, 0x00000032,
    0x00000000, 0x8c028440, 0x00000021, 0x00020e20, 0x000000ce, 0x00000040, 0x00002000, 0x00000455,
    0x01002031, 0x87f38141, 0x01400140, 0x00008800, 0x00000010, 0x00000100, 0x00010800, 0x00000010,
    0x00000100, 0x00000100, 0x00000100, 0x0000014a, 0x00018800, 0x00000240, 0x00080000, 0x804c0540,
    0x01020e31, 0x00000040, 0x00002000, 0x01302031, 0x2300844a, 0x00000000, 0x00000000, 0x22808542,
    0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000,
};

static constexpr ConstantRun kChainCReadOutConstants0[] = {
    {0, 0xc8f8, 1},
    {1, 0x4829, 1},
    {3, 0x3c00, 1},
    {4, 0x0d7f, 1},
    {5, 0x1087, 1},
    {6, 0x1377, 1},
    {7, 0x1626, 1},
    {8, 0x1910, 1},
    {9, 0x1c2b, 1},
    {10, 0x1eda, 1},
    {11, 0x21a0, 1},
    {12, 0x249b, 1},
    {13, 0x2781, 1},
    {14, 0x2a12, 1},
    {15, 0x2cdb, 1},
    {16, 0x2fa1, 1},
    {17, 0x31d6, 1},
    {18, 0x344e, 1},
    {19, 0x360a, 1},
    {20, 0x3800, 1},
    {21, 0x38fb, 1},
    {22, 0x39d9, 1},
    {23, 0x3a8a, 1},
    {24, 0x3b0c, 1},
    {25, 0x3b65, 1},
    {26, 0x3b9f, 1},
    {27, 0x3bc4, 1},
    {28, 0x3bdb, 1},
    {29, 0x3be9, 1},
    {30, 0x3bf2, 1},
    {31, 0x3bf8, 1},
    {32, 0x3bfb, 1},
    {33, 0x3bfd, 1},
    {34, 0x3bfe, 1},
    {35, 0x3bff, 2},
    {37, 0x09ac, 1},
    {38, 0x170c, 1},
    {39, 0x163e, 1},
    {40, 0x3be6, 1},
    {41, 0x0001, 2},
};

static constexpr OracleChainTensor kChainCReadOutInputs0[] = {
    {5, {1, 16, 128, 1, 8192, 64}, 131072, 5, 2},
    {6, {1, 16, 128, 1, 8192, 64}, 131072, 5, 2},
    {7, {1, 16, 128, 1, 8192, 64}, 131072, 5, 2},
};

static constexpr OracleChainTensor kChainCReadOutOutput0 = {
    4, {1, 16, 128, 1, 8192, 64}, 131072, 5, 2,
};

static constexpr OracleChainTemplate kChainTemplates[] = {
    {16, 128, 1, kChainCReadOutText0, std::size(kChainCReadOutText0), 3,
     kChainCReadOutConstants0, std::size(kChainCReadOutConstants0), 128,
     0x00000030, 0x00000007, 0x00000000,
     kChainCReadOutInputs0, std::size(kChainCReadOutInputs0),
     &kChainCReadOutOutput0},
};

// Apple provides four 128 KiB tensor allocations for this chain.

