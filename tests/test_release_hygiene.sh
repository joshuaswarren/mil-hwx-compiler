#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"

for path in \
    tests/oracles \
    plugins/H16G/Resources/h16g_attention_64k_skeleton.bin \
    plugins/H16G/Resources/h16g_attention_h4_s64_d64.constants \
    plugins/H16G/Resources/h16g_attention_h4_s64_d64.td \
    plugins/H16G/Resources/h16g_conv1x1_64k.bin \
    plugins/H16G/Resources/h16g_w8a8_80k_skeleton.bin \
    plugins/H16G/Resources/h16g_w8a8_c64_s64_l4.td \
    plugins/H16G/Encoding/H16GLinearProgramEncoder.h \
    plugins/H16G/Encoding/H16GLinearProgramEncoder.mm \
    lib/HWX/HWXBuilder.mm \
    lib/IR/ANEPrimitiveIR.mm \
    lib/IR/ANEMachineIR.mm \
    plugins/H16G/H16GPatternPlugin.mm \
    plugins/H16G/H16GLowering.mm \
    plugins/H16G/H16GConvReluEmitter.mm \
    plugins/H16G/H16GAttentionEmitter.mm \
    plugins/H16G/H16GW8A8Emitter.mm; do
    if [[ -e "$path" ]]; then
        echo "release tree contains excluded legacy or generated artifact: $path" >&2
        exit 1
    fi
done

if grep -REq 'static const uint32_t k(StartMatmul128|Scale128|ReduceMax128|CenterExp128|ReduceSum128|Division128|EndMatmul128)\[\]' \
        plugins/H16G/Encoding; then
    echo "release tree contains an opaque measured linear-program packet" >&2
    exit 1
fi

if find research -type f -name '*.hwx' -print -quit | grep -q .; then
    echo "release tree contains Apple-generated research HWX files" >&2
    exit 1
fi

if grep -REq --exclude=test_no_pattern_shortcuts.sh \
        --exclude=test_release_hygiene.sh \
        'tests/oracles|plugins/H16G/Resources/.*\.(bin|td|constants)|research/.*\.hwx' \
        Makefile include lib plugins tests tools; then
    echo "source or tests still reference excluded binary artifacts" >&2
    exit 1
fi

echo "release hygiene: PASS"
