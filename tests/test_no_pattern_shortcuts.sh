#!/bin/bash
set -euo pipefail

for source in lib/Driver/ANECompiler.mm tools/mil-hwxc.mm; do
    if grep -Eq 'H16GPatternPlugin|H16GConvReluEmitter|H16GAttentionEmitter|H16GW8A8Emitter|HWXBuilder' "$source"; then
        echo "legacy pattern shortcut remains in production source: $source" >&2
        exit 1
    fi
done

build_command=$(make -Bn build/mil-hwxc)
if grep -Eq 'H16GPatternPlugin|H16GConvReluEmitter|H16GAttentionEmitter|H16GW8A8Emitter|HWXBuilder|ANEPrimitiveIR\.mm|ANEMachineIR\.mm|plugins/H16G/Resources|skeleton' <<<"$build_command"; then
    echo "production CLI still links a legacy pattern or skeleton emitter" >&2
    exit 1
fi

if grep -Eq 'H16GTaskPacket|writePacket|static const uint32_t k(Split|Transpose|Matmul|Scale|Reduce|Center|Reciprocal|Normalize)' \
        plugins/H16G/Encoding/H16GMixedTaskEncoder.mm; then
    echo "mixed-task encoder still inserts a pre-decoded register image" >&2
    exit 1
fi

if grep -REq 'dataWithContentsOfFile|plugins/H16G/Resources|skeleton|descriptorRow' \
        plugins/H16G/Encoding lib/HWX/HWXObjectWriter.mm \
        lib/Driver/ANEStagedCompiler.mm; then
    echo "production encoding path reads or patches an external descriptor artifact" >&2
    exit 1
fi

if grep -Eq 'sourceFunction\.name|matmul_gelu|flashattention|flash_attention|deltanet|gated_delta' \
        plugins/H16G/Encoding/H16GProgramAssembler.mm \
        plugins/H16G/Encoding/H16GSRAMChainEncoder.mm \
        plugins/H16G/Encoding/H16GTaskEncoder.mm; then
    echo "task assembly contains a graph-name or workload-name shortcut" >&2
    exit 1
fi

if grep -REq 'deltanet|delta_net|gated_delta|chunked_delta' \
        lib/Planning lib/Transform plugins/H16G/Encoding; then
    echo "production lowering contains a DeltaNet workload-name shortcut" >&2
    exit 1
fi

if nm build/mil-hwxc | grep -Eq \
        'H16GPatternPlugin|H16GConvReluEmitter|H16GAttentionEmitter|H16GW8A8Emitter|HWXBuilder|ANEMachineIR|ANEPrimitiveIR'; then
    echo "production binary contains a legacy pattern, skeleton, or old-IR symbol" >&2
    exit 1
fi

echo "production staged-route guard: PASS"
