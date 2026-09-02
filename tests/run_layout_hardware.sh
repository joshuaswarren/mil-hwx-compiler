#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || exit 2

model_key=56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache
layout_cache=$cache_root/layout_exec/$model_key
layout_conv_cache=$cache_root/layout_conv_exec/$model_key

make build/prepare_layout build/layout_exec \
    build/prepare_layout_conv build/layout_conv_exec -j4
sudo -n mkdir -p "$layout_cache" "$layout_conv_cache"

run_layout_case() {
    local operation=$1 channels=$2 spatial=$3 block=$4
    local root=build/hardware-layout-${operation}-c${channels}-s${spatial}-b${block}
    ./build/prepare_layout "$root" "$operation" "$channels" "$spatial" "$block"
    local artifact=$root/program-0.hwx
    printf 'HWX layout op=%s C=%s S=%s B=%s hash=%s\n' \
        "$operation" "$channels" "$spatial" "$block" \
        "$(shasum -a 256 "$artifact"|awk '{print $1}')"
    sudo -n install -m 0644 "$artifact" "$layout_cache/model.hwx"
    ./build/layout_exec "$root" "$layout_cache/model.hwx" \
        "$operation" "$channels" "$spatial" "$block"
}

for channels in 8 16 24 32;do run_layout_case s2d "$channels" 128 4;done
for channels in 32 48 64 80 96;do run_layout_case s2d "$channels" 64 4;done
for channels in 8 16 24 32;do run_layout_case s2d "$channels" 128 8;done
for channels in 256 320 384 448 512;do run_layout_case d2s "$channels" 32 4;done
for channels in 512 1024 1536;do run_layout_case d2s "$channels" 16 8;done

for channels in 8 16 24 32;do
    root=build/hardware-layout-conv-c${channels}
    ./build/prepare_layout_conv "$root" "$channels"
    artifact=$root/program-0.hwx
    printf 'HWX layout-conv C=%s hash=%s\n' "$channels" \
        "$(shasum -a 256 "$artifact"|awk '{print $1}')"
    sudo -n install -m 0644 "$artifact" "$layout_conv_cache/model.hwx"
    ./build/layout_conv_exec "$root" "$layout_conv_cache/model.hwx" "$channels"
done

echo "S2D/D2S and fused layout M4 hardware sweep: PASS"
