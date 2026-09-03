#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "depthwise hardware sweep requires arm64 macOS" >&2
    exit 2
fi

model_key=56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
os_build=$(sw_vers -buildVersion)
cache_dir=/Library/Caches/com.apple.aned/$os_build/InMemoryModelCache/depthwise_exec/$model_key

make build/prepare_depthwise build/depthwise_exec -j4
sudo -n mkdir -p "$cache_dir"

for channels in 64 128 256 512; do
    output_root=build/hardware-depthwise-c$channels
    ./build/prepare_depthwise "$channels" "$output_root"
    artifact=$output_root/bundle/program-0.hwx
    hash=$(shasum -a 256 "$artifact" | awk '{print $1}')
    printf 'HWX depthwise channels=%s hash=%s\n' "$channels" "$hash"
    sudo -n install -m 0644 "$artifact" "$cache_dir/model.hwx"
    ./build/depthwise_exec "$output_root/bundle" "$cache_dir/model.hwx"
done

echo "depthwise M4 hardware sweep: PASS"
