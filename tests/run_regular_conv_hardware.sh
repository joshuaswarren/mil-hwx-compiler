#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "regular Conv hardware sweep requires arm64 macOS" >&2
    exit 2
fi
model_key=56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
cache_dir=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/regular_conv_exec/$model_key
make build/prepare_regular_conv build/regular_conv_exec -j4
sudo -n mkdir -p "$cache_dir"
for geometry in "64 32 3" "64 64 3" "64 64 5" "128 32 3" "128 64 3" "128 64 5"; do
    read -r channels spatial kernel <<<"$geometry"
    output_root=build/hardware-regular-c${channels}-s${spatial}-k${kernel}
    ./build/prepare_regular_conv "$channels" "$spatial" "$kernel" "$output_root"
    artifact=$output_root/bundle/program-0.hwx
    printf 'HWX regular_conv C=%s S=%s K=%s hash=%s\n' \
        "$channels" "$spatial" "$kernel" "$(shasum -a 256 "$artifact" | awk '{print $1}')"
    sudo -n install -m 0644 "$artifact" "$cache_dir/model.hwx"
    ./build/regular_conv_exec "$output_root/bundle" "$cache_dir/model.hwx"
done
echo "regular Conv M4 hardware sweep: PASS"
