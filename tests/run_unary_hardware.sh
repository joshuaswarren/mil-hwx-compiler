#!/bin/bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd);cd "$repo_dir"
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || exit 2
model_key=56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
cache_dir=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/unary_exec/$model_key
make build/prepare_unary build/unary_exec -j4
sudo -n mkdir -p "$cache_dir"
for geometry in "sigmoid 128" "sigmoid 256" "sigmoid 512" \
                "sigmoid 1024" "sigmoid 2048" "relu 256" "tanh 256" \
                "gelu 256" "silu 256" "exp 256" "log 256" "sqrt 256" \
                "rsqrt 256" "reciprocal 256";do
    read -r operation size <<<"$geometry";root=build/hardware-unary-${operation}-n${size}
    ./build/prepare_unary "$operation" "$size" "$root";artifact=$root/bundle/program-0.hwx
    printf 'HWX unary op=%s N=%s hash=%s\n' "$operation" "$size" \
        "$(shasum -a 256 "$artifact"|awk '{print $1}')"
    sudo -n install -m 0644 "$artifact" "$cache_dir/model.hwx"
    ./build/unary_exec "$operation" "$root/bundle" "$cache_dir/model.hwx"
done
echo "unary pointwise M4 hardware sweep: PASS"
