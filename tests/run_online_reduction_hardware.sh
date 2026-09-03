#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "online-reduction hardware gate requires arm64 macOS" >&2
    exit 2
fi

make build/mil-hwxc build/online_reduction_exec -j4
bundle_dir=$(mktemp -d /tmp/mil-hwx-online.XXXXXX)
./build/mil-hwxc --mil tests/fixtures/fa2_fp16_s128_d128.mil \
    --model-root tests/models/conv_relu --target H16G --output "$bundle_dir"

cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/online_reduction_exec
hashes=()
artifacts=("$bundle_dir"/program-*.hwx)
for artifact in "${artifacts[@]}"; do
    digest=$(shasum -a 256 "$artifact" | awk '{print toupper($1)}')
    key=${digest}_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
    directory=$cache_root/$key
    sudo -n mkdir -p "$directory"
    sudo -n install -m 0644 "$artifact" "$directory/model.hwx"
    hashes+=("$key")
done
./build/online_reduction_exec "$bundle_dir" "${hashes[@]}"
echo "online-reduction M4 hardware gate: PASS"
