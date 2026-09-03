#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "chunked DeltaNet hardware gate requires arm64 macOS" >&2
    exit 2
fi

make build/mil-hwxc build/chunked_deltanet_exec -j4
bundle_dir=$(mktemp -d /tmp/mil-hwx-chunked-delta.XXXXXX)
./build/mil-hwxc \
    --mil tests/fixtures/chunked_deltanet_fp16_c128_d128.mil \
    --model-root tests/models/conv_relu --target H16G --output "$bundle_dir"

artifact_count=$(find "$bundle_dir" -name 'program-*.hwx' -type f | wc -l | tr -d ' ')
if [[ $artifact_count != 58 ]]; then
    echo "expected 58 generic artifacts, found $artifact_count" >&2
    exit 3
fi

cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/chunked_deltanet_exec
hashes=()
for ((index=0; index<artifact_count; ++index)); do
    artifact=$bundle_dir/program-$index.hwx
    digest=$(shasum -a 256 "$artifact" | awk '{print toupper($1)}')
    slot=$(printf '%064X' "$index")
    key=${digest}_${slot}_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
    directory=$cache_root/$key
    sudo -n mkdir -p "$directory"
    sudo -n install -m 0644 "$artifact" "$directory/model.hwx"
    hashes+=("$key")
done
./build/chunked_deltanet_exec \
    tests/fixtures/chunked_deltanet_fp16_c128_d128.mil \
    "$bundle_dir" "${hashes[@]}"
echo "chunked DeltaNet M4 hardware gate: PASS"
