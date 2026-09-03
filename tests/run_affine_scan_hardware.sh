#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "affine-scan hardware gate requires arm64 macOS" >&2
    exit 2
fi

make build/mil-hwxc build/affine_scan_exec -j4
bundle_dir=$(mktemp -d /tmp/mil-hwx-scan.XXXXXX)
./build/mil-hwxc --mil tests/fixtures/affine_scan_fp16_4.mil \
    --model-root tests/models/conv_relu --target H16G --output "$bundle_dir"

cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/affine_scan_exec
hashes=()
for index in {0..7}; do
    artifact=$bundle_dir/program-$index.hwx
    digest=$(shasum -a 256 "$artifact" | awk '{print toupper($1)}')
    slot=$(printf '%064X' "$index")
    key=${digest}_${slot}_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
    directory=$cache_root/$key
    sudo -n mkdir -p "$directory"
    sudo -n install -m 0644 "$artifact" "$directory/model.hwx"
    hashes+=("$key")
done
./build/affine_scan_exec "$bundle_dir" "${hashes[@]}"
echo "affine-scan M4 hardware gate: PASS"
