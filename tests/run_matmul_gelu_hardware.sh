#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "matmul-GELU hardware gate requires arm64 macOS" >&2
    exit 2
fi

make build/mil-hwxc build/matmul_gelu_exec -j4
cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/matmul_gelu_exec
for size in 128 256; do
    bundle_dir=$(mktemp -d /tmp/mil-hwx-matmul-gelu.XXXXXX)
    ./build/mil-hwxc --mil "tests/fixtures/matmul_gelu_${size}.mil" \
        --model-root tests/models/conv_relu --target H16G \
        --output "$bundle_dir"
    hashes=()
    index=0
    while [[ -f $bundle_dir/program-$index.hwx ]]; do
        artifact=$bundle_dir/program-$index.hwx
        digest=$(shasum -a 256 "$artifact" | awk '{print toupper($1)}')
        slot=$(printf '%064X' "$index")
        key=${digest}_${slot}_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
        directory=$cache_root/$key
        sudo -n mkdir -p "$directory"
        sudo -n install -m 0644 "$artifact" "$directory/model.hwx"
        hashes+=("$key")
        index=$((index + 1))
    done
    [[ $index -eq 1 ]]
    ./build/matmul_gelu_exec "$bundle_dir" "${hashes[@]}"
done
echo "matmul-GELU M4 hardware gate: PASS"
