#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "online-reduction fallback gate requires arm64 macOS" >&2
    exit 2
fi

make build/mil-hwxc build/test_compiler_e2e build/online_reduction_exec -j4
./build/test_compiler_e2e

# The forced-fallback gate proves that the standalone primitive programs stay
# usable on hardware. It compiles its own bundle with composition disabled so
# the eight standalone programs are exercised even when the planner would
# otherwise coalesce adjacent tasks.
bundle_dir=build/fa2-fallback
rm -rf "$bundle_dir"
ANE_DISABLE_PROGRAM_COMPOSITION=1 ./build/mil-hwxc \
    --mil tests/fixtures/fa2_fp16_s128_d128.mil \
    --model-root tests/models/conv_relu --target H16G --output "$bundle_dir"

cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/online_reduction_exec
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
if [[ $index -ne 8 ]]; then
    echo "forced fallback expected 8 standalone programs, found $index" >&2
    exit 1
fi
./build/online_reduction_exec "$bundle_dir" "${hashes[@]}"
echo "online-reduction forced fallback M4 hardware gate: PASS"
