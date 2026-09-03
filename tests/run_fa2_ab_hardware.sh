#!/bin/bash
# FA2-only compiler A/B benchmark.
#
# Unlike run_compiler_ab_hardware.sh this entry point compiles, provisions,
# and times only the FA2 fixture. No affine-scan or matmul-GELU program runs
# inside the timed window, so a powermetrics capture around this script sees
# only FA2 traffic on the ANE.
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "FA2 A/B benchmark requires arm64 macOS" >&2
    exit 2
fi

warmup=${ANE_BENCHMARK_WARMUP:-20}
iterations=${ANE_BENCHMARK_ITERATIONS:-2000}
batches=${ANE_BENCHMARK_BATCHES:-5}
fixture=${ANE_FA2_FIXTURE:-tests/fixtures/fa2_fp16_s128_d128.mil}
cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/compiler_ab_benchmark
temporary_root=$(mktemp -d /tmp/mil-hwx-fa2-ab.XXXXXX)
trap 'rm -rf "$temporary_root"' EXIT

make build/mil-hwxc build/compiler_ab_benchmark -j4

bundle_dir=$temporary_root/fa2
./build/mil-hwxc --mil "$fixture" \
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
if [[ $index -eq 0 ]]; then
    echo "no research HWX artifacts for fa2" >&2
    exit 3
fi

printf 'MACHINE model=%s os_build=%s warmup=%s iterations=%s batches=%s\n' \
    "$(sysctl -n hw.model)" "$(sw_vers -buildVersion)" \
    "$warmup" "$iterations" "$batches"
printf 'PROGRAMS workload=fa2 count=%s\n' "$index"
./build/compiler_ab_benchmark fa2 "$fixture" "$bundle_dir" \
    "$warmup" "$iterations" "$batches" "${hashes[@]}"
echo "FA2 A/B benchmark: COMPLETE"
