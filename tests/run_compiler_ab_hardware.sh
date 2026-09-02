#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "compiler A/B benchmark requires arm64 macOS" >&2
    exit 2
fi

warmup=${ANE_BENCHMARK_WARMUP:-20}
iterations=${ANE_BENCHMARK_ITERATIONS:-1000}
batches=${ANE_BENCHMARK_BATCHES:-5}
cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/compiler_ab_benchmark
temporary_root=$(mktemp -d /tmp/mil-hwx-compiler-ab.XXXXXX)
trap 'rm -rf "$temporary_root"' EXIT

make build/mil-hwxc build/compiler_ab_benchmark -j4

printf 'MACHINE model=%s os_build=%s warmup=%s iterations=%s batches=%s\n' \
    "$(sysctl -n hw.model)" "$(sw_vers -buildVersion)" \
    "$warmup" "$iterations" "$batches"

run_case() {
    local workload=$1
    local fixture=$2
    local bundle_dir=$temporary_root/$workload
    ./build/mil-hwxc --mil "$fixture" \
        --model-root tests/models/conv_relu --target H16G \
        --output "$bundle_dir"

    local hashes=()
    local index=0
    while [[ -f $bundle_dir/program-$index.hwx ]]; do
        local artifact=$bundle_dir/program-$index.hwx
        local digest
        digest=$(shasum -a 256 "$artifact" | awk '{print toupper($1)}')
        local slot
        slot=$(printf '%064X' "$index")
        local key=${digest}_${slot}_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
        local directory=$cache_root/$key
        sudo -n mkdir -p "$directory"
        sudo -n install -m 0644 "$artifact" "$directory/model.hwx"
        hashes+=("$key")
        index=$((index + 1))
    done
    if [[ $index -eq 0 ]]; then
        echo "no research HWX artifacts for $workload" >&2
        return 3
    fi
    ./build/compiler_ab_benchmark "$workload" "$fixture" "$bundle_dir" \
        "$warmup" "$iterations" "$batches" "${hashes[@]}"
}

# Establish fresh CPU-reference correctness before timing either compiler.
bash tests/run_online_reduction_hardware.sh
bash tests/run_affine_scan_hardware.sh
bash tests/run_matmul_gelu_hardware.sh

run_case fa2 tests/fixtures/fa2_fp16_s128_d128.mil
if run_case affine-scan tests/fixtures/affine_scan_fp16_4.mil; then
    :
else
    status=$?
    if [[ $status -ne 5 ]]; then
        exit "$status"
    fi
fi
run_case matmul-gelu tests/fixtures/matmul_gelu_256.mil

echo "compiler A/B benchmark: COMPLETE"
