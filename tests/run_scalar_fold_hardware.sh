#!/bin/bash
# M4 gate for the scalar-fold compositions on non-attention graphs.
#
# Each workload is compiled twice: once with composition enabled, which must
# yield one program, and once with composition disabled, which must yield the
# standalone programs. Both bundles run on the ANE against a CPU reference.
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "scalar fold gate requires arm64 macOS" >&2
    exit 2
fi

make build/mil-hwxc build/scalar_fold_exec -j4
cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/scalar_fold_exec
temporary_root=$(mktemp -d /tmp/mil-hwx-scalar-fold.XXXXXX)
trap 'rm -rf "$temporary_root"' EXIT

run_bundle() {
    local workload=$1
    local fixture=$2
    local expected_programs=$3
    local disable=$4
    local bundle_dir=$temporary_root/$workload-$disable
    if [[ $disable == 1 ]]; then
        ANE_DISABLE_PROGRAM_COMPOSITION=1 ./build/mil-hwxc --mil "$fixture" \
            --model-root tests/models/conv_relu --target H16G \
            --output "$bundle_dir"
    else
        ./build/mil-hwxc --mil "$fixture" \
            --model-root tests/models/conv_relu --target H16G \
            --output "$bundle_dir"
    fi
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
    if [[ $index -ne $expected_programs ]]; then
        echo "$workload composition=$((1 - disable)) expected $expected_programs programs, found $index" >&2
        return 1
    fi
    ./build/scalar_fold_exec "$workload" "$bundle_dir" "${hashes[@]}"
}

run_bundle matmul-scale tests/fixtures/matmul_scale_128.mil 1 0
run_bundle matmul-scale tests/fixtures/matmul_scale_128.mil 2 1
run_bundle scale-exp tests/fixtures/scale_exp_128.mil 1 0
run_bundle scale-exp tests/fixtures/scale_exp_128.mil 2 1
echo "scalar fold M4 hardware gate: PASS"
