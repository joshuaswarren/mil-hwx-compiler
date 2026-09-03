#!/bin/bash
# Profile every program submission of one compiled MIL fixture on the M4.
#
#   bash tests/run_program_profile_hardware.sh [FIXTURE] [WARMUP] [REPEATS]
#
# Defaults to the FA2 fixture. Set ANE_DISABLE_PROGRAM_COMPOSITION=1 to
# profile the forced standalone path.
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "program profile requires arm64 macOS" >&2
    exit 2
fi

fixture=${1:-tests/fixtures/fa2_fp16_s128_d128.mil}
warmup=${2:-20}
repeats=${3:-500}
cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/profile_program_chain
temporary_root=$(mktemp -d /tmp/mil-hwx-profile.XXXXXX)
trap 'rm -rf "$temporary_root"' EXIT

make build/mil-hwxc build/profile_program_chain -j4

bundle_dir=$temporary_root/bundle
./build/mil-hwxc --mil "$fixture" \
    --model-root tests/models/conv_relu --target H16G --output "$bundle_dir"

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
    echo "no research HWX artifacts for $fixture" >&2
    exit 3
fi

printf 'MACHINE model=%s os_build=%s fixture=%s warmup=%s repeats=%s\n' \
    "$(sysctl -n hw.model)" "$(sw_vers -buildVersion)" "$fixture" \
    "$warmup" "$repeats"
./build/profile_program_chain "$bundle_dir" "$warmup" "$repeats" "${hashes[@]}"
