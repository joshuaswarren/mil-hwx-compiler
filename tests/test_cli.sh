#!/bin/bash
set -euo pipefail

compiler=./build/mil-hwxc
repo_root=$(pwd)
compile_case() {
    local name=$1
    local fixture=$2
    local model=$3
    local output="build/cli-${name}"
    local repeat="build/cli-${name}-repeat"
    mkdir -p "$output"
    "$compiler" --mil "$fixture" --model-root "$model" \
        --target H16G --output "$output"
    mkdir -p "$repeat"
    "$compiler" --mil "$fixture" --model-root "$model" \
        --target H16G --output "$repeat" >/dev/null
    cmp "$output/program-0.hwx" "$repeat/program-0.hwx"
    test -s "$output/program-0.hwx"
    test -s "$output/manifest.json"
}

compile_case conv tests/fixtures/conv_relu.mil tests/models/conv_relu
compile_case attention tests/fixtures/attention.mil tests/models/attention
compile_case w8a8 tests/fixtures/w8a8_conv_chain.mil tests/models/w8a8

if "$compiler" --mil tests/fixtures/conv_relu.mil \
    --model-root tests/models/conv_relu \
    --target H15 --output build/cli-invalid-target >/dev/null 2>&1; then
    echo "unsupported target unexpectedly compiled" >&2
    exit 1
fi

if "$compiler" --unknown value >/dev/null 2>&1; then
    echo "unknown CLI option unexpectedly succeeded" >&2
    exit 1
fi

if "$compiler" --mil tests/fixtures/conv_relu.mil \
    --model-root tests/models/conv_relu --target H16G \
    --output build/cli-legacy-resources \
    --resources /tmp/removed-h16g-resources >/dev/null 2>&1; then
    echo "legacy descriptor-resource option unexpectedly succeeded" >&2
    exit 1
fi

mkdir -p build/cli-no-resources
(
    cd /tmp
    "$repo_root/build/mil-hwxc" \
        --mil "$repo_root/tests/fixtures/attention.mil" \
        --model-root "$repo_root/tests/models/attention" \
        --target H16G --output "$repo_root/build/cli-no-resources" >/dev/null
)
test -s build/cli-no-resources/program-0.hwx

echo "compiler CLI: PASS"
