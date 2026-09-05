#!/usr/bin/env bash
set -euo pipefail

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "H13 hardware test requires an arm64 Mac" >&2
    exit 2
fi
if (( $# < 3 )); then
    echo "usage: $0 MODEL.mil MODEL_ROOT INPUT=FP16_FILE [INPUT=FP16_FILE ...]" >&2
    exit 64
fi

mil=$1
model_root=$2
shift 2
for input in "$@"; do
    if [[ $input != *=* || ! -f ${input#*=} ]]; then
        echo "invalid input assignment: $input" >&2
        exit 64
    fi
done

cache_root=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/h13_exec
if ! mkdir -p "$cache_root" 2>/dev/null || [[ ! -w $cache_root ]]; then
    echo "H13 hardware test cannot provision the aned cache: $cache_root is not writable" >&2
    exit 2
fi

make build/mil-hwxc build/h13_exec
work=$(mktemp -d /tmp/mil-hwx-h13.XXXXXX)
trap 'rm -rf "$work"' EXIT
package=$work/package
expected=$work/expected
mkdir -p "$expected"
./build/mil-hwxc --mil "$mil" --model-root "$model_root" \
    --output "$package" --target H13 --format hwx

reference_args=()
runner_args=()
for input in "$@"; do
    reference_args+=(--input "$input")
    runner_args+=(--input "$input")
done
outputs_list=$work/outputs
python3 - "$package/manifest.json" >"$outputs_list" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1], encoding='utf-8'))
for name, tensor in manifest['tensors'].items():
    if tensor['role'] == 'output':
        print(name)
PY
if [[ ! -s $outputs_list ]]; then
    echo "H13 package has no output tensors" >&2
    exit 1
fi
while IFS= read -r name; do
    path=$expected/$name.fp16
    reference_args+=(--output "$name=$path")
    runner_args+=(--expected "$name=$path")
done <"$outputs_list"
python3 tools/h13_reference.py "$mil" --model-root "$model_root" "${reference_args[@]}"

index=0
while [[ -f $package/program-$index.hwx ]]; do
    artifact=$package/program-$index.hwx
    digest=$(shasum -a 256 "$artifact" | awk '{print toupper($1)}')
    slot=$(printf '%064X' "$index")
    key=${digest}_${slot}_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
    directory=$cache_root/$key
    if ! mkdir -p "$directory" || ! install -m 0644 "$artifact" "$directory/model.hwx"; then
        echo "H13 hardware test failed to provision writable cache entry: $directory" >&2
        exit 2
    fi
    runner_args+=(--cache "$key")
    index=$((index + 1))
done
if (( index == 0 )); then
    echo "H13 compiler produced no HWX programs" >&2
    exit 1
fi

./build/h13_exec "$package" "${runner_args[@]}"
