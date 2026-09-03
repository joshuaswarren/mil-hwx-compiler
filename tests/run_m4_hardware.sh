#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "M4 hardware verification requires arm64 macOS" >&2
    exit 2
fi

model_key=56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
os_build=$(sw_vers -buildVersion)
cache_root=/Library/Caches/com.apple.aned/$os_build/InMemoryModelCache

make build/mil-hwxc build/prepare_w8a8_model \
    build/conv_exec build/attention_exec build/w8a8_exec -j4
./build/prepare_w8a8_model build/models/w8a8-hardware

./build/mil-hwxc --mil tests/fixtures/conv_relu.mil \
    --model-root tests/models/conv_relu \
    --target H16G --output build/hardware-conv
./build/mil-hwxc --mil tests/fixtures/attention.mil \
    --model-root tests/models/attention \
    --target H16G --output build/hardware-attention
./build/mil-hwxc --mil tests/fixtures/w8a8_conv_chain.mil \
    --model-root build/models/w8a8-hardware \
    --target H16G \
    --output build/hardware-w8a8

check_hash() {
    local file=$1
    local expected=$2
    local actual
    actual=$(shasum -a 256 "$file" | awk '{print $1}')
    if [[ $actual != "$expected" ]]; then
        echo "hash mismatch for $file: $actual" >&2
        exit 3
    fi
    printf "HWX hash=%s file=%s\n" "$actual" "$file"
}

check_hash build/hardware-conv/program-0.hwx \
    991aa7a3ec28f5603beb090f2ca244921370d4081948bea30ad5074499ca1266
check_hash build/hardware-attention/program-0.hwx \
    90d014ab1cdb6288d490780b8d9d8565bc84037efade13d506d91067126678d0
check_hash build/hardware-w8a8/program-0.hwx \
    38f1858486a352de2145d888601b9c50526459f75775377a22080b80a8627013

provision() {
    local executable=$1
    local artifact=$2
    local directory=$cache_root/$executable/$model_key
    sudo -n mkdir -p "$directory"
    sudo -n install -m 0644 "$artifact" "$directory/model.hwx"
    printf "%s" "$directory/model.hwx"
}

conv_cache=$(provision conv_exec build/hardware-conv/program-0.hwx)
attention_cache=$(provision attention_exec \
    build/hardware-attention/program-0.hwx)
w8a8_cache=$(provision w8a8_exec build/hardware-w8a8/program-0.hwx)

./build/conv_exec build/hardware-conv "$conv_cache"
./build/attention_exec build/hardware-attention "$attention_cache"
./build/w8a8_exec build/hardware-w8a8 "$w8a8_cache"

echo "all M4 hardware tests: PASS"
