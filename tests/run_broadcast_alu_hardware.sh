#!/bin/bash
set -euo pipefail

repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
    echo "broadcast ALU hardware gate requires arm64 macOS" >&2
    exit 2
fi
model_key=56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
cache_dir=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/broadcast_alu_exec/$model_key
make build/prepare_broadcast_alu build/broadcast_alu_exec -j4
sudo -n mkdir -p "$cache_dir"
for case_name in scale matrix_sub matrix_mul row_add row_mul row_max; do
    bundle_dir=build/hardware-broadcast-$case_name
    ./build/prepare_broadcast_alu "$case_name" "$bundle_dir"
    sudo -n install -m 0644 "$bundle_dir/program-0.hwx" "$cache_dir/model.hwx"
    ./build/broadcast_alu_exec "$case_name" "$bundle_dir"
done
echo "broadcast ALU M4 hardware gate: PASS"
