#!/bin/bash
set -euo pipefail
repo_dir=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo_dir"
[[ $(uname -s) == Darwin && $(uname -m) == arm64 ]] || exit 2
model_key=A8FA2E340D752C36B7B5E31658D96FA6D6C85A8185A72B72CBE1305C3F4806AA_B9CB7B429B3392F8A4C6A77826C0698CF7719A8F59ED06176DE99C6D593F661B_E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855
cache_dir=/Library/Caches/com.apple.aned/$(sw_vers -buildVersion)/InMemoryModelCache/reduce_exec/$model_key
make build/prepare_reduce build/reduce_exec -j4
sudo -n mkdir -p "$cache_dir"
geometries=("32 8 8 1" "64 8 8 1" "128 8 8 1" "64 16 16 1" \
            "64 32 32 1" "1 64 64 3" "1 128 128 3" "32 64 16 2")
for operation in reduce_sum reduce_mean reduce_max;do
    for geometry in "${geometries[@]}";do
        read -r channels height width axis <<<"$geometry"
        root=build/hardware-${operation}-c${channels}-h${height}-w${width}-a${axis}
        ./build/prepare_reduce "$operation" "$channels" "$height" "$width" \
            "$axis" "$root"
        artifact=$root/bundle/program-0.hwx
        printf 'HWX reduce op=%s C=%s H=%s W=%s axis=%s hash=%s\n' \
            "$operation" "$channels" "$height" "$width" "$axis" \
            "$(shasum -a 256 "$artifact"|awk '{print $1}')"
        sudo -n install -m 0644 "$artifact" "$cache_dir/model.hwx"
        ./build/reduce_exec "$operation" "$channels" "$height" "$width" \
            "$axis" "$root/bundle" "$cache_dir/model.hwx"
    done
done
echo "reduction M4 hardware sweep: PASS"
