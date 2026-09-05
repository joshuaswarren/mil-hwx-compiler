#!/usr/bin/env bash
# Compile one MIL program for H13, run it through libane on a Linux ANE host,
# and compare every output with tools/h13_reference.py.
#
#   bash tests/run_h13_linux_hardware.sh MIL MODEL_ROOT NAME=input.fp16 ...
#
# Requires the omarchy branch of joshuaswarren/omarchy-ane built at
# $ANE_CHECKOUT (default ~/src/omarchy-ane): libane/libane.a and
# bindings/python/dylib/libane_python.so, and a loaded ane.ko exposing
# /dev/accel/accel0. Nothing is submitted until every check passes.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mil=${1:?usage: $0 MIL MODEL_ROOT NAME=input.fp16 ...}
model_root=${2:?usage: $0 MIL MODEL_ROOT NAME=input.fp16 ...}
shift 2
checkout=${ANE_CHECKOUT:-$HOME/src/omarchy-ane}
library=$checkout/bindings/python/dylib/libane_python.so
device=${ANE_DEVICE:-/dev/accel/accel0}

[[ $(uname -s) == Linux && $(uname -m) == aarch64 ]] || {
    echo "H13 Linux hardware test requires an aarch64 Linux host" >&2; exit 2; }
[[ -c $device ]] || { echo "no ANE device node at $device; is ane.ko loaded?" >&2; exit 2; }
[[ -r $device && -w $device ]] || { echo "$device is not accessible by $(id -un)" >&2; exit 2; }
[[ -f $library ]] || { echo "missing $library; build libane and bindings/python/dylib on the omarchy branch" >&2; exit 2; }
[[ $(git -C "$checkout" rev-parse --abbrev-ref HEAD) == omarchy ]] || {
    echo "$checkout must be on the omarchy branch (ANEC header 0x1000)" >&2; exit 2; }
grep -q 'ANEC_HEADER_SIZE   0x1000UL' "$checkout/libane/ane.c" || {
    echo "libane at $checkout does not read the ANEC payload at 0x1000" >&2; exit 2; }
[[ $# -gt 0 ]] || { echo "at least one NAME=input.fp16 binding is required" >&2; exit 2; }

make -C "$repo" build/mil-hwxc
package=$(mktemp -d /tmp/mil-hwx-h13-linux.XXXXXX)
rmdir "$package"
"$repo/build/mil-hwxc" --target H13 --mil "$mil" --model-root "$model_root" --output "$package"
python3 "$repo/research/inspect_anec.py" "$package" > /dev/null

inputs=()
for binding in "$@"; do inputs+=(--input "$binding"); done
outputs=$(python3 - "$mil" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
print(" ".join(re.findall(r'->\s*\((.*?)\)', text)[-1].replace(",", " ").split()))
PY
)
output_args=()
for name in $outputs; do output_args+=(--output "$name=$package/$name.out.fp16"); done

printf 'H13 LINUX host=%s kernel=%s libane=%s device=%s package=%s\n' \
    "$(hostname)" "$(uname -r)" "$(git -C "$checkout" rev-parse --short HEAD)" "$device" "$package"
python3 "$repo/tools/h13_run_linux.py" "$package" --mil "$mil" --model-root "$model_root" \
    "${inputs[@]}" "${output_args[@]}" --libane-library "$library"
echo "H13 Linux hardware gate: PASS"
