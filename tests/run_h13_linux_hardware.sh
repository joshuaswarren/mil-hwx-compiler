#!/usr/bin/env bash
# Compile one MIL program for H13, run it through libane on a Linux ANE host,
# and compare every output with tools/h13_reference.py.
#
#   bash tests/run_h13_linux_hardware.sh MIL MODEL_ROOT NAME=input.fp16 ...
#
# Host, device-tree, module, device-node and libane gates come from
# tests/h13_first_run/preflight.sh, which also prints the identities the
# handoff records. The reviewed dispatch plan is written next to the package.
# Nothing is submitted until every check passes.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mil=${1:?usage: $0 MIL MODEL_ROOT NAME=input.fp16 ...}
model_root=${2:?usage: $0 MIL MODEL_ROOT NAME=input.fp16 ...}
shift 2
checkout=${ANE_CHECKOUT:-$HOME/src/omarchy-ane}
library=$checkout/bindings/python/dylib/libane_python.so
device=${ANE_DEVICE:-/dev/accel/accel0}

[[ $# -gt 0 ]] || { echo "at least one NAME=input.fp16 binding is required" >&2; exit 2; }
ANE_CHECKOUT=$checkout ANE_DEVICE=$device bash "$repo/tests/h13_first_run/preflight.sh" || exit 2

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

python3 "$repo/tools/h13_run_linux.py" "$package" --mil "$mil" \
    --model-root "$model_root" --dry-run "${inputs[@]}" "${output_args[@]}" \
    > "$package/plan.json"

printf 'H13 LINUX host=%s kernel=%s libane=%s device=%s package=%s\n' \
    "$(hostname)" "$(uname -r)" "$(git -C "$checkout" rev-parse --short HEAD)" "$device" "$package"
python3 "$repo/tools/h13_run_linux.py" "$package" --mil "$mil" --model-root "$model_root" \
    "${inputs[@]}" "${output_args[@]}" --libane-library "$library"
echo "H13 Linux hardware gate: PASS"
