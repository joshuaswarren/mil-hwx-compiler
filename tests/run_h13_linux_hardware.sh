#!/usr/bin/env bash
# Compile one MIL program for H13, run it through libane on a Linux ANE host,
# and compare every output with tools/h13_reference.py.
#
#   bash tests/run_h13_linux_hardware.sh [--package DIR] MIL MODEL_ROOT NAME=input.fp16 ...
#
# The compiler binary is located (or built) BEFORE identity checking, so the
# binary preflight digests is exactly the binary that executes.
# ANE_COMPILER_BIN pins an isolated reviewed build and is never rebuilt.
# --package DIR skips compilation entirely and runs the exact
# already-compiled package. Identity gates come from
# tests/h13_first_run/preflight.sh against the reviewed identity file
# (ANE_REVIEWED_IDENTITIES); nothing is submitted until every check passes,
# and the reviewed dispatch plan is written next to the package. Packages
# carry a compiler.sha256 provenance file recording the attested binary that
# produced them; --package verifies it against the attested binary.
#
# PREFLIGHT_ROOT is a fixture-only seam and is refused here: submission paths
# always run against the real host tree.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
package=
if [[ -n ${PREFLIGHT_ROOT:-} ]]; then
    echo "PREFLIGHT_ROOT is the preflight fixture seam; refusing to submit against a fake host tree" >&2
    exit 2
fi
if [[ ${1:-} == --package ]]; then
    package=${2:?"usage: $0 --package DIR MIL MODEL_ROOT NAME=input.fp16 ..."}
    shift 2
fi
mil=${1:?"usage: $0 [--package DIR] MIL MODEL_ROOT NAME=input.fp16 ..."}
model_root=${2:?"usage: $0 [--package DIR] MIL MODEL_ROOT NAME=input.fp16 ..."}
shift 2
checkout=${ANE_CHECKOUT:-$HOME/src/omarchy-ane}
library=$checkout/bindings/python/dylib/libane_python.so
device=${ANE_DEVICE:-/dev/accel/accel0}
compiler=${ANE_COMPILER_BIN:-$repo/build/mil-hwxc}

[[ $# -gt 0 ]] || { echo "at least one NAME=input.fp16 binding is required" >&2; exit 2; }
if [[ -n $package ]]; then
    [[ -f $package/manifest.json ]] || { echo "no manifest.json in $package; pass the exact compiled package directory" >&2; exit 2; }
    recorded=$(cut -d' ' -f1 "$package/compiler.sha256" 2>/dev/null)
    attested=$(sha256sum "$compiler" | cut -d' ' -f1)
    if [[ $recorded != "$attested" ]]; then
        echo "$package records compiler ${recorded:-<none>}, not the pinned $compiler ($attested)" >&2
        exit 2
    fi
elif [[ ! -e $compiler ]]; then
    if [[ -n ${ANE_COMPILER_BIN:-} ]]; then
        echo "ANE_COMPILER_BIN $compiler is missing; refusing to rebuild a pinned compiler" >&2
        exit 2
    fi
    make -C "$repo" build/mil-hwxc
fi

# Identity checking happens after any build and before any execution: the
# binary digested here is the binary used below. Never rebuild past this line.
ANE_CHECKOUT=$checkout ANE_DEVICE=$device ANE_COMPILER_BIN=$compiler \
    bash "$repo/tests/h13_first_run/preflight.sh" || exit 2

if [[ -z $package ]]; then
    package=$(mktemp -d /tmp/mil-hwx-h13-linux.XXXXXX)
    rmdir "$package"
    "$compiler" --target H13 --mil "$mil" --model-root "$model_root" --output "$package"
    printf '%s  mil-hwxc\n' "$(sha256sum "$compiler" | cut -d' ' -f1)" > "$package/compiler.sha256"
fi
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
# --device 0 matches the node preflight attested: only /dev/accel/accel0 may
# be attested, so the attested node and the opened index cannot drift apart.
python3 "$repo/tools/h13_run_linux.py" "$package" --mil "$mil" --model-root "$model_root" \
    "${inputs[@]}" "${output_args[@]}" --libane-library "$library" --device 0
echo "H13 Linux hardware gate: PASS"
