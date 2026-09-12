#!/usr/bin/env bash
# Gate and identify a Linux ANE host before anything is submitted.
#
#   bash tests/h13_first_run/preflight.sh
#
# Checks the host, the device tree, the loaded module, the device node and the
# libane build, then prints the identities the hardware handoff records. Every
# failure exits 2 and names what to fix; nothing here touches the device.
#
# ANE_CHECKOUT        libane checkout (default ~/src/omarchy-ane)
# ANE_DEVICE          device node (default /dev/accel/accel0)
# ANE_EXPECTED_COMMIT exact driver/libane source commit
set -uo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checkout=${ANE_CHECKOUT:-$HOME/src/omarchy-ane}
device=${ANE_DEVICE:-/dev/accel/accel0}
expected_commit=${ANE_EXPECTED_COMMIT:-f261a6cb537aca62f267ad3d01beda0d6877544c}
expected_compiler=783dbe138d6cd170abe20854697498f60ea725b5
library=$checkout/bindings/python/dylib/libane_python.so
archive=$checkout/libane/libane.a
module=$checkout/ane/ane.ko
status=0

fail() { printf 'PREFLIGHT FAIL %s\n' "$1" >&2; status=2; }
identity() { printf 'identity %-24s %s\n' "$1" "$2"; }
value() { tr -d '\0' < "$1" 2>/dev/null | tr '\n' ' '; }

[[ $(uname -s) == Linux ]] || fail "host is $(uname -s), not Linux"
[[ $(uname -m) == aarch64 ]] || fail "host is $(uname -m), not aarch64"
identity host "$(hostname)"
identity kernel "$(uname -r) $(uname -m)"

node=$(printf '%s\n' /proc/device-tree/soc/ane@* | head -1)
if [[ -d $node ]]; then
    identity dt-node "${node#/proc/device-tree/soc/}"
    identity dt-compatible "$(value "$node/compatible")"
    identity dt-status "$(value "$node/status")"
else
    fail "no /proc/device-tree/soc/ane@* node; boot the ANE device-tree entry"
fi

if [[ -d /sys/module/ane ]]; then
    loaded_srcversion=$(cat /sys/module/ane/srcversion 2>/dev/null)
    identity module-srcversion "${loaded_srcversion:-unknown}"
    if [[ -f $module ]]; then
        built_srcversion=$(modinfo -F srcversion "$module" 2>/dev/null)
        identity module-build "$module $(sha256sum "$module" | cut -d' ' -f1)"
        identity module-build-srcversion "${built_srcversion:-unknown}"
        [[ -n $loaded_srcversion && $loaded_srcversion == "$built_srcversion" ]] ||
            fail "loaded module srcversion '${loaded_srcversion:-unknown}' differs from canonical build '${built_srcversion:-unknown}'"
    else
        fail "missing canonical module build $module"
    fi
    parameters=
    for parameter in /sys/module/ane/parameters/*; do
        [[ -f $parameter ]] || continue
        parameters+="$(basename "$parameter")=$(cat "$parameter" 2>/dev/null) "
    done
    identity module-parameters "${parameters:-none}"
else
    fail "ane module is not loaded; run the jwm1 bring-up ladder first"
fi
platform=$(printf '%s\n' /sys/bus/platform/devices/*.ane | head -1)
if [[ -d $platform && -L $platform/driver ]]; then
    identity platform-device "$(basename "$platform")"
    identity platform-driver "$(basename "$(readlink -f "$platform/driver")")"
    control=$(cat "$platform/power/control" 2>/dev/null)
    runtime_status=$(cat "$platform/power/runtime_status" 2>/dev/null)
    identity runtime-pm "$control"
    identity runtime-status "${runtime_status:-unknown}"
    [[ $control == on ]] || fail "runtime PM is '$control', not 'on'; pin it before submitting"
    [[ $runtime_status == active ]] || fail "runtime status is '${runtime_status:-unknown}', not 'active'"
else
    fail "no bound /sys/bus/platform/devices/*.ane; the module bound nothing"
fi

if [[ -c $device ]]; then
    identity device "$device $(stat -c '%A %u:%g %t,%T' "$device")"
    [[ -r $device && -w $device ]] || fail "$device is not readable and writable by $(id -un)"
    users=$(fuser "$device" 2>/dev/null || true)
    identity device-users "${users:-none}"
    [[ -z $users ]] || fail "$device has live user(s): $users"
else
    fail "no character device at $device"
fi

if git -C "$checkout" rev-parse --git-dir >/dev/null 2>&1; then
    branch=$(git -C "$checkout" rev-parse --abbrev-ref HEAD)
    commit=$(git -C "$checkout" rev-parse HEAD)
    identity libane-checkout "$checkout"
    identity libane-commit "$commit on $branch"
    identity libane-dirty "$(git -C "$checkout" status --porcelain | wc -l) modified path(s)"
    [[ $commit == "$expected_commit" ]] || fail "$checkout is at '$commit', not canonical '$expected_commit'"
    [[ -z $(git -C "$checkout" status --porcelain) ]] || fail "$checkout has local modifications"
    abi=$(grep -E '^#define[[:space:]]+ANE_ABI_MAJOR[[:space:]]+' \
        "$checkout/ane/src/uapi/drm/ane_accel.h" 2>/dev/null | awk '{print $3}')
    identity driver-abi "${abi:-unknown}"
    [[ $abi == 1 ]] || fail "driver/libane ABI is '${abi:-unknown}', not 1"
    header=$(grep -o 'ANEC_HEADER_SIZE[[:space:]]*0x[0-9a-fA-F]*' \
        "$checkout/libane/ane.c" 2>/dev/null | head -1 | grep -o '0x[0-9a-fA-F]*')
    identity libane-anec-header "${header:-unknown}"
    [[ $header == 0x1000 ]] || fail "libane reads the ANEC payload at ${header:-unknown}, not 0x1000"
else
    fail "no git checkout at $checkout"
fi
for artifact in "$archive" "$library"; do
    if [[ -f $artifact ]]; then
        identity "$(basename "$artifact")" "$(stat -c '%s bytes %y' "$artifact") $(sha256sum "$artifact" | cut -d' ' -f1)"
    else
        fail "missing $artifact; build libane and bindings/python/dylib"
    fi
done

compiler_commit=$(git -C "$repo" rev-parse HEAD)
identity compiler-repo "$repo"
identity compiler-commit "$compiler_commit on $(git -C "$repo" rev-parse --abbrev-ref HEAD)"
identity compiler-dirty "$(git -C "$repo" status --porcelain | wc -l) modified path(s)"
[[ $compiler_commit == "$expected_compiler" ]] || fail "compiler is at '$compiler_commit', not canonical '$expected_compiler'"

if [[ $status -eq 0 ]]; then
    echo "H13 preflight: PASS (no device access performed)"
else
    echo "H13 preflight: FAIL; do not submit" >&2
fi
echo "reminder: after a hung submission, reboot. Never rmmod and reload ane -" \
     "the remove path may have in-flight resources, and the driver refuses unload after a wedge."
exit $status
