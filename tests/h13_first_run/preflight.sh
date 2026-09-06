#!/usr/bin/env bash
# Gate and identify a Linux ANE host before anything is submitted.
#
#   bash tests/h13_first_run/preflight.sh
#
# Checks the host, the device tree, the loaded module, the device node and the
# libane build, then prints the identities the hardware handoff records. Every
# failure exits 2 and names what to fix; nothing here touches the device.
#
# ANE_CHECKOUT   libane checkout (default ~/src/omarchy-ane)
# ANE_DEVICE     device node (default /dev/accel/accel0)
set -uo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
checkout=${ANE_CHECKOUT:-$HOME/src/omarchy-ane}
device=${ANE_DEVICE:-/dev/accel/accel0}
library=$checkout/bindings/python/dylib/libane_python.so
archive=$checkout/libane/libane.a
status=0

fail() { printf 'PREFLIGHT FAIL %s\n' "$1" >&2; status=2; }
identity() { printf 'identity %-18s %s\n' "$1" "$2"; }
value() { tr -d '\0' < "$1" 2>/dev/null | tr '\n' ' '; }

# 1. Host.
[[ $(uname -s) == Linux ]] || fail "host is $(uname -s), not Linux"
[[ $(uname -m) == aarch64 ]] || fail "host is $(uname -m), not aarch64"
identity host "$(hostname)"
identity kernel "$(uname -r) $(uname -m)"

# 2. Device-tree node. The engine window must be the task-manager block
# (0x26bc04000 on t8103); a node pointing at 0x26a000000 is the tree that
# made every TM access land outside the block.
node=$(printf '%s\n' /proc/device-tree/soc/ane@* | head -1)
if [[ -d $node ]]; then
    identity dt-node "${node#/proc/device-tree/soc/}"
    identity dt-compatible "$(value "$node/compatible")"
    identity dt-status "$(value "$node/status")"
else
    fail "no /proc/device-tree/soc/ane@* node; boot the ANE device-tree entry"
fi

# 3. Module and platform device.
if [[ -d /sys/module/ane ]]; then
    identity module-srcversion "$(cat /sys/module/ane/srcversion 2>/dev/null)"
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
if [[ -d $platform ]]; then
    identity platform-device "$(basename "$platform")"
    identity platform-driver "$(basename "$(readlink -f "$platform/driver" 2>/dev/null)")"
    control=$(cat "$platform/power/control" 2>/dev/null)
    identity runtime-pm "$control"
    # Autosuspend invalidates DART TLBs about a second later, including the
    # dart0 apple-dart owns, and resets the SoC.
    [[ $control == on ]] || fail "runtime PM is '$control', not 'on'; pin it before submitting"
else
    fail "no /sys/bus/platform/devices/*.ane; the module bound nothing"
fi

# 4. Device node.
if [[ -c $device ]]; then
    identity device "$device $(stat -c '%A %u:%g %t,%T' "$device")"
    [[ -r $device && -w $device ]] || fail "$device is not readable and writable by $(id -un)"
else
    fail "no character device at $device"
fi

# 5. libane build. The compiler writes a 0x1000-byte ANEC header, so the
# loader must read the payload there.
if [[ -d $checkout/.git ]]; then
    branch=$(git -C "$checkout" rev-parse --abbrev-ref HEAD)
    identity libane-checkout "$checkout"
    identity libane-commit "$(git -C "$checkout" rev-parse --short HEAD) on $branch"
    identity libane-dirty "$(git -C "$checkout" status --porcelain | wc -l) modified path(s)"
    [[ $branch == omarchy ]] || fail "$checkout is on '$branch', not 'omarchy'"
    header=$(grep -o 'ANEC_HEADER_SIZE[[:space:]]*0x[0-9a-fA-F]*' \
        "$checkout/libane/ane.c" 2>/dev/null | head -1 | grep -o '0x[0-9a-fA-F]*')
    identity libane-anec-header "${header:-unknown}"
    [[ $header == 0x1000 ]] || fail "libane reads the ANEC payload at ${header:-unknown}, not 0x1000"
else
    fail "no git checkout at $checkout"
fi
for artifact in "$archive" "$library"; do
    if [[ -f $artifact ]]; then
        identity "$(basename "$artifact")" "$(stat -c '%s bytes %y' "$artifact")"
    else
        fail "missing $artifact; build libane and bindings/python/dylib"
    fi
done

# 6. Compiler identity, so a device result names the package that produced it.
identity compiler-repo "$repo"
identity compiler-commit "$(git -C "$repo" rev-parse --short HEAD) on \
$(git -C "$repo" rev-parse --abbrev-ref HEAD)"
identity compiler-dirty "$(git -C "$repo" status --porcelain | wc -l) modified path(s)"

if [[ $status -eq 0 ]]; then
    echo "H13 preflight: PASS (no device access performed)"
else
    echo "H13 preflight: FAIL; do not submit" >&2
fi
echo "reminder: after a hung submission, reboot. Never rmmod and reload ane -" \
     "the remove path leaks IOVA mappings and a second insmod fails bo_init."
exit $status
