#!/usr/bin/env bash
# Gate and identify a Linux ANE host before anything is submitted.
#
#   ANE_REVIEWED_IDENTITIES=reviewed.env bash tests/h13_first_run/preflight.sh
#
# Checks the host, the device tree, the loaded module, the platform binding,
# the device node and the libane build against a reviewed identity file, then
# prints the identities the hardware handoff records. Every failure exits 2
# and names what to fix; nothing here touches the device.
#
# Reviewed provenance is explicit and fail-closed: there is no default file
# and no approved default identity. After a review that pins the firmware,
# device tree, module, libane and compiler bytes, write a key=value file (one
# KEY=VALUE per line, never sourced) and point ANE_REVIEWED_IDENTITIES at it.
# Required keys, every one of which must be present:
#
#   firmware               reviewed firmware/OS identity; recorded for the
#                          handoff, not locally verifiable without the device
#   dt-compatible          expected FIRST compatible string of the ANE node
#   module-srcversion      expected /sys/module/ane/srcversion
#   module-ko              path of the reviewed ane.ko bytes that were inserted
#   module-ko-sha256       sha256 of that ane.ko
#   libane-python-sha256   sha256 of bindings/python/dylib/libane_python.so
#   libane-archive-sha256  sha256 of libane/libane.a
#   compiler-commit        full commit sha of the compiler source the binary
#                          was built from
#   compiler-sha256        sha256 of the compiler binary that will execute
#
# The reviewed manifest holds private host provenance: keep it outside the
# repository (gitignored .unlazy/m1-linux-delivery/private/ works). No
# approved identity values exist until a review pins them.
#
# A missing file, a missing key, or any observed/reviewed mismatch refuses the
# host. Branch names and dirty state are recorded but never gate.
#
#   ANE_REVIEWED_IDENTITIES  reviewed identity file (required, fail-closed)
#   ANE_CHECKOUT             libane checkout or worktree (default ~/src/omarchy-ane)
#   ANE_DEVICE               device node; only /dev/accel/accel0 (index 0)
#                            may be attested, which is what the runner opens
#   ANE_COMPILER_BIN         compiler binary to identify (default build/mil-hwxc)
#   PREFLIGHT_ROOT           test seam: /proc and /sys prefix (default /);
#                            the hardware wrapper refuses this seam
set -uo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
root=${PREFLIGHT_ROOT:-}
checkout=${ANE_CHECKOUT:-$HOME/src/omarchy-ane}
device=${ANE_DEVICE:-}
compiler=${ANE_COMPILER_BIN:-$repo/build/mil-hwxc}
library=$checkout/bindings/python/dylib/libane_python.so
archive=$checkout/libane/libane.a
status=0

fail() { printf 'PREFLIGHT FAIL %s\n' "$1" >&2; status=2; }
identity() { printf 'identity %-20s %s\n' "$1" "$2"; }
prop() { tr '\0' '\n' < "$1" 2>/dev/null | head -1; }
digest() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1; }
online() { grep -c '^processor' "$root/proc/cpuinfo" 2>/dev/null; }
cells() { # last byte of a 4-byte DT cell-count property, in decimal
    set -- $(od -An -tu1 "$1" 2>/dev/null)
    [[ $# == 4 && $4 -ge 1 && $4 -le 4 ]] && printf '%s' "$4"
}

# 0. Reviewed provenance, fail-closed. Key presence is validated in this
# shell: a subshell $(...) cannot carry the refusal status out.
reviewed=${ANE_REVIEWED_IDENTITIES:-}
if [[ -n $reviewed && -f $reviewed ]]; then
    identity reviewed "$reviewed"
else
    reviewed=/dev/null
    fail "no reviewed identity file; set ANE_REVIEWED_IDENTITIES to the reviewed key=value file; refusing an unreviewed host"
fi
for key in firmware dt-compatible module-srcversion module-ko module-ko-sha256 \
           libane-python-sha256 libane-archive-sha256 compiler-commit compiler-sha256; do
    [[ -n $(sed -n "s/^$key=//p" "$reviewed" 2>/dev/null | head -1) ]] || \
        fail "reviewed identities omit $key; the review must pin it before submission"
done
require() { sed -n "s/^$1=//p" "$reviewed" 2>/dev/null | head -1; }
check() { # observed reviewed label; a mismatch refuses the host
    { [[ -n $2 ]] && [[ $1 == "$2" ]]; } || \
        fail "$3: observed '$1' does not match reviewed '$2'"
}
r_firmware=$(require firmware)
r_dt=$(require dt-compatible)
r_srcver=$(require module-srcversion)
r_ko=$(require module-ko)
r_ko_digest=$(require module-ko-sha256)
r_lib=$(require libane-python-sha256)
r_arc=$(require libane-archive-sha256)
r_commit=$(require compiler-commit)
r_binary=$(require compiler-sha256)
identity reviewed-firmware "$r_firmware (recorded, not locally verified)"

# 1. Host: Linux on aarch64 with all eight CPUs online. The historical frozen
# device-tree override booted with a single CPU, so a degraded tree refuses
# here instead of qualifying.
[[ $(uname -s) == Linux ]] || fail "host is $(uname -s), not Linux"
[[ $(uname -m) == aarch64 ]] || fail "host is $(uname -m), not aarch64"
identity host "$(hostname)"
identity kernel "$(uname -r) $(uname -m)"
cpus=$(online)
cpus=${cpus:-0}
identity cpus "$cpus online"
[[ $cpus == 8 ]] || fail "host has $cpus online CPUs, not 8; a degraded or overridden device tree loses cores - refuse to qualify"

# 2. Device-tree node. Compatible (first entry), status and the complete
# engine reg (address 0x26bc04000 plus 0x24000 window, per the parent's cell
# counts) are validated against the reviewed value, not just printed.
node=$(printf '%s\n' "$root"/proc/device-tree/soc/ane@* | head -1)
if [[ -d $node ]]; then
    compatible=$(prop "$node/compatible")
    dtstatus=$(prop "$node/status")
    reg=$(od -An -v -tx1 "$node/reg" 2>/dev/null | tr -d ' \n')
    acells=$(cells "$node/../#address-cells")
    scells=$(cells "$node/../#size-cells")
    identity dt-node "${node#*/soc/}"
    identity dt-compatible "$compatible"
    identity dt-status "$dtstatus"
    identity dt-engine-reg "${reg:-none} ($acells/$scells cells)"
    check "$compatible" "$r_dt" "dt-compatible"
    [[ $dtstatus == okay ]] || fail "device-tree status is '$dtstatus', not 'okay'"
    if [[ $acells =~ ^[1-4]$ && $scells =~ ^[1-4]$ ]]; then
        printf -v regexpect '%0*x%0*x' $((acells * 8)) 0x26bc04000 \
            $((scells * 8)) 0x24000
        [[ $reg == "$regexpect" ]] || \
            fail "ANE reg is ${reg:-missing}, not the t8103 task-manager window 0x26bc04000+0x24000"
    else
        fail "cannot decode reg cell counts ($acells/$scells); refusing an unreadable tree"
    fi
else
    fail "no ANE node under the running device-tree; the booted tree is not a reviewed one - refuse"
fi

# 3. Compare loaded srcversion and the reviewed module file separately.
# The platform device is the OF device of the validated node (OF names it
# <unit-address>.<nodename>) and must be bound to the ane driver.
moddir=$root/sys/module/ane
platform=$root/sys/bus/platform/devices/26bc04000.ane
if [[ -d $moddir ]]; then
    srcversion=$(cat "$moddir/srcversion" 2>/dev/null)
    parameters=
    for parameter in "$moddir"/parameters/*; do
        [[ -f $parameter ]] || continue
        parameters+="$(basename "$parameter")=$(cat "$parameter" 2>/dev/null) "
    done
    identity module-srcversion "$srcversion"
    identity module-parameters "${parameters:-none}"
    [[ -z $parameters ]] || fail "module parameters are outside the reviewed parameter-free driver configuration: $parameters"
    check "$srcversion" "$r_srcver" "module-srcversion"
    kodigest=$(digest "$r_ko")
    identity module-ko "$r_ko"
    identity module-ko-sha256 "${kodigest:-missing $r_ko}"
    check "$kodigest" "$r_ko_digest" "module-ko-sha256"
else
    fail "ane module is not loaded; submission requires the reviewed module bytes loaded in an explicitly owned window"
fi
if [[ -d $platform ]]; then
    identity platform-device "26bc04000.ane"
    driver=$(basename "$(readlink -f "$platform/driver" 2>/dev/null)")
    identity platform-driver "$driver"
    [[ $driver == ane ]] || fail "platform device 26bc04000.ane is bound to '${driver:-none}', not the ane driver"
    control=$(cat "$platform/power/control" 2>/dev/null)
    identity runtime-pm "$control"
    # Autosuspend invalidates DART TLBs about a second later, including the
    # dart0 apple-dart owns, and resets the SoC.
    [[ $control == on ]] || fail "runtime PM is '$control', not 'on'; refusing"
else
    fail "no platform device at sys/bus/platform/devices/26bc04000.ane; the reviewed DT node bound nothing"
fi

# 4. Device node. The runner opens device index 0, so attesting anything but
# /dev/accel/accel0 would attest one node and open another.
accel0=$root/dev/accel/accel0
if [[ -z $device ]]; then
    device=$accel0
fi
if [[ $device != "$accel0" ]]; then
    fail "refusing $device: the runner opens index 0, so only $accel0 may be attested"
elif [[ -c $device || (-n $root && -f $device) ]]; then
    # Under the PREFLIGHT_ROOT test seam a regular file stands in for the
    # character node; on a real host only a character device qualifies.
    identity device "$device $(stat -c '%A %u:%g %t,%T' "$device" 2>/dev/null)"
    [[ -r $device && -w $device ]] || fail "$device is not readable and writable by $(id -un)"
else
    fail "no character device at $device"
fi

# 5. libane build. The library and archive are gated by reviewed sha256 of
# the exact bytes (identity provenance), not by presence, size, branch name
# or a source-text macro.
if git -C "$checkout" rev-parse --git-dir > /dev/null 2>&1; then
    identity libane-checkout "$checkout"
    identity libane-commit "$(git -C "$checkout" rev-parse --short HEAD) on \
$(git -C "$checkout" rev-parse --abbrev-ref HEAD)"
    identity libane-dirty "$(git -C "$checkout" status --porcelain | wc -l) modified path(s)"
    libdigest=$(digest "$library")
    arcdigest=$(digest "$archive")
    identity libane-python-sha256 "${libdigest:-missing $library}"
    identity libane-archive-sha256 "${arcdigest:-missing $archive}"
    check "$libdigest" "$r_lib" "libane_python.so sha256"
    check "$arcdigest" "$r_arc" "libane.a sha256"
else
    fail "no git checkout or worktree at $checkout"
fi

# 6. Compiler identity: the exact binary that will execute is compared with
# the reviewed digest and the repo commit with the reviewed commit, so a
# device result names the reviewed source and bytes that produced it.
identity compiler-repo "$repo"
identity compiler-bin "$compiler"
identity compiler-commit "$(git -C "$repo" rev-parse --short HEAD) on \
$(git -C "$repo" rev-parse --abbrev-ref HEAD)"
identity compiler-dirty "$(git -C "$repo" status --porcelain | wc -l) modified path(s)"
comdigest=$(digest "$compiler")
identity compiler-sha256 "${comdigest:-missing $compiler}"
check "$comdigest" "$r_binary" "compiler binary sha256"
comcommit=$(git -C "$repo" rev-parse HEAD 2>/dev/null)
check "$comcommit" "$r_commit" "compiler commit"

if [[ $status -eq 0 ]]; then
    echo "H13 preflight: PASS (no device access performed)"
else
    echo "H13 preflight: FAIL; do not submit" >&2
fi
echo "recovery, retries and ownership follow the agreed qualification procedure; this script takes none of those actions."
exit $status
