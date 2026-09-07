#!/usr/bin/env python3
"""Deterministic preflight validation against fixture /proc and /sys trees.

Nothing here touches a device or a real ANE host: the fixture tree stands in
for /proc and /sys (PREFLIGHT_ROOT), a stub uname stands in for the M1
kernel, and a regular file stands in for the /dev/accel/accel0 character
node under the seam. The fully reviewed fixture must pass every gate; each
named single-fault mutation must exit 2 and name the failure. Fixture values
are test data, never claimed hardware identities.
"""

import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "tests/h13_first_run/preflight.sh"

KO_BYTES = b"fixture ane.ko bytes"
KO_BYTES_ALT = b"unreviewed ane.ko bytes"
LIB_BYTES = b"fixture libane_python.so bytes"
ARCHIVE_BYTES = b"fixture libane.a bytes"
COMPILER_BYTES = b"fixture mil-hwxc bytes"
SRCVERSION = "REVIEWEDSRCVERSION"
COMPATIBLE = "apple,t8103-ane"
ENGINE_BASE = 0x26BC04000
ENGINE_SIZE = 0x24000
REG = struct.pack(">Q", ENGINE_BASE) + struct.pack(">I", ENGINE_SIZE)
REG_ALT_BASE = struct.pack(">Q", 0x26A000000) + struct.pack(">I", ENGINE_SIZE)
REG_SMALL_SIZE = struct.pack(">Q", ENGINE_BASE) + struct.pack(">I", 0x1000)
UNAME_STUB = """#!/bin/sh
case "$1" in
  -s) echo Linux ;;
  -m) echo aarch64 ;;
  -r) echo 7.1.6-fixture ;;
  *) echo fixture ;;
esac
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Fixture:
    """A qualified host: eight CPUs, okay ANE node with the complete t8103
    engine reg (2 address cells / 1 size cell), loaded reviewed module, bound
    ane platform device with pinned runtime PM, accel0 node, reviewed libane
    artifacts and compiler binary, and a complete reviewed file. The checkout
    branch is deliberately not 'omarchy': branch text must not gate."""

    def __init__(self, work):
        self.work = work
        self.root = work / "root"
        self.checkout = work / "omarchy-ane"
        self.compiler = work / "mil-hwxc"
        self.ko = work / "ane.ko"
        self.reviewed = work / "reviewed.env"
        self.extra_env = {}
        bin_dir = work / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "uname").write_text(UNAME_STUB)
        (bin_dir / "uname").chmod(0o755)

        soc = self.root / "proc/device-tree/soc"
        node = soc / "ane@26bc04000"
        node.mkdir(parents=True, exist_ok=True)
        (soc / "#address-cells").write_bytes(struct.pack(">I", 2))
        (soc / "#size-cells").write_bytes(struct.pack(">I", 1))
        (self.root / "proc/cpuinfo").write_text(
            "".join(f"processor\t: {core}\n" for core in range(8)))
        (node / "compatible").write_bytes(
            COMPATIBLE.encode() + b"\0" + b"apple,ane\0")
        (node / "status").write_bytes(b"okay\0")
        (node / "reg").write_bytes(REG)

        module = self.root / "sys/module/ane/parameters"
        module.mkdir(parents=True, exist_ok=True)
        (self.root / "sys/module/ane/srcversion").write_text(SRCVERSION + "\n")
        platform = self.root / "sys/bus/platform/devices/26bc04000.ane"
        platform.mkdir(parents=True, exist_ok=True)
        drivers = self.root / "sys/bus/platform/drivers/ane"
        drivers.mkdir(parents=True, exist_ok=True)
        driver_link = platform / "driver"
        driver_link.unlink(missing_ok=True)
        driver_link.symlink_to(drivers)
        (platform / "power").mkdir(parents=True, exist_ok=True)
        (platform / "power/control").write_text("on\n")

        self.accel0 = self.root / "dev/accel/accel0"
        self.accel0.parent.mkdir(parents=True, exist_ok=True)
        self.accel0.write_bytes(b"")  # seam stand-in for the character node

        subprocess.run(["git", "init", "-q", str(self.checkout)], check=True)
        subprocess.run(["git", "-C", str(self.checkout), "symbolic-ref", "HEAD",
                        "refs/heads/fixture-branch"], check=True)
        (self.checkout / "libane").mkdir(parents=True, exist_ok=True)
        (self.checkout / "libane/ane.c").write_text("/* fixture loader */\n")
        dylib = self.checkout / "bindings/python/dylib"
        dylib.mkdir(parents=True, exist_ok=True)
        (dylib / "libane_python.so").write_bytes(LIB_BYTES)
        (self.checkout / "libane/libane.a").write_bytes(ARCHIVE_BYTES)
        subprocess.run(["git", "-C", str(self.checkout), "-c", "user.name=fixture",
                        "-c", "user.email=fixture@invalid", "commit",
                        "--allow-empty", "-q", "-m", "fixture"], check=True)

        self.compiler.write_bytes(COMPILER_BYTES)
        self.ko.write_bytes(KO_BYTES)
        head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        self.reviewed.write_text(
            "firmware=fixture-firmware\n"
            f"dt-compatible={COMPATIBLE}\n"
            f"module-srcversion={SRCVERSION}\n"
            f"module-ko={self.ko}\n"
            f"module-ko-sha256={sha(KO_BYTES)}\n"
            f"libane-python-sha256={sha(LIB_BYTES)}\n"
            f"libane-archive-sha256={sha(ARCHIVE_BYTES)}\n"
            f"compiler-commit={head}\n"
            f"compiler-sha256={sha(COMPILER_BYTES)}\n")

    def node(self):
        return self.root / "proc/device-tree/soc/ane@26bc04000"

    def platform(self):
        return self.root / "sys/bus/platform/devices/26bc04000.ane"

    def run(self, **overrides):
        env = dict(os.environ,
                   PATH=f"{self.work / 'bin'}:{os.environ['PATH']}",
                   PREFLIGHT_ROOT=str(self.root),
                   ANE_CHECKOUT=str(self.checkout),
                   ANE_DEVICE=str(self.accel0),
                   ANE_COMPILER_BIN=str(self.compiler),
                   ANE_REVIEWED_IDENTITIES=str(self.reviewed))
        env.update(self.extra_env)
        env.update(overrides)
        return subprocess.run(["bash", str(PREFLIGHT)], env=env,
                              capture_output=True, text=True, check=False)


def as_worktree(checkout):
    """Convert the fixture checkout to gitfile format, as any linked
    worktree looks: .git is a file, not a directory."""
    gitdir = checkout.parent / "fixture.git"
    os.rename(checkout / ".git", gitdir)
    (checkout / ".git").write_text(f"gitdir: {gitdir}\n")


def refusals():
    """Named single-fault mutations; each must make preflight exit 2."""
    return {
        "unreviewed-dt-compatible":
            lambda f: (f.node() / "compatible").write_bytes(b"apple,t6000-ane\0"),
        "compatible-only-second-entry-matches":
            lambda f: (f.node() / "compatible").write_bytes(
                b"apple,t6000-ane\0" + COMPATIBLE.encode() + b"\0"),
        "disabled-dt-status":
            lambda f: (f.node() / "status").write_bytes(b"disabled\0"),
        "wrong-engine-reg-address":
            lambda f: (f.node() / "reg").write_bytes(REG_ALT_BASE),
        "truncated-engine-reg-size":
            lambda f: (f.node() / "reg").write_bytes(REG_SMALL_SIZE),
        "missing-reg-size-cells":
            lambda f: (f.root / "proc/device-tree/soc/#size-cells").unlink(),
        "missing-ane-node":
            lambda f: shutil.move(str(f.node()),
                                  str(f.root / "proc/device-tree/soc/moved")),
        "degraded-cpu-count":
            lambda f: (f.root / "proc/cpuinfo").write_text(
                "".join(f"processor\t: {core}\n" for core in range(7))),
        "foreign-module-srcversion":
            lambda f: (f.root / "sys/module/ane/srcversion").write_text("OTHERSRC\n"),
        "unreviewed-module-parameters":
            lambda f: (f.root / "sys/module/ane/parameters/ane_skip_dart_invalidate").write_text("1\n"),
        "unreviewed-module-bytes":
            lambda f: f.ko.write_bytes(KO_BYTES_ALT),
        "platform-device-renamed":
            lambda f: f.platform().rename(f.platform().with_name("renamed.ane")),
        "platform-driver-unbound":
            lambda f: (f.platform() / "driver").unlink(),
        "runtime-pm-unpinned":
            lambda f: (f.platform() / "power/control").write_text("auto\n"),
        "non-accel0-device":
            lambda f: setattr(f, "extra_env",
                              {"ANE_DEVICE": str(f.root / "dev/accel/accel1")}),
        "unreviewed-libane-library":
            lambda f: (f.checkout / "bindings/python/dylib/libane_python.so"
                       ).write_bytes(b"rebuilt bytes"),
        "unreviewed-libane-archive":
            lambda f: (f.checkout / "libane/libane.a").write_bytes(b"rebuilt bytes"),
        "unreviewed-compiler-binary":
            lambda f: f.compiler.write_bytes(b"rebuilt compiler bytes"),
        "unreviewed-compiler-commit":
            lambda f: reviewed_replace(f.reviewed, "compiler-commit", "0" * 40),
    }


def reviewed_replace(reviewed, key, value):
    lines = reviewed.read_text().splitlines(keepends=True)
    reviewed.write_text("".join(
        f"{key}={value}\n" if line.startswith(key + "=") else line
        for line in lines))


def reviewed_without(reviewed, key):
    reviewed.write_text("".join(
        line for line in reviewed.read_text().splitlines(keepends=True)
        if not line.startswith(key + "=")))


def main():
    with tempfile.TemporaryDirectory(prefix="h13-preflight-good-") as work:
        fixture = Fixture(Path(work))
        good = fixture.run()
        assert good.returncode == 0, \
            f"qualified fixture refused:\n{good.stdout}\n{good.stderr}"
        assert "H13 preflight: PASS (no device access performed)" in good.stdout
        assert "PREFLIGHT FAIL" not in good.stderr, good.stderr
        assert "fixture-branch" in good.stdout, "libane branch not recorded"
        print("PASS qualified fixture passes every gate "
              "(branch text is recorded, never gates)")

        worktree = Fixture(Path(work))
        as_worktree(worktree.checkout)
        result = worktree.run()
        assert result.returncode == 0, \
            f"worktree-style checkout refused:\n{result.stdout}\n{result.stderr}"
        print("PASS worktree-style checkout (.git file) passes")

        noenv = fixture.run(ANE_REVIEWED_IDENTITIES="")
        assert noenv.returncode == 2 and "no reviewed identity file" in noenv.stderr, \
            f"missing reviewed file did not refuse:\n{noenv.stdout}{noenv.stderr}"
        for key in ("firmware", "dt-compatible", "module-srcversion", "module-ko",
                    "module-ko-sha256", "libane-python-sha256",
                    "libane-archive-sha256", "compiler-commit", "compiler-sha256"):
            assert f"reviewed identities omit {key}" in noenv.stderr, key
        print("PASS missing reviewed identity file refuses and names every key")

        absent = fixture.run(ANE_REVIEWED_IDENTITIES=str(fixture.work / "nope.env"))
        assert absent.returncode == 2 and "no reviewed identity file" in absent.stderr
        print("PASS reviewed file path that does not exist refuses")

        no_firmware = Fixture(fixture.work)
        reviewed_without(no_firmware.reviewed, "firmware")
        result = no_firmware.run()
        assert result.returncode == 2 and "omit firmware" in result.stderr, \
            f"review missing only firmware passed (subshell status lost):\n{result.stdout}{result.stderr}"
        print("PASS review missing only firmware refuses in the parent shell")

        incomplete = Fixture(fixture.work)
        reviewed_without(incomplete.reviewed, "compiler-sha256")
        result = incomplete.run()
        assert result.returncode == 2 and "omit compiler-sha256" in result.stderr, \
            f"incomplete review did not refuse:\n{result.stdout}{result.stderr}"
        print("PASS incomplete reviewed file refuses on the missing key")

        missing = Fixture(fixture.work)
        missing.compiler.unlink()
        result = missing.run()
        assert result.returncode == 2 and "compiler binary sha256" in result.stderr, \
            f"missing compiler binary did not refuse:\n{result.stdout}{result.stderr}"
        print("PASS missing compiler binary refuses")

    for name, mutate in refusals().items():
        with tempfile.TemporaryDirectory(prefix=f"h13-preflight-{name}-") as work:
            fixture = Fixture(Path(work))
            mutate(fixture)
            result = fixture.run()
            assert result.returncode == 2, \
                f"{name}: expected refusal, got 0:\n{result.stdout}\n{result.stderr}"
            assert "PREFLIGHT FAIL" in result.stderr, name
            print(f"PASS refuse {name}")

    print("H13 preflight validation: all cases behave as gated")


if __name__ == "__main__":
    sys.exit(main())
