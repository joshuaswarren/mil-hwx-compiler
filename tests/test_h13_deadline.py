#!/usr/bin/env python3
"""Host test for the H13 whole-device-phase qualification deadline."""

import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h13_run_linux", ROOT / "tools" / "h13_run_linux.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ForbiddenAdapter:
    def execute(self, *args, **kwargs):
        raise AssertionError("device execute must not be reached")


def write_package(root):
    mil = root / "add.mil"
    mil.write_text(
        "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
        "  func main<ios18>(tensor<fp16, [1, 64, 1, 1]> a, tensor<fp16, [1, 64, 1, 1]> b) {\n"
        "    tensor<fp16, [1, 64, 1, 1]> c = add(x = a, y = b)[name = string(\"first\")];\n"
        "    tensor<fp16, [1, 64, 1, 1]> y = add(x = c, y = b)[name = string(\"second\")];\n"
        "  } -> (y);\n}\n")
    models = root / "models"
    models.mkdir()
    result = subprocess.run(
        [str(ROOT / "build" / "mil-hwxc"), "--target", "H13", "--mil", str(mil),
         "--model-root", str(models), "--output", str(root / "pkg")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    inputs = {name: root / f"{name}.fp16" for name in ("a", "b")}
    for path in inputs.values():
        path.write_bytes(bytes(128))
    return root / "pkg", mil, inputs

def check_unmangled_symbols():
    names = []

    class Function:
        pass

    class Library:
        def __getattr__(self, name):
            names.append(name)
            return Function()

    original = runner.ctypes.CDLL
    runner.ctypes.CDLL = lambda *_args, **_kwargs: Library()
    try:
        runner.LibANEAdapter("unused.so")
    finally:
        runner.ctypes.CDLL = original
    for name in ("__ane_src_size", "__ane_dst_size", "__ane_send", "__ane_read"):
        assert name in names, names
        assert f"_LibANEAdapter{name}" not in names, names


def main():
    check_unmangled_symbols()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        package, mil, inputs = write_package(root)
        clock = {"tick": 0.0}

        def now():
            return clock["tick"]

        for invalid in (0.0, -1.0, math.nan, math.inf, -math.inf):
            try:
                runner.run_package(package, mil, root / "models", inputs,
                                   ForbiddenAdapter(), deadline_seconds=invalid, now=now)
                assert False, f"accepted invalid deadline {invalid}"
            except ValueError as error:
                assert "positive finite" in str(error), error

        try:
            runner.run_package(package, mil, root / "models", inputs,
                               ForbiddenAdapter(), now=now)
            assert False, "device execution accepted a missing deadline"
        except ValueError as error:
            assert "requires --deadline-seconds" in str(error), error

        manifest = json.loads((package / "manifest.json").read_text())
        assert manifest["dispatchPlan"] == [0, 1], manifest["dispatchPlan"]
        calls = []

        class CompletingAdapter:
            def execute(self, _anec, _kernel, _inputs, output_sizes):
                calls.append(len(calls))
                clock["tick"] = 0.5
                return [bytes(size) for size in output_sizes]

        original_intermediate_buffer = runner._intermediate_buffer

        def forbidden_intermediate_buffer(*_args):
            raise AssertionError("expired runner allocated the next program input")

        runner._intermediate_buffer = forbidden_intermediate_buffer
        try:
            runner.run_package(package, mil, root / "models", inputs,
                               CompletingAdapter(), deadline_seconds=0.5, now=now)
            assert False, "expired qualification submitted the second program"
        except ValueError as error:
            assert "whole-run qualification deadline" in str(error), error
            assert "before program 1" in str(error), error
        finally:
            runner._intermediate_buffer = original_intermediate_buffer
        assert calls == [0], calls
        class WrongAdapter:
            def execute(self, _anec, _kernel, _inputs, output_sizes):
                return [bytes([0xff]) * size for size in output_sizes]

        try:
            runner.run_package(package, mil, root / "models", inputs,
                               WrongAdapter(), deadline_seconds=1.0, now=now)
            assert False, "accepted incorrect device output"
        except ValueError as error:
            assert "preserved" in str(error), error
        assert (package / "y.failed.fp16").read_bytes() == bytes([0xff]) * 128

        dry_manifest, _, plan = runner.run_package(
            package, mil, root / "models", inputs, None, now=now)
        assert dry_manifest["dispatchPlan"] == [0, 1]
        assert plan["deviceCalls"] is False

        command = [sys.executable, str(ROOT / "tools" / "h13_run_linux.py"),
                   str(package), "--mil", str(mil), "--model-root", str(root / "models"),
                   "--input", f"a={inputs['a']}", "--input", f"b={inputs['b']}",
                   "--output", f"y={root / 'y.fp16'}"]
        missing = subprocess.run(
            command + ["--libane-library", str(root / "missing.so")],
            capture_output=True, text=True)
        assert missing.returncode != 0
        assert "requires --deadline-seconds" in missing.stderr, missing.stderr
        malformed = subprocess.run(
            command + ["--dry-run", "--deadline-seconds", "nan"],
            capture_output=True, text=True)
        assert malformed.returncode != 0
        assert "positive finite" in malformed.stderr, malformed.stderr

        environment = os.environ.copy()
        environment.pop("H13_DEADLINE_SECONDS", None)
        hardware = subprocess.run(
            ["bash", str(ROOT / "tests" / "run_h13_linux_hardware.sh"),
             str(mil), str(root / "models"), f"a={inputs['a']}"],
            capture_output=True, text=True, env=environment)
        assert hardware.returncode != 0
        assert "H13_DEADLINE_SECONDS" in hardware.stderr, hardware.stderr
    print("H13_DEADLINE_OK")


if __name__ == "__main__":
    sys.exit(main())
