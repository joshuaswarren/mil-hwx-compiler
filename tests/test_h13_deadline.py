#!/usr/bin/env python3
"""Host test: the qualification deadline stops issuing programs before the
next device submit and never reaches libane when already expired."""

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h13_run_linux", ROOT / "tools" / "h13_run_linux.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ForbiddenAdapter:
    def execute(self, *args, **kwargs):
        raise AssertionError("device execute must not be reached once the "
                             "qualification deadline has expired")


def write_package(root):
    mil = root / "add.mil"
    mil.write_text(
        "program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
        "  func main<ios18>(tensor<fp16, [1, 64, 1, 1]> a, tensor<fp16, [1, 64, 1, 1]> b) {\n"
        "    tensor<fp16, [1, 64, 1, 1]> y = add(x = a, y = b)[name = string(\"binary\")];\n"
        "  } -> (y);\n}\n")
    models = root / "models"
    models.mkdir()
    result = __import__("subprocess").run(
        [str(ROOT / "build" / "mil-hwxc"), "--target", "H13", "--mil", str(mil),
         "--model-root", str(models), "--output", str(root / "pkg")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    inputs = {name: root / f"{name}.fp16" for name in ("a", "b")}
    for path in inputs.values():
        path.write_bytes(bytes(128))
    return root / "pkg", mil, inputs


def main():
    with tempfile.TemporaryDirectory() as tmp:
        package, mil, inputs = write_package(Path(tmp))
        clock = {"tick": 0.0}

        def now():
            return clock["tick"]

        expired = None
        try:
            runner.run_package(package, mil, ROOT / "tests", inputs,
                               ForbiddenAdapter(), deadline_seconds=0, now=now)
        except ValueError as error:
            expired = str(error)
        assert expired and "deadline" in expired and "before program 0" in expired, expired

        class DeviceReached(Exception):
            pass

        class LiveAdapter:
            def execute(self, *args, **kwargs):
                raise DeviceReached

        reached = None
        try:
            runner.run_package(package, mil, ROOT / "tests", inputs, LiveAdapter(), now=now)
        except DeviceReached:
            reached = True
        except ValueError as error:
            assert "deadline" not in str(error), error
            reached = "passed deadline; numeric gate rejected stub output"
        assert reached, "no-deadline run must reach the device phase"
    print("H13_DEADLINE_OK")


if __name__ == "__main__":
    sys.exit(main())
