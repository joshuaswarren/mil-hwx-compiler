#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
prefix=${GNUSTEP_PREFIX:-$HOME/.local/mil-hwx-gnustep}
config="$prefix/bin/gnustep-config"
mode=${1:-all}

if [[ $(uname -s) != Linux ]]; then
    echo "Linux host required" >&2
    exit 1
fi
if [[ ! -x "$config" ]]; then
    GNUSTEP_PREFIX="$prefix" "$repo/scripts/bootstrap-linux-toolchain.sh"
fi

export PATH="$prefix/bin:$PATH"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export GNUSTEP_CONFIG_FILE="$prefix/etc/GNUstep/GNUstep.conf"

build_compiler() {
    make -C "$repo" -B build/mil-hwxc \
        GNUSTEP_PREFIX="$prefix" GNUSTEP_CONFIG="$config"
    local links
    links=$(ldd "$repo/build/mil-hwxc")
    printf '%s\n' "$links"
    [[ "$links" == *"$prefix/lib/libgnustep-base.so"* ]]
    [[ "$links" == *"$prefix/lib/libobjc.so"* ]]
}

emit_programs() {
    make -C "$repo" build/mil-hwxc \
        GNUSTEP_PREFIX="$prefix" GNUSTEP_CONFIG="$config"
    local output="$repo/build/linux-compiler-smoke"
    mkdir -p "$output/conv_relu" "$output/w8a8"
    "$repo/build/mil-hwxc" \
        --mil "$repo/tests/fixtures/conv_relu.mil" \
        --model-root "$repo/tests/models/conv_relu" \
        --target H16G --output "$output/conv_relu"
    "$repo/build/mil-hwxc" \
        --mil "$repo/tests/fixtures/w8a8_conv_chain.mil" \
        --model-root "$repo/tests/models/w8a8" \
        --target H16G --output "$output/w8a8"
    python3 "$repo/research/inspect_hwx.py" \
        "$output/conv_relu/program-0.hwx" | tee "$output/conv_relu.inspect"
    python3 "$repo/research/inspect_hwx.py" \
        "$output/w8a8/program-0.hwx" | tee "$output/w8a8.inspect"
    python3 - "$output" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
for name in ("conv_relu", "w8a8"):
    image = root / name / "program-0.hwx"
    manifest = json.loads((root / name / "manifest.json").read_text())
    parsed = (root / f"{name}.inspect").read_text()
    assert image.stat().st_size > 0
    assert manifest["target"] == "H16G"
    assert "program_descriptor" in parsed
    assert "tensor_descriptor" in parsed
    assert "__TEXT/__text" in parsed
PY
    sha256sum "$output/conv_relu/program-0.hwx" \
        "$output/w8a8/program-0.hwx"
}
run_software_tests() {
    local base_libraries
    base_libraries=$("$config" --base-libs)
    local targets=(
        build/test_operation_graph
        build/test_hwx_object_writer
        build/test_program_composition
    )
    make -C "$repo" -B "${targets[@]}" GNUSTEP_PREFIX="$prefix" GNUSTEP_CONFIG="$config" FRAMEWORKS="$base_libraries -lcrypto"
    "$repo/build/test_operation_graph"
    "$repo/build/test_hwx_object_writer"
    "$repo/build/test_program_composition"
    make -C "$repo" test-h13 test-hwx-inspection GNUSTEP_PREFIX="$prefix" GNUSTEP_CONFIG="$config"
}

check_hygiene() {
    python3 - "$repo" <<'PY'
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
files = [root / "Makefile", *(
    path for directory in ("include", "lib", "plugins", "tools")
    for path in (root / directory).rglob("*") if path.is_file()
)]
needles = ("ANECCompile", "ANECompiler.framework")
fixtures = ("conv_relu.mil", "attention.mil", "w8a8_conv_chain.mil")
text = "\n".join(path.read_text(errors="ignore") for path in files)
for needle in (*needles, *fixtures):
    assert needle not in text, f"production source contains forbidden selector: {needle}"
positive_control = "call ANECCompile; select conv_relu.mil"
assert any(needle in positive_control for needle in needles)
assert any(fixture in positive_control for fixture in fixtures)
PY
}

case "$mode" in
    build-only)
        build_compiler
        echo "linux compiler build: PASS"
        ;;
    emit)
        emit_programs
        echo "linux compiler emission: PASS"
        ;;
    hygiene)
        check_hygiene
        echo "linux compiler hygiene: PASS"
        ;;
    software-tests)
        run_software_tests
        echo "linux compiler software tests: PASS"
        ;;
    all)
        build_compiler
        run_software_tests
        emit_programs
        check_hygiene
        echo "linux compiler build: PASS"
        echo "linux compiler software tests: PASS"
        echo "linux compiler emission: PASS"
        echo "linux compiler hygiene: PASS"
        ;;
    *)
        echo "usage: $0 [all|build-only|software-tests|emit|hygiene]" >&2
        exit 64
        ;;
esac
