#!/usr/bin/env python3
"""Re-mint the F5 chain and capture the raw hwx for the worktree-side decoder."""
import sys, os, subprocess, tempfile, json, hashlib
from pathlib import Path

sys.path.insert(0, '/home/joshuawarren/src/mil-hwx-h14-integ-wt/research')
import mint_oracles as om

ROOT = Path('/tmp/hwx_capture_batch5')
ROOT.mkdir(exist_ok=True)
TOOL = Path('/private/tmp/ane-compile-hwx')

CASES = [
    ("env_chain_creadout_h16_s128",
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 128, 1]> o, tensor<fp16, [1, 16, 128, 1]> gate, tensor<fp16, [1, 16, 128, 1]> h_residual) {
    tensor<fp16, [1, 16, 128, 1]> sg = sigmoid(x = gate)[name = string("sg")];
    tensor<fp16, [1, 16, 128, 1]> gated = mul(x = o, y = sg)[name = string("gated")];
    tensor<fp16, [1, 16, 128, 1]> h = add(x = gated, y = h_residual)[name = string("h")];
  } -> (h);
}
"""),
    ("env_bcast_mul_1x16x1x128_runtime_1x16x1x1",
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 128]> x, tensor<fp16, [1, 16, 1, 1]> z) {
    tensor<fp16, [1, 16, 1, 128]> y = mul(x = x, y = z)[name = string("y")];
  } -> (y);
}
"""),
    ("env_bcast_mul_1x16x1x1_runtime_1x16x1x128",
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 1]> z, tensor<fp16, [1, 16, 1, 128]> x) {
    tensor<fp16, [1, 16, 1, 128]> y = mul(x = z, y = x)[name = string("y")];
  } -> (y);
}
"""),
    ("env_bcast_add_1x16x1x128_runtime_1x16x1x1",
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 128]> x, tensor<fp16, [1, 16, 1, 1]> z) {
    tensor<fp16, [1, 16, 1, 128]> y = add(x = x, y = z)[name = string("y")];
  } -> (y);
}
"""),
    ("env_bcast_add_1x16x1x1_runtime_1x16x1x128",
     """program(1.3)
[buildInfo = dict<string, string>({})]
{
  func main<ios18>(tensor<fp16, [1, 16, 1, 1]> z, tensor<fp16, [1, 16, 1, 128]> x) {
    tensor<fp16, [1, 16, 1, 128]> y = add(x = z, y = x)[name = string("y")];
  } -> (y);
}
"""),
]

for name, mil in CASES:
    capture = ROOT / name / "capture"
    capture.mkdir(parents=True, exist_ok=True)
    (capture / "model.mil").write_text(mil)
    (capture / "weights.bin").write_bytes(b"")
    compiled = ROOT / name / "compiled"
    if compiled.exists():
        import shutil; shutil.rmtree(compiled)
    compiled.mkdir()
    result = subprocess.run([str(TOOL), str(capture), str(compiled), "h14"],
                            capture_output=True, text=True, timeout=120)
    hwx = compiled / "model.hwx"
    if result.returncode == 0 and hwx.is_file():
        data = hwx.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        print(f"{name}: decoded {len(data)} bytes sha256={sha[:16]}...")
        # Mirror under stable filename
        target = ROOT / f"{name}.hwx"
        target.write_bytes(data)
    else:
        print(f"{name}: REJECTED rc={result.returncode} err={(result.stderr or result.stdout).strip()[:300]}")