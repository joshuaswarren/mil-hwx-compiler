"""Round 4: remint the uncovered encoder convolutions with the distinct
per-element payload, so the constant-section permutations become readable.

The 2026-09-16 conv captures compiled per-constant uniform payloads (every
section halfword identical), which qualified Apple's task streams but left
every packing permutation invisible. This campaign re-spells the refused
forms with the ``uint16(index + 1)`` wrapping payload the parity harness
already knows (``pattern: uint16_le_index_plus_one_wrapping``), one running
value across every constant region:

- F1 W-major in-projection: ``(1,1024,1,375) -> (1,2048,1,375)`` k1x1 valid
  (the rank-3 spell lowers through exactly this surface);
- F2 depthwise with bias, both W-major kernel placements;
- F4 padconv on the W-padded ``[1,8,375,750]`` surface;
- F7 bias1 pointwise at ``[1,256,750,32]`` (bias placement);
- F8/F9 at the ``[1,256,375,16]`` surface pair;
- F5/F6 multi-tap stride-2 sections, plus the tiny ``c1_n4``/``c4_n8``/
  ``c8_n8`` controls, with distinct payloads so the layouts decode.

    python3 mint_encoder_conv_idx.py --host macstudio --force \\
        --output oracles-idx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "2026-09-16-compiler-leftover"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R2))
sys.path.insert(0, str(HERE.parents[1] / "research"))

import mint_encoder_leftover2 as r2  # noqa: E402
import mint_oracles as om  # noqa: E402


def encoder_conv_idx(x_shape, w_shape, out_shape, stride, groups,
                     pad_type, bias, tag):
    """Round 2's rank-4 same/valid builder, finished with the distinct
    per-element payload and a case name that cannot collide with the
    uniform-payload corpus."""
    blobs = r2.Blobs()
    weight = blobs.constant("w", w_shape)
    bias_argument = ""
    if bias:
        b = blobs.constant("b", (w_shape[0],))
        bias_argument = f"bias = {b}, "
    body = list(blobs.const_scalars()) + blobs.body
    stride_kind = f"tensor<int32, [{len(stride)}]>"
    pad = (0,) * (2 * len(stride))
    body.append(
        f'string pt = const()[name = string("pt"), val = string("{pad_type}")];')
    body.append(f'{stride_kind} st = const()[name = string("st"), '
                f'val = {stride_kind}([{", ".join(map(str, stride))}])];')
    body.append(f'tensor<int32, [{len(pad)}]> pd = const()[name = string("pd"), '
                f'val = tensor<int32, [{len(pad)}]>([{", ".join(map(str, pad))}])];')
    body.append('tensor<int32, [2]> dl = const()[name = string("dl"), '
                'val = tensor<int32, [2]>([1, 1])];')
    body.append(f'int32 gp = const()[name = string("gp"), val = int32({groups})];')
    body.append(
        f'{om.tensor_type(out_shape)} y = conv({bias_argument}dilations = dl, '
        f'groups = gp, pad = pd, pad_type = pt, strides = st, '
        f'weight = {weight}, x = x)[name = string("y")];')
    name = (f"encoder_conv_idx_c{x_shape[1]}_n{w_shape[0]}_k{w_shape[2]}"
            f"x{w_shape[3]}_s{stride[0]}_g{groups}_bias{int(bias)}_"
            f"{pad_type}_{tag}" if tag else
            f"encoder_conv_idx_c{x_shape[1]}_n{w_shape[0]}_k{w_shape[2]}"
            f"x{w_shape[3]}_s{stride[0]}_g{groups}_bias{int(bias)}_{pad_type}")
    item = om.env_case(name, "conv_probe", {
        "kernel": w_shape[2], "input_channels": x_shape[1],
        "output_channels": w_shape[0], "spatial": x_shape[2],
        "spatial_width": x_shape[3],
        "stride": stride[0], "groups": groups, "bias": bias,
        "pad_type": pad_type, "input_shape": list(x_shape),
        "output_shape": list(out_shape), "weight_shape": list(w_shape),
        "rank": 4, "weight_mode": "index",
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return r2.finish_index_case(item, blobs)


def campaign():
    cases = [
        # F1: the W-major in-projection respell the rank-3 spell lowers
        # through. The 2026-09-16 H-major capture cannot serve it.
        encoder_conv_idx((1, 1024, 1, 375), (2048, 1024, 1, 1),
                         (1, 2048, 1, 375), (1, 1), 1, "valid", False, "wmaj"),
        # F2: the bias-bearing rank-3 depthwise, respelled both ways.
        encoder_conv_idx((1, 1024, 1, 375), (1024, 1, 1, 9),
                         (1, 1024, 1, 375), (1, 1), 1024, "same", True, "wmaj"),
        encoder_conv_idx((1, 1024, 375, 1), (1024, 1, 9, 1),
                         (1, 1024, 375, 1), (1, 1), 1024, "same", True, "hmaj"),
        # F4: the W-padded rel-pos surface the padconv respell needs.
        encoder_conv_idx((1, 8, 375, 750), (8, 1, 1, 1),
                         (1, 8, 375, 750), (1, 1), 8, "valid", False, "pad"),
        # F7: the bias1 pointwise whose uniform capture hid the bias
        # placement.
        encoder_conv_idx((1, 256, 750, 32), (256, 256, 1, 1),
                         (1, 256, 750, 32), (1, 1), 1, "valid", True, "pw750"),
        # F9: the small pointwise surface pair.
        encoder_conv_idx((1, 256, 375, 16), (256, 256, 1, 1),
                         (1, 256, 375, 16), (1, 1), 1, "valid", True, "pw375"),
        # F8: the second stem-depthwise surface pair.
        encoder_conv_idx((1, 256, 750, 32), (256, 1, 3, 3),
                         (1, 256, 375, 16), (2, 2), 256, "same", True, "dw2"),
        # F6: the stem depthwise whose section never reproduced.
        encoder_conv_idx((1, 256, 1500, 64), (256, 1, 3, 3),
                         (1, 256, 750, 32), (2, 2), 256, "same", True, "dw1"),
        # F5: the stem conv, plus the tiny stride-2 control.
        encoder_conv_idx((1, 1, 3000, 128), (256, 1, 3, 3),
                         (1, 256, 1500, 64), (2, 2), 1, "same", True, "stem"),
        encoder_conv_idx((1, 1, 8, 8), (4, 1, 3, 3),
                         (1, 4, 4, 4), (2, 2), 1, "same", True, "tiny"),
        # Grouped pointwise controls at the sizes whose lane structure the
        # derived packer never modeled.
        encoder_conv_idx((1, 4, 8, 8), (8, 4, 1, 1),
                         (1, 8, 8, 8), (1, 1), 1, "valid", False, "tiny"),
        encoder_conv_idx((1, 8, 8, 8), (8, 1, 1, 1),
                         (1, 8, 8, 8), (1, 1), 8, "valid", False, "tiny"),
    ]
    unique: dict[str, dict] = {}
    for item in cases:
        unique.setdefault(item["name"], item)
    return list(unique.values())


def local_run(args: argparse.Namespace) -> int:
    selected = [item for item in campaign()
                if not args.case or args.case in item["name"]]
    if args.list:
        for item in selected:
            print(item["name"])
        print(f"cases={len(selected)}")
        return 0
    if sys.platform != "darwin":
        raise SystemExit("--local requires macOS")
    tool = Path(args.oracle_tool)
    if not tool.is_file():
        raise SystemExit(f"oracle tool is missing: {tool}")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    decoded = rejected = 0
    for item in selected:
        status, record = r2.run_case_retain(item, output, tool,
                                            args.source_commit)
        decoded += status == "decoded"
        rejected += status == "rejected"
        tasks = len(record.get("task_descriptors") or [])
        const = (record.get("constant_section") or {}).get("size")
        print(f"h13 {item['name']} {status} tasks={tasks} const={const}"
              + ("" if record.get("error") is None
                 else f" error={record['error'][:120]}"), flush=True)
    print(f"SUMMARY cases={len(selected)} decoded={decoded} "
          f"rejected={rejected}")
    return 0


def remote_run(args: argparse.Namespace) -> int:
    import shlex
    import shutil
    import subprocess
    root = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", args.host,
         "mktemp -d /tmp/mil-hwx-encoder-r4.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-encoder-r4."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    research = script.parents[2] / "research"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script),
             str(R2 / "mint_encoder_leftover2.py"),
             str(research / "mint_oracles.py"),
             str(research / "h13_td.py"),
             str(research / "mint_chain_probes.py"),
             f"{args.host}:{root}/"], check=True)
        command = [
            "python3", f"{root}/{script.name}", "--local",
            "--oracle-tool", args.oracle_tool,
            "--output", f"{root}/oracles",
            "--source-commit", args.source_commit,
        ]
        if args.case:
            command.extend(["--case", args.case])
        if args.list:
            command.append("--list")
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", args.host,
             " ".join(shlex.quote(value) for value in command)], check=False)
        if not args.list:
            staging = output / "_staging"
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir()
            subprocess.run(
                ["scp", "-q", "-r", f"{args.host}:{root}/oracles/.",
                 str(staging)], check=True)
            for path in staging.rglob("*"):
                if path.is_file():
                    shutil.copy2(path, output / path.name)
            shutil.rmtree(staging)
        return result.returncode
    finally:
        subprocess.run(["ssh", "-o", "BatchMode=yes", args.host,
                        f"rm -rf -- {shlex.quote(root)}"], check=False)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--host")
    parser.add_argument("--oracle-tool",
                        default="/tmp/h13-oracle/bin/ane-compile-hwx")
    parser.add_argument("--output", default=str(
        Path(__file__).resolve().parent / "oracles-idx"))
    parser.add_argument("--case", help="substring selecting case names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default="f5cd370")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.host:
        return remote_run(args)
    return local_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
