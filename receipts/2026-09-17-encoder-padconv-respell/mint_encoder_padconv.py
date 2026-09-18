#!/usr/bin/env python3
"""F4 round: mint the padconv respell itself through the oracle tool.

Round 2 landed the W-padded *surface* row (`pad_type=valid` on
``[1,8,375,750]``) but never tried the encoder's actual spelling: a rank-4
``conv`` with ``pad_type="custom"`` and the asymmetric pad ``[0,0,1,0]`` on
the 749-wide input. Every checked-in conv record is same/valid spelled, and
the layout campaign's ``pad`` op is refused in every form, so this closes the
one untested spelling. Controls isolate the custom spelling from the
asymmetric pad. Rejections are findings and are retained.

    python3 receipts/2026-09-17-encoder-padconv-respell/mint_encoder_padconv.py \
        --host macstudio
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2 = HERE.parent / "2026-09-16-compiler-leftover"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(R2))
sys.path.insert(0, str(HERE.parents[1] / "research"))

import mint_encoder_leftover2 as r2  # noqa: E402
import mint_conv_probes as mc  # noqa: E402
import mint_oracles as om  # noqa: E402


def encoder_conv_pad(x_shape, w_shape, out_shape, stride, groups, pad,
                     bias, tag):
    """The round-2 rank-4 builder with the encoder's explicit pad vector."""
    blobs = r2.Blobs()
    weight = blobs.constant("w", w_shape)
    bias_argument = ""
    if bias:
        b = blobs.constant("b", (w_shape[0],))
        bias_argument = f"bias = {b}, "
    body = list(blobs.const_scalars()) + blobs.body
    stride_kind = f"tensor<int32, [{len(stride)}]>"
    body.append('string pt = const()[name = string("pt"), '
                'val = string("custom")];')
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
    name = (f"encoder_conv_pad_c{x_shape[1]}_n{w_shape[0]}_k{w_shape[2]}"
            f"x{w_shape[3]}_s{stride[0]}_g{groups}_bias{int(bias)}_"
            f"p{''.join(map(str, pad))}_{tag}")
    item = om.env_case(name, "conv_probe", {
        "kernel": w_shape[2], "input_channels": x_shape[1],
        "output_channels": w_shape[0], "spatial": x_shape[2],
        "spatial_width": x_shape[3],
        "stride": stride[0], "groups": groups, "bias": bias,
        "pad_type": "custom", "pad": list(pad),
        "input_shape": list(x_shape),
        "output_shape": list(out_shape), "weight_shape": list(w_shape),
        "rank": 4, "weight_mode": "index",
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    return r2.finish_index_case(item, blobs)


def encoder_conv_known_pad(x_shape, w_shape, out_shape, stride, groups, pad,
                           bias, plane, tag):
    """The padconv respell with one bit-plane probe payload, so the section's
    weight bytes name themselves across the baseline and every plane."""
    elements = w_shape[0] * (w_shape[1] // groups) * w_shape[2] * w_shape[3]
    weight, _ = om.blobfile("w", w_shape)
    body = [
        'string pt = const()[name = string("pt"), val = string("custom")];',
        f'tensor<int32, [{len(stride)}]> st = const()[name = string("st"), '
        f'val = tensor<int32, [{len(stride)}]>([{", ".join(map(str, stride))}])];',
        f'tensor<int32, [{len(pad)}]> pd = const()[name = string("pd"), '
        f'val = tensor<int32, [{len(pad)}]>([{", ".join(map(str, pad))}])];',
        'tensor<int32, [2]> dl = const()[name = string("dl"), '
        'val = tensor<int32, [2]>([1, 1])];',
        f'int32 gp = const()[name = string("gp"), val = int32({groups})];',
        weight,
        f'{om.tensor_type(out_shape)} y = conv(dilations = dl, '
        f'groups = gp, pad = pd, pad_type = pt, strides = st, '
        f'weight = w, x = x)[name = string("y")];',
    ]
    name = (f"encoder_conv_known_pad_c{x_shape[1]}_n{w_shape[0]}"
            f"_k{w_shape[2]}x{w_shape[3]}_s{stride[0]}_g{groups}"
            f"_bias{int(bias)}_p{''.join(map(str, pad))}"
            f"_{tag}_plane{plane if plane is not None else 'base'}")
    item = om.env_case(name, "conv_probe", {
        "kernel": w_shape[2], "input_channels": x_shape[1],
        "output_channels": w_shape[0], "spatial": x_shape[2],
        "spatial_width": x_shape[3],
        "stride": stride[0], "groups": groups, "bias": bias,
        "pad_type": "custom", "pad": list(pad),
        "input_shape": list(x_shape),
        "output_shape": list(out_shape), "weight_shape": list(w_shape),
        "rank": 4, "weight_mode": "known",
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    item["weights"] = om.blob(mc.known_weights(elements, plane))
    item["weights_description"] = {
        "storage": "BLOBFILE", "shapes": [list(w_shape)],
        "payload_bytes": elements * 2,
        "value": f"known_weights plane {plane}",
        "pattern": "low_high_bit_plane",
        "low": mc.LOW_PATTERN, "high": mc.HIGH_PATTERN,
    }
    return item


def payload_blob(payload: bytes) -> bytes:
    """The round-2 multi-record blob layout (one constant) with an explicit
    u16 payload, so a section delta names weight bytes and not blob-header
    structure. Mirrors Blobs.blob_bytes_index without touching it."""
    data_start = (64 + r2.Blobs.RECORD_CAPACITY * r2.Blobs.RECORD_BYTES
                  + 0x3F) & ~0x3F
    blob = bytearray(data_start + len(payload))
    struct.pack_into("<II", blob, 0, 1, 2)
    struct.pack_into("<IIQQ", blob, 64, 0xDEADBEEF, 1, len(payload),
                     data_start)
    blob[data_start:data_start + len(payload)] = payload
    return bytes(blob)


def encoder_conv_pad_payload(x_shape, w_shape, out_shape, stride, groups,
                             pad, bias, tag, payload):
    """The padconv respell with the round-2 multi-record blob format but an
    explicit u16 weight payload, so section deltas isolate weight carriage
    from blob-header structure."""
    blobs = r2.Blobs()
    weight = blobs.constant("w", w_shape)
    body = list(blobs.const_scalars()) + blobs.body
    body.append('string pt = const()[name = string("pt"), '
                'val = string("custom")];')
    body.append('tensor<int32, [2]> st = const()[name = string("st"), '
                'val = tensor<int32, [2]>([1, 1])];')
    body.append(f'tensor<int32, [{len(pad)}]> pd = const()'
                f'[name = string("pd"), val = tensor<int32, [{len(pad)}]>'
                f'([{", ".join(map(str, pad))}])];')
    body.append('tensor<int32, [2]> dl = const()[name = string("dl"), '
                'val = tensor<int32, [2]>([1, 1])];')
    body.append(f'int32 gp = const()[name = string("gp"), val = int32({groups})];')
    body.append(
        f'{om.tensor_type(out_shape)} y = conv(dilations = dl, '
        f'groups = gp, pad = pd, pad_type = pt, strides = st, '
        f'weight = {weight}, x = x)[name = string("y")];')
    name = (f"encoder_conv_padpay_{tag}_w{'x'.join(str(v) for v in payload)}"
            if len(payload) <= 4 else
            f"encoder_conv_padpay_{tag}_{len(payload)}e_"
            f"{payload[0]:04x}_{payload[1]:04x}")
    item = om.env_case(name, "conv_probe", {
        "kernel": w_shape[2], "input_channels": x_shape[1],
        "output_channels": w_shape[0], "spatial": x_shape[2],
        "spatial_width": x_shape[3],
        "stride": stride[0], "groups": groups, "bias": bias,
        "pad_type": "custom", "pad": list(pad),
        "input_shape": list(x_shape),
        "output_shape": list(out_shape), "weight_shape": list(w_shape),
        "rank": 4, "weight_mode": "explicit",
    }, f"{om.tensor_type(x_shape)} x", body, "y", None, None)
    item["weights"] = payload_blob(struct.pack(
        f"<{len(payload)}H", *payload))
    item["weights_description"] = {
        "storage": "BLOBFILE", "shapes": [list(shape) for shape in blobs.shapes],
        "record_offsets": [blobs.offset(index)
                           for index in range(len(blobs.shapes))],
        "payload_bytes": len(payload) * 2,
        "value": f"explicit u16 payload {tag}",
        "pattern": "explicit_u16",
    }
    return item


def campaign():
    f4 = ((1, 8, 375, 749), (8, 1, 1, 1), (1, 8, 375, 750), (1, 1), 8,
          (0, 0, 1, 0), False)
    cases = [
        # F4: the encoder's actual respell -- asymmetric W pad on the
        # 749-wide input at the captured geometry.
        encoder_conv_pad(*f4, "f4"),
        # Tiny control of the same asymmetric pad.
        encoder_conv_pad((1, 8, 8, 7), (8, 1, 1, 1), (1, 8, 8, 8),
                         (1, 1), 8, (0, 0, 1, 0), False, "tiny"),
        # Symmetric custom control: a `same`-equivalent pad spelled custom,
        # to separate the spelling from the asymmetry.
        encoder_conv_pad((1, 4, 8, 8), (8, 4, 3, 3), (1, 8, 8, 8),
                         (1, 1), 1, (1, 1, 1, 1), False, "sym"),
    ]
    # Bit-plane probes at the F4 geometry: the captured custom-pad section
    # carries too few distinct values to be the weight permutation, so the
    # probes name whichever bytes (section or task stream) hold the weights.
    for plane in (None, 0, 1, 2, -1):
        cases.append(encoder_conv_known_pad(*f4, plane, "f4"))
    # Same-format, different-value runs: the round-2 blob layout with three
    # explicit payloads. Control A must reproduce the landed f4 section
    # byte-for-byte; B and C name which section bytes carry weight values.
    f4_padconv = ((1, 8, 375, 749), (8, 1, 1, 1), (1, 8, 375, 750),
                  (1, 1), 8, (0, 0, 1, 0), False)
    cases.append(encoder_conv_pad_payload(*f4_padconv, "ctl",
                                          [1, 2, 3, 4, 5, 6, 7, 8]))
    cases.append(encoder_conv_pad_payload(*f4_padconv, "even",
                                          [2, 4, 6, 8, 10, 12, 14, 16]))
    cases.append(encoder_conv_pad_payload(*f4_padconv, "low",
                                          [0x3801] * 8))
    return cases


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
         "mktemp -d /tmp/mil-hwx-padconv.XXXXXX"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not root.startswith("/tmp/mil-hwx-padconv."):
        raise RuntimeError(f"unexpected remote temporary path: {root!r}")
    script = Path(__file__).resolve()
    research = script.parents[2] / "research"
    r2_script = R2 / "mint_encoder_leftover2.py"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["scp", "-q", str(script), str(r2_script),
             str(research / "mint_oracles.py"),
             str(research / "h13_td.py"),
             str(research / "mint_conv_probes.py"),
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
        Path(__file__).resolve().parent / "oracles-padconv"))
    parser.add_argument("--case", help="substring selecting case names")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-commit", default="073c7b8")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    if args.host:
        return remote_run(args)
    return local_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
