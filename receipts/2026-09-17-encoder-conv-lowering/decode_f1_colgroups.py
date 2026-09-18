"""Decode the F1 colgroup order from the u32-pair capture.

Payload: one running uint32(pair+1) stream - source element e carries
low16(e//2 + 1) at even e, high16(e//2 + 1) at odd e.  With row-major dense
weights e = c*1024 + r, parity of e is parity of r, so a halfword reveals

    r even:  v = (512*(c mod 128) + r//2 + 1) mod 2^16  -> c mod 128, r//2
    r odd:   v = c // 128  (+1 carry when c mod 128 == 127, r >= 1022)

Decode result: the 4 MiB section is 16 regions x 8 slots; slot (g, j) is a
16-lane plane over 1024 consecutive rows holding colgroup

    G(g, j, lane) = 8*B(g, j) + 4*(g mod 2) + j,  lane = c mod 16
    B(g, j)       = 4*(3 - g//4) + ((g//2) mod 2) + 2*(j mod 2)

i.e. column c = 16*G + lane.  All 128 colgroups of the [2048, 1024] weight
appear exactly once - the u16 payload could not see B (blocks 64 apart) nor
the g//4 / j-bit permutations, which is exactly the round-2 refusal.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

REDUCTION = 1024
OUTPUTS = 2048
GROUPS = 16  # regions
SLOTS = 8
LANES = 16
REGION = 262144  # halfwords * 2 bytes
SLOT = 32768
ALIAS = 64 * 1024


def payload_halfwords() -> list[int]:
    elements = OUTPUTS * REDUCTION
    hw = [0] * elements
    for pair in range(elements // 2):
        value = pair + 1
        hw[2 * pair] = value & 0xFFFF
        hw[2 * pair + 1] = (value >> 16) & 0xFFFF
    return hw


def block_index(g: int, j: int) -> int:
    """The c//128 block stored at region g, slot j."""
    return 4 * (3 - g // 4) + ((g // 2) % 2) + 2 * (j % 2)


def colgroup(g: int, j: int) -> int:
    """The 16-column colgroup stored at region g, slot j."""
    return 8 * block_index(g, j) + 4 * (g % 2) + j // 2


def reconstruct(payload: list[int]) -> bytes:
    packed = bytearray(REGION * GROUPS)
    mv = memoryview(packed).cast("H")
    for g in range(GROUPS):
        for j in range(SLOTS):
            base = (g * REGION + j * SLOT) // 2
            column = 16 * colgroup(g, j)
            for lane in range(LANES):
                cursor = base + lane
                source = (column + lane) * REDUCTION
                row = struct.pack(f"<{REDUCTION}H",
                                  *payload[source:source + REDUCTION])
                mv[cursor:cursor + REDUCTION * LANES:LANES] = \
                    memoryview(row).cast("H")
    return bytes(packed)


def stripe_gate(halfwords: list[int], section: bytes) -> dict:
    """Capture-side distinctness gate.

    Under the u16 payload every position is invariant under e -> e + ALIAS,
    so a section of this geometry could never qualify colgroup order beyond
    64 columns.  The u32 payload must: every odd row stripe must be 16
    copies of one c//128 block value (small), every even row stripe must
    carry 16 distinct spread values, and the only uniformity break may be
    the 16 exact payload-carry stripes.
    """
    rows = uniform_odd = spread_even = boundary = bad = 0
    for g in range(GROUPS):
        for j in range(SLOTS):
            base = g * REGION // 2 + j * SLOT // 2
            for r in range(REDUCTION):
                row = halfwords[base + r * LANES: base + r * LANES + LANES]
                rows += 1
                if r % 2 == 0:
                    if len(set(row)) == LANES:
                        spread_even += 1
                    else:
                        bad += 1
                elif len(set(row)) == 1:
                    uniform_odd += 1
                elif (r in (1022, 1023) and len(set(row)) == 2
                      and row[15] == row[14] + 1):
                    boundary += 1
                else:
                    bad += 1
    return {
        "rows": rows,
        "even_rows_spread": spread_even,
        "odd_rows_uniform_block": uniform_odd,
        "boundary_carry_stripes": boundary,
        "violations": bad,
        "u16_payload_would_show_alias": True,
        "u16_alias_note": "e -> e + 64*1024 preserves every uint16; "
                          "colgroups 64 apart were indistinguishable",
    }


def main() -> int:
    here = Path(__file__).resolve().parent
    u32dir = here / "oracles-u32/h13"
    name = "encoder_conv_idx_c1024_n2048_k1x1_s1_g1_bias0_valid_wmaj_u32"
    u32data = (u32dir / f"{name}.const.bin").read_bytes()
    halfwords = list(struct.unpack(f"<{len(u32data) // 2}H", u32data))

    gate = stripe_gate(halfwords, u32data)
    print("== capture distinctness gate ==")
    print(json.dumps(gate, indent=2))
    if gate["violations"]:
        print("GATE FAILED")
        return 1

    print("== reconstruct u32 capture from decoded layout ==")
    rebuilt32 = reconstruct(payload_halfwords())
    ok32 = rebuilt32 == u32data
    print("u32 byte-exact:", ok32)

    print("== cross-check round-2 u16 capture ==")
    r2dir = Path(os.environ.get("F1_U16_DIR", "")) if False else \
        here.parent / "oracles-idx"
    r2name = name.replace("_u32", "")
    candidates = [here / "oracles-idx" / f"{r2name}.const.bin",
                  Path("/home/joshuawarren/.config/superpowers/worktrees"
                       "/mil-hwx-compiler/slice-lastdim/receipts"
                       "/2026-09-17-encoder-conv-lowering/oracles-idx"
                       f"/{r2name}.const.bin")]
    r2data = next((p.read_bytes() for p in candidates if p.is_file()), None)
    ok16 = None
    if r2data is not None:
        u16 = [(e + 1) & 0xFFFF for e in range(OUTPUTS * REDUCTION)]
        ok16 = reconstruct(u16) == r2data
        print("u16 byte-exact:", ok16)
    else:
        print("round-2 u16 capture not found locally; skipped")

    order = [colgroup(g, j) for g in range(GROUPS) for j in range(SLOTS)]
    result = {
        "schema": "f1-colgroup-order-u32",
        "geometry": "16 regions x 8 slots; slot = 16-lane plane over 1024 "
                    "consecutive rows; lane stride 16 halfwords",
        "colgroup_formula": "G(g,j) = 8*(4*(3 - g//4) + ((g//2) mod 2) + "
                            "2*(j mod 2)) + 4*(g mod 2) + j",
        "section_order_colgroups": order,
        "distinctness_gate": gate,
        "u32_byte_exact": ok32,
        "u16_round2_byte_exact": ok16,
        "covers_columns_ge_128": True,
    }
    out = here / "colgroup-order-u32.json"
    out.write_text(json.dumps(result, indent=2))
    print("wrote", out)
    print("section colgroup order (region-major):", order)
    return 0 if ok32 and (ok16 in (True, None)) else 1


if __name__ == "__main__":
    import os
    raise SystemExit(main())
