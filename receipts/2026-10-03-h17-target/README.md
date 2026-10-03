# H15-H18 oracles and the H17/H18 parity targets (2026-10-03)

Host-only work. Apple's compiler ran on one M1 Ultra (macOS 26.6.2) through
`/tmp/h13-oracle/bin/ane-compile-hwx` (sha256 `4013c743...4091e7`); no ANE
device was used. No HWX bytes are stored here or in `research/oracles`.

## Measured

- Subtypes: `h15` 6, `h16` 7, `h17` 9, `h18` 10; `h19` exits 1 for every
  input tried.
- Campaign: the 316 H14-campaign cases matching `binary_*`, `unary_*`,
  `env_act_*`, `env_bcast_*`, `matmul_*`, `linear_*`, minted with
  `research/mint_oracles.py` at 677f7f1. Per target: 259 decoded, 57
  rejected; per-case acceptance equals the committed H14 record for all 316.
  Every decoded object splits and decodes with zero unknown register
  addresses and zero truncated records.
- Determinism: two mint runs (48e37d3, 677f7f1) agree on every decoded field
  of all 1036 accepted h15-h18 objects.
- Controls: re-minted h13 and h14 records for the same 316 cases equal the
  committed records in every decoded field (task descriptors, program
  descriptor, tensor descriptors, constant section, error status) with the
  rebuilt compiler tool; only the whole-file hash differs.
- H16-H18 kernel table: in 89 objects per target Apple adds a `__KERN_0`
  segment after `__TEXT` and a matching resource slot. In the 33 elementwise
  cases that read a kernel table, its 128 bytes have the same SHA-256 as the
  H14 `__TEXT/__const` section of the same case; `__TEXT/__const` becomes 16
  KiB of zeros.
- Task streams: 0 of 259 accepted cases have identical `__TEXT/__text` between
  h16 and h17, or between h17 and h18; the constant sections are identical in
  all 259.
- Surface order: six naming probes of one broadcast `add`
  (`[1,64,16,16] + [1,1,16,16]`) under h17 lay the surfaces out in ascending
  MIL-name order, independent of argument order and of input/output role:
  `x,y,z`; `p,q,r`; `p(small),q,r`; `a(out),b,c`; `p(small),q,r`;
  `x(small),y(out),z`.
- Parity: `make test-h17-parity` and `make test-h18-parity` pass 165 cases
  each and refuse 7 off-point inputs by name.

## Not measured

No H15-H18 object has executed on hardware. Matvec, batched broadcast, H15 and
H16 have decoded oracles but no compiler target. The ISA values printed by
`research/inspect_hwx.py` are the reference parser's table, not a field of the
object.
