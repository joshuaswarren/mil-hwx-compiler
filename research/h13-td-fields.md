# H13 matvec task-descriptor and constant-section fields

## Scope and provenance

This report documents the H13 `matmul` (`transpose_y=true`) programs in the
checked-in oracle campaign under `research/oracles/h13`, minted on
`MacStudio.local` (Apple M1 Ultra, macOS 26.6.2 arm64) with
`/tmp/h13-oracle/bin/ane-compile-hwx`, SHA-256
`b1bab437e2da0d26e65799698b63d8ad592d5455eec5da64c5877799b08abcbe`. All 36
accepted matmuls per target use `transpose_y=true`; all 36 `transpose_y=false`
attempts were rejected by Apple's compiler.

Two additional research drivers back the field derivations here. Neither
retains Apple HWX bytes:

- `research/mint_matvec_probes.py` mints matmuls whose weight payload carries a
  known pattern (uniform `fp16 0.5` cannot distinguish one constant-section
  permutation from another). It covers `zero`, `mask`, `index`, per-element
  `onehot<logical>`, and `rowstain<n>` / `colstain<k>` payloads, and records
  only the same hashes and decoded-word tables `mint_oracles.py` records.
- `research/analyze_matvec_oracles.py` reports, for every decoded word, whether
  it is invariant across the 36 cases or varies with M, K, or N.

Apple's compiler was also asked for the Mach-O segment table of two freshly
compiled matvec objects, on the host, to resolve the surface addresses. Only
segment metadata was read; no object bytes left the host.

## Program shape

Every decoded matmul in the grid M ∈ {1, 2, 8, 64}, K ∈ {256, 512, 1024},
N ∈ {256, 512, 1024} is one program with two tasks, except
`matmul_m64_k256_n1024_ty1`, which has three:

| Field | Value |
|---|---|
| Task 0, "preparation" | 126 words (504 bytes), `header[7] = 0x200` |
| Task 1, "compute" | 157 words (628 bytes), `header[7] = 0` |
| Task stream | 1140 bytes of tasks in a 1152-byte constant offset (0x480) |
| `task_words_minus_one` | 125, so the program declares task 0's size |
| Task 1's size | task 0's `header[1][24:16] + 1` = 157 words |
| Constant section | exactly `K * N * 2` bytes |
| Surfaces | input channel 5, output channel 4, no separate kernel resource |

`matmul_m64_k256_n1024_ty1` instead emits 126 + 157 + 126 words, links task 2 at
`0x500`, and takes a 1792-byte (0x700) constant offset; task 2 is a second copy
task whose destination is the output surface.

## Constant section: Apple's weight packing

The section is a pure permutation of the `K * N` fp16 halfwords the compiler
reads for the `[N, K]` weight constant. In halfword units, with

- `g = min(16, N / 16)` interleaved weight rows per plane,
- `planes = N / g`,
- `p = n / g` the plane of output column `n`,
- `q = (p % 16) * (planes / 16) + p / 16` the destination plane,

the element `W[n][k]` lands at

```
destination = q * (K * g) + k * g + (n % g)
```

`plugins/H13/H13Program.cpp:packMatvecWeights` implements exactly this.

Evidence: 502 of 505 pattern probes reproduce Apple's whole-section SHA-256
bit-for-bit, including every `(K, N)` in the shipped grid and the wider set
(K, N) ∈ {16, 32, 64, 128, 256, 512, 1024}²  that stays inside the compiler's
accepted range. The three mismatches are `K ∈ {8, 16}`, where the section is
padded to a 64-byte row stride and its size is no longer `K * N * 2`; those
shapes are outside the shipped envelope. All 36 checked-in matmul oracles also
match, and `tests/test_h13_parity.py` re-derives that equality from the
compiler's own output for every case.

The permutation was read out, not guessed. A `zero` payload leaves only the
weight file's own leading bytes in the section, which located the first
destinations (`0 -> 0`, `2 -> N/8`); `onehot` payloads then pinned individual
destinations by brute-forcing the 2 KiB chunk hashes; and `rowstain` /
`colstain` payloads confirmed that each weight row stays inside one plane.

The 16-way outer grouping in `q` is not arbitrary: task 1 programs 16 kernel-DMA
descriptors whose offsets and sizes are `K * N * 2 / 16` bytes each
(`0x1f848`-`0x1f884` offsets, `0x1f888`-`0x1f8c4` sizes), so the permutation is
what makes each of the 16 DMA chunks hold every sixteenth row group.

Note on oracle weight payloads: `research/mint_oracles.py` writes its blob
entry metadata unaligned relative to the CoreML `BlobMetadata` layout, so the
declared data offset is 0 and both Apple's compiler and this repository's
`ANEBlobResolver` read the weight from file byte 0. The 128-byte blob header is
therefore part of the weight stream the oracle compiled, which is why 56 of the
uniform-`0.5` section's halfwords are not `0x3800`. Parity and numerical
correctness coincide: both compilers permute the same input bytes.

## Program descriptor and surface addresses

Apple's H13 matvec object carries a `__DATA/__bss` scratch allocation below
every surface, and lays the output surface out before the input:

```
scratch  = align(max(M * K * 2, K * 64), 0x4000)   __DATA/__bss at 0x30000000
output   = scratch                                 __FVMLIB/__data, align(M*N*2, 0x4000)
input    = output + align(M * N * 2, 0x4000)       __FVMLIB/__const, align(M*K*2, 0x4000)
text     = input  + align(M * K * 2, 0x4000)       __TEXT/__text, file offset 0x4000
constant = align(text + task stream, 0x40)         __TEXT/__const
```

This reproduces `resource_addresses`, `text_address`, and `constant_address` for
all 35 two-task cases. `__TEXT` keeps segment flags `0x0` even with the scratch
present. `matmul_m64_k256_n1024_ty1` uses a 0x20000 scratch that the two-task
formula does not predict; its template carries the decoded value.

The scratch is not a DMA destination in the two-task form: task 0's tile-DMA
destination word `0x17800` is `0x000000c0`, while every task that writes the
output surface sets bit 26 (`0x040000c1`), and task 0 configures the L2 result
registers instead. The ANEC container therefore keeps the same channels as
before (input 5, output 4) and declares no extra channel.

## Task words

Word counts below are over the 35 two-task cases.

### Task 0, preparation (126 words, 21 vary)

| Word | Value |
|---|---|
| `0x0000c`, `0x00010` | `M` |
| `0x1380c`, `0x13814`, `0x13818` | `M * K * 2` |
| `0x13810` | `K * 2` |
| `header[1]` | `0x009c0000` for M ≤ 8; `0x009c0001` for M = 64. Bits 24:16 are task 1's size minus one; the low bits are unresolved. |
| `0x00000`, `0x00014` | packed 16-bit pair, `0x00010100` at K = 256, `0x00010200` at K = 512, `0x00010400` at K = 1024, and `0x00080100` in the three-task case |
| `0x00034`, `0x04808`, `0x04810`, `0x04814`, `0x04818`, `0x04838`, `0x0483c`, `0x04840`, `0x17808`, `0x1780c`, `0x17810`, `0x17814` | vary with M, K, or N without a resolved formula; the templates carry the decoded values |

The remaining 105 words are invariant and are emitted verbatim, including
`header[2] = 0x00000408`, `header[4] = 0x00000068`, `header[6] = 0x1000d800`,
`header[7] = 0x00000200`, `header[8] = 0x05823025`, `0x0001c = 0x5000a021`,
`0x04804 = 0x00500172`, `0x04830 = 0x00500170`, and `0x0c800 = 0x00000082`.

### Task 1, compute (157 words, 49 vary)

| Word | Value |
|---|---|
| `0x0000c` | `K` |
| `0x00010` | `N` |
| `0x17808`, `0x17810`, `0x17814` | `M * N * 2` |
| `0x1780c` | `N * 2` |
| `0x1f848`-`0x1f884` | 16 kernel-DMA offsets, `i * K * N * 2 / 16` |
| `0x1f888`-`0x1f8c4` | 16 kernel-DMA sizes, `K * N * 2 / 16` |
| `header[1]` | `N / 64 - 1` |
| `0x00000`, `0x00014`, `0x00034`, `0x0480c`, `0x04810`, `0x04814`, `0x04834`, `0x1380c`, `0x13810`, `0x13814`, `0x13818` | vary with M, K, or N without a resolved formula |

Invariant task 1 words include `header[6] = 0x2000b800`,
`header[8] = 0x05024023`, `header[9] = 0x00000021`, `0x0001c = 0x5000b021`,
`0x00034 = 0x10244405` for M ≤ 8, `0x0c800 = 0x00000082`,
`0x0c804 = 0x00101c00`, `0x0c810 = 0x00003c00`, and
`0x17818 = 0x01302031`.

## Coverage and limits

- Covered as one program: every (M, K, N) with M ∈ {1, 2, 8, 64} and
  K, N ∈ {256, 512, 1024}, i.e. all 36 decoded matmul oracles.
- `transpose_y=false` is handled by transposing the constant exactly on the
  host and using the same encoder; a transposed single-row `x` reduces to the
  same contiguous surface and uses the same program.
- K > 1024, N > 1024, or any geometry off the grid keeps the existing
  source-qualified encoder with its reduction chunking and 512-column slices.
- Unresolved: the low bits of task 0's `header[1]`, the packed pair at
  `0x00000`/`0x00014`, the L2 configuration words listed above, the three-task
  scratch size, and which ANEC channel Apple's runtime would use if a matvec
  ever needed the scratch as a DMA surface.
