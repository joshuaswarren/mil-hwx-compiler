# Encoder GPU-leftover inventory + transpose lowering (2026-09-17)

Host-only CPU/compiler work. No jw16/jwm1 contact, no ANE execution, no GPU
lock taken (jw16 held by MlxBarrierChurn per Main), no SET writes, no
mlx-omarchy edits.

## Inventory: what the pinned Parakeet encoder still runs on GPU

Baseline: `mlx-omarchy/receipts/2026-09-14-encoder-leftover.md` (fail-H13
probe, 6718-op encoder MIL), updated by the qualification ladder in this repo
(parity 846 → 861 → 874 → **878**). Current placement default is ABC (attention,
o-proj, FFN mm1/mm2 islands); ABCG (fused XOR-only FFN chain) is opt-in and
loses E2E, so GPU leftovers below are on top of the four ANE islands.

| leftover family (encoder count) | status at session start | covered by |
| --- | --- | --- |
| rank-3 `linear` ×5 geometries, B=8 bmm, FFN chain d1024 s375 | qualified 861→874 (`H13LinearTemplates.inc`) | landed, not yet wired into a jw16 placement |
| rank-3 unaries `silu`/`sigmoid` ×3 forms, softmax/LN ×3, γ/β-peel + GLU/FFN/residual/pos-bias broadcasts ×9 | qualified at 861 | landed |
| **`transpose` ×73** — 48 r3 `[0,2,1]` `[1,375,1024]↔[1,1024,375]`, 24 r4 `[0,2,1,3]` `[1,8,375,128]→[1,375,8,128]`, 1 r4 `[1,256,375,16]→[1,375,256,16]` | decoded Apple oracles (`receipts/2026-09-16-compiler-leftover/`, 1 task, 504 B stream, zero constant section) but **no lowering** — compiler refused `h13.nonfoldable-transpose` | **lowered this session** (below): 72/73 forms |
| `slice_by_index` last-dim ×24 (`[1,8,375,749]`→`[1,8,375,375]`) | decoded (1 task) but no lowering; last-dim rewrite costs 101 MB | still GPU; next cheapest after transpose |
| `conv` ×101 (nine encoder forms: pointwise k1×1, depthwise k9×1/k9×9, grouped padconv, stem convs) | decoded rank-4 respells exist; rect conv keys not in the table | still GPU |
| `concat` ×96, LN-affine ×120, linear-op chains | **Apple-refused** (`callback_status=1`), not an emitter gap | structural elimination: B=8 bmm respell removes the 72 head concats; planner peels LN γ/β into norm + broadcasts (all decoded) |

GPU wall attribution (mlx-omarchy `2026-09-16-encoder-attribution-2.md`) puts
the largest GPU-busy buckets on the 194 leftover fp16 linears (coopmat,
rounding-pinned) and pointwise f32 residue — the transpose/slice/conv
lowerings attack the *coverage* hole (the encoder MIL cannot compile whole on
H13 at all, exit 65), not those buckets directly.

## Landed change: transpose lowering (72/73 forms)

The cheapest real leftover: Apple's decoded programs are a single strided-copy
task (504-byte stream, zero constant section) for every rank-3/rank-4 encoder
surface pair. The one uncovered form is the r4 inverse
`[1,256,375,16]→[1,375,256,16]` (no Apple oracle exists for that direction);
it still refuses `h13.nonfoldable-transpose`, and so does every non-table
geometry.

- `receipts/2026-09-17-encoder-transpose-lowering/emit_transposes.py` —
  regenerates `plugins/H13/H13TransposeTemplates.inc` from the four decoded
  records (`receipts/2026-09-16-compiler-leftover/oracles-round1/encoder_transpose_*.json`,
  now also copied to `research/oracles/h13/` for the parity suite). Word
  reconstruction (header words + per-record header + block values) verified
  byte-identical against the checked-in `kEncoderUnaryTask0` template.
- `H13Program.cpp/.h` — `OracleTransposeTemplate` table keyed by the
  Apple-normalized elementwise triples of input and result (rank 3
  `[1,A,B]→(1,A,B)`, rank 4 `[1,C,H,W]→(C,H,W)`); `supportsTransposeParity`
  / `encodeTransposeParity` build the 1-task program with
  `elementwiseTensor` surfaces on both sides (`taskSurfaceChannels {5,4,6,7}`,
  input channel 5, output channel 4, matching the recorded selector words).
- `ANEH13Compiler.mm` — `transposeParityShapes` (fp16, perm exactly
  `[0,2,1]`/`[0,2,1,3]`, table-covered pair); the rewrite-pass transpose
  handler materializes covered forms as a whole `transpose` operation instead
  of rejecting; `lowerOperation` and the schedule loop treat it as a
  whole-tensor program with encoder name `apple-parity-transpose`.
- `tests/test_h13_transpose_cli.py` (new, wired into `make test-h13`) — the
  four covered forms compile; uncovered r4 inverse, uncovered shape, and
  identity-perm-returned-input refuse with their exact codes.
- `tests/test_h13_layout_cli.py` — two cases pinned the old refusal contract
  and were updated to the new (strictly better) behavior: `transpose-encoder-conv`
  now refuses at the conv itself (`h13.conv-outside-envelope`), and
  `transpose-encoder-residual-add` now compiles (transpose program + native
  add path) where it previously refused outright.
- `tests/test_h13_parity.py` — `encoder_structure` transpose family selected
  (4 cases), encoder `apple-parity-transpose`, expected family count 4.

## Verification (this worktree, Linux omp-studio-local)

| check | result |
| --- | --- |
| `make build/mil-hwxc` (clang++-14, GNUstep, -Werror) | PASS |
| `build/test_h13_encoding` / `build/test_h13_anec` | PASS (H13_ENCODING_OK, rc 0) |
| `python3 tests/test_h13_transpose_cli.py build/mil-hwxc` | PASS (4 covered, 3 refusals) |
| `python3 tests/test_h13_parity.py build/mil-hwxc` | **PASS — 878 cases / 1756 artifacts** (was 874; +4 transpose, task streams word-identical to the Apple oracles, both anec and hwx) |
| `python3 tests/test_h13_layout_cli.py build/mil-hwxc` | PASS |
| `make test-h13` battery | PASS (all 17 CLI suites incl. the new transpose suite) |
| `make test-h13-reference` | FAIL — **pre-existing**: same `ValueError: slice has incorrect fields` (`research/inspect_anec.py:180`) on the baseline binary `~/src/mil-hwx-compiler/build/mil-hwxc` at a8392e6 |
| `make test-h13-simulation` | FAIL — **pre-existing**: `NameError: destination` inside `tests/test_h13_package_simulation.py`, identical on the baseline binary |
| parity rerun after layout-test edits | PASS (unchanged) |

No device gate: the transpose programs are byte-compared against decoded
Apple task streams; a jw16 mint does not exist for the transpose family and
jw16 GPU time was locked (MlxBarrierChurn), so hardware execution was not
attempted. No GPU-parity or E2E claim is made.

## Refs

| repo | ref |
| --- | --- |
| mil-hwx-compiler | `agent/encoder-transpose-lowering` (this worktree, branched from `ffn-mm2-plane` @ a8392e6) |
| ane-linux-experiments | `agent/ffn-chain-fusion` @ 7f01833 (main checkout; also `receipts/parakeet-compiler-coverage` @ 5476d17) |
| oracle provenance | mil-hwx-compiler `encoder-leftover-oracles` @ c1ed80e, host MacStudio.local, tool sha256 3d13fc85… |
