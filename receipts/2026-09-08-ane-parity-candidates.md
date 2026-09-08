# ANE candidate compilation provenance (2026-09-08)

This receipt records that the model-shape ANE candidate packages referenced
by the omarchy-ane ane-parity branch (96f5ea3) and the mlx-omarchy ane-parity
branch (8f81cd88) were produced by this compiler tree at 42fd0bb.

## Environment

- Host: omp-studio-local (x86-64 Linux)
- Compiler source: ~/src/mil-hwx-compiler, branch ane-parity, base
  42fd0bb09f2f361b573df4dc314b18c1c92f8ccf (fork main HEAD; bundle pulled
  from jwm1-linux:~/src/ane-eightcore-20260906/compiler.bundle)
- Toolchain bootstrapped under ~/.local/mil-hwx-gnustep via
  scripts/verify-linux-compiler.sh all (pinned libobjc2 a1faad6,
  tools-make d0349cc, libs-base 3d7013e14). Binary produced at
  build/mil-hwxc, ELF x86-64.
- Runtime verification target: jwm1-linux (aarch64, Linux 7.1.6-1-1-ARCH).
  The compiler binary runs on x86-64 hosts only; the produced packages
  are device-independent bytes that were shipped to M1 via scp and
  validated via tools/h13_run_linux.py --dry-run (device-free).

## Build identity (no code changes, branch base only)

- 42fd0bb docs: transformer-layer mapping, conv packing, and fusion rules
  in the knowledge base
- Working tree clean, no patches applied. The ane-parity branch is a
  marker for the candidate compilation provenance only; no H13 lowering
  changes are part of this scope.

## BLOBFILE chunk header format that the compiler requires

At the offset named in the BLOBFILE MIL attribute (= uint64(64) in these
candidates), the file must contain:

    bytes  0..3  : 0xDEADBEEF  (uint32 LE)
    bytes  8..15 : payloadLength  (uint64 LE) >= expected bytes
    bytes 16..23 : payloadOffset  (uint64 LE) -> start of fp16 data

The first 64 bytes before that offset are user-defined; the lifecycle6
matvec package uses zero bytes there, which works.

## Cases

| name | M | K | N | programs | package size |
|---|---:|---:|---:|---:|---:|
| b1-gemv-k896-n4864 | 1 | 896 | 4864 | 96 | 9.1 MB |
| b2-gemv-k4864-n896 | 1 | 4864 | 896 | 147 | 9.3 MB |
| b3-gemm-m64-k896-n4864 | 64 | 896 | 4864 | refused: H13 intermediate physical writes must not overlap |
| b3-gemm-m32-k896-n4864 | 32 | 896 | 4864 | same refusal |
| b3-gemm-m16-k896-n4864 | 16 | 896 | 4864 | same refusal |
| b3-gemm-m8-k896-n4864 | 8 | 896 | 4864 | same refusal |
| b3-gemm-m2-k896-n4864 | 2 | 896 | 4864 | same refusal |

## Compiler command

    ~/src/mil-hwx-compiler/build/mil-hwxc \
        --target H13 \
        --mil <case>/model.mil \
        --model-root <case>/models \
        --output <case>/pkg

The output directory must be absent or empty per the README contract.
