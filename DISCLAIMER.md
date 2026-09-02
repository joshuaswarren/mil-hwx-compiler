# Project scope and disclaimer

This repository is independent research into the Apple Neural Engine. It is
not affiliated with, endorsed by, or supported by Apple Inc.

## What this project is not

- It is not a replacement for Core ML, MLX, or Apple's ANE compiler.
- It is not a production SDK or a supported platform API.
- It is not suitable for safety-critical, security-critical, or real-time use.
- It is not compatible with App Store distribution in its current form.
- It does not promise support across macOS releases or Apple chip generations.

The tested runtime uses private Apple interfaces. Those interfaces can change
or disappear without notice. The compiler currently covers measured H16G
operation and geometry families on the M4 configuration named in
`docs/VERIFICATION.md`. Unsupported inputs fail closed.

## Copyright and provenance

The MIT License covers original source code, tests, documentation, and data
authored for this project. It cannot grant rights in third-party material.

The repository does not include Apple-compiled HWX containers, extracted Task
Descriptor rows, or Apple-compiled container skeletons. Regression tests store
expected hashes and decoded field values rather than the corresponding Apple
binary output.

`plugins/H16G/Encoding/*EncoderData.inc` contains measurements recorded from
compiler output for specific operations and geometries. These tables describe
observed interface behavior; users remain responsible for determining whether
their use and distribution are permitted where they live.

No Apple source code was available or used. The production compiler source is
an independent implementation. It does not call `ANECCompile` or copy an Apple
container at compile time. The legacy v1 path is retained only as a regression
reference and is excluded from the production build.

Apple, Apple Neural Engine, Core ML, and macOS are trademarks of Apple Inc.

## Legal context

This section provides context, not legal advice or a guarantee that a
particular use is lawful.

The project was developed by studying lawfully available software and hardware
to identify functional interfaces needed for interoperability. United States
cases discussing intermediate copying during reverse engineering include
*Atari Games Corp. v. Nintendo of America Inc.*, 975 F.2d 832 (Fed. Cir.
1992), *Sega Enterprises Ltd. v. Accolade, Inc.*, 977 F.2d 1510 (9th Cir.
1992), and *Sony Computer Entertainment, Inc. v. Connectix Corp.*, 203 F.3d
596 (9th Cir. 2000).

Those decisions are fact-specific. *Atari* distinguished reverse engineering
used to understand unprotected ideas and processes from copying protected
expression, and it rejected reliance on material obtained without
authorization. Section 1201(f) of the Digital Millennium Copyright Act also
contains a limited reverse-engineering exception for interoperability, subject
to its statutory conditions.

The MIT License and this disclaimer do not provide permission to use Apple's
software, services, trademarks, or confidential material. Users are
responsible for compliance with applicable licenses and law and should obtain
legal advice for commercial distribution.

## Warranty

The software is provided under the warranty disclaimer in `LICENSE`. Use of
private APIs, generated hardware programs, and system cache provisioning is at
the user's own risk.
