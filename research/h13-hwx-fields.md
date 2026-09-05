# H13 HWX fields

This table records the structural contract observed in `fresh-w4.hwx.sample`. The sample remains outside this repository. Tests store decoded expectations and compare compiler output without copying the Apple-generated file.

| Field | Observed value | Compiler output |
|---|---:|---:|
| Mach-O CPU type | `0x80` | `0x80` |
| Mach-O CPU subtype | `4` | `4` |
| ISA generation | `7` | `7` |
| `__TEXT` file offset | `0x4000` | `0x4000` |
| `__TEXT,__text` size | `0x274` | `0x274` |
| `__TEXT,__const` content offset | `0x280` | `0x280` |
| `__TEXT,__text` alignment | `2^14` | `2^14` |
| `__TEXT,__const` alignment | `2^6` | `2^6` |
| Program descriptor command size | `0x880` | `0x880` |
| Program descriptor kind | `1` | `1` |
| Program descriptor `+0x0c` | `0x21a` | `0x21a` |
| Program descriptor text address | `__text` | `__text` |
| Program descriptor constant address | `__const` | `__const` |
| Program descriptor input/output resources | input VM addresses, then output VM address | input VM addresses, then output VM address |
| Program descriptor `+0x818` | task words minus one | task words minus one |
| Program descriptor `+0x81c` | `1` | `1` |
| Program descriptor `+0x824` | `0xffff` | `0xffff` |
| Program descriptor `+0x83c` | `0x878` | `0x878` |
| Program descriptor `+0x850` | `0x11` | `0x11` |
| Program descriptor `+0x858` | `4` | `4` |
| Program descriptor `+0x860` | `1` | `1` |
| Program descriptor `+0x868` | `9` | `9` |
| Program descriptor `+0x86c` | `8` | `8` |
| Program descriptor name | `net` | `net` |
| Compiler metadata command size | `0x728` | `0x728` |
| Relocation kind | local section relocation, info high byte `0x05` | local section relocation, info high byte `0x05` |
| Tensor element code | `5` (fp16) | `5` (fp16) |

The H13 ANEC payload is reconstructed as a `0x1000`-byte header followed by the HWX bytes from `__TEXT,__text` through the end of `__TEXT,__const`. The task descriptor starts at ANEC offset `0x1000`; constants start at ANEC offset `0x1280`. Runtime IOSurface channels are output `4`, first input `5`, and optional second input `6`. Logical fp16 channel values use a 64-byte physical channel stride.

## Unknown semantics retained or bounded

The names of several program-descriptor fields are unknown. The compiler preserves the observed values at `+0x0c`, `+0x824`, `+0x83c`, `+0x850`, `+0x858`, `+0x860`, `+0x868`, and `+0x86c` rather than assigning unsupported meanings to them.

The Apple compiler metadata text, build identity, and debug symbols are not required by the decoded load contract. The writer keeps the observed `0x728` metadata command size, emits a deterministic H13 identity string, and emits only deterministic binding symbols. It does not invent Apple tool versions or kernel symbol names.

The fixture has two constant relocations. Compiler-generated H13 matvec tasks contain 16 constant DMA addresses at task offsets `0x74` through `0xb0`; the writer emits those 16 relocations. Programs without constants emit no constant relocations.

The fixture has one `0x4000` text allocation and 128 constant bytes. Compiler output aligns the text segment to `0x4000` while allowing the compiler's larger matvec constant region. Tensor VM allocations come from the H13 ANEC layout rather than the fixture's specific tensor sizes.
