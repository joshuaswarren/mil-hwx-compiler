#import "H16GSRAMChainEncoder.h"

#import "H16GLUTEncoder.h"
#import "H16GTDWriter.h"

static NSString *const H16GSRAMChainEncoderErrorDomain =
    @"ANE.H16G.SRAMChainEncoder";

static void setError(NSError **error, NSString *message) {
    if (error)
        *error = [NSError errorWithDomain:H16GSRAMChainEncoderErrorDomain
            code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

typedef NS_ENUM(NSUInteger, H16GSRAMBridgeKind) {
    H16GSRAMBridgeKindNone,
    H16GSRAMBridgeKindExpTiles,
    H16GSRAMBridgeKindRowMaximum,
    H16GSRAMBridgeKindCenteredMatrix,
    H16GSRAMBridgeKindExponentialMatrix,
    H16GSRAMBridgeKindRowSum,
    H16GSRAMBridgeKindReciprocalRow,
};

typedef NS_ENUM(NSUInteger, H16GSRAMEgressKind) {
    H16GSRAMEgressKindSRAM,
    H16GSRAMEgressKindExternal,
};

typedef BOOL (*H16GSRAMFormWriter)(H16GTDWriter *, NSUInteger,
                                    NSUInteger, NSError **);

typedef struct {
    const char *operation;
    H16GSRAMBridgeKind requiredBridge;
    H16GSRAMBridgeKind producedBridge;
    H16GSRAMEgressKind egress;
    NSUInteger byteLength;
    NSUInteger taskCount;
    NSUInteger recordCount;
    uint32_t formatCode;
    const NSInteger *kernelRelocationOffsets;
    NSUInteger kernelRelocationCount;
    const char *constantOperation;
    BOOL retainsProducedValue;
    BOOL preservesRetainedValue;
    BOOL requiresRetainedValue;
    H16GSRAMFormWriter writer;
} H16GSRAMTaskForm;

static BOOL writeExpExternalToSRAM(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x37u << 16) | (uint32_t)taskIndex, @"exp.ExpExternalToSRAM.field_0x00");
    FORM_WORD(0x008, 0x00000068u, @"exp.ExpExternalToSRAM.field_0x08");
    FORM_WORD(0x010, 0x00fff868u, @"exp.ExpExternalToSRAM.field_0x10");
    FORM_WORD(0x01c, 0x00000001u, @"exp.ExpExternalToSRAM.field_0x1c");
    FORM_WORD(0x020, 0x00020001u, @"exp.ExpExternalToSRAM.field_0x20");
    FORM_WORD(0x024, 0x00001540u, @"exp.ExpExternalToSRAM.field_0x24");
    FORM_WORD(0x028, 0x00010240u, @"exp.ExpExternalToSRAM.field_0x28");
    FORM_WORD(0x02c, 0x00009584u, @"exp.ExpExternalToSRAM.field_0x2c");
    FORM_WORD(0x030, 0x00000021u, @"exp.ExpExternalToSRAM.field_0x30");
    FORM_WORD(0x034, 0x00000080u, @"exp.ExpExternalToSRAM.field_0x34");
    FORM_WORD(0x038, 0x00010001u, @"exp.ExpExternalToSRAM.field_0x38");
    FORM_WORD(0x03c, 0x00000080u, @"exp.ExpExternalToSRAM.field_0x3c");
    FORM_WORD(0x040, 0x00000080u, @"exp.ExpExternalToSRAM.field_0x40");
    FORM_WORD(0x044, 0x00000001u, @"exp.ExpExternalToSRAM.field_0x44");
    FORM_WORD(0x048, 0x93618005u, @"exp.ExpExternalToSRAM.field_0x48");
    FORM_WORD(0x04c, 0x00000080u, @"exp.ExpExternalToSRAM.field_0x4c");
    FORM_WORD(0x050, 0x00000080u, @"exp.ExpExternalToSRAM.field_0x50");
    FORM_WORD(0x054, 0x00000001u, @"exp.ExpExternalToSRAM.field_0x54");
    FORM_WORD(0x058, 0x00014000u, @"exp.ExpExternalToSRAM.field_0x58");
    FORM_WORD(0x05c, 0x00000002u, @"exp.ExpExternalToSRAM.field_0x5c");
    FORM_WORD(0x060, 0x00011100u, @"exp.ExpExternalToSRAM.field_0x60");
    FORM_WORD(0x064, 0x00000008u, @"exp.ExpExternalToSRAM.field_0x64");
    FORM_WORD(0x068, 0x00100000u, @"exp.ExpExternalToSRAM.field_0x68");
    FORM_WORD(0x06c, 0x80041342u, @"exp.ExpExternalToSRAM.field_0x6c");
    FORM_WORD(0x070, 0x000000ceu, @"exp.ExpExternalToSRAM.field_0x70");
    FORM_WORD(0x074, 0x00000100u, @"exp.ExpExternalToSRAM.field_0x74");
    FORM_WORD(0x078, 0x0000135au, @"exp.ExpExternalToSRAM.field_0x78");
    FORM_WORD(0x07c, 0x01002031u, @"exp.ExpExternalToSRAM.field_0x7c");
    FORM_WORD(0x080, 0x803e9041u, @"exp.ExpExternalToSRAM.field_0x80");
    FORM_WORD(0x084, 0x00500172u, @"exp.ExpExternalToSRAM.field_0x84");
    FORM_WORD(0x088, 0x00500130u, @"exp.ExpExternalToSRAM.field_0x88");
    FORM_WORD(0x08c, 0x00008000u, @"exp.ExpExternalToSRAM.field_0x8c");
    FORM_WORD(0x090, 0x00000100u, @"exp.ExpExternalToSRAM.field_0x90");
    FORM_WORD(0x094, 0x00000100u, @"exp.ExpExternalToSRAM.field_0x94");
    FORM_WORD(0x098, 0x00000100u, @"exp.ExpExternalToSRAM.field_0x98");
    FORM_WORD(0x09c, 0x00000100u, @"exp.ExpExternalToSRAM.field_0x9c");
    FORM_WORD(0x0a0, 0x80021052u, @"exp.ExpExternalToSRAM.field_0xa0");
    FORM_WORD(0x0a4, 0x00500170u, @"exp.ExpExternalToSRAM.field_0xa4");
    FORM_WORD(0x0a8, 0x00000100u, @"exp.ExpExternalToSRAM.field_0xa8");
    FORM_WORD(0x0ac, 0x80049240u, @"exp.ExpExternalToSRAM.field_0xac");
    FORM_WORD(0x0b0, 0x00010082u, @"exp.ExpExternalToSRAM.field_0xb0");
    FORM_WORD(0x0b4, 0x00123c0cu, @"exp.ExpExternalToSRAM.field_0xb4");
    FORM_WORD(0x0b8, 0x00003dc5u, @"exp.ExpExternalToSRAM.field_0xb8");
    FORM_WORD(0x0bc, 0x24009586u, @"exp.ExpExternalToSRAM.field_0xbc");
    FORM_WORD(0x0c8, 0x23009344u, @"exp.ExpExternalToSRAM.field_0xc8");
    FORM_WORD(0x0d4, 0x22001340u, @"exp.ExpExternalToSRAM.field_0xd4");
    FORM_WORD(0x0d8, 0x00000021u, @"exp.ExpExternalToSRAM.field_0xd8");
#undef FORM_WORD
    return YES;
}

static BOOL writeReduceSumSRAMToExternal(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x2du << 16) | (uint32_t)taskIndex, @"reduce_sum.ReduceSumSRAMToExternal.field_0x00");
    FORM_WORD(0x008, 0x04000000u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x08");
    FORM_WORD(0x010, 0x00fff860u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x10");
    FORM_WORD(0x020, 0x00000009u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x20");
    FORM_WORD(0x024, 0x00001540u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x24");
    FORM_WORD(0x028, 0x00000080u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x28");
    FORM_WORD(0x02c, 0x00010001u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x2c");
    FORM_WORD(0x030, 0x00000020u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x30");
    FORM_WORD(0x034, 0x00000004u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x34");
    FORM_WORD(0x038, 0x00000080u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x38");
    FORM_WORD(0x03c, 0x95418005u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x3c");
    FORM_WORD(0x040, 0x00000020u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x40");
    FORM_WORD(0x044, 0x00000004u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x44");
    FORM_WORD(0x048, 0x00000080u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x48");
    FORM_WORD(0x04c, 0x00000004u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x4c");
    FORM_WORD(0x050, 0x00000050u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x50");
    FORM_WORD(0x054, 0x00000025u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x54");
    FORM_WORD(0x05c, 0x800c1041u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x5c");
    FORM_WORD(0x060, 0x00000120u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x60");
    FORM_WORD(0x064, 0x00000100u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x64");
    FORM_WORD(0x068, 0x00000040u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x68");
    FORM_WORD(0x06c, 0x00009052u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x6c");
    FORM_WORD(0x070, 0x0000014au, @"reduce_sum.ReduceSumSRAMToExternal.field_0x70");
    FORM_WORD(0x074, 0x00008000u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x74");
    FORM_WORD(0x078, 0x80801445u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x78");
    FORM_WORD(0x07c, 0x00000040u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x7c");
    FORM_WORD(0x080, 0x01302031u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x80");
    FORM_WORD(0x084, 0x83119640u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x84");
    FORM_WORD(0x088, 0x00000012u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x88");
    FORM_WORD(0x08c, 0x00a000a0u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x8c");
    FORM_WORD(0x090, ((uint32_t)taskIndex << 16) | 0xe0u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x90");
    FORM_WORD(0x098, 0x007f0000u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x98");
    FORM_WORD(0x09c, 0x20010701u, @"reduce_sum.ReduceSumSRAMToExternal.field_0x9c");
    FORM_WORD(0x0a0, 0x23809442u, @"reduce_sum.ReduceSumSRAMToExternal.field_0xa0");
    FORM_WORD(0x0ac, 0x22001440u, @"reduce_sum.ReduceSumSRAMToExternal.field_0xac");
    FORM_WORD(0x0b0, 0x01000031u | ((uint32_t)taskIndex << 16), @"reduce_sum.ReduceSumSRAMToExternal.field_0xb0");
#undef FORM_WORD
    return YES;
}

static BOOL writeReduceMaxExternalToSRAM(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x2eu << 16) | (uint32_t)taskIndex, @"reduce_max.ReduceMaxExternalToSRAM.field_0x00");
    FORM_WORD(0x008, 0x00000068u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x08");
    FORM_WORD(0x010, 0x00000068u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x10");
    FORM_WORD(0x01c, 0x00000001u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x1c");
    FORM_WORD(0x020, 0x00020001u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x20");
    FORM_WORD(0x024, 0x00010001u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x24");
    FORM_WORD(0x028, 0x00000080u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x28");
    FORM_WORD(0x02c, 0x00000080u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x2c");
    FORM_WORD(0x030, 0x00000001u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x30");
    FORM_WORD(0x034, 0x93618005u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x34");
    FORM_WORD(0x038, 0x00000080u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x38");
    FORM_WORD(0x03c, 0x00000080u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x3c");
    FORM_WORD(0x040, 0x00000001u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x40");
    FORM_WORD(0x044, 0x00014000u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x44");
    FORM_WORD(0x048, 0x00000002u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x48");
    FORM_WORD(0x04c, 0x00011100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x4c");
    FORM_WORD(0x050, 0x00000008u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x50");
    FORM_WORD(0x054, 0x00100000u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x54");
    FORM_WORD(0x058, 0x80041342u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x58");
    FORM_WORD(0x05c, 0x000000ceu, @"reduce_max.ReduceMaxExternalToSRAM.field_0x5c");
    FORM_WORD(0x060, 0x00000100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x60");
    FORM_WORD(0x064, 0x0000135au, @"reduce_max.ReduceMaxExternalToSRAM.field_0x64");
    FORM_WORD(0x068, 0x01002031u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x68");
    FORM_WORD(0x06c, 0x803e9041u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x6c");
    FORM_WORD(0x070, 0x00500172u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x70");
    FORM_WORD(0x074, 0x00500130u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x74");
    FORM_WORD(0x078, 0x00008000u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x78");
    FORM_WORD(0x07c, 0x00000100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x7c");
    FORM_WORD(0x080, 0x00000100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x80");
    FORM_WORD(0x084, 0x00000100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x84");
    FORM_WORD(0x088, 0x00000100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x88");
    FORM_WORD(0x08c, 0x80021052u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x8c");
    FORM_WORD(0x090, 0x00500170u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x90");
    FORM_WORD(0x094, 0x00000100u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x94");
    FORM_WORD(0x098, 0x80021241u, @"reduce_max.ReduceMaxExternalToSRAM.field_0x98");
    FORM_WORD(0x09c, 0x00103c0cu, @"reduce_max.ReduceMaxExternalToSRAM.field_0x9c");
    FORM_WORD(0x0a0, 0x00003c00u, @"reduce_max.ReduceMaxExternalToSRAM.field_0xa0");
    FORM_WORD(0x0a4, 0x23009344u, @"reduce_max.ReduceMaxExternalToSRAM.field_0xa4");
    FORM_WORD(0x0b0, 0x22001340u, @"reduce_max.ReduceMaxExternalToSRAM.field_0xb0");
    FORM_WORD(0x0b4, 0x00000021u, @"reduce_max.ReduceMaxExternalToSRAM.field_0xb4");
#undef FORM_WORD
    return YES;
}

static BOOL writeSubSRAMAndExternalToSRAM(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x1fu << 16) | (uint32_t)taskIndex, @"sub.SubSRAMAndExternalToSRAM.field_0x00");
    FORM_WORD(0x010, 0x00fff820u, @"sub.SubSRAMAndExternalToSRAM.field_0x10");
    FORM_WORD(0x01c, 0x00000003u, @"sub.SubSRAMAndExternalToSRAM.field_0x1c");
    FORM_WORD(0x020, 0x00000001u, @"sub.SubSRAMAndExternalToSRAM.field_0x20");
    FORM_WORD(0x024, 0x00010001u, @"sub.SubSRAMAndExternalToSRAM.field_0x24");
    FORM_WORD(0x028, 0x00000020u, @"sub.SubSRAMAndExternalToSRAM.field_0x28");
    FORM_WORD(0x02c, 0x00000004u, @"sub.SubSRAMAndExternalToSRAM.field_0x2c");
    FORM_WORD(0x030, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0x30");
    FORM_WORD(0x034, 0x95418005u, @"sub.SubSRAMAndExternalToSRAM.field_0x34");
    FORM_WORD(0x038, 0x00000020u, @"sub.SubSRAMAndExternalToSRAM.field_0x38");
    FORM_WORD(0x03c, 0x00000004u, @"sub.SubSRAMAndExternalToSRAM.field_0x3c");
    FORM_WORD(0x040, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0x40");
    FORM_WORD(0x044, 0x00000004u, @"sub.SubSRAMAndExternalToSRAM.field_0x44");
    FORM_WORD(0x048, 0x00000050u, @"sub.SubSRAMAndExternalToSRAM.field_0x48");
    FORM_WORD(0x04c, 0x00000025u, @"sub.SubSRAMAndExternalToSRAM.field_0x4c");
    FORM_WORD(0x054, 0x00009040u, @"sub.SubSRAMAndExternalToSRAM.field_0x54");
    FORM_WORD(0x058, 0x00000004u, @"sub.SubSRAMAndExternalToSRAM.field_0x58");
    FORM_WORD(0x05c, 0x00000120u, @"sub.SubSRAMAndExternalToSRAM.field_0x5c");
    FORM_WORD(0x060, 0xb0009045u, @"sub.SubSRAMAndExternalToSRAM.field_0x60");
    FORM_WORD(0x064, 0x00000100u, @"sub.SubSRAMAndExternalToSRAM.field_0x64");
    FORM_WORD(0x068, 0x00000040u, @"sub.SubSRAMAndExternalToSRAM.field_0x68");
    FORM_WORD(0x06c, ((uint32_t)taskIndex << 16) | 0x8800u, @"sub.SubSRAMAndExternalToSRAM.field_0x6c");
    FORM_WORD(0x070, 0x00000010u, @"sub.SubSRAMAndExternalToSRAM.field_0x70");
    FORM_WORD(0x074, 0x00001140u, @"sub.SubSRAMAndExternalToSRAM.field_0x74");
    FORM_WORD(0x078, 0x00000002u, @"sub.SubSRAMAndExternalToSRAM.field_0x78");
    FORM_WORD(0x080, (0x21u << 16) | (uint32_t)(taskIndex + 1), @"sub.SubSRAMAndExternalToSRAM.field_0x80");
    FORM_WORD(0x090, 0x00fff820u, @"sub.SubSRAMAndExternalToSRAM.field_0x90");
    FORM_WORD(0x09c, 0x00000003u, @"sub.SubSRAMAndExternalToSRAM.field_0x9c");
    FORM_WORD(0x0a0, 0x00000001u, @"sub.SubSRAMAndExternalToSRAM.field_0xa0");
    FORM_WORD(0x0a4, 0x00010001u, @"sub.SubSRAMAndExternalToSRAM.field_0xa4");
    FORM_WORD(0x0a8, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0xa8");
    FORM_WORD(0x0ac, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0xac");
    FORM_WORD(0x0b0, 0x00000001u, @"sub.SubSRAMAndExternalToSRAM.field_0xb0");
    FORM_WORD(0x0b4, 0x9d418005u, @"sub.SubSRAMAndExternalToSRAM.field_0xb4");
    FORM_WORD(0x0b8, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0xb8");
    FORM_WORD(0x0bc, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0xbc");
    FORM_WORD(0x0c0, 0x00000001u, @"sub.SubSRAMAndExternalToSRAM.field_0xc0");
    FORM_WORD(0x0c4, 0x00000080u, @"sub.SubSRAMAndExternalToSRAM.field_0xc4");
    FORM_WORD(0x0c8, 0x00000030u, @"sub.SubSRAMAndExternalToSRAM.field_0xc8");
    FORM_WORD(0x0cc, 0x00000044u, @"sub.SubSRAMAndExternalToSRAM.field_0xcc");
    FORM_WORD(0x0d0, 0x00000010u, @"sub.SubSRAMAndExternalToSRAM.field_0xd0");
    FORM_WORD(0x0d8, 0x80a41042u, @"sub.SubSRAMAndExternalToSRAM.field_0xd8");
    FORM_WORD(0x0dc, 0x00400100u, @"sub.SubSRAMAndExternalToSRAM.field_0xdc");
    FORM_WORD(0x0e0, 0x00000100u, @"sub.SubSRAMAndExternalToSRAM.field_0xe0");
    FORM_WORD(0x0e4, ((uint32_t)taskIndex << 16) | 0x8800u, @"sub.SubSRAMAndExternalToSRAM.field_0xe4");
    FORM_WORD(0x0e8, 0x00000010u, @"sub.SubSRAMAndExternalToSRAM.field_0xe8");
    FORM_WORD(0x0ec, 0x80011053u, @"sub.SubSRAMAndExternalToSRAM.field_0xec");
    FORM_WORD(0x0f0, 0x00010000u, @"sub.SubSRAMAndExternalToSRAM.field_0xf0");
    FORM_WORD(0x0f4, 0x00000110u, @"sub.SubSRAMAndExternalToSRAM.field_0xf4");
    FORM_WORD(0x0f8, 0x80011140u, @"sub.SubSRAMAndExternalToSRAM.field_0xf8");
    FORM_WORD(0x0fc, 0x000c0000u, @"sub.SubSRAMAndExternalToSRAM.field_0xfc");
    FORM_WORD(0x100, 0x3fb8a000u, @"sub.SubSRAMAndExternalToSRAM.field_0x100");
#undef FORM_WORD
    return YES;
}

static BOOL writeExpSRAMToSRAM(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x3bu << 16) | (uint32_t)taskIndex, @"exp.ExpSRAMToSRAM.field_0x00");
    FORM_WORD(0x004, 0x00000001u, @"exp.ExpSRAMToSRAM.field_0x04");
    FORM_WORD(0x010, 0x00fff820u, @"exp.ExpSRAMToSRAM.field_0x10");
    FORM_WORD(0x01c, 0x00000003u, @"exp.ExpSRAMToSRAM.field_0x1c");
    FORM_WORD(0x020, 0x00010001u, @"exp.ExpSRAMToSRAM.field_0x20");
    FORM_WORD(0x024, 0xffc01540u, @"exp.ExpSRAMToSRAM.field_0x24");
    FORM_WORD(0x028, 0x00040240u, @"exp.ExpSRAMToSRAM.field_0x28");
    FORM_WORD(0x02c, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x2c");
    FORM_WORD(0x030, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x30");
    FORM_WORD(0x034, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x34");
    FORM_WORD(0x038, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x38");
    FORM_WORD(0x03c, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x3c");
    FORM_WORD(0x040, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x40");
    FORM_WORD(0x044, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x44");
    FORM_WORD(0x048, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x48");
    FORM_WORD(0x04c, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x4c");
    FORM_WORD(0x050, ((uint32_t)taskIndex << 16) | 0x1551u, @"exp.ExpSRAMToSRAM.field_0x50");
    FORM_WORD(0x054, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x54");
    FORM_WORD(0x058, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x58");
    FORM_WORD(0x05c, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x5c");
    FORM_WORD(0x060, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x60");
    FORM_WORD(0x064, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x64");
    FORM_WORD(0x068, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x68");
    FORM_WORD(0x06c, ((uint32_t)taskIndex << 16) | 0x20u, @"exp.ExpSRAMToSRAM.field_0x6c");
    FORM_WORD(0x070, 0x00009584u, @"exp.ExpSRAMToSRAM.field_0x70");
    FORM_WORD(0x074, ((uint32_t)taskIndex << 16) | 0x21u, @"exp.ExpSRAMToSRAM.field_0x74");
    FORM_WORD(0x078, 0x00000080u, @"exp.ExpSRAMToSRAM.field_0x78");
    FORM_WORD(0x07c, 0x00010001u, @"exp.ExpSRAMToSRAM.field_0x7c");
    FORM_WORD(0x080, 0x00000080u, @"exp.ExpSRAMToSRAM.field_0x80");
    FORM_WORD(0x084, 0x00000080u, @"exp.ExpSRAMToSRAM.field_0x84");
    FORM_WORD(0x088, 0x00000001u, @"exp.ExpSRAMToSRAM.field_0x88");
    FORM_WORD(0x08c, 0x93618005u, @"exp.ExpSRAMToSRAM.field_0x8c");
    FORM_WORD(0x090, 0x00000080u, @"exp.ExpSRAMToSRAM.field_0x90");
    FORM_WORD(0x094, 0x00000080u, @"exp.ExpSRAMToSRAM.field_0x94");
    FORM_WORD(0x098, 0x00000001u, @"exp.ExpSRAMToSRAM.field_0x98");
    FORM_WORD(0x09c, 0x00014000u, @"exp.ExpSRAMToSRAM.field_0x9c");
    FORM_WORD(0x0a0, 0x00000080u, @"exp.ExpSRAMToSRAM.field_0xa0");
    FORM_WORD(0x0a4, 0x00044400u, @"exp.ExpSRAMToSRAM.field_0xa4");
    FORM_WORD(0x0ac, 0x00100000u, @"exp.ExpSRAMToSRAM.field_0xac");
    FORM_WORD(0x0b0, 0x800a1041u, @"exp.ExpSRAMToSRAM.field_0xb0");
    FORM_WORD(0x0b4, 0x00000108u, @"exp.ExpSRAMToSRAM.field_0xb4");
    FORM_WORD(0x0b8, 0x00010000u, @"exp.ExpSRAMToSRAM.field_0xb8");
    FORM_WORD(0x0bc, 0x00000110u, @"exp.ExpSRAMToSRAM.field_0xbc");
    FORM_WORD(0x0c0, 0x80029052u, @"exp.ExpSRAMToSRAM.field_0xc0");
    FORM_WORD(0x0c4, 0x00000140u, @"exp.ExpSRAMToSRAM.field_0xc4");
    FORM_WORD(0x0c8, 0x00008000u, @"exp.ExpSRAMToSRAM.field_0xc8");
    FORM_WORD(0x0cc, 0x00000100u, @"exp.ExpSRAMToSRAM.field_0xcc");
    FORM_WORD(0x0d0, 0x80049240u, @"exp.ExpSRAMToSRAM.field_0xd0");
    FORM_WORD(0x0d4, 0x00010082u, @"exp.ExpSRAMToSRAM.field_0xd4");
    FORM_WORD(0x0d8, 0x00123c0cu, @"exp.ExpSRAMToSRAM.field_0xd8");
    FORM_WORD(0x0dc, 0x00003c00u, @"exp.ExpSRAMToSRAM.field_0xdc");
    FORM_WORD(0x0e0, 0x24009586u, @"exp.ExpSRAMToSRAM.field_0xe0");
#undef FORM_WORD
    return YES;
}

static BOOL writeReduceSumSRAMToSRAM(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x1du << 16) | (uint32_t)taskIndex, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x00");
    FORM_WORD(0x010, 0x00fff820u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x10");
    FORM_WORD(0x01c, 0x00000003u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x1c");
    FORM_WORD(0x020, 0x00000001u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x20");
    FORM_WORD(0x024, 0x00010001u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x24");
    FORM_WORD(0x028, 0x00000020u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x28");
    FORM_WORD(0x02c, 0x00000004u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x2c");
    FORM_WORD(0x030, 0x00000080u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x30");
    FORM_WORD(0x034, 0x95418005u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x34");
    FORM_WORD(0x038, 0x00000020u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x38");
    FORM_WORD(0x03c, 0x00000004u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x3c");
    FORM_WORD(0x040, 0x00000080u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x40");
    FORM_WORD(0x044, 0x00000004u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x44");
    FORM_WORD(0x048, 0x00000050u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x48");
    FORM_WORD(0x04c, 0x00000025u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x4c");
    FORM_WORD(0x054, 0x800e1041u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x54");
    FORM_WORD(0x058, 0x00000120u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x58");
    FORM_WORD(0x05c, 0x00008000u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x5c");
    FORM_WORD(0x060, 0x00000100u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x60");
    FORM_WORD(0x064, 0x00000040u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x64");
    FORM_WORD(0x068, 0x00009053u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x68");
    FORM_WORD(0x06c, 0x00010000u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x6c");
    FORM_WORD(0x070, 0x00000010u, @"reduce_sum.ReduceSumSRAMToSRAM.field_0x70");
#undef FORM_WORD
    return YES;
}

static BOOL writeReciprocalSRAMToSRAM(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x3bu << 16) | (uint32_t)taskIndex, @"reciprocal.ReciprocalSRAMToSRAM.field_0x00");
    FORM_WORD(0x010, 0x00fff820u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x10");
    FORM_WORD(0x01c, 0x00000003u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x1c");
    FORM_WORD(0x020, ((uint32_t)(taskIndex - 1) << 16) | 1u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x20");
    FORM_WORD(0x024, 0xffc01540u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x24");
    FORM_WORD(0x028, ((uint32_t)(taskIndex + 1) << 16) | 0x240u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x28");
    FORM_WORD(0x02c, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x2c");
    FORM_WORD(0x030, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x30");
    FORM_WORD(0x034, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x34");
    FORM_WORD(0x038, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x38");
    FORM_WORD(0x03c, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x3c");
    FORM_WORD(0x040, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x40");
    FORM_WORD(0x044, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x44");
    FORM_WORD(0x048, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x48");
    FORM_WORD(0x04c, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x4c");
    FORM_WORD(0x050, ((uint32_t)(taskIndex - 2) << 16) | 0x1551u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x50");
    FORM_WORD(0x054, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x54");
    FORM_WORD(0x058, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x58");
    FORM_WORD(0x05c, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x5c");
    FORM_WORD(0x060, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x60");
    FORM_WORD(0x064, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x64");
    FORM_WORD(0x068, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x68");
    FORM_WORD(0x06c, ((uint32_t)taskIndex << 16) | 0x20u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x6c");
    FORM_WORD(0x070, 0x00009584u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x70");
    FORM_WORD(0x074, ((uint32_t)taskIndex << 16) | 0x21u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x74");
    FORM_WORD(0x078, 0x00000080u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x78");
    FORM_WORD(0x07c, 0x00010001u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x7c");
    FORM_WORD(0x080, 0x00000001u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x80");
    FORM_WORD(0x084, 0x00000080u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x84");
    FORM_WORD(0x088, 0x00000001u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x88");
    FORM_WORD(0x08c, 0x93618005u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x8c");
    FORM_WORD(0x090, 0x00000001u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x90");
    FORM_WORD(0x094, 0x00000080u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x94");
    FORM_WORD(0x098, 0x00000001u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x98");
    FORM_WORD(0x09c, 0x00014000u, @"reciprocal.ReciprocalSRAMToSRAM.field_0x9c");
    FORM_WORD(0x0a0, 0x00000080u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xa0");
    FORM_WORD(0x0a4, 0x00844400u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xa4");
    FORM_WORD(0x0a8, 0x00000030u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xa8");
    FORM_WORD(0x0ac, 0x00100000u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xac");
    FORM_WORD(0x0b0, 0x800a1041u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xb0");
    FORM_WORD(0x0b4, 0x00400100u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xb4");
    FORM_WORD(0x0b8, 0x00010000u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xb8");
    FORM_WORD(0x0bc, 0x00000010u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xbc");
    FORM_WORD(0x0c0, 0x80029052u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xc0");
    FORM_WORD(0x0c4, 0x00000140u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xc4");
    FORM_WORD(0x0c8, 0x00010800u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xc8");
    FORM_WORD(0x0cc, 0x00000020u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xcc");
    FORM_WORD(0x0d0, 0x80049240u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xd0");
    FORM_WORD(0x0d4, 0x00010082u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xd4");
    FORM_WORD(0x0d8, 0x00123c0cu, @"reciprocal.ReciprocalSRAMToSRAM.field_0xd8");
    FORM_WORD(0x0dc, 0x00003c00u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xdc");
    FORM_WORD(0x0e0, 0x24009586u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xe0");
    FORM_WORD(0x0e4, 0x00000080u, @"reciprocal.ReciprocalSRAMToSRAM.field_0xe4");
#undef FORM_WORD
    return YES;
}

static BOOL writeMulSRAMPairToExternal(H16GTDWriter *writer,
    NSUInteger base, NSUInteger taskIndex, NSError **error) {
#define FORM_WORD(offset, value, name) do { \
    if (![writer writeUInt32:(value) atOffset:base + (offset) \
                         field:(name) error:error]) return NO; \
} while (0)
    FORM_WORD(0x000, (0x32u << 16) | (uint32_t)taskIndex, @"mul.MulSRAMPairToExternal.field_0x00");
    FORM_WORD(0x008, 0x04000000u, @"mul.MulSRAMPairToExternal.field_0x08");
    FORM_WORD(0x010, 0x00fff860u, @"mul.MulSRAMPairToExternal.field_0x10");
    FORM_WORD(0x020, 0x00000009u, @"mul.MulSRAMPairToExternal.field_0x20");
    FORM_WORD(0x024, 0x00001540u, @"mul.MulSRAMPairToExternal.field_0x24");
    FORM_WORD(0x028, 0x00000080u, @"mul.MulSRAMPairToExternal.field_0x28");
    FORM_WORD(0x02c, 0x00010001u, @"mul.MulSRAMPairToExternal.field_0x2c");
    FORM_WORD(0x030, 0x00000080u, @"mul.MulSRAMPairToExternal.field_0x30");
    FORM_WORD(0x034, 0x00000080u, @"mul.MulSRAMPairToExternal.field_0x34");
    FORM_WORD(0x038, 0x00000001u, @"mul.MulSRAMPairToExternal.field_0x38");
    FORM_WORD(0x03c, 0x9d418005u, @"mul.MulSRAMPairToExternal.field_0x3c");
    FORM_WORD(0x040, 0x00000080u, @"mul.MulSRAMPairToExternal.field_0x40");
    FORM_WORD(0x044, 0x00000080u, @"mul.MulSRAMPairToExternal.field_0x44");
    FORM_WORD(0x048, 0x00000001u, @"mul.MulSRAMPairToExternal.field_0x48");
    FORM_WORD(0x04c, 0x00000080u, @"mul.MulSRAMPairToExternal.field_0x4c");
    FORM_WORD(0x050, 0x00000030u, @"mul.MulSRAMPairToExternal.field_0x50");
    FORM_WORD(0x054, 0x00000017u, @"mul.MulSRAMPairToExternal.field_0x54");
    FORM_WORD(0x058, 0x00000010u, @"mul.MulSRAMPairToExternal.field_0x58");
    FORM_WORD(0x060, 0xc0a51042u, @"mul.MulSRAMPairToExternal.field_0x60");
    FORM_WORD(0x064, 0x00000104u, @"mul.MulSRAMPairToExternal.field_0x64");
    FORM_WORD(0x068, 0x00008000u, @"mul.MulSRAMPairToExternal.field_0x68");
    FORM_WORD(0x06c, 0x00000100u, @"mul.MulSRAMPairToExternal.field_0x6c");
    FORM_WORD(0x070, 0x00010800u, @"mul.MulSRAMPairToExternal.field_0x70");
    FORM_WORD(0x074, 0x00000020u, @"mul.MulSRAMPairToExternal.field_0x74");
    FORM_WORD(0x078, 0x0000014au, @"mul.MulSRAMPairToExternal.field_0x78");
    FORM_WORD(0x07c, 0x00001053u, @"mul.MulSRAMPairToExternal.field_0x7c");
    FORM_WORD(0x080, 0x00011800u, @"mul.MulSRAMPairToExternal.field_0x80");
    FORM_WORD(0x084, 0x00001140u, @"mul.MulSRAMPairToExternal.field_0x84");
    FORM_WORD(0x088, 0x00080004u, @"mul.MulSRAMPairToExternal.field_0x88");
    FORM_WORD(0x08c, 0x81001444u, @"mul.MulSRAMPairToExternal.field_0x8c");
    FORM_WORD(0x090, 0x00000100u, @"mul.MulSRAMPairToExternal.field_0x90");
    FORM_WORD(0x094, 0x01302031u, @"mul.MulSRAMPairToExternal.field_0x94");
    FORM_WORD(0x098, 0x83119640u, @"mul.MulSRAMPairToExternal.field_0x98");
    FORM_WORD(0x09c, 0x00000012u, @"mul.MulSRAMPairToExternal.field_0x9c");
    FORM_WORD(0x0a0, 0x00a000a0u, @"mul.MulSRAMPairToExternal.field_0xa0");
    FORM_WORD(0x0a4, ((uint32_t)taskIndex << 16) | 0xe0u, @"mul.MulSRAMPairToExternal.field_0xa4");
    FORM_WORD(0x0ac, 0x007f0000u, @"mul.MulSRAMPairToExternal.field_0xac");
    FORM_WORD(0x0b0, 0x20010701u, @"mul.MulSRAMPairToExternal.field_0xb0");
    FORM_WORD(0x0b4, 0x23809442u, @"mul.MulSRAMPairToExternal.field_0xb4");
    FORM_WORD(0x0c0, 0x22001440u, @"mul.MulSRAMPairToExternal.field_0xc0");
    FORM_WORD(0x0c4, 0x01000031u | ((uint32_t)taskIndex << 16), @"mul.MulSRAMPairToExternal.field_0xc4");
#undef FORM_WORD
    return YES;
}

static const NSInteger kRelocations0[] = {0xc0};
static const NSInteger kRelocations4[] = {0xe4};
static const NSInteger kRelocations6[] = {0xe4};

static const H16GSRAMTaskForm kSRAMTaskForms[] = {
    {"exp", H16GSRAMBridgeKindNone,
     H16GSRAMBridgeKindExpTiles, H16GSRAMEgressKindSRAM,
     0xe0, 1, 16, 0,
     kRelocations0, 1, "exp",
     NO, NO,
     NO, writeExpExternalToSRAM},
    {"reduce_sum", H16GSRAMBridgeKindExpTiles,
     H16GSRAMBridgeKindNone, H16GSRAMEgressKindExternal,
     0xb4, 1, 15, 0,
     nullptr, 0, nullptr,
     NO, NO,
     NO, writeReduceSumSRAMToExternal},
    {"reduce_max", H16GSRAMBridgeKindNone,
     H16GSRAMBridgeKindRowMaximum, H16GSRAMEgressKindSRAM,
     0xc0, 1, 16, 1,
     nullptr, 0, nullptr,
     NO, NO,
     NO, writeReduceMaxExternalToSRAM},
    {"sub", H16GSRAMBridgeKindRowMaximum,
     H16GSRAMBridgeKindCenteredMatrix, H16GSRAMEgressKindSRAM,
     0x110, 2, 16, 0,
     nullptr, 0, nullptr,
     NO, NO,
     NO, writeSubSRAMAndExternalToSRAM},
    {"exp", H16GSRAMBridgeKindCenteredMatrix,
     H16GSRAMBridgeKindExponentialMatrix, H16GSRAMEgressKindSRAM,
     0xf0, 1, 14, 0,
     kRelocations4, 1, "exp",
     YES, NO,
     NO, writeExpSRAMToSRAM},
    {"reduce_sum", H16GSRAMBridgeKindExponentialMatrix,
     H16GSRAMBridgeKindRowSum, H16GSRAMEgressKindSRAM,
     0x80, 1, 14, 0,
     nullptr, 0, nullptr,
     NO, YES,
     NO, writeReduceSumSRAMToSRAM},
    {"reciprocal", H16GSRAMBridgeKindRowSum,
     H16GSRAMBridgeKindReciprocalRow, H16GSRAMEgressKindSRAM,
     0xf0, 1, 14, 0,
     kRelocations6, 1, "reciprocal",
     NO, YES,
     NO, writeReciprocalSRAMToSRAM},
    {"mul", H16GSRAMBridgeKindReciprocalRow,
     H16GSRAMBridgeKindNone, H16GSRAMEgressKindExternal,
     0xc8, 1, 14, 0,
     nullptr, 0, nullptr,
     NO, NO,
     YES, writeMulSRAMPairToExternal},
};

static NSArray<NSValue *> *taskFormPlan(NSArray<NSString *> *operations) {
    if (operations.count < 2) return nil;
    H16GSRAMBridgeKind liveBridge = H16GSRAMBridgeKindNone;
    BOOL retainedValue = NO;
    NSMutableArray<NSValue *> *plan = [NSMutableArray array];
    for (NSUInteger index = 0; index < operations.count; ++index) {
        NSString *operation = operations[index];
        H16GSRAMEgressKind expectedEgress = index + 1 == operations.count
            ? H16GSRAMEgressKindExternal : H16GSRAMEgressKindSRAM;
        const H16GSRAMTaskForm *selected = nullptr;
        for (const H16GSRAMTaskForm &form : kSRAMTaskForms) {
            if (![operation isEqualToString:
                    [NSString stringWithUTF8String:form.operation]] ||
                form.requiredBridge != liveBridge ||
                form.egress != expectedEgress ||
                (form.requiresRetainedValue && !retainedValue)) continue;
            selected = &form;
            break;
        }
        if (!selected) return nil;
        [plan addObject:[NSValue valueWithPointer:selected]];
        liveBridge = selected->producedBridge;
        if (!selected->preservesRetainedValue) retainedValue = NO;
        if (selected->retainsProducedValue) retainedValue = YES;
    }
    return liveBridge == H16GSRAMBridgeKindNone ? [plan copy] : nil;
}

static H16GEncodedTDProgram *encodeTaskForms128(
    NSArray<NSValue *> *plan, NSError **error) {
    NSUInteger byteLength = 0x10;
    NSUInteger recordCount = 0;
    uint32_t formatCode = 0;
    for (NSValue *value in plan) {
        const H16GSRAMTaskForm *form =
            static_cast<const H16GSRAMTaskForm *>(value.pointerValue);
        byteLength += form->byteLength;
        recordCount += form->recordCount;
        formatCode = MAX(formatCode, form->formatCode);
    }
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:byteLength];
    if (![writer writeUInt32:1 atOffset:0 field:@"program.start" error:error])
        return nil;
    NSMutableArray<NSNumber *> *relocations = [NSMutableArray array];
    NSUInteger base = 0x10;
    NSUInteger taskIndex = 0;
    for (NSValue *value in plan) {
        const H16GSRAMTaskForm *form =
            static_cast<const H16GSRAMTaskForm *>(value.pointerValue);
        if (!form->writer(writer, base, taskIndex, error)) return nil;
        for (NSUInteger index = 0; index < form->kernelRelocationCount; ++index)
            [relocations addObject:@(base + form->kernelRelocationOffsets[index])];
        base += form->byteLength;
        taskIndex += form->taskCount;
    }
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:relocations
        programRecordCount:recordCount programFormatCode:formatCode
        scratchByteLength:0];
}

@implementation H16GSRAMChainEncoder
+ (H16GEncodedTDProgram *)encodeStageOperations:
    (NSArray<NSString *> *)stageOperations
    rows:(NSUInteger)rows columns:(NSUInteger)columns
    error:(NSError **)error {
    if (rows != 128 || columns != 128) {
        setError(error, @"SRAM task chain requires measured 128x128 geometry");
        return nil;
    }
    NSArray<NSValue *> *plan = taskFormPlan(stageOperations);
    if (!plan) {
        setError(error, @"SRAM task chain has incompatible producer and consumer forms");
        return nil;
    }
    return encodeTaskForms128(plan, error);
}

+ (NSData *)constantRegionForStageOperations:
    (NSArray<NSString *> *)stageOperations error:(NSError **)error {
    NSArray<NSValue *> *plan = taskFormPlan(stageOperations);
    if (!plan) {
        setError(error, @"SRAM task chain has no compatible constant layout");
        return nil;
    }
    NSMutableData *constants = [NSMutableData data];
    for (NSValue *value in plan) {
        const H16GSRAMTaskForm *form =
            static_cast<const H16GSRAMTaskForm *>(value.pointerValue);
        if (!form->constantOperation) continue;
        NSData *region = [H16GLUTEncoder constantRegionForOperationName:
            [NSString stringWithUTF8String:form->constantOperation] error:error];
        if (!region) return nil;
        [constants appendData:region];
    }
    return [constants copy];
}

+ (NSUInteger)taskCountForStageOperations:
    (NSArray<NSString *> *)stageOperations {
    NSArray<NSValue *> *plan = taskFormPlan(stageOperations);
    if (!plan) return 0;
    NSUInteger count = 0;
    for (NSValue *value in plan) {
        const H16GSRAMTaskForm *form =
            static_cast<const H16GSRAMTaskForm *>(value.pointerValue);
        count += form->taskCount;
    }
    return count;
}

+ (H16GEncodedTDProgram *)encodeProducerOperationName:
    (NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    rows:(NSUInteger)rows columns:(NSUInteger)columns
    error:(NSError **)error {
    return [self encodeStageOperations:
        @[producerOperationName, consumerOperationName]
        rows:rows columns:columns error:error];
}
@end
