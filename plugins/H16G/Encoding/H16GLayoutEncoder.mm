#import "H16GLayoutEncoder.h"

#import "H16GTDWriter.h"

#import <math.h>

typedef struct {
    NSUInteger offset;
    uint32_t value;
} H16GPacketWord;

static const uint32_t kS2DDMA3Recipe[] = {
    0x00000001,0x00000000,0x00000000,0x00000000,0x00390000,0x0000007e,0x00000068,0x00000000,
    0x00000068,0x00000000,0x00000000,0x00000000,0x00050001,0x00010001,0x00000040,0x00000040,
    0x00000001,0x93658005,0x00000040,0x00000040,0x00000001,0x00000020,0x00014000,0x00000010,
    0x10233300,0x00000000,0x00100000,0x80241342,0x000000ce,0x00000080,0x00002000,0x0000135a,
    0x01002031,0x00009041,0x00500172,0x00500130,0x98039045,0x00000040,0x00000040,0x00000040,
    0x00000040,0x0050017a,0x00000400,0x80021241,0x00103c0c,0x00003c00,0x81029444,0x00020000,
    0x00000800,0x00000040,0x01302031,0x23009344,0x00000000,0x00000000,0x21809442,0x00000000,
    0x00000000,0x22001340,0x00000021,0x22001440,0x010000f1,0x00000000,0x00000000,0x00000000,
    0x00370001,0x0000008a,0x00000000,0x00000000,0x00000020,0x00000000,0x00000000,0x00000000,
    0x00050001,0x00008001,0x00000001,0x00000040,0x93648005,0x00000001,0x00000040,0x00000080,
    0x00014000,0x00000040,0x00240000,0x00000000,0x00100000,0x802c1342,0x0000004e,0x00020000,
    0x00002000,0x00000040,0x0000135a,0x01002031,0x803c1041,0x00000162,0x00000010,0x00000110,
    0x00000100,0x00000100,0x00009052,0x0000084a,0x00004400,0x80021241,0x00103c0c,0x00003c00,
    0x81029444,0x00004000,0x00000040,0x00000080,0x08302031,0x21809344,0x00000000,0x00000000,
    0x21809442,0x00800000,0x00000000,0x22001340,0x10010041,0x22001440,0x010100f1,0x00000000,
    0x00430002,0x00000017,0x04000000,0x00000000,0x00000060,0x00000000,0x00000000,0x00000000,
    0x00040009,0x00001540,0x00000080,0x00008001,0x00000001,0x00000010,0x93648005,0x00000001,
    0x00000010,0x00000200,0x00014000,0x00000010,0x101c0000,0x00000000,0x00100000,0x802c1342,
    0x0000004e,0x00010000,0x00000040,0x00000080,0x0000135a,0x08002031,0x803c1041,0x00000842,
    0x00000010,0x00000110,0x00000100,0x00000100,0x00029052,0x00000142,0x00001100,0x00000020,
    0x00000020,0x00000020,0x00000020,0x80021241,0x00103c0c,0x00003c00,0x81021444,0x00000040,
    0x00000400,0x01302031,0x83119640,0x00000012,0x00a000a0,0x000200e0,0x00000000,0x007f0000,
    0x20010701,0x21809344,0x00800000,0x00000000,0x23809442,0x00000000,0x00000000,0x22001340,
    0x10020041,0x22001440,0x00020031,
};

static const uint32_t kS2DDMA3WideRecipe[] = {
    0x00000001,0x00000000,0x00000000,0x00000000,0x00390000,0x0000007e,0x00000068,0x00000000,
    0x00000068,0x00000000,0x00000000,0x00000000,0x00050001,0x00010001,0x00000080,0x00000080,
    0x00000001,0x93658005,0x00000080,0x00000080,0x00000001,0x00000008,0x00014000,0x00000010,
    0x10233300,0x00000000,0x00100000,0x80241342,0x000000ce,0x00000100,0x00008000,0x0000135a,
    0x01002031,0x00009041,0x00500172,0x00500130,0x98039045,0x00000040,0x00000040,0x00000040,
    0x00000040,0x0050017a,0x00000400,0x80021241,0x00103c0c,0x00003c00,0x81029444,0x00010000,
    0x00000200,0x00000040,0x01302031,0x23009344,0x00000000,0x00000000,0x21809442,0x00000000,
    0x00000000,0x22001340,0x00000021,0x22001440,0x010000f1,0x00000000,0x00000000,0x00000000,
    0x00390001,0x0000008a,0x00000000,0x00000000,0x00000020,0x00000000,0x00000000,0x00000000,
    0x00050001,0x00010001,0x00000001,0x00000080,0x00000020,0x93658005,0x00000001,0x00000080,
    0x00000020,0x00000020,0x00014000,0x00000080,0x00240000,0x00000000,0x00100000,0x802c1342,
    0x0000004e,0x00010000,0x00000800,0x00000040,0x0000135a,0x01002031,0x803c1041,0x00000162,
    0x00000010,0x00000210,0x00000200,0x00000200,0x00009052,0x0000084a,0x00010800,0x80021241,
    0x00103c0c,0x00003c00,0x81029444,0x00002000,0x00000040,0x00000100,0x08302031,0x21809344,
    0x00000000,0x00000000,0x21809442,0x00800000,0x00000000,0x22001340,0x10010041,0x22001440,
    0x010100f1,0x00000000,0x00000000,0x00000000,0x00410002,0x00000013,0x04000000,0x00000000,
    0x00000060,0x00000000,0x00000000,0x00000000,0x00050009,0x00001540,0x00000080,0x00010001,
    0x00000001,0x00000020,0x00000020,0x93658005,0x00000001,0x00000020,0x00000020,0x00000080,
    0x00024000,0x00000010,0x30243300,0x00000001,0x00100000,0x802c1342,0x0000004e,0x00008000,
    0x00000040,0x00000100,0x0000135a,0x08002031,0x803c1041,0x00000842,0x00000010,0x00000210,
    0x00000200,0x00000200,0x00009052,0x0000014a,0x00002100,0x80021241,0x00103c0c,0x00003c00,
    0x81021444,0x00000040,0x00000800,0x01302031,0x83119640,0x00000012,0x00a000a0,0x000200e0,
    0x00000000,0x007f0000,0x20010701,0x21809344,0x00800000,0x00000000,0x23809442,0x00000000,
    0x00000000,0x22001340,0x10020041,0x22001440,0x01020031,
};

static const uint32_t kD2SDMA3Recipe[] = {
    0x00000001,0x00000000,0x00000000,0x00000000,0x00310000,0x00000007,0x00000068,0x00000000,
    0x00000068,0x00000000,0x00000000,0x000001ff,0x00030001,0x00010001,0x00000020,0x00000020,
    0x00000001,0x93658005,0x00000020,0x00000020,0x00000001,0x00000100,0x00014000,0x00000010,
    0x00022200,0x00000020,0x00100000,0x80241342,0x000000ce,0x00000040,0x00000800,0x0000135a,
    0x01002031,0x803e9041,0x00500172,0x00500130,0x00080000,0x00000040,0x00000050,0x00000040,
    0x00000040,0x800a1052,0x00500170,0x00004000,0x00000040,0x80021241,0x00103c0c,0x00003c00,
    0x23009344,0x00000000,0x00000000,0x22001340,0x00000021,0x00000000,0x00000000,0x00000000,
    0x002b0001,0x000000f5,0x00000000,0x00000000,0x00000020,0x00000000,0x00000000,0x000001ff,
    0x00050001,0x00010001,0x00000020,0x00000080,0x00000001,0x93658005,0x00000020,0x00000080,
    0x00000001,0x00000040,0x00014000,0x00000010,0x10233300,0x00000000,0x00100000,0x80281041,
    0x00000120,0x00001000,0x00000040,0x00009052,0x0000014a,0x00080000,0x80021241,0x00103c0c,
    0x00003c00,0x81029444,0x00020000,0x00001000,0x00000040,0x01302031,0x21809442,0x00000000,
    0x00000000,0x22001440,0x010100f1,0x00000000,0x00400002,0x000000fd,0x04000000,0x00000000,
    0x00000060,0x00000000,0x00000000,0x00000000,0x00050009,0x00001540,0x00000080,0x00010001,
    0x00000001,0x00000080,0x00000080,0x91658005,0x00000001,0x00000080,0x00000080,0x00000010,
    0x00084000,0x00000010,0x30241100,0x00100000,0x802c1342,0x0000004e,0x00020000,0x00000400,
    0x00000040,0x0000135a,0x01002031,0x803c1041,0x00000142,0x00000010,0x00000820,0x00000800,
    0x00000800,0x00009052,0x0000014a,0x00008200,0x80021241,0x00103c0c,0x00003c00,0x81021444,
    0x00000100,0x00008000,0x01302031,0x83119640,0x00000012,0x00a000a0,0x000200e0,0x00000000,
    0x007f0000,0x20010701,0x21809344,0x00000000,0x00000000,0x23809442,0x00000000,0x00000000,
    0x22001340,0x10020041,0x22001440,0x01020031,
};

static BOOL containsOffset(const NSUInteger *offsets, NSUInteger count,
                           NSUInteger offset) {
    for (NSUInteger i = 0; i < count; ++i)
        if (offsets[i] == offset) return YES;
    return NO;
}

static BOOL writeRecipe(H16GTDWriter *writer, const uint32_t *words,
                        NSUInteger wordCount, const NSUInteger *derivedOffsets,
                        NSUInteger derivedCount, NSString *family,
                        NSError **error) {
    for (NSUInteger i = 0; i < wordCount; ++i) {
        NSUInteger offset = i * sizeof(uint32_t);
        if (words[i] == 0 ||
            containsOffset(derivedOffsets, derivedCount, offset)) continue;
        NSString *field = [NSString stringWithFormat:@"%@.packet.0x%03lx",
            family, (unsigned long)offset];
        if (![writer writeUInt32:words[i] atOffset:offset field:field error:error])
            return NO;
    }
    return YES;
}

static BOOL put(H16GTDWriter *writer, NSUInteger offset, uint64_t value,
                NSString *field, NSError **error) {
    if (value > UINT32_MAX) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain code:6
            userInfo:@{NSLocalizedDescriptionKey:[NSString stringWithFormat:
                @"field '%@' value 0x%llx exceeds H16G u32", field, value]}];
        return NO;
    }
    return [writer writeUInt32:(uint32_t)value atOffset:offset
                         field:field error:error];
}

static H16GEncodedTDProgram *encodeS2D(NSUInteger channels,
                                      NSUInteger spatial,
                                      NSUInteger block,
                                      NSError **error) {
    BOOL wideB4 = block == 4 && spatial == 128 && channels >= 8 &&
        channels <= 32 && channels % 8 == 0;
    BOOL narrowB4 = block == 4 && spatial == 64 && channels >= 32 &&
        channels <= 96 && channels % 16 == 0;
    BOOL narrowB8 = block == 8 && spatial == 128 && channels >= 8 &&
        channels <= 32 && channels % 8 == 0;
    if (!wideB4 && !narrowB4 && !narrowB8) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain
            code:10 userInfo:@{NSLocalizedDescriptionKey:
                @"S2D shape is legal but no measured three-TD packet variant covers it"}];
        return nil;
    }
    if (wideB4) {
        static const NSUInteger derived[] = {
            0x014,0x054,0x0bc,0x0c0,0x104,0x144,0x164,
            0x168,0x1ac,0x1cc,0x1f4,0x23c,0x25c,0x2d0,
        };
        H16GTDWriter *writer = [[H16GTDWriter alloc]
            initWithByteLength:sizeof(kS2DDMA3WideRecipe)];
        if (!writeRecipe(writer,kS2DDMA3WideRecipe,
            sizeof(kS2DDMA3WideRecipe)/sizeof(kS2DDMA3WideRecipe[0]),
            derived,sizeof(derived)/sizeof(derived[0]),@"s2d.wide",error))
            return nil;
        uint64_t elements=(uint64_t)channels*spatial*spatial;
        uint64_t bytes=2*elements;
        const H16GPacketWord fields[] = {
            {0x014,(uint32_t)((bytes*253)>>19)},
            {0x054,(uint32_t)channels}, {0x0bc,(uint32_t)(bytes/4)},
            {0x0c0,(uint32_t)(channels*spatial/2)},
            {0x104,(uint32_t)((bytes*69)>>17)},
            {0x144,(uint32_t)(channels*block)},
            {0x164,(uint32_t)(bytes/4)},
            {0x168,(uint32_t)(elements/64)},
            {0x1ac,(uint32_t)(8*channels*spatial)},
            {0x1cc,(uint32_t)(bytes*32)},
            {0x1f4,(uint32_t)(channels*19/8)},
            {0x23c,(uint32_t)(channels*block*block)},
            {0x25c,(uint32_t)(elements/4)},
            {0x2d0,(uint32_t)(bytes*32)},
        };
        for (NSUInteger i=0;i<sizeof(fields)/sizeof(fields[0]);++i)
            if (!put(writer,fields[i].offset,fields[i].value,
                     @"s2d.wide.geometry",error)) return nil;
        return [[H16GEncodedTDProgram alloc] initWithData:writer.data
            kernelRelocationOffsets:@[] programRecordCount:6
            programFormatCode:(uint32_t)((bytes*69)>>17)
            scratchByteLength:(NSUInteger)(bytes*36)];
    }
    static const NSUInteger derived[] = {
        0x014,0x038,0x03c,0x048,0x04c,0x054,0x074,0x078,
        0x0bc,0x0c0,0x104,0x12c,0x138,0x13c,0x144,0x15c,
        0x160,0x190,0x1a4,0x1c4,0x1e4,0x224,0x244,0x2c8,
    };
    H16GTDWriter *writer = [[H16GTDWriter alloc]
        initWithByteLength:sizeof(kS2DDMA3Recipe)];
    if (!writeRecipe(writer,kS2DDMA3Recipe,
        sizeof(kS2DDMA3Recipe)/sizeof(kS2DDMA3Recipe[0]),derived,
        sizeof(derived)/sizeof(derived[0]),@"s2d",error)) return nil;
    uint64_t elements=(uint64_t)channels*spatial*spatial, bytes=2*elements;
    const H16GPacketWord fields[] = {
        {0x014,(uint32_t)((bytes*253)>>19)}, {0x038,(uint32_t)spatial},
        {0x03c,(uint32_t)spatial}, {0x048,(uint32_t)spatial},
        {0x04c,(uint32_t)spatial}, {0x054,(uint32_t)channels},
        {0x074,(uint32_t)(2*spatial)}, {0x078,(uint32_t)(2*spatial*spatial)},
        {0x0bc,(uint32_t)(2*bytes/block)},
        {0x0c0,(uint32_t)(4*channels*spatial/block)},
        {0x104,(uint32_t)((bytes*69)>>17)}, {0x12c,(uint32_t)spatial},
        {0x138,(uint32_t)spatial}, {0x13c,(uint32_t)(channels*block)},
        {0x144,(uint32_t)spatial}, {0x15c,(uint32_t)(2*bytes/block)},
        {0x160,(uint32_t)(elements/(4*block))},
        {0x190,(uint32_t)(272*spatial)},
        {0x1a4,(uint32_t)(8*channels*spatial)},
        {0x1c4,(uint32_t)(bytes*32)}, {0x1e4,(uint32_t)((bytes*23)>>18)},
        {0x224,(uint32_t)(elements/256)}, {0x244,(uint32_t)(elements/2)},
        {0x2c8,(uint32_t)(bytes*32)},
    };
    for (NSUInteger i=0;i<sizeof(fields)/sizeof(fields[0]);++i)
        if (!put(writer,fields[i].offset,fields[i].value,@"s2d.geometry",error))
            return nil;
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:@[] programRecordCount:6
        programFormatCode:(uint32_t)((bytes*69)>>17)
        scratchByteLength:(NSUInteger)(bytes*36)];
}

static H16GEncodedTDProgram *encodeD2S(NSUInteger channels,
                                      NSUInteger spatial,
                                      NSUInteger block,
                                      NSError **error) {
    BOOL b4Family = block == 4 && spatial == 32 && channels >= 256 &&
        channels <= 512 && channels % 64 == 0;
    BOOL b8Family = block == 8 && spatial == 16 && channels >= 512 &&
        channels <= 1536 && channels % 512 == 0;
    if (!b4Family && !b8Family) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain
            code:11 userInfo:@{NSLocalizedDescriptionKey:
                @"D2S shape is legal but no measured 0x290 packet family covers it"}];
        return nil;
    }
    static const NSUInteger derived[] = {
        0x014,0x02c,0x038,0x03c,0x048,0x04c,0x054,0x078,
        0x090,0x094,0x09c,0x0a0,0x0ac,0x0b0,0x0e4,0x0fc,
        0x108,0x118,0x124,0x144,0x148,0x154,0x168,0x16c,
        0x194,0x1dc,0x1f8,0x1fc,
    };
    H16GTDWriter *writer = [[H16GTDWriter alloc]
        initWithByteLength:sizeof(kD2SDMA3Recipe)];
    if (!writeRecipe(writer,kD2SDMA3Recipe,
        sizeof(kD2SDMA3Recipe)/sizeof(kD2SDMA3Recipe[0]),derived,
        sizeof(derived)/sizeof(derived[0]),@"d2s",error)) return nil;
    uint64_t elements=(uint64_t)channels*spatial*spatial, bytes=2*elements;
    uint64_t outputChannels=channels/((uint64_t)block*block);
    if (outputChannels/2+1 >= 32) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain code:7
            userInfo:@{NSLocalizedDescriptionKey:
                @"D2S three-TD mask exceeds its measured 32-bit field"}];
        return nil;
    }
    uint64_t mask=(1ULL<<(outputChannels/2+1))-1;
    const H16GPacketWord fields[] = {
        {0x014,(uint32_t)(bytes*block/262144-1)}, {0x02c,(uint32_t)mask},
        {0x038,(uint32_t)spatial}, {0x03c,(uint32_t)spatial},
        {0x048,(uint32_t)spatial}, {0x04c,(uint32_t)spatial},
        {0x054,(uint32_t)channels}, {0x078,(uint32_t)(64*spatial)},
        {0x090,(uint32_t)bytes}, {0x094,(uint32_t)(2*spatial)},
        {0x09c,(uint32_t)(2*spatial)}, {0x0a0,(uint32_t)(2*spatial)},
        {0x0ac,(uint32_t)(2*channels*spatial)},
        {0x0b0,(uint32_t)(2*spatial)},
        {0x0e4,(uint32_t)nearbyint((double)bytes*245.0/524288.0)},
        {0x0fc,(uint32_t)mask}, {0x108,(uint32_t)spatial},
        {0x118,(uint32_t)spatial}, {0x124,(uint32_t)(channels/block)},
        {0x144,(uint32_t)(elements/64)}, {0x148,(uint32_t)(2*spatial)},
        {0x154,(uint32_t)bytes}, {0x168,(uint32_t)(elements/2)},
        {0x16c,(uint32_t)(channels*spatial/2)},
        {0x194,(uint32_t)((bytes*253)>>19)},
        {0x1dc,(uint32_t)(elements>>14)}, {0x1f8,(uint32_t)(elements/2)},
        {0x1fc,(uint32_t)(elements/256)},
    };
    for (NSUInteger i=0;i<sizeof(fields)/sizeof(fields[0]);++i)
        if (!put(writer,fields[i].offset,fields[i].value,@"d2s.geometry",error))
            return nil;
    return [[H16GEncodedTDProgram alloc] initWithData:writer.data
        kernelRelocationOffsets:@[] programRecordCount:6
        programFormatCode:(uint32_t)((bytes*253)>>19)
        scratchByteLength:(NSUInteger)(bytes*32)];
}

@implementation H16GLayoutEncoder
+ (H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
    inputShape:(NSArray<NSNumber *> *)inputShape
    outputShape:(NSArray<NSNumber *> *)outputShape
    blockSize:(NSUInteger)blockSize
    strategy:(ANETileStrategy)strategy
    error:(NSError **)error {
    BOOL s2d=[operationName isEqualToString:@"space_to_depth"];
    BOOL d2s=[operationName isEqualToString:@"depth_to_space"];
    if ((!s2d&&!d2s)||inputShape.count!=4||outputShape.count!=4||
        strategy!=ANETileStrategyLayoutDMA3) {
        if (error) *error=[NSError errorWithDomain:H16GTDWriterErrorDomain code:8
            userInfo:@{NSLocalizedDescriptionKey:
                @"layout encoder requires a planned three-TD S2D/D2S transform"}];
        return nil;
    }
    NSUInteger channels=inputShape[1].unsignedIntegerValue;
    NSUInteger height=inputShape[2].unsignedIntegerValue;
    NSUInteger width=inputShape[3].unsignedIntegerValue;
    if (height!=width) {
        if (error) *error=[NSError errorWithDomain:H16GTDWriterErrorDomain code:9
            userInfo:@{NSLocalizedDescriptionKey:
                @"decoded layout packet family currently requires square spatial geometry"}];
        return nil;
    }
    return s2d ? encodeS2D(channels,width,blockSize,error)
               : encodeD2S(channels,width,blockSize,error);
}
@end
