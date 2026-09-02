#import "H16GRegularConvEncoder.h"

#import "H16GTDWriter.h"

typedef struct {
    NSUInteger channels;
    NSUInteger spatial;
    NSUInteger kernel;
    uint32_t descriptorCount;
    uint32_t bankRows;
    uint32_t tileMode;
    uint32_t window;
    uint32_t stride;
    uint32_t inputBytes;
} H16GRegularConvGeometry;

static const H16GRegularConvGeometry kGeometry[] = {
    {64,32,3,0x04,0x20,0x222200,0x1050,0x1000,0x20a00},
    {64,64,3,0x10,0x40,0x210000,0x2000,0x2000,0x80000},
    {64,64,5,0x2b,0x40,0x211100,0x2000,0x2000,0x80000},
    {128,32,3,0x0f,0x20,0x222200,0x2050,0x2000,0x40a00},
    {128,64,3,0x3e,0x40,0x211100,0x4000,0x4000,0x100000},
    {128,64,5,0xad,0x40,0x211100,0x4000,0x4000,0x100000},
};

static const H16GRegularConvGeometry *lookup(NSUInteger channels,
                                              NSUInteger spatial,
                                              NSUInteger kernel) {
    for (const H16GRegularConvGeometry &row : kGeometry)
        if (row.channels == channels && row.spatial == spatial &&
            row.kernel == kernel) return &row;
    return nullptr;
}

static const uint32_t kC64Base[119] = {
    0x00000001,0,0,0,0x00730000,0x00000010,0x04000068,0,
    0x00fff868,0,0,0,0x00050009,0xffc01540,0x000102c0,0x00000021,
    0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,
    0x00031551,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,
    0x000f1559,0x00001200,0x00002400,0x00003600,0x00004800,0x00005a00,0x00006c00,0x00007e00,
    0x00009000,0x0000a200,0x0000b400,0x0000c600,0x0000d800,0x0000ea00,0x0000fc00,0x00010e00,
    0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,
    0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,0x00001200,
    0x00010001,0x00000040,0x00000040,0x00000040,0x93498005,0x00000040,0x00000040,0x00000040,
    0x5042a0c3,0x00000040,0x00210000,0x00000002,0x00100000,0x800c1342,0x000000ce,0x00000080,
    0x00002000,0x0000135a,0x01002031,0x00009041,0x00500172,0x00500130,0x98039045,0x00000080,
    0x00002000,0x00002000,0x00002000,0x0050017a,0x00080000,0x80021241,0x00103c00,0x00003c00,
    0x81009444,0x00000080,0x00002000,0x01302031,0x83109640,0x00000012,0x00a000a0,0,
    0x007f0000,0x20010701,0x24009544,0,0,0x23009344,0,0,
    0x23809442,0,0,0x22001340,0x00000021,0x22001440,0x01000031,
};

static const uint32_t kC128Base[118] = {
    0x00000001,0,0,0,0x00720000,0x0000003e,0x04000068,0,
    0x00fff868,0,0,0,0x00050009,0xffc01540,0x000102c0,0x00000021,
    0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,
    0x00031551,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,
    0x000f1559,0x00004800,0x00009000,0x0000d800,0x00012000,0x00016800,0x0001b000,0x0001f800,
    0x00024000,0x00028800,0x0002d000,0x00031800,0x00036000,0x0003a800,0x0003f000,0x00043800,
    0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,
    0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,0x00004800,
    0x00010001,0x00000040,0x00000040,0x00000080,0x91498005,0x00000040,0x00000040,0x00000080,
    0x5042a0c3,0x00000040,0x00211100,0x00100000,0x800c1342,0x000000ce,0x00000080,0x00002000,
    0x0000135a,0x01002031,0x00009041,0x00500172,0x00500130,0x98039045,0x00000080,0x00004000,
    0x00004000,0x00004000,0x0050017a,0x00100000,0x80021241,0x00103c00,0x00003c00,0x81009444,
    0x00000080,0x00002000,0x01302031,0x83109640,0x00000012,0x00a000a0,0,0x007f0000,
    0x20010701,0x24009544,0,0,0x23009344,0,0,0x23809442,
    0,0,0x22001340,0x00000021,0x22001440,0x01000031,
};

@implementation H16GRegularConvEncoder
+ (BOOL)supportsChannels:(NSUInteger)channels spatial:(NSUInteger)spatial
                   kernel:(NSUInteger)kernel {
    return lookup(channels,spatial,kernel) != nullptr;
}

+ (NSUInteger)kernelRelocationOffsetForChannels:(NSUInteger)channels {
    return channels == 64 ? 0x1ac : (channels == 128 ? 0x1a8 : NSNotFound);
}

+ (NSData *)encodeWithChannels:(NSUInteger)channels
                        spatial:(NSUInteger)spatial
                         kernel:(NSUInteger)kernel
                          error:(NSError **)error {
    const H16GRegularConvGeometry *geometry = lookup(channels,spatial,kernel);
    if (!geometry) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain
            code:7 userInfo:@{NSLocalizedDescriptionKey:
                @"regular Conv geometry has no decoded C64/C128 H16G packet family"}];
        return nil;
    }
    BOOL c64Family = channels == 64;
    NSUInteger wordCount = c64Family ? 119 : 118;
    const uint32_t *base = c64Family ? kC64Base : kC128Base;
    H16GTDWriter *writer = [[H16GTDWriter alloc]
        initWithByteLength:wordCount*4];
    for (NSUInteger index = 0; index < wordCount; ++index)
        if (![writer writeUInt32:base[index] atOffset:index*4
                           field:@"regular_conv.invariant" error:error]) return nil;
    uint32_t step = (uint32_t)(channels*channels*kernel*kernel/8);
    for (NSUInteger index = 0; index < 31; ++index) {
        uint32_t value = index < 15 ? (uint32_t)(index+1)*step : step;
        if (![writer writeUInt32:value atOffset:0x84+index*4
                           field:@"regular_conv.weight_tap" error:error]) return nil;
    }
    const NSUInteger commonOffsets[] = {0x104,0x108,0x114,0x118};
    for (NSUInteger offset : commonOffsets)
        if (![writer writeUInt32:(uint32_t)spatial atOffset:offset
                           field:@"regular_conv.spatial" error:error]) return nil;
    uint32_t kernelWord = kernel == 3 ? 0x5042a0c3 : 0x5084a145;
    if (![writer writeUInt32:geometry->descriptorCount atOffset:0x014
                       field:@"regular_conv.descriptor_count" error:error] ||
        ![writer writeUInt32:(uint32_t)channels atOffset:0x10c
                       field:@"regular_conv.input_channels" error:error] ||
        ![writer writeUInt32:(uint32_t)channels atOffset:0x11c
                       field:@"regular_conv.output_channels" error:error] ||
        ![writer writeUInt32:kernelWord atOffset:0x120
                       field:@"regular_conv.kernel" error:error]) return nil;
    uint32_t rowBytes = (uint32_t)(spatial*2);
    uint32_t planeBytes = (uint32_t)(spatial*spatial*2);
    if (c64Family) {
        const struct { NSUInteger offset; uint32_t value; } fields[] = {
            {0x124,geometry->bankRows},{0x128,geometry->tileMode},{0x12c,2},
            {0x13c,rowBytes},{0x140,planeBytes},{0x15c,rowBytes},
            {0x160,geometry->window},{0x164,geometry->stride},
            {0x168,geometry->stride},{0x170,geometry->inputBytes},
            {0x184,rowBytes},{0x188,planeBytes},
        };
        for (const auto &field : fields)
            if (![writer writeUInt32:field.value atOffset:field.offset
                               field:@"regular_conv.geometry" error:error]) return nil;
    } else {
        const struct { NSUInteger offset; uint32_t value; } fields[] = {
            {0x124,geometry->bankRows},{0x128,geometry->tileMode},
            {0x138,rowBytes},{0x13c,planeBytes},{0x158,rowBytes},
            {0x15c,geometry->window},{0x160,geometry->stride},
            {0x164,geometry->stride},{0x16c,geometry->inputBytes},
            {0x180,rowBytes},{0x184,planeBytes},
        };
        for (const auto &field : fields)
            if (![writer writeUInt32:field.value atOffset:field.offset
                               field:@"regular_conv.geometry" error:error]) return nil;
    }
    return writer.data;
}
@end
