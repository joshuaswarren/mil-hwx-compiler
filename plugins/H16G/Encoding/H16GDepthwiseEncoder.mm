#import "H16GDepthwiseEncoder.h"

#import "H16GTDWriter.h"

typedef struct {
    NSUInteger channels;
    uint32_t descriptorCount;
    uint32_t tileWord;
    uint32_t workingSetBytes;
} H16GDepthwiseGeometry;

static const H16GDepthwiseGeometry kDepthwiseGeometry[] = {
    {64, 15, 0x40, 0x80000},
    {128, 30, 0x40, 0x100000},
    {256, 62, 0x3e, 0x1f8000},
    {512, 126, 0x18, 0x1a0000},
};

static const H16GDepthwiseGeometry *lookupGeometry(NSUInteger channels,
                                                    NSUInteger spatial) {
    if (spatial != 64) return nullptr;
    for (const H16GDepthwiseGeometry &row : kDepthwiseGeometry)
        if (row.channels == channels) return &row;
    return nullptr;
}

static const uint32_t kDepthwiseBase[120] = {
    0x00000001,0,0,0,0x00740000,0x0000000f,0x04000068,0,
    0x00fff868,0,0,0,0x00050009,0xffc01540,0x000102c0,0x00000021,
    0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,
    0x00031551,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,0x00000021,
    0x000f1559,0x00000080,0x00000100,0x00000180,0x00000200,0x00000280,0x00000300,0x00000380,
    0x00000400,0x00000480,0x00000500,0x00000580,0x00000600,0x00000680,0x00000700,0x00000780,
    0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,
    0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,0x00000080,
    0x00010001,0x00000040,0x00000040,0x00000040,0x93698005,0x00000040,0x00000040,0x00000040,
    0x5042a0c3,0x00014000,0x00000040,0x00210000,0,0x00100000,0x800c1342,0x000000ce,
    0x00000080,0x00002000,0x0000135a,0x01002031,0x00009041,0x00500172,0x00500130,0x98039045,
    0x00000080,0x00002000,0x00002000,0x00002000,0x0050017a,0x00080000,0x80021241,0x00103c00,
    0x00003c00,0x81009444,0x00000080,0x00002000,0x01302031,0x83109640,0x00000012,0x00a000a0,
    0,0x007f0000,0x20010701,0x24009544,0,0,0x23009344,0,
    0,0x23809442,0,0,0x22001340,0x00000021,0x22001440,0x01000031,
};

@implementation H16GDepthwiseEncoder
+ (BOOL)supportsChannels:(NSUInteger)channels spatial:(NSUInteger)spatial {
    return lookupGeometry(channels, spatial) != nullptr;
}

+ (NSData *)encode3x3WithChannels:(NSUInteger)channels
                           spatial:(NSUInteger)spatial
                             error:(NSError **)error {
    const H16GDepthwiseGeometry *geometry = lookupGeometry(channels, spatial);
    if (!geometry) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain
            code:6 userInfo:@{NSLocalizedDescriptionKey:
                @"depthwise 3x3 has no decoded H16G schedule outside C64/C128/C256/C512 at S64"}];
        return nil;
    }
    H16GTDWriter *writer = [[H16GTDWriter alloc] initWithByteLength:0x1e0];
    for (NSUInteger index = 0; index < 120; ++index)
        if (![writer writeUInt32:kDepthwiseBase[index] atOffset:index * 4
                           field:@"depthwise.invariant" error:error]) return nil;
    uint32_t step = (uint32_t)channels + 64;
    for (NSUInteger index = 0; index < 31; ++index) {
        uint32_t value = index < 15 ? (uint32_t)(index + 1) * step : step;
        if (![writer writeUInt32:value atOffset:0x84 + index * 4
                           field:@"depthwise.weight_tap" error:error]) return nil;
    }
    uint32_t channelRowBytes = (uint32_t)(channels * spatial * 2);
    const struct { NSUInteger offset; uint32_t value; NSString *field; } fields[] = {
        {0x014,geometry->descriptorCount,@"depthwise.descriptor_count"},
        {0x10c,(uint32_t)channels,@"depthwise.input_channels"},
        {0x11c,(uint32_t)channels,@"depthwise.output_channels"},
        {0x128,geometry->tileWord,@"depthwise.tile_word"},
        {0x164,channelRowBytes,@"depthwise.row_stride_a"},
        {0x168,channelRowBytes,@"depthwise.row_stride_b"},
        {0x16c,channelRowBytes,@"depthwise.row_stride_c"},
        {0x174,geometry->workingSetBytes,@"depthwise.working_set"},
    };
    for (const auto &field : fields)
        if (![writer writeUInt32:field.value atOffset:field.offset
                           field:field.field error:error]) return nil;
    return writer.data;
}
@end
