#import "H16GConvEncoder.h"
#import "H16GTDWriter.h"

static BOOL put(H16GTDWriter *writer, NSUInteger offset, uint32_t value,
                NSString *field, NSError **error) {
    return [writer writeUInt32:value atOffset:offset field:field error:error];
}

typedef struct {
    NSUInteger inputChannels;
    NSUInteger outputChannels;
    NSUInteger spatial;
    uint32_t descriptorCount;
    uint32_t bankRows;
    uint32_t tileMode;
    uint32_t tileGroups;
    uint32_t window;
    uint32_t strideA;
    uint32_t strideB;
    uint32_t inputBytes;
} H16GConv1x1Geometry;

static const H16GConv1x1Geometry kConv1x1Geometry[] = {
    {128,256,32,0x0c,0x20,0x220000,0x4,0x2050,0x2000,0x2000,0x40a00},
    {160,160,64,0x27,0x40,0x210000,0x4,0x5000,0x5000,0x5000,0x140000},
    {192,192,64,0x2f,0x40,0x210000,0x4,0x6000,0x6000,0x6000,0x180000},
    {256,256,128,0xf7,0x01,0x200000,0x4,0x10000,0x10000,0x10000,0x10000},
    {256,256,32,0x11,0x20,0x220000,0x4,0x4050,0x4000,0x4000,0x80a00},
    {256,256,64,0x3f,0x02,0x210000,0x4,0x8000,0x8000,0x8000,0x10000},
    {320,320,64,0x4f,0x02,0x210000,0x4,0xa000,0xa000,0xa000,0x14000},
    {32,32,64,0x07,0x04,0x1a0000,0xa,0x1050,0x1000,0x1000,0x4140},
    {384,384,64,0x60,0x02,0x210000,0x4,0xc000,0xc000,0xc000,0x18000},
    {448,448,64,0x71,0x22,0x210000,0x4,0xe000,0xe000,0xe000,0x1dc000},
    {512,512,128,0x1f2,0x01,0x200000,0x4,0x20000,0x20000,0x20000,0x20000},
    {512,512,32,0x26,0x20,0x222200,0x4,0x8050,0x8000,0x8000,0x100a00},
    {512,512,64,0x82,0x02,0x210000,0x4,0x10000,0x10000,0x10000,0x20000},
    {64,64,128,0x3d,0x80,0x211100,0x2,0x2000,0x2000,0x2000,0x100000},
    {64,64,32,0x03,0x20,0x220000,0x2,0x1050,0x1000,0x1000,0x20a00},
    {64,64,64,0x0f,0x40,0x210000,0x2,0x2000,0x2000,0x2000,0x80000},
};

static const H16GConv1x1Geometry *lookupGeometry(
    NSUInteger inputChannels, NSUInteger outputChannels, NSUInteger spatial) {
    for (const H16GConv1x1Geometry &row : kConv1x1Geometry)
        if (row.inputChannels == inputChannels &&
            row.outputChannels == outputChannels && row.spatial == spatial)
            return &row;
    return nullptr;
}

static BOOL writeControlPrologue(H16GTDWriter *w, NSError **error) {
    if (!put(w,0x000,0x00000001,@"td.version",error) ||
        !put(w,0x010,0x00720000,@"conv.op_class",error) ||
        !put(w,0x018,0x04000068,@"conv.dispatch",error) ||
        !put(w,0x020,0x00fff868,@"conv.dispatch_limit",error) ||
        !put(w,0x030,0x00050009,@"conv.control",error) ||
        !put(w,0x034,0xffc01540,@"conv.control_mask",error) ||
        !put(w,0x038,0x000102c0,@"conv.control_shape",error)) return NO;
    for (NSUInteger off=0x03c; off<=0x05c; off+=4)
        if (!put(w,off,0x21,@"conv.prologue_chain",error)) return NO;
    if (!put(w,0x060,0x00031551,@"conv.prologue_barrier",error)) return NO;
    for (NSUInteger off=0x064; off<=0x07c; off+=4)
        if (!put(w,off,0x21,@"conv.prologue_chain",error)) return NO;
    return put(w,0x080,0x000f1559,@"weight.tap_header",error);
}

static BOOL writeCommandTail(H16GTDWriter *w, NSError **error) {
    const NSUInteger offsets[] = {
        0x12c,0x130,0x134,0x140,0x144,0x148,0x14c,0x150,0x154,0x168,
        0x17c,0x188,0x18c,0x190,0x194,0x19c,0x1a0,0x1a4,0x1b0,0x1bc,
        0x1c8,0x1cc,0x1d0,0x1d4
    };
    const uint32_t values[] = {
        0x00100000,0x800c1342,0x000000ce,0x0000135a,0x01002031,
        0x00009041,0x00500172,0x00500130,0x98039045,0x0050017a,
        0x81009444,0x01302031,0x83109640,0x00000012,0x00a000a0,
        0x007f0000,0x20010701,0x24009544,0x23009344,0x23809442,
        0x22001340,0x00000021,0x22001440,0x01000031
    };
    for (NSUInteger i=0; i<sizeof(offsets)/sizeof(offsets[0]); ++i)
        if (!put(w,offsets[i],values[i],@"conv.command",error)) return NO;
    return YES;
}

@implementation H16GConvEncoder
+ (BOOL)supportsConv1x1WithInputChannels:(NSUInteger)inputChannels
                          outputChannels:(NSUInteger)outputChannels
                                  spatial:(NSUInteger)spatial {
    return lookupGeometry(inputChannels, outputChannels, spatial) != nullptr;
}

+ (NSData *)encodeConv1x1WithInputChannels:(NSUInteger)inputChannels
                            outputChannels:(NSUInteger)outputChannels
                                    spatial:(NSUInteger)spatial
                             bytesPerWeight:(NSUInteger)bytesPerWeight
                                numericMode:(ANELegalNumericMode)numericMode
                               reluEpilogue:(BOOL)reluEpilogue
                                       error:(NSError **)error {
    const H16GConv1x1Geometry *geometry =
        lookupGeometry(inputChannels, outputChannels, spatial);
    if (!geometry || (bytesPerWeight != 1 && bytesPerWeight != 2)) {
        if (error) *error = [NSError errorWithDomain:H16GTDWriterErrorDomain code:2
            userInfo:@{NSLocalizedDescriptionKey:
                @"Conv1x1 geometry has no decoded H16G SRAM schedule"}];
        return nil;
    }
    H16GTDWriter *w = [[H16GTDWriter alloc] initWithByteLength:0x1d8];
    if (!writeControlPrologue(w,error)) return nil;
    uint32_t step = (uint32_t)(inputChannels * outputChannels / 8);
    for (NSUInteger i=0; i<31; ++i) {
        uint32_t value = i < 15 ? (uint32_t)(i + 1) * step : step;
        if (!put(w,0x084+i*4,value,@"weight.tap",error)) return nil;
    }
    uint32_t modeWord = numericMode == ANELegalNumericModeW8A8Packed
        ? 0xb3418005u : 0x93418005u;
    uint32_t dmaWord = numericMode == ANELegalNumericModeFP16
        ? 0x80021241u : (numericMode == ANELegalNumericModeW8A8InputBoundary
            ? 0x80041240u : 0x80049240u);
    uint32_t dtype = bytesPerWeight == 1 ? 0x00000081u :
        (reluEpilogue ? 0x00113c00u : 0x00103c00u);
    NSUInteger planeBytes = spatial * spatial * 2;
    if (!put(w,0x014,geometry->descriptorCount,@"output.descriptor_count",error) ||
        !put(w,0x100,0x00010001,@"conv.geometry_header",error) ||
        !put(w,0x104,(uint32_t)spatial,@"input.width",error) ||
        !put(w,0x108,(uint32_t)spatial,@"input.height",error) ||
        !put(w,0x10c,(uint32_t)inputChannels,@"input.channels",error) ||
        !put(w,0x110,modeWord,@"conv.numeric_mode",error) ||
        !put(w,0x114,(uint32_t)spatial,@"output.width",error) ||
        !put(w,0x118,(uint32_t)spatial,@"output.height",error) ||
        !put(w,0x11c,(uint32_t)outputChannels,@"output.channels",error) ||
        !put(w,0x120,geometry->bankRows,@"conv.bank_rows",error) ||
        !put(w,0x124,geometry->tileMode,@"conv.tile_mode",error) ||
        !put(w,0x128,geometry->tileGroups,@"conv.tile_groups",error) ||
        !put(w,0x138,(uint32_t)(2*spatial),@"input.row_bytes",error) ||
        !put(w,0x13c,(uint32_t)planeBytes,@"input.plane_bytes",error) ||
        !put(w,0x158,(uint32_t)(2*spatial),@"output.row_bytes",error) ||
        !put(w,0x15c,geometry->window,@"sram.window",error) ||
        !put(w,0x160,geometry->strideA,@"sram.row_stride_a",error) ||
        !put(w,0x164,geometry->strideB,@"sram.row_stride_b",error) ||
        !put(w,0x16c,geometry->inputBytes,@"sram.input_bytes",error) ||
        !put(w,0x170,dmaWord,@"dma.configuration",error) ||
        !put(w,0x174,dtype,@"conv.dtype_epilogue",error) ||
        !put(w,0x178,bytesPerWeight==1?0x00003000:0x00003c00,@"conv.scale",error) ||
        !put(w,0x180,(uint32_t)(2*spatial),@"store.row_bytes",error) ||
        !put(w,0x184,(uint32_t)planeBytes,@"store.plane_bytes",error) ||
        !writeCommandTail(w,error)) return nil;
    return w.data;
}
@end
