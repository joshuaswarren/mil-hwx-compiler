#import "H16GMatmulEncoder.h"

#include <initializer_list>

static NSString *const H16GMatmulEncoderErrorDomain = @"ANE.H16G.MatmulEncoder";

@implementation H16GMatmulPostOperationEncoding
- (instancetype)initWithTDProgram:(H16GEncodedTDProgram *)tdProgram
                     kernelRegion:(NSData *)kernelRegion {
    self = [super init];
    if (self) {
        _tdProgram = tdProgram;
        _kernelRegion = [kernelRegion copy];
    }
    return self;
}
@end

// These are decoded invariant packet words, not an HWX container or binary
// template. Shape-, tile-, stride-, and address-bearing fields are written by
// the functions below before a task stream is returned.
static const uint32_t kSingleFamilyWords[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x00310000, 0x0000000f,
    0x00000068, 0x00000000, 0x00fff868, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00010001, 0x00000200, 0x00000001, 0x00000200, 0x95418005,
    0x00000200, 0x00000001, 0x00000200, 0x00000001, 0x00000020, 0x00000009,
    0x00000000, 0x80081342, 0x000000ce, 0x00000400, 0x0000135a, 0x01002031,
    0x00009041, 0x00500172, 0x00500130, 0x98039045, 0x00000400, 0x00080000,
    0x00080000, 0x00080000, 0x0050087a, 0x00080000, 0x80801445, 0x00002000,
    0x08302031, 0x23009344, 0x00000000, 0x00000000, 0x21809442, 0x00000000,
    0x00000000, 0x22001340, 0x00000021, 0x22001440, 0x010000f1, 0x00000000,
    0x00000000, 0x00000000, 0x00810001, 0x00000017, 0x04000000, 0x00000000,
    0x00fff860, 0x00000000, 0x00000000, 0x00000000, 0x0005000b, 0x00000000,
    0x80201540, 0x000202c0, 0x00020000, 0x000f1559, 0x00002000, 0x00004000,
    0x00006000, 0x00008000, 0x0000a000, 0x0000c000, 0x0000e000, 0x00010000,
    0x00012000, 0x00014000, 0x00016000, 0x00018000, 0x0001a000, 0x0001c000,
    0x0001e000, 0x00002000, 0x00002000, 0x00002000, 0x00002000, 0x00002000,
    0x00002000, 0x00002000, 0x00002000, 0x00002000, 0x00002000, 0x00002000,
    0x00002000, 0x00002000, 0x00002000, 0x00002000, 0x00002000, 0x00010001,
    0x00000200, 0x00000001, 0x00000200, 0x91418005, 0x00000200, 0x00000001,
    0x00000200, 0x00000001, 0x20200000, 0x00100000, 0x80081342, 0x000000ce,
    0x00000400, 0x0000135a, 0x01002031, 0x803c1041, 0x00100142, 0x00000100,
    0x00020000, 0x00020000, 0x00020000, 0x00009052, 0x0000014a, 0x00020000,
    0x80049240, 0x00010082, 0x00103c00, 0x00003c00, 0x80801445, 0x00000400,
    0x01302031, 0x83119640, 0x00000012, 0x00a000a0, 0x000100e0, 0x00000000,
    0x007f0000, 0x20010701, 0x21809544, 0x00000000, 0x00000000, 0x23809344,
    0x00000000, 0x00000000, 0x24009442, 0x00000000, 0x00000000, 0x22001340,
    0x00010021, 0x22001440, 0x01010031, 0x22001548, 0x00010041, 0x22001549,
    0x00010041, 0x2200154a, 0x00010041, 0x2200154b, 0x00010041, 0x2200154c,
    0x00010041, 0x2200154d, 0x00010041, 0x2200154e, 0x00010041, 0x2200154f,
    0x00010041, 0x22001550, 0x00010041, 0x22001551, 0x00010041, 0x22001552,
    0x00010041, 0x22001553, 0x00010041, 0x22001554, 0x00010041, 0x22001555,
    0x00010041, 0x22001556, 0x00010041, 0x22001557, 0x00010041,
};
// 185 words / 0x2e4 bytes

static const uint32_t kSingle256Words[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x002f0000, 0x00000003,
    0x00000068, 0x00000000, 0x00fff868, 0x00000000, 0x00000000, 0x00000000,
    0x00000001, 0x00008002, 0x00000001, 0x00000100, 0x8aa08006, 0x00000001,
    0x00000100, 0x00000001, 0x00000020, 0x00000008, 0x00000000, 0x80081342,
    0x000000ce, 0x00000200, 0x0000135a, 0x01002031, 0x00009041, 0x00500172,
    0x00500130, 0x98039045, 0x00000200, 0x00020000, 0x00020000, 0x00020000,
    0x0050087a, 0x00020000, 0x80801445, 0x00001000, 0x08302031, 0x23009344,
    0x00000000, 0x00000000, 0x21809442, 0x00000000, 0x00000000, 0x22001340,
    0x00000021, 0x22001440, 0x010000f1, 0x00000000, 0x007f0001, 0x00000005,
    0x04000000, 0x00000000, 0x00fff860, 0x00000000, 0x00000000, 0x00000000,
    0x0005000b, 0x00000000, 0x80201540, 0x000202c0, 0x00010000, 0x000f1559,
    0x00001000, 0x00002000, 0x00003000, 0x00004000, 0x00005000, 0x00006000,
    0x00007000, 0x00008000, 0x00009000, 0x0000a000, 0x0000b000, 0x0000c000,
    0x0000d000, 0x0000e000, 0x0000f000, 0x00001000, 0x00001000, 0x00001000,
    0x00001000, 0x00001000, 0x00001000, 0x00001000, 0x00001000, 0x00001000,
    0x00001000, 0x00001000, 0x00001000, 0x00001000, 0x00001000, 0x00001000,
    0x00001000, 0x00008002, 0x00000001, 0x00000100, 0x88a08006, 0x00000001,
    0x00000100, 0x00000001, 0x20200000, 0x00100000, 0x80081342, 0x000000ce,
    0x00000200, 0x0000135a, 0x01002031, 0x803c1041, 0x00100142, 0x00000100,
    0x00010000, 0x00010000, 0x00010000, 0x00009052, 0x0000014a, 0x00010000,
    0x80049240, 0x00010082, 0x00103c00, 0x00003c00, 0x80801445, 0x00000200,
    0x01302031, 0x83119640, 0x00000012, 0x00a000a0, 0x000100e0, 0x00000000,
    0x007f0000, 0x20010701, 0x21809544, 0x00000000, 0x00000000, 0x23809344,
    0x00000000, 0x00000000, 0x24009442, 0x00000000, 0x00000000, 0x22001340,
    0x00010021, 0x22001440, 0x01010031, 0x22001548, 0x00010041, 0x22001549,
    0x00010041, 0x2200154a, 0x00010041, 0x2200154b, 0x00010041, 0x2200154c,
    0x00010041, 0x2200154d, 0x00010041, 0x2200154e, 0x00010041, 0x2200154f,
    0x00010041, 0x22001550, 0x00010041, 0x22001551, 0x00010041, 0x22001552,
    0x00010041, 0x22001553, 0x00010041, 0x22001554, 0x00010041, 0x22001555,
    0x00010041, 0x22001556, 0x00010041, 0x22001557, 0x00010041,
};
// 179 words / 0x2cc bytes

static const uint32_t kTiledHeader[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000,
};
// 4 words / 0x10 bytes
static const uint32_t kTiledPrologue[] = {
    0x00350000, 0x000000f5, 0x00000068, 0x00000000, 0x00000068, 0x00000000,
    0x00000000, 0x00000000, 0x00050001, 0x00010001, 0x00000800, 0x00000001,
    0x00000800, 0x93618005, 0x00000800, 0x00000001, 0x00000800, 0x00014000,
    0x00000001, 0x00200000, 0x00000000, 0x00100000, 0x80081342, 0x000000ce,
    0x00001000, 0x0000135a, 0x01002031, 0x00009041, 0x00500172, 0x00500130,
    0x98039045, 0x00000100, 0x00080000, 0x00080000, 0x00080000, 0x0050087a,
    0x00080000, 0x80021241, 0x00103c0c, 0x00003c00, 0x80801445, 0x00008000,
    0x08302031, 0x23009344, 0x00000000, 0x00000000, 0x21809442, 0x00000000,
    0x00000000, 0x22001340, 0x00000021, 0x22001440, 0x010000f1,
};
// 53 words / 0xd4 bytes
static const uint32_t kTiledFirst[] = {
    0x00000000, 0x00000000, 0x00000000, 0x007a0001, 0x00000099, 0x00000000,
    0x00000000, 0x00fff820, 0x00000000, 0x00000000, 0x00000000, 0x00050003,
    0x00000000, 0x80201540, 0x00020240, 0x00080000, 0x000f1559, 0x00008000,
    0x00010000, 0x00018000, 0x00020000, 0x00028000, 0x00030000, 0x00038000,
    0x00040000, 0x00048000, 0x00050000, 0x00058000, 0x00060000, 0x00068000,
    0x00070000, 0x00078000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00010001, 0x00000800, 0x00000001, 0x00000800, 0x91418005, 0x00000800,
    0x00000001, 0x00000100, 0x00000001, 0x20200000, 0x00100000, 0x80081342,
    0x000000ee, 0x00001000, 0x0000135a, 0x01002031, 0x803c1041, 0x00000142,
    0x00000100, 0x00080000, 0x00080000, 0x00080000, 0x00009052, 0x0000014a,
    0x00080000, 0x80049240, 0x00010082, 0x00103c00, 0x00003c00, 0x80801445,
    0x00001000, 0x01302031, 0x21809544, 0x00700000, 0x00000000, 0x23809344,
    0x00000000, 0x00000000, 0x24009442, 0x00700000, 0x00000000, 0x22001340,
    0x000100e1, 0x22001440, 0x01010031,
};
// 93 words / 0x174 bytes
static const uint32_t kTiledMiddle[] = {
    0x22001548, 0x000100c1, 0x22001549, 0x000100c1, 0x2200154a, 0x000100c1,
    0x2200154b, 0x000100c1, 0x2200154c, 0x000100c1, 0x2200154d, 0x000100c1,
    0x2200154e, 0x000100c1, 0x2200154f, 0x000100c1, 0x22001550, 0x000100c1,
    0x22001551, 0x000100c1, 0x22001552, 0x000100c1, 0x22001553, 0x000100c1,
    0x22001554, 0x000100c1, 0x22001555, 0x000100c1, 0x22001556, 0x000100c1,
    0x22001557, 0x000100c1, 0x00000000, 0x00000000, 0x007a0002, 0x00000099,
    0x00000000, 0x00000000, 0x00fff820, 0x00000000, 0x00000000, 0x00000000,
    0x00050003, 0x00000000, 0x80201540, 0x00030240, 0x00080000, 0x000f1559,
    0x00008000, 0x00010000, 0x00018000, 0x00020000, 0x00028000, 0x00030000,
    0x00038000, 0x00040000, 0x00048000, 0x00050000, 0x00058000, 0x00060000,
    0x00068000, 0x00070000, 0x00078000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00010001, 0x00000800, 0x00000001, 0x00000800, 0x91418005,
    0x00000800, 0x00000001, 0x00000100, 0x00000001, 0x20200000, 0x00100000,
    0x80081342, 0x000000ee, 0x00001000, 0x0000135a, 0x01002031, 0x803c1041,
    0x00000142, 0x00000100, 0x00080000, 0x00080000, 0x00080000, 0x00009052,
    0x0000014a, 0x00080000, 0x80049240, 0x00010082, 0x00103c00, 0x00003c00,
    0x80801445, 0x00001000, 0x01302031, 0x21809544, 0x00600000, 0x00000000,
    0x23809344, 0x00000000, 0x00000000, 0x24009442, 0x00600000, 0x00000000,
    0x22001340, 0x000200e1, 0x22001440, 0x01020031,
};
// 124 words / 0x1f0 bytes
static const uint32_t kTiledLast[] = {
    0x22001548, 0x000700c1, 0x22001549, 0x000700c1, 0x2200154a, 0x000700c1,
    0x2200154b, 0x000700c1, 0x2200154c, 0x000700c1, 0x2200154d, 0x000700c1,
    0x2200154e, 0x000700c1, 0x2200154f, 0x000700c1, 0x22001550, 0x000700c1,
    0x22001551, 0x000700c1, 0x22001552, 0x000700c1, 0x22001553, 0x000700c1,
    0x22001554, 0x000700c1, 0x22001555, 0x000700c1, 0x22001556, 0x000700c1,
    0x22001557, 0x000700c1, 0x00000000, 0x00000000, 0x00810008, 0x00000099,
    0x04000000, 0x00000000, 0x00fff860, 0x00000000, 0x00000000, 0x00000000,
    0x0005000b, 0x00000000, 0x80201540, 0x000902c0, 0x00080000, 0x000f1559,
    0x00008000, 0x00010000, 0x00018000, 0x00020000, 0x00028000, 0x00030000,
    0x00038000, 0x00040000, 0x00048000, 0x00050000, 0x00058000, 0x00060000,
    0x00068000, 0x00070000, 0x00078000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000, 0x00008000,
    0x00008000, 0x00010001, 0x00000800, 0x00000001, 0x00000800, 0x91418005,
    0x00000800, 0x00000001, 0x00000100, 0x00000001, 0x20200000, 0x00100000,
    0x80081342, 0x000000ce, 0x00001000, 0x0000135a, 0x01002031, 0x803c1041,
    0x00000142, 0x00000100, 0x00080000, 0x00080000, 0x00080000, 0x00009052,
    0x0000014a, 0x00080000, 0x80049240, 0x00010082, 0x00103c00, 0x00003c00,
    0x80801445, 0x00001000, 0x01302031, 0x83119640, 0x00000012, 0x00a000a0,
    0x000800e0, 0x00000000, 0x007f0000, 0x20010701, 0x21809544, 0x00000000,
    0x00000000, 0x23809344, 0x00000000, 0x00000000, 0x24009442, 0x00000000,
    0x00000000, 0x22001340, 0x000800c1, 0x22001440, 0x01080031,
};
// 131 words / 0x20c bytes
// decoded block count in reference: 9

static void setWord(NSMutableData *data, NSUInteger index, uint32_t value) {
    ((uint32_t *)data.mutableBytes)[index] = value;
}

typedef struct {
    __unsafe_unretained NSString *operationName;
    NSUInteger size;
    NSUInteger plainWordCount;
    NSUInteger blockHeaderWord;
    uint32_t expectedBlockHeader;
    NSUInteger taskControlWord;
    NSUInteger geometryInsertionWord;
    NSUInteger completionInsertionWord;
    NSUInteger outputModeWord;
    NSUInteger launchInsertionWord;
} H16GMatmulPostOperationLayout;

static const H16GMatmulPostOperationLayout kPostOperationLayouts[] = {
    {@"gelu", 128, 185, 56, 0x00810001, 66, 68, 101, 128, 140},
    {@"gelu", 256, 179, 52, 0x007f0001, 62, 64, 97, 122, 134},
};

static const H16GMatmulPostOperationLayout *postOperationLayout(
    NSUInteger size, NSString *operationName) {
    for (const H16GMatmulPostOperationLayout &row : kPostOperationLayouts)
        if (row.size == size &&
            [row.operationName isEqualToString:operationName]) return &row;
    return nullptr;
}

static BOOL insertWords(NSMutableData *data, NSUInteger wordIndex,
                        const uint32_t *words, NSUInteger count) {
    if (wordIndex > data.length / sizeof(uint32_t)) return NO;
    [data replaceBytesInRange:NSMakeRange(wordIndex * sizeof(uint32_t), 0)
                    withBytes:words length:count * sizeof(uint32_t)];
    return YES;
}

static NSData *unpaddedKernelRegion(NSData *kernelRegion) {
    const uint8_t *bytes = (const uint8_t *)kernelRegion.bytes;
    NSUInteger length = kernelRegion.length;
    while (length > 0 && bytes[length - 1] == 0) --length;
    length = MIN(kernelRegion.length, (length + 1) & ~(NSUInteger)1);
    return [kernelRegion subdataWithRange:NSMakeRange(0, length)];
}

static void appendWords(NSMutableData *data, const uint32_t *words,
                        NSUInteger count) {
    [data appendBytes:words length:count * sizeof(uint32_t)];
}

static NSUInteger largestPowerOfTwoAtMost(NSUInteger value) {
    NSUInteger result = 1;
    while (result <= value / 2) result *= 2;
    return result;
}

static void schedule(NSUInteger size, NSUInteger *tileCount,
                     NSUInteger *rowsPerTile) {
    const NSUInteger outputBudget = 1 << 20;
    if (size * size * 2 <= outputBudget) {
        *tileCount = 1;
        *rowsPerTile = size;
        return;
    }
    NSUInteger rowLimit = MIN((NSUInteger)256,
        outputBudget / (size * 2));
    NSUInteger rows = largestPowerOfTwoAtMost(rowLimit);
    *rowsPerTile = rows;
    *tileCount = (size + rows - 1) / rows;
}

static NSMutableData *encodeSingleFamily(NSUInteger size) {
    NSMutableData *data = [NSMutableData dataWithBytes:kSingleFamilyWords
        length:sizeof(kSingleFamilyWords)];
    uint32_t tensorBytes = (uint32_t)(size * size * 2);
    uint32_t stride = (uint32_t)(size * 2);
    uint32_t size16 = (uint32_t)(size * 16);
    uint32_t size256 = (uint32_t)(size * 256);
    uint32_t log2Size = 0;
    for (NSUInteger value = size; value > 1; value >>= 1) ++log2Size;
    const NSUInteger tensorByteWords[] = {35,36,37,39};
    for (NSUInteger index : tensorByteWords) setWord(data,index,tensorBytes);
    const NSUInteger dimensionWords[] = {14,16,18,20,102,104,106,108};
    for (NSUInteger index : dimensionWords) setWord(data,index,(uint32_t)size);
    const NSUInteger strideWords[] = {27,34,114,131};
    for (NSUInteger index : strideWords) setWord(data,index,stride);
    const NSUInteger size256Words[] = {68,120,121,122,125};
    for (NSUInteger index : size256Words) setWord(data,index,size256);
    setWord(data,41,size16);
    for (NSUInteger lane = 0; lane < 15; ++lane)
        setWord(data,70 + lane,size16 * (uint32_t)(lane + 1));
    for (NSUInteger index = 85; index <= 100; ++index)
        setWord(data,index,size16);
    setWord(data,23,log2Size);
    if (size == 128) {
        setWord(data,5,0x00);
        setWord(data,57,0x01);
        setWord(data,124,0x0040014a);
    } else {
        setWord(data,5,0x0f);
        setWord(data,57,0x17);
        setWord(data,124,0x0000014a);
    }
    return data;
}

static uint32_t rowsLow(NSUInteger rows) { return rows == 256 ? 0xe1 : 0x21; }
static uint32_t rowControl(NSUInteger rows) { return rows == 256 ? 0xee : 0xce; }

static NSMutableData *patchedBlock(const uint32_t *words, NSUInteger count) {
    return [NSMutableData dataWithBytes:words length:count * sizeof(uint32_t)];
}

static NSData *encodeTiled(NSUInteger size, NSUInteger tiles,
                           NSUInteger rows) {
    NSMutableData *result = [NSMutableData data];
    appendWords(result,kTiledHeader,sizeof(kTiledHeader)/sizeof(uint32_t));

    NSMutableData *prologue = patchedBlock(kTiledPrologue,
        sizeof(kTiledPrologue)/sizeof(uint32_t));
    setWord(prologue,1,0);
    for (NSUInteger index : {10u,12u,14u,16u})
        setWord(prologue,index,(uint32_t)size);
    setWord(prologue,24,(uint32_t)(2*size));
    for (NSUInteger index : {32u,33u,34u,36u})
        setWord(prologue,index,(uint32_t)(size*256));
    setWord(prologue,41,(uint32_t)(size*16));
    [result appendData:prologue];

    uint32_t slab = (uint32_t)(rows * size * 2);
    NSMutableData *first = patchedBlock(kTiledFirst,
        sizeof(kTiledFirst)/sizeof(uint32_t));
    setWord(first,4,0);
    setWord(first,15,(uint32_t)(size*256));
    for (NSUInteger lane=0;lane<16;++lane)
        setWord(first,17+lane,(uint32_t)(size*16*(lane+1)));
    for (NSUInteger index=32;index<48;++index)
        setWord(first,index,(uint32_t)(size*16));
    for (NSUInteger index : {49u,51u,53u})
        setWord(first,index,(uint32_t)size);
    setWord(first,55,(uint32_t)(size-(tiles-1)*rows));
    setWord(first,60,rowControl(rows));
    setWord(first,61,(uint32_t)(2*size));
    for (NSUInteger index : {67u,68u,69u,72u})
        setWord(first,index,(uint32_t)(size*256));
    setWord(first,78,(uint32_t)(2*size));
    setWord(first,81,(uint32_t)(tiles-1)*slab);
    setWord(first,87,(uint32_t)(tiles-1)*slab);
    setWord(first,90,0x10000|rowsLow(rows));
    [result appendData:first];

    for (NSUInteger tile=1;tile<=tiles-2;++tile) {
        NSMutableData *middle = patchedBlock(kTiledMiddle,
            sizeof(kTiledMiddle)/sizeof(uint32_t));
        for (NSUInteger lane=0;lane<16;++lane)
            setWord(middle,1+2*lane,(uint32_t)(0xc1+tile*0x10000));
        setWord(middle,34,(uint32_t)(0x7a0001+tile));
        setWord(middle,35,0);
        setWord(middle,45,(uint32_t)(0x20240+tile*0x10000));
        setWord(middle,46,(uint32_t)(size*256));
        for (NSUInteger lane=0;lane<15;++lane)
            setWord(middle,48+lane,(uint32_t)(size*16*(lane+1)));
        for (NSUInteger index=63;index<79;++index)
            setWord(middle,index,(uint32_t)(size*16));
        for (NSUInteger index : {80u,82u,84u})
            setWord(middle,index,(uint32_t)size);
        setWord(middle,86,(uint32_t)rows);
        setWord(middle,91,rowControl(rows));
        setWord(middle,92,(uint32_t)(2*size));
        for (NSUInteger index : {98u,99u,100u,103u})
            setWord(middle,index,(uint32_t)(size*256));
        setWord(middle,109,(uint32_t)(2*size));
        setWord(middle,112,(uint32_t)(tiles-1-tile)*slab);
        setWord(middle,118,(uint32_t)(tiles-1-tile)*slab);
        setWord(middle,121,(uint32_t)((tile+1)*0x10000)|rowsLow(rows));
        setWord(middle,123,(uint32_t)(0x01010031+tile*0x10000));
        [result appendData:middle];
    }

    NSMutableData *last = patchedBlock(kTiledLast,
        sizeof(kTiledLast)/sizeof(uint32_t));
    for (NSUInteger lane=0;lane<16;++lane)
        setWord(last,1+2*lane,(uint32_t)((tiles-1)*0x10000)|0xc1);
    setWord(last,34,(uint32_t)(0x810000|tiles));
    setWord(last,35,0);
    setWord(last,45,(uint32_t)((tiles+1)*0x10000)|0x2c0);
    setWord(last,46,(uint32_t)(size*256));
    for (NSUInteger lane=0;lane<15;++lane)
        setWord(last,48+lane,(uint32_t)(size*16*(lane+1)));
    for (NSUInteger index=63;index<79;++index)
        setWord(last,index,(uint32_t)(size*16));
    for (NSUInteger index : {80u,82u,84u})
        setWord(last,index,(uint32_t)size);
    setWord(last,86,(uint32_t)rows);
    setWord(last,92,(uint32_t)(2*size));
    for (NSUInteger index : {98u,99u,100u,103u})
        setWord(last,index,(uint32_t)(size*256));
    setWord(last,109,(uint32_t)(2*size));
    setWord(last,114,(uint32_t)(tiles*0x10000)|0xe0);
    setWord(last,128,(uint32_t)(tiles*0x10000)|0xc1);
    setWord(last,130,(uint32_t)(0x01000031|tiles*0x10000));
    [result appendData:last];
    for (NSUInteger lane=0;lane<16;++lane) {
        uint32_t pair[] = {(uint32_t)(0x22001548+lane),
                           (uint32_t)(tiles*0x10000)|0x41};
        appendWords(result,pair,2);
    }
    return result;
}

@implementation H16GMatmulEncoder
+ (BOOL)supportsSquareSize:(NSUInteger)size {
    if (size == 128 || size == 256 || size == 512) return YES;
    return size >= 768 && size <= 4096 && size % 128 == 0;
}
+ (NSUInteger)tileCountForSquareSize:(NSUInteger)size {
    if (![self supportsSquareSize:size]) return 0;
    NSUInteger tiles=0,rows=0; schedule(size,&tiles,&rows); return tiles;
}
+ (NSUInteger)rowsPerTileForSquareSize:(NSUInteger)size {
    if (![self supportsSquareSize:size]) return 0;
    NSUInteger tiles=0,rows=0; schedule(size,&tiles,&rows); return rows;
}
+ (NSUInteger)outputScaleWordIndexForSquareSize:(NSUInteger)size {
    // Decoded from Apple matmul-then-scalar-mul oracles: the only word that
    // changes is the fp16 output scale (1.0 = 0x3c00 in the plain program).
    if (size == 128) return 129;
    if (size == 256) return 123;
    return NSNotFound;
}
+ (H16GEncodedTDProgram *)encodeSquareSize:(NSUInteger)size
                                      error:(NSError **)error {
    if (![self supportsSquareSize:size]) {
        if (error) *error=[NSError errorWithDomain:H16GMatmulEncoderErrorDomain
            code:1 userInfo:@{NSLocalizedDescriptionKey:
            @"square fp16 matmul requires N in {128,256,512} or a multiple of 128 in [768,4096]"}];
        return nil;
    }
    NSUInteger tiles=0,rows=0; schedule(size,&tiles,&rows);
    NSData *data = size == 256
        ? [NSData dataWithBytes:kSingle256Words length:sizeof(kSingle256Words)]
        : (tiles == 1 ? encodeSingleFamily(size)
                      : encodeTiled(size,tiles,rows));
    NSUInteger tensorBytes=size*size*2;
    NSUInteger slabBytes=rows*size*2;
    NSUInteger workingSet=tiles==1 ? tensorBytes
        : tensorBytes+(tiles-1)*slabBytes;
    if (workingSet > 0x2000000) workingSet=tensorBytes;
    NSUInteger records=tiles==1 ? 31 : 14*tiles+4;
    return [[H16GEncodedTDProgram alloc] initWithData:data
        kernelRelocationOffsets:@[] programRecordCount:records
        programFormatCode:0 scratchByteLength:workingSet];
}
+ (H16GMatmulPostOperationEncoding *)
    encodeSquareSize:(NSUInteger)size
    postOperationName:(NSString *)postOperationName
    kernelRegion:(NSData *)kernelRegion
    error:(NSError **)error {
    const H16GMatmulPostOperationLayout *layout =
        postOperationLayout(size, postOperationName);
    H16GEncodedTDProgram *plain = layout
        ? [self encodeSquareSize:size error:error] : nil;
    if (!layout || !plain ||
        plain.data.length != layout->plainWordCount * sizeof(uint32_t)) {
        if (error)
            *error = [NSError errorWithDomain:H16GMatmulEncoderErrorDomain
                code:2 userInfo:@{NSLocalizedDescriptionKey:
                    @"matmul post-operation has no decoded field row"}];
        return nil;
    }
    NSMutableData *data = [plain.data mutableCopy];
    uint32_t *words = (uint32_t *)data.mutableBytes;
    if (words[layout->blockHeaderWord] != layout->expectedBlockHeader ||
        words[layout->taskControlWord] != 0x80201540 ||
        words[layout->outputModeWord] != 0x00103c00) {
        if (error)
            *error = [NSError errorWithDomain:H16GMatmulEncoderErrorDomain
                code:3 userInfo:@{NSLocalizedDescriptionKey:
                    @"matmul packet family does not match its post-operation row"}];
        return nil;
    }
    words[layout->blockHeaderWord] += 0x00070000;
    words[layout->taskControlWord] |= 0x00008000;
    words[layout->outputModeWord] |= 0x00020000;

    const uint32_t launch[] = {0x24809586, 0x00000000, 0x00000000};
    const uint32_t completion[] = {0x00009584, 0x00010021, 0x00000080};
    uint32_t geometry = (uint32_t)(size * 16);
    if (!insertWords(data, layout->launchInsertionWord,
                     launch, sizeof(launch) / sizeof(uint32_t)) ||
        !insertWords(data, layout->completionInsertionWord,
                     completion, sizeof(completion) / sizeof(uint32_t)) ||
        !insertWords(data, layout->geometryInsertionWord, &geometry, 1)) {
        if (error)
            *error = [NSError errorWithDomain:H16GMatmulEncoderErrorDomain
                code:4 userInfo:@{NSLocalizedDescriptionKey:
                    @"matmul post-operation insertion is outside the task stream"}];
        return nil;
    }
    H16GEncodedTDProgram *program = [[H16GEncodedTDProgram alloc]
        initWithData:data kernelRelocationOffsets:plain.kernelRelocationOffsets
        programRecordCount:plain.programRecordCount
        programFormatCode:plain.programFormatCode
        scratchByteLength:plain.scratchByteLength];
    return [[H16GMatmulPostOperationEncoding alloc]
        initWithTDProgram:program
        kernelRegion:unpaddedKernelRegion(kernelRegion)];
}
@end
