#import "H16GBoundaryConversionEncoder.h"

static NSString *const H16GBoundaryConversionEncoderErrorDomain =
    @"ANE.H16G.BoundaryConversionEncoder";

// Decoded invariant task words for a standalone quantize fp16->int8 at
// [1, 64, 64, 64] with scale 0x1p-3 (minted on macstudio, h16).
static const uint32_t kQuantizeFP16Int8Words[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x003e0000, 0x0000000f,
    0x04000068, 0x00000000, 0x00fff868, 0x00000000, 0x00000000, 0x00000000,
    0x00000009, 0x00001540, 0x00000080, 0x00018000, 0x0000001a, 0x00000040,
    0x00000040, 0x00000040, 0x95418005, 0x00000040, 0x00000040, 0x00000040,
    0x00000040, 0x00000020, 0x00000036, 0x00000000, 0x800c1342, 0x000000ce,
    0x00000080, 0x00002000, 0x0000135a, 0x01002031, 0x00009041, 0x00500172,
    0x00500130, 0x98039045, 0x00000080, 0x00002000, 0x00002000, 0x00002000,
    0x0050013a, 0x00080000, 0x00001142, 0x41000000, 0x81009444, 0x00000040,
    0x00001000, 0x01301000, 0x83109640, 0x00000012, 0x00a000a0, 0x00000000,
    0x007f0000, 0x20010701, 0x23009344, 0x00000000, 0x00000000, 0x23809442,
    0x00000000, 0x00000000, 0x22001340, 0x00000021, 0x22001440, 0x01000031,
};
// 66 words / 0x108 bytes

// Decoded invariant task words for a standalone dequantize int8->fp16 at
// [1, 64, 64, 64] with scale 0x1p-3 (minted on macstudio, h16).
static const uint32_t kDequantizeInt8FP16Words[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x003e0000, 0x0000000f,
    0x04000068, 0x00000000, 0x00fff868, 0x00000000, 0x00000000, 0x00000000,
    0x00000009, 0x00001540, 0x00000080, 0x00018000, 0x00000029, 0x00000040,
    0x00000040, 0x00000040, 0x95418005, 0x00000040, 0x00000040, 0x00000040,
    0x00000040, 0x00000020, 0x00000036, 0x00000000, 0x800c1342, 0x000000ce,
    0x00000040, 0x00001000, 0x0000135a, 0x01001000, 0x00009041, 0x00500132,
    0x00500130, 0x98039045, 0x00000040, 0x00001000, 0x00001000, 0x00001000,
    0x0050017a, 0x00040000, 0x00001142, 0x3e000000, 0x81009444, 0x00000080,
    0x00002000, 0x01302031, 0x83109640, 0x00000012, 0x00a000a0, 0x00000000,
    0x007f0000, 0x20010701, 0x23009344, 0x00000000, 0x00000000, 0x23809442,
    0x00000000, 0x00000000, 0x22001340, 0x00000021, 0x22001440, 0x01000031,
};
// 66 words / 0x108 bytes

// Decoded invariant task words for a standalone cast fp16->bool at
// [1, 375, 375] (minted on macstudio, h16).
static const uint32_t kCastFP16ToBoolWords[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x003c0000, 0x00000008,
    0x04000068, 0x00000000, 0x00fff868, 0x00000000, 0x00000000, 0x00000000,
    0x00000009, 0x00001540, 0x00000080, 0x00018000, 0x0000000a, 0x00000177,
    0x00000177, 0x00000001, 0x95418005, 0x00000177, 0x00000177, 0x00000001,
    0x00000177, 0x00000020, 0x00000009, 0x00000000, 0x80041342, 0x000000ce,
    0x00000300, 0x0000135a, 0x01002031, 0x00009041, 0x00500172, 0x00500130,
    0x98039045, 0x000002f0, 0x000002f0, 0x000002f0, 0x000002f0, 0x0050013a,
    0x00044d90, 0x00001140, 0x000000c0, 0x81001444, 0x00000180, 0x01300000,
    0x83109640, 0x00000012, 0x00a000a0, 0x00000000, 0x007f0000, 0x20010701,
    0x23009344, 0x00000000, 0x00000000, 0x23809442, 0x00000000, 0x00000000,
    0x22001340, 0x00000021, 0x22001440, 0x01000031,
};
// 64 words / 0x100 bytes

// Decoded invariant task words for a standalone cast fp16->fp32 at
// [1, 64, 64] (minted on macstudio, h16).
static const uint32_t kCastFP16ToFP32Words[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x003d0000, 0x00000000,
    0x04000068, 0x00000000, 0x00000068, 0x00000000, 0x00000000, 0x00000000,
    0x00040009, 0x00001540, 0x00000080, 0x00010001, 0x00000040, 0x00000040,
    0x00000001, 0x93618005, 0x00000040, 0x00000040, 0x00000001, 0x00014000,
    0x00000010, 0x00010000, 0x00000030, 0x00100000, 0x80041342, 0x000000ce,
    0x00000080, 0x0000135a, 0x01002031, 0x00009041, 0x00500172, 0x00500130,
    0x98039045, 0x00000080, 0x00000080, 0x00000080, 0x00000080, 0x005001fa,
    0x00000800, 0x80021241, 0x00103c0c, 0x00003c00, 0x81001444, 0x00000100,
    0x01303103, 0x83109640, 0x00000012, 0x00a000a0, 0x00000000, 0x007f0000,
    0x20010701, 0x23009344, 0x00000000, 0x00000000, 0x23809442, 0x00000000,
    0x00000000, 0x22001340, 0x00000021, 0x22001440, 0x01000031,
};
// 65 words / 0x104 bytes

// Decoded invariant task words for a standalone cast fp32->fp16 at
// [1, 64, 64] (minted on macstudio, h16).
static const uint32_t kCastFP32ToFP16Words[] = {
    0x00000001, 0x00000000, 0x00000000, 0x00000000, 0x003d0000, 0x00000000,
    0x04000068, 0x00000000, 0x00000068, 0x00000000, 0x00000000, 0x00000000,
    0x00050009, 0x00001540, 0x00000080, 0x00010001, 0x00000040, 0x00000040,
    0x00000001, 0x93618005, 0x00000040, 0x00000040, 0x00000001, 0x00014000,
    0x00000020, 0x00020000, 0x00000038, 0x00100000, 0x80041342, 0x000000ce,
    0x00000100, 0x0000135a, 0x01003103, 0x00009041, 0x005001f2, 0x00500130,
    0x98039045, 0x00000080, 0x00000080, 0x00000080, 0x00000080, 0x0050017a,
    0x00001000, 0x80021241, 0x00103c0c, 0x00003c00, 0x81001444, 0x00000080,
    0x01302031, 0x83109640, 0x00000012, 0x00a000a0, 0x00000000, 0x007f0000,
    0x20010701, 0x23009344, 0x00000000, 0x00000000, 0x23809442, 0x00000000,
    0x00000000, 0x22001340, 0x00000021, 0x22001440, 0x01000031,
};
// 65 words / 0x104 bytes

typedef struct {
    ANEElementType sourceType;
    ANEElementType destinationType;
    const uint32_t *words;
    NSUInteger wordCount;
    uint32_t formatCode;
    NSArray<NSNumber *> *shape;
} BoundaryForm;

static BOOL shapeEquals(NSArray<NSNumber *> *shape, NSArray<NSNumber *> *want) {
    return [shape isEqualToArray:want];
}

@implementation H16GBoundaryConversionEncoder
+ (H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                   sourceType:(ANEElementType)sourceType
                              destinationType:(ANEElementType)destinationType
                                        shape:(NSArray<NSNumber *> *)shape
                                        error:(NSError **)error {
    const uint32_t *words = NULL;
    NSUInteger wordCount = 0;
    uint32_t formatCode = 0;
    if ([operationName isEqualToString:@"quantize"] &&
        sourceType == ANEElementTypeFP16 &&
        destinationType == ANEElementTypeInt8 &&
        shapeEquals(shape, @[@1, @64, @64, @64])) {
        words = kQuantizeFP16Int8Words;
        wordCount = sizeof(kQuantizeFP16Int8Words) / 4;
        formatCode = 0x0f;
    } else if ([operationName isEqualToString:@"dequantize"] &&
               sourceType == ANEElementTypeInt8 &&
               destinationType == ANEElementTypeFP16 &&
               shapeEquals(shape, @[@1, @64, @64, @64])) {
        words = kDequantizeInt8FP16Words;
        wordCount = sizeof(kDequantizeInt8FP16Words) / 4;
        formatCode = 0x0f;
    } else if ([operationName isEqualToString:@"cast"] &&
               sourceType == ANEElementTypeFP16 &&
               destinationType == ANEElementTypeBool &&
               shapeEquals(shape, @[@1, @375, @375])) {
        words = kCastFP16ToBoolWords;
        wordCount = sizeof(kCastFP16ToBoolWords) / 4;
        formatCode = 0x08;
    } else if ([operationName isEqualToString:@"cast"] &&
               sourceType == ANEElementTypeFP16 &&
               destinationType == ANEElementTypeFP32 &&
               shapeEquals(shape, @[@1, @64, @64])) {
        words = kCastFP16ToFP32Words;
        wordCount = sizeof(kCastFP16ToFP32Words) / 4;
        formatCode = 0x00;
    } else if ([operationName isEqualToString:@"cast"] &&
               sourceType == ANEElementTypeFP32 &&
               destinationType == ANEElementTypeFP16 &&
               shapeEquals(shape, @[@1, @64, @64])) {
        words = kCastFP32ToFP16Words;
        wordCount = sizeof(kCastFP32ToFP16Words) / 4;
        formatCode = 0x00;
    }
    if (!words) {
        if (error) *error = [NSError errorWithDomain:
            H16GBoundaryConversionEncoderErrorDomain code:1 userInfo:@{
            NSLocalizedDescriptionKey:
                @"no decoded h16 boundary-conversion form for this operation, "
                "type pair and shape (Apple rejects every int32 surface)"}];
        return nil;
    }
    NSData *data = [NSData dataWithBytes:words length:wordCount * sizeof(uint32_t)];
    return [[H16GEncodedTDProgram alloc] initWithData:data
        kernelRelocationOffsets:@[] programRecordCount:14
        programFormatCode:formatCode];
}
@end
