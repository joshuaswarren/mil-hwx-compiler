#import "H16GReduceEncoder.h"

static NSString *const H16GReduceEncoderErrorDomain = @"ANE.H16G.ReduceEncoder";

#include "H16GReduceEncoderData.inc"

typedef struct {
    NSString *__unsafe_unretained operation;
    NSUInteger channels;
    NSUInteger height;
    NSUInteger width;
    NSUInteger axis;
    const uint32_t *words;
    NSUInteger wordCount;
    NSUInteger taskCount;
    NSUInteger recordCount;
    uint32_t formatCode;
} H16GReductionGeometry;

#define WORD_COUNT(array) (sizeof(array) / sizeof((array)[0]))
#define REDUCE_ROW(op,c,h,w,a,array,tasks,records,format) \
    {op,c,h,w,a,array,WORD_COUNT(array),tasks,records,format}
static const H16GReductionGeometry kReductionGeometries[] = {
    REDUCE_ROW(@"reduce_sum",32,8,8,1,kSumC32H8W8A1TDWords,2,18,0),
    REDUCE_ROW(@"reduce_sum",64,8,8,1,kSumC64H8W8A1TDWords,2,18,0),
    REDUCE_ROW(@"reduce_sum",128,8,8,1,kSumC128H8W8A1TDWords,2,18,0),
    REDUCE_ROW(@"reduce_sum",64,16,16,1,kSumC64H16W16A1TDWords,2,18,1),
    REDUCE_ROW(@"reduce_sum",64,32,32,1,kSumC64H32W32A1TDWords,2,18,4),
    REDUCE_ROW(@"reduce_sum",1,64,64,3,kSumC1H64W64A3TDWords,1,16,0),
    REDUCE_ROW(@"reduce_sum",1,128,128,3,kSumC1H128W128A3TDWords,1,16,1),
    REDUCE_ROW(@"reduce_sum",32,64,16,2,kSumC32H64W16A2TDWords,3,19,2),

    REDUCE_ROW(@"reduce_mean",32,8,8,1,kMeanC32H8W8A1TDWords,2,18,0),
    REDUCE_ROW(@"reduce_mean",64,8,8,1,kMeanC64H8W8A1TDWords,2,18,0),
    REDUCE_ROW(@"reduce_mean",128,8,8,1,kMeanC128H8W8A1TDWords,2,18,0),
    REDUCE_ROW(@"reduce_mean",64,16,16,1,kMeanC64H16W16A1TDWords,2,18,1),
    REDUCE_ROW(@"reduce_mean",64,32,32,1,kMeanC64H32W32A1TDWords,2,18,4),
    REDUCE_ROW(@"reduce_mean",1,64,64,3,kMeanC1H64W64A3TDWords,1,16,0),
    REDUCE_ROW(@"reduce_mean",1,128,128,3,kMeanC1H128W128A3TDWords,1,16,1),
    REDUCE_ROW(@"reduce_mean",32,64,16,2,kMeanC32H64W16A2TDWords,3,19,2),

    REDUCE_ROW(@"reduce_max",32,8,8,1,kMaxC32H8W8A1TDWords,1,16,0),
    REDUCE_ROW(@"reduce_max",64,8,8,1,kMaxC64H8W8A1TDWords,1,16,0),
    REDUCE_ROW(@"reduce_max",128,8,8,1,kMaxC128H8W8A1TDWords,1,16,0),
    REDUCE_ROW(@"reduce_max",64,16,16,1,kMaxC64H16W16A1TDWords,1,16,0),
    REDUCE_ROW(@"reduce_max",64,32,32,1,kMaxC64H32W32A1TDWords,1,16,1),
    REDUCE_ROW(@"reduce_max",1,64,64,3,kMaxC1H64W64A3TDWords,1,16,0),
    REDUCE_ROW(@"reduce_max",1,128,128,3,kMaxC1H128W128A3TDWords,1,16,1),
    REDUCE_ROW(@"reduce_max",32,64,16,2,kMaxC32H64W16A2TDWords,3,19,2),
};
#undef REDUCE_ROW
#undef WORD_COUNT

@interface H16GReduceEncoding ()
@property(nonatomic, readwrite, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readwrite, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readwrite) NSUInteger taskCount;
@property(nonatomic, readwrite) NSUInteger inputRowStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputPlaneStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputBatchStrideBytes;
@property(nonatomic, readwrite) NSUInteger inputStorageByteLength;
@property(nonatomic, readwrite) NSUInteger outputRowStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputPlaneStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputBatchStrideBytes;
@property(nonatomic, readwrite) NSUInteger outputStorageByteLength;
@end

@implementation H16GReduceEncoding
@end

static const H16GReductionGeometry *geometry(NSString *operation,
                                              NSArray<NSNumber *> *shape,
                                              NSUInteger axis) {
    if (shape.count != 4 || shape[0].unsignedIntegerValue != 1) return NULL;
    NSUInteger channels=shape[1].unsignedIntegerValue;
    NSUInteger height=shape[2].unsignedIntegerValue;
    NSUInteger width=shape[3].unsignedIntegerValue;
    for (const H16GReductionGeometry &row : kReductionGeometries) {
        if ([operation isEqualToString:row.operation] &&
            channels == row.channels && height == row.height &&
            width == row.width && axis == row.axis)
            return &row;
    }
    return NULL;
}

static NSArray<NSNumber *> *reducedShape(NSArray<NSNumber *> *inputShape,
                                          NSUInteger axis) {
    NSMutableArray<NSNumber *> *shape = [inputShape mutableCopy];
    shape[axis] = @1;
    return shape;
}

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GReduceEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

@implementation H16GReduceEncoder
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape
                         axis:(NSUInteger)axis {
    return geometry(operationName,inputShape,axis) != NULL;
}

+ (H16GReduceEncoding *)encodeOperationName:(NSString *)operationName
                                    inputShape:(NSArray<NSNumber *> *)inputShape
                                          axis:(NSUInteger)axis
                                         error:(NSError **)error {
    const H16GReductionGeometry *row = geometry(operationName,inputShape,axis);
    if (!row) {
        setError(error,@"reduction requires a measured H16G operation, axis, and input geometry");
        return nil;
    }
    NSData *data = [NSData dataWithBytes:row->words
        length:row->wordCount * sizeof(uint32_t)];
    H16GEncodedTDProgram *program = [[H16GEncodedTDProgram alloc]
        initWithData:data kernelRelocationOffsets:@[]
        programRecordCount:row->recordCount programFormatCode:row->formatCode
        scratchByteLength:0];
    NSArray<NSNumber *> *outputShape = reducedShape(inputShape,axis);
    NSUInteger inputChannels=inputShape[1].unsignedIntegerValue;
    NSUInteger inputHeight=inputShape[2].unsignedIntegerValue;
    NSUInteger inputWidth=inputShape[3].unsignedIntegerValue;
    NSUInteger inputRowBytes=alignUp(inputWidth * sizeof(uint16_t),64);
    NSUInteger inputPlaneBytes=inputHeight * inputRowBytes;
    NSUInteger inputBatchBytes=inputChannels * inputPlaneBytes;
    NSUInteger channels=outputShape[1].unsignedIntegerValue;
    NSUInteger height=outputShape[2].unsignedIntegerValue;
    NSUInteger width=outputShape[3].unsignedIntegerValue;
    NSUInteger rowBytes=alignUp(width * sizeof(uint16_t),64);
    NSUInteger planeBytes=height * rowBytes;
    NSUInteger batchBytes=channels * planeBytes;
    H16GReduceEncoding *encoding = [H16GReduceEncoding new];
    encoding.tdProgram=program; encoding.outputShape=outputShape;
    encoding.taskCount=row->taskCount;
    encoding.inputRowStrideBytes=inputRowBytes;
    encoding.inputPlaneStrideBytes=inputPlaneBytes;
    encoding.inputBatchStrideBytes=inputBatchBytes;
    encoding.inputStorageByteLength=inputBatchBytes;
    encoding.outputRowStrideBytes=rowBytes;
    encoding.outputPlaneStrideBytes=planeBytes;
    encoding.outputBatchStrideBytes=batchBytes;
    encoding.outputStorageByteLength=batchBytes;
    return encoding;
}
@end
