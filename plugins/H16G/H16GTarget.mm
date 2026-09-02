#import "H16GTarget.h"

@implementation H16GOptionalUInt
- (instancetype)initUnavailableWithProvenance:(NSString *)provenance {
    self = [super init];
    if (self) {
        _available = NO;
        _provenance = [provenance copy];
    }
    return self;
}
- (instancetype)initWithValue:(NSUInteger)value provenance:(NSString *)provenance {
    self = [super init];
    if (self) {
        _available = YES;
        _value = value;
        _provenance = [provenance copy];
    }
    return self;
}
@end

@implementation H16GTaskCapability
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry {
    self = [super init];
    if (self) {
        _stageOperations = [stageOperations copy];
        _inputShapes = [inputShapes copy];
        _outputShape = [outputShape copy];
        _numericMode = numericMode;
        _outputStorage = outputStorage;
        _packetFamily = packetFamily;
        _geometry = geometry;
    }
    return self;
}
@end

static NSArray<H16GTaskCapability *> *taskCapabilityRows(void) {
    static NSArray<H16GTaskCapability *> *rows;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        NSMutableArray<H16GTaskCapability *> *result = [NSMutableArray array];
        for (NSNumber *sizeNumber in @[@128, @256]) {
            NSUInteger size = sizeNumber.unsignedIntegerValue;
            NSArray<NSNumber *> *matrix = @[@1, @(size), @(size)];
            NSArray<NSNumber *> *image = @[@1, @1, @(size), @(size)];
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[@"matmul", @"reshape"]
                inputShapes:@[matrix, matrix] outputShape:image
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilySquareMatmul geometry:size]];
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[@"gelu"] inputShapes:@[image]
                outputShape:image numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyUnaryLUT geometry:size]];
        }
        NSArray<NSNumber *> *matrix128 = @[@1, @1, @128, @128];
        NSArray<NSNumber *> *row128 = @[@1, @1, @128, @1];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"matmul"]
            inputShapes:@[matrix128, matrix128] outputShape:matrix128
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilySquareMatmul geometry:128]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"exp"] inputShapes:@[matrix128]
            outputShape:matrix128 numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilyUnaryLUT geometry:128]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"mul"] inputShapes:@[matrix128]
            outputShape:matrix128 numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilyScalarScale geometry:128]];
        for (NSString *operation in @[@"sub", @"mul"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[matrix128, row128] outputShape:matrix128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyMatrixRowALU geometry:128]];
        for (NSString *operation in @[@"add", @"mul", @"max"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[row128, row128] outputShape:row128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyRowStateALU geometry:128]];
        for (NSString *operation in @[@"reduce_max", @"reduce_sum"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation] inputShapes:@[matrix128]
                outputShape:row128 numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilyReduction geometry:128]];
        [result addObject:[[H16GTaskCapability alloc]
            initWithStageOperations:@[@"real_div"]
            inputShapes:@[matrix128, row128] outputShape:matrix128
            numericMode:ANELegalNumericModeFP16
            outputStorage:ANEScheduledBridgeStorageExternal
            packetFamily:H16GTaskPacketFamilyMatrixRowDivision geometry:128]];
        for (NSString *operation in @[@"add", @"mul"])
            [result addObject:[[H16GTaskCapability alloc]
                initWithStageOperations:@[operation]
                inputShapes:@[matrix128, matrix128] outputShape:matrix128
                numericMode:ANELegalNumericModeFP16
                outputStorage:ANEScheduledBridgeStorageExternal
                packetFamily:H16GTaskPacketFamilySquareALU geometry:128]];
        rows = [result copy];
    });
    return rows;
}

@implementation H16GTarget
+ (instancetype)currentTarget {
    static H16GTarget *target;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        target = [[H16GTarget alloc] init];
    });
    return target;
}
- (instancetype)init {
    self = [super init];
    if (self) {
        _identifier = @"H16G";
        _sramBankCount = 64;
        _sramGranuleBytes = 16;
        _computeSetCount = 8;
        _ioSurfaceAlignment = 16384;
        _outputSlabBytes = 1024 * 1024;
        _workingSetBytes = 2 * 1024 * 1024;
        _maximumTensorDimension = 65536;
        _minimumSurfaceRowBytes = 64;
        _layoutDMAProgramMaxInputBytes = 1024 * 1024;
        _layoutBlockSizes = @[@2, @4, @8];
        _maximumProgramTaskCount = [[H16GOptionalUInt alloc]
            initUnavailableWithProvenance:@"pending direct task-count probe"];
    }
    return self;
}
- (BOOL)validateSpaceDepthInputShape:(NSArray<NSNumber *> *)inputShape
                         outputShape:(NSArray<NSNumber *> *)outputShape
                           blockSize:(NSUInteger)blockSize
                        depthToSpace:(BOOL)depthToSpace
                              reason:(NSString **)reason {
    BOOL supportedBlock = [_layoutBlockSizes containsObject:@(blockSize)];
    if (inputShape.count != 4 || outputShape.count != 4 || !supportedBlock) {
        if (reason) *reason = @"layout transform requires rank-4 NCHW and a measured block size";
        return NO;
    }
    NSUInteger n = inputShape[0].unsignedIntegerValue;
    NSUInteger channels = inputShape[1].unsignedIntegerValue;
    NSUInteger height = inputShape[2].unsignedIntegerValue;
    NSUInteger width = inputShape[3].unsignedIntegerValue;
    NSUInteger blockArea = blockSize * blockSize;
    if (n != 1 || outputShape[0].unsignedIntegerValue != 1 ||
        channels == 0 || height == 0 || width == 0 ||
        height > _maximumTensorDimension || width > _maximumTensorDimension) {
        if (reason) *reason = @"layout transform requires one positive, bounded NCHW batch";
        return NO;
    }
    NSUInteger expectedChannels = 0;
    NSUInteger expectedHeight = 0;
    NSUInteger expectedWidth = 0;
    if (depthToSpace) {
        if (channels % blockArea != 0) {
            if (reason) *reason = @"depth_to_space input channels must be divisible by block squared";
            return NO;
        }
        expectedChannels = channels / blockArea;
        expectedHeight = height * blockSize;
        expectedWidth = width * blockSize;
    } else {
        if (height % blockSize != 0 || width % blockSize != 0) {
            if (reason) *reason = @"space_to_depth spatial dimensions must be divisible by block";
            return NO;
        }
        expectedChannels = channels * blockArea;
        expectedHeight = height / blockSize;
        expectedWidth = width / blockSize;
    }
    BOOL shapeMatches = outputShape[1].unsignedIntegerValue == expectedChannels &&
        outputShape[2].unsignedIntegerValue == expectedHeight &&
        outputShape[3].unsignedIntegerValue == expectedWidth;
    if (!shapeMatches || expectedChannels > _maximumTensorDimension ||
        expectedHeight > _maximumTensorDimension ||
        expectedWidth > _maximumTensorDimension) {
        if (reason) *reason = @"layout output type does not match the block transform";
        return NO;
    }
    return YES;
}
- (BOOL)supportsOperationKind:(ANEOperationKind)kind {
    switch (kind) {
        case ANEOperationKindConv:
        case ANEOperationKindMatmul:
        case ANEOperationKindALU:
        case ANEOperationKindLUT:
        case ANEOperationKindReduce:
        case ANEOperationKindLayout:
        case ANEOperationKindQuantize:
        case ANEOperationKindDequantize:
            return YES;
        case ANEOperationKindConstant:
        case ANEOperationKindHighLevel:
        case ANEOperationKindUnsupported:
            return NO;
    }
}
- (BOOL)supportsFusedEpilogue:(NSString *)operationName
                 producerKind:(ANEOperationKind)producerKind {
    return [operationName isEqualToString:@"relu"] &&
        (producerKind == ANEOperationKindConv ||
         producerKind == ANEOperationKindMatmul);
}
- (H16GTaskCapability *)taskCapabilityForStageOperations:
    (NSArray<NSString *> *)stageOperations
    inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
    outputShape:(NSArray<NSNumber *> *)outputShape
    numericMode:(ANELegalNumericMode)numericMode
    outputStorage:(ANEScheduledBridgeStorage)outputStorage {
    for (H16GTaskCapability *row in taskCapabilityRows())
        if ([row.stageOperations isEqualToArray:stageOperations] &&
            [row.inputShapes isEqualToArray:inputShapes] &&
            [row.outputShape isEqualToArray:outputShape] &&
            row.numericMode == numericMode &&
            row.outputStorage == outputStorage) return row;
    return nil;
}
@end
