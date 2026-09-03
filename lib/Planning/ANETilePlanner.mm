#import "ANETilePlanner.h"

static NSUInteger largestPowerOfTwoAtMost(NSUInteger value) {
    NSUInteger result = 1;
    while (result <= value / 2) result *= 2;
    return result;
}

static NSUInteger elementBytes(ANEElementType type) {
    switch (type) {
        case ANEElementTypeFP16: return 2;
        case ANEElementTypeFP32:
        case ANEElementTypeInt32: return 4;
        case ANEElementTypeInt8:
        case ANEElementTypeBool: return 1;
        case ANEElementTypeUInt64: return 8;
        case ANEElementTypeString:
        case ANEElementTypeInvalid: return 0;
    }
}

static NSUInteger tensorBytes(ANEValueType *type) {
    NSUInteger elements = 1;
    for (NSNumber *dimension in type.shape)
        elements *= dimension.unsignedIntegerValue;
    return elements * elementBytes(type.elementType);
}

static ANEGraphArgument *singleValue(ANEGraphArgument *argument) {
    while (argument.kind == ANEGraphArgumentKindCall &&
           argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument;
}

static NSUInteger constantOperand(ANEOperationNode *node, NSString *name) {
    ANEGraphValue *value = node.sourceOperation.operands[name].value;
    for (ANEOperationNode *input in node.inputs) {
        if (![input.identifier isEqualToString:value.name]) continue;
        NSString *text = singleValue(input.sourceOperation.attributes[@"val"]).text;
        return text ? (NSUInteger)text.longLongValue : 0;
    }
    return 0;
}

static ANETilePlan *layoutPlan(ANEOperationNode *node, H16GTarget *target) {
    ANEValueType *inputType = node.sourceOperation.operands[@"x"].value.type;
    NSArray<NSNumber *> *inputShape = inputType.shape;
    NSArray<NSNumber *> *outputShape = node.outputType.shape;
    NSUInteger block = constantOperand(node, @"block_size");
    BOOL depthToSpace = [node.operationName isEqualToString:@"depth_to_space"];
    NSUInteger bytes = tensorBytes(inputType);
    ANETileStrategy strategy;
    NSUInteger descriptors;
    if (!depthToSpace && block == 2) {
        strategy = ANETileStrategyLayoutConv;
        descriptors = 1;
    } else if (bytes <= target.layoutDMAProgramMaxInputBytes) {
        strategy = ANETileStrategyLayoutDMA3;
        descriptors = 3;
    } else if (depthToSpace) {
        strategy = ANETileStrategyLayoutLoadStream;
        NSUInteger outputChannels = outputShape[1].unsignedIntegerValue;
        descriptors = MAX((NSUInteger)1, (outputChannels + 31) / 32);
    } else {
        strategy = ANETileStrategyLayoutStoreStream;
        NSUInteger outputChannels = outputShape[1].unsignedIntegerValue;
        descriptors = MAX((NSUInteger)1, (outputChannels + 31) / 32);
    }
    NSUInteger rows = outputShape[2].unsignedIntegerValue;
    return [[ANETilePlan alloc] initWithRows:rows count:descriptors
        strategy:strategy inputShape:inputShape outputShape:outputShape
        descriptorCount:descriptors];
}

@implementation ANETilePlanner
+ (ANETilePlan *)tilePlanForNode:(ANEOperationNode *)node
                          target:(H16GTarget *)target {
    NSArray<NSNumber *> *shape = node.outputType.shape;
    NSUInteger rows = shape.count ? shape[shape.count - 2 < shape.count
        ? shape.count - 2 : 0].unsignedIntegerValue : 1;
    if ([node.operationName isEqualToString:@"space_to_depth"] ||
        [node.operationName isEqualToString:@"depth_to_space"])
        return layoutPlan(node, target);
    if (node.kind != ANEOperationKindMatmul || shape.count < 2)
        return [[ANETilePlan alloc] initWithRows:rows count:1];
    NSUInteger columns = shape.lastObject.unsignedIntegerValue;
    NSUInteger bytes = rows * columns * 2;
    if (bytes <= target.outputSlabBytes)
        return [[ANETilePlan alloc] initWithRows:rows count:1];
    NSUInteger limit = MIN((NSUInteger)256, target.outputSlabBytes / (columns * 2));
    NSUInteger tileRows = largestPowerOfTwoAtMost(MAX((NSUInteger)1, limit));
    NSUInteger count = (rows + tileRows - 1) / tileRows;
    return [[ANETilePlan alloc] initWithRows:tileRows count:count
        strategy:ANETileStrategyMatrixRows inputShape:@[] outputShape:shape
        descriptorCount:count];
}
@end
