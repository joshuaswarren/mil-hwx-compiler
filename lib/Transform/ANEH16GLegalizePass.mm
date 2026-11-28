#import "ANEH16GLegalizePass.h"

static ANEOperationNode *operandNode(ANEOperationGraph *graph,
                                     ANEOperationNode *node,
                                     NSString *name);

static ANEPhysicalLayout layoutForNode(ANEOperationNode *node) {
    if (node.kind == ANEOperationKindConv) return ANEPhysicalLayoutNCHW;
    if ([node.operationName isEqualToString:@"space_to_depth"] ||
        [node.operationName isEqualToString:@"depth_to_space"])
        return ANEPhysicalLayoutNCHW;
    if (node.kind == ANEOperationKindMatmul)
        return node.outputType.shape.count > 2 ? ANEPhysicalLayoutBHSD
                                               : ANEPhysicalLayoutMatrix;
    for (ANEOperationNode *input in node.inputs)
        if (input.physicalLayout != ANEPhysicalLayoutUnknown)
            return input.physicalLayout;
    switch (node.outputType.shape.count) {
        case 0:
        case 1: return ANEPhysicalLayoutLinear;
        case 2: return ANEPhysicalLayoutMatrix;
        case 4: return ANEPhysicalLayoutTensor4D;
        default: return ANEPhysicalLayoutLinear;
    }
}

static ANEGraphArgument *singleValue(ANEGraphArgument *argument) {
    while (argument.kind == ANEGraphArgumentKindCall &&
           argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument;
}

static NSUInteger constantUnsignedValue(ANEOperationGraph *graph,
                                        ANEOperationNode *node,
                                        NSString *operandName) {
    ANEOperationNode *constant = operandNode(graph, node, operandName);
    ANEGraphArgument *value = constant.sourceOperation.attributes[@"val"];
    NSString *text = singleValue(value).text;
    return text ? (NSUInteger)text.longLongValue : 0;
}

static BOOL validateLayoutTransform(ANEOperationGraph *graph,
                                    ANEOperationNode *node,
                                    H16GTarget *target,
                                    ANEDiagnosticEngine *diagnostics) {
    BOOL isS2D = [node.operationName isEqualToString:@"space_to_depth"];
    BOOL isD2S = [node.operationName isEqualToString:@"depth_to_space"];
    if (!isS2D && !isD2S) return YES;
    ANEOperationNode *input = operandNode(graph, node, @"x");
    ANEValueType *inputType = input ? input.outputType : nil;
    if (!inputType) {
        ANEGraphValue *value = node.sourceOperation.operands[@"x"].value;
        for (ANEGraphValue *candidate in graph.sourceFunction.inputs)
            if ([candidate.name isEqualToString:value.name]) inputType = candidate.type;
    }
    NSUInteger block = constantUnsignedValue(graph, node, @"block_size");
    NSString *reason = nil;
    BOOL valid = inputType && inputType.elementType == ANEElementTypeFP16 &&
        node.outputType.elementType == ANEElementTypeFP16 &&
        [target validateSpaceDepthInputShape:inputType.shape
            outputShape:node.outputType.shape blockSize:block
            depthToSpace:isD2S reason:&reason];
    if (valid) return YES;
    [diagnostics emitSeverity:ANEDiagnosticSeverityError
                         code:@"h16g.legalize.invalid-layout-shape"
                      message:reason ?: @"invalid space/depth transform"
                        range:node.sourceOperation.range];
    return NO;
}

static ANEOperationNode *operandNode(ANEOperationGraph *graph,
                                     ANEOperationNode *node,
                                     NSString *name) {
    ANEGraphValue *value = node.sourceOperation.operands[name].value;
    return value ? [graph nodeForValueName:value.name] : nil;
}

static BOOL isQuantizedWeightNode(ANEOperationNode *node) {
    return node && [node.operationName isEqualToString:
                    @"constexpr_affine_dequantize"];
}

static BOOL isPackedActivationBoundary(ANEOperationNode *node) {
    if (!node || ![node.operationName isEqualToString:@"dequantize"]) return NO;
    for (ANEOperationNode *input in node.inputs)
        if ([input.operationName isEqualToString:@"quantize"]) return YES;
    return NO;
}

static BOOL hasQuantizedOutput(ANEOperationNode *node) {
    for (ANEOperationNode *user in node.users)
        if ([user.operationName isEqualToString:@"quantize"]) return YES;
    return NO;
}

static BOOL isW8A8Conv(ANEOperationGraph *graph, ANEOperationNode *node) {
    return node && node.kind == ANEOperationKindConv &&
        isQuantizedWeightNode(operandNode(graph, node, @"weight"));
}

static BOOL quantizeFoldsIntoBoundary(ANEOperationGraph *graph,
                                      ANEOperationNode *node) {
    return isW8A8Conv(graph, operandNode(graph, node, @"input"));
}

static BOOL dequantizeFoldsIntoBoundary(ANEOperationGraph *graph,
                                        ANEOperationNode *node) {
    if (node.users.count == 0) return NO;
    for (ANEOperationNode *user in node.users)
        if (!isW8A8Conv(graph, user) ||
            operandNode(graph, user, @"x") != node) return NO;
    return YES;
}

static ANEElementType castSourceType(ANEOperationGraph *graph,
                                     ANEOperationNode *node) {
    ANEOperationNode *input = operandNode(graph, node, @"x");
    if (input && input.outputType) return input.outputType.elementType;
    ANEGraphValue *value = node.sourceOperation.operands[@"x"].value;
    for (ANEGraphValue *candidate in graph.sourceFunction.inputs)
        if ([candidate.name isEqualToString:value.name])
            return candidate.type.elementType;
    return ANEElementTypeInvalid;
}

static NSString *elementTypeName(ANEElementType type) {
    switch (type) {
        case ANEElementTypeFP16: return @"fp16";
        case ANEElementTypeFP32: return @"fp32";
        case ANEElementTypeInt8: return @"int8";
        case ANEElementTypeInt32: return @"int32";
        case ANEElementTypeUInt64: return @"uint64";
        case ANEElementTypeBool: return @"bool";
        case ANEElementTypeString: return @"string";
        default: return @"?";
    }
}

static BOOL decodedCastDirection(ANEOperationGraph *graph,
                                 ANEOperationNode *node) {
    ANEElementType source = castSourceType(graph, node);
    ANEElementType destination = node.outputType.elementType;
    return (source == ANEElementTypeFP16 &&
            (destination == ANEElementTypeBool ||
             destination == ANEElementTypeFP32)) ||
        (source == ANEElementTypeFP32 && destination == ANEElementTypeFP16);
}

static ANELegalNumericMode numericModeForConv(ANEOperationGraph *graph,
                                               ANEOperationNode *conv) {
    ANEOperationNode *weight = operandNode(graph, conv, @"weight");
    if (!isQuantizedWeightNode(weight)) return ANELegalNumericModeFP16;
    ANEOperationNode *activation = operandNode(graph, conv, @"x");
    BOOL packedInput = isPackedActivationBoundary(activation);
    if (!packedInput) return ANELegalNumericModeW8A8InputBoundary;
    return hasQuantizedOutput(conv) ? ANELegalNumericModeW8A8Packed
                                    : ANELegalNumericModeW8A8OutputBoundary;
}

@implementation ANEH16GLegalizePass
+ (BOOL)runOnGraph:(ANEOperationGraph *)graph
             target:(H16GTarget *)target
        diagnostics:(ANEDiagnosticEngine *)diagnostics {
    for (ANEOperationNode *node in graph.nodes) {
        if (node.kind == ANEOperationKindConstant) {
            [node applyPhysicalLayout:ANEPhysicalLayoutUnknown
                          numericMode:ANELegalNumericModeFP16];
            continue;
        }
        if (![target supportsOperationKind:node.kind]) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                 code:@"h16g.legalize.unsupported-operation"
                              message:[NSString stringWithFormat:
                                  @"H16G has no decoded primitive for '%@'",
                                  node.operationName]
                                range:node.sourceOperation ? node.sourceOperation.range
                                                           : ANESourceRangeMake(
                                                               ANESourceLocationMake(0, 1, 1),
                                                               ANESourceLocationMake(0, 1, 1))];
            return NO;
        }
        if (node.kind == ANEOperationKindCast &&
            !decodedCastDirection(graph, node)) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                 code:@"h16g.legalize.unsupported-operation"
                              message:[NSString stringWithFormat:
                                  @"H16G has no decoded primitive for 'cast' "
                                  "%@->%@ (Apple rejects every int32 surface)",
                                  elementTypeName(
                                      castSourceType(graph, node)),
                                  elementTypeName(node.outputType.elementType)]
                                range:node.sourceOperation ? node.sourceOperation.range
                                                           : ANESourceRangeMake(
                                                               ANESourceLocationMake(0, 1, 1),
                                                               ANESourceLocationMake(0, 1, 1))];
            return NO;
        }
        if (!validateLayoutTransform(graph, node, target, diagnostics))
            return NO;
        ANELegalNumericMode mode = node.kind == ANEOperationKindConv
            ? numericModeForConv(graph, node) : ANELegalNumericModeFP16;
        [node applyPhysicalLayout:layoutForNode(node) numericMode:mode];
    }
    for (ANEOperationNode *node in graph.nodes) {
        if (node.kind == ANEOperationKindQuantize) {
            if (quantizeFoldsIntoBoundary(graph, node))
                [node markFoldedIntoNumericBoundary];
        } else if (node.kind == ANEOperationKindDequantize) {
            if (dequantizeFoldsIntoBoundary(graph, node))
                [node markFoldedIntoNumericBoundary];
        }
    }
    return YES;
}
@end
