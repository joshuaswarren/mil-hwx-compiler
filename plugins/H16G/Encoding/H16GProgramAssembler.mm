#import "H16GProgramAssembler.h"

#import "ANEHWXArtifact.h"
#import "ANEProgramPartition.h"
#import "H16GEncodedTask.h"
#import "H16GTarget.h"
#import "H16GSRAMChainEncoder.h"
#import "H16GTaskComposer.h"
#import "H16GTaskEncoder.h"
#import "HWXImage.h"
#import "HWXObjectWriter.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

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

static NSUInteger elementCount(ANEValueType *type) {
    NSUInteger count = 1;
    for (NSNumber *dimension in type.shape)
        count *= dimension.unsignedIntegerValue;
    return count;
}

static NSUInteger alignUp(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void shapeDimensions(NSArray<NSNumber *> *shape, NSUInteger *n,
                            NSUInteger *c, NSUInteger *h, NSUInteger *w) {
    *n = 1; *c = 1; *h = 1; *w = 1;
    if (shape.count == 4) {
        *n = shape[0].unsignedIntegerValue;
        *c = shape[1].unsignedIntegerValue;
        *h = shape[2].unsignedIntegerValue;
        *w = shape[3].unsignedIntegerValue;
    } else if (shape.count == 3) {
        *n = shape[0].unsignedIntegerValue;
        *h = shape[1].unsignedIntegerValue;
        *w = shape[2].unsignedIntegerValue;
    } else if (shape.count == 2) {
        *h = shape[0].unsignedIntegerValue;
        *w = shape[1].unsignedIntegerValue;
    } else if (shape.count == 1) {
        *w = shape[0].unsignedIntegerValue;
    }
}

static void physicalLayout(NSArray<NSNumber *> *shape, ANEElementType type,
                           NSUInteger *row, NSUInteger *plane,
                           NSUInteger *batch, NSUInteger *storage) {
    NSUInteger n = 1, c = 1, h = 1, w = 1;
    shapeDimensions(shape, &n, &c, &h, &w);
    *row = MAX(w * elementBytes(type), (NSUInteger)64);
    *plane = h * *row;
    *batch = c * *plane;
    *storage = n * *batch;
}

static ANEValueType *typeForIdentifier(ANEOperationGraph *graph,
                                       NSString *identifier) {
    for (ANEGraphValue *input in graph.sourceFunction.inputs)
        if ([input.name isEqualToString:identifier]) return input.type;
    return [graph nodeForValueName:identifier].outputType;
}

static NSNumber *scalarOperandForStage(ANEOperationGraph *graph,
                                       ANEScheduledStage *stage,
                                       NSArray<NSString *> *tensorInputs);
static NSNumber *scalarOperandForTask(ANEOperationGraph *graph,
                                      ANEScheduledTask *task,
                                      NSArray<NSString *> *tensorInputs);

static ANESourceRange rangeForTask(ANEOperationGraph *graph,
                                   ANEScheduledTask *task) {
    ANEOperationNode *node = [graph
        nodeForValueName:task.stages.firstObject.sourceNodeIdentifier];
    if (node.sourceOperation) return node.sourceOperation.range;
    ANESourceLocation location = ANESourceLocationMake(0, 1, 1);
    return ANESourceRangeMake(location, location);
}

static NSArray<NSString *> *boundaryInputs(ANEOperationGraph *graph,
                                            ANEScheduledTask *task) {
    NSMutableSet<NSString *> *stageOutputs = [NSMutableSet set];
    for (ANEScheduledStage *stage in task.stages)
        [stageOutputs addObject:stage.outputIdentifier];
    NSMutableArray<NSString *> *inputs = [NSMutableArray array];
    NSMutableSet<NSString *> *seen = [NSMutableSet set];
    for (ANEScheduledStage *stage in task.stages) {
        for (NSString *identifier in stage.inputIdentifiers) {
            if ([stageOutputs containsObject:identifier] ||
                [seen containsObject:identifier]) continue;
            ANEOperationNode *producer = [graph nodeForValueName:identifier];
            if (producer && producer.kind == ANEOperationKindConstant) continue;
            ANEValueType *type = typeForIdentifier(graph, identifier);
            if (!type || type.kind != ANEValueTypeKindTensor) continue;
            [seen addObject:identifier];
            [inputs addObject:identifier];
        }
    }
    return [inputs copy];
}

static NSArray<NSString *> *stageOperations(ANEScheduledTask *task) {
    NSMutableArray<NSString *> *operations = [NSMutableArray array];
    for (ANEScheduledStage *stage in task.stages)
        [operations addObject:stage.operationName];
    return [operations copy];
}

static NSArray<NSArray<NSNumber *> *> *shapesForIdentifiers(
    ANEOperationGraph *graph, NSArray<NSString *> *identifiers) {
    NSMutableArray<NSArray<NSNumber *> *> *shapes = [NSMutableArray array];
    for (NSString *identifier in identifiers) {
        ANEValueType *type = typeForIdentifier(graph, identifier);
        if (!type) return nil;
        [shapes addObject:type.shape];
    }
    return [shapes copy];
}

static HWXObjectBinding *objectBinding(NSString *identifier,
                                       NSString *shortName,
                                       NSArray<NSNumber *> *shape,
                                       ANEElementType elementType,
                                       HWXObjectBindingRole role) {
    NSString *symbol = role == HWXObjectBindingRoleOutput
        ? [identifier stringByAppendingString:@"@output"] : identifier;
    NSUInteger row = 0, plane = 0, batch = 0, storage = 0;
    physicalLayout(shape, elementType, &row, &plane, &batch, &storage);
    return [[HWXObjectBinding alloc] initWithSymbol:symbol
        shortName:shortName role:role elementType:elementType shape:shape
        rowStrideBytes:row planeStrideBytes:plane batchStrideBytes:batch
        storageByteLength:storage];
}

static ANEHWXBinding *runtimeBinding(NSString *identifier,
                                     NSArray<NSNumber *> *shape,
                                     ANEElementType elementType,
                                     ANESurfaceRole role,
                                     NSInteger index) {
    ANEValueType *type = [[ANEValueType alloc] initWithKind:ANEValueTypeKindTensor
        elementType:elementType shape:shape];
    NSUInteger bytes = elementCount(type) * elementBytes(elementType);
    NSUInteger rowBytes = 0, planeBytes = 0, batchBytes = 0, storage = 0;
    physicalLayout(shape, elementType, &rowBytes, &planeBytes, &batchBytes,
                   &storage);
    return [[ANEHWXBinding alloc] initWithIdentifier:identifier role:role
        logicalByteLength:bytes allocationByteLength:alignUp(storage, 0x4000)
        ioSurfaceIndex:index rowStrideBytes:rowBytes
        planeStrideBytes:planeBytes batchStrideBytes:batchBytes];
}

static ANEHWXArtifact *artifactForEncodedTask(
    H16GEncodedTask *encoded, ANESourceRange range,
    ANEDiagnosticEngine *diagnostics) {
    if (encoded.inputIdentifiers.count != encoded.inputShapes.count) return nil;
    NSArray<NSString *> *operations = encoded.stageOperations;
    NSMutableArray<HWXObjectBinding *> *objectBindings =
        [NSMutableArray array];
    NSMutableArray<ANEHWXBinding *> *runtimeBindings =
        [NSMutableArray array];
    NSInteger ioIndex = 0;
    for (NSUInteger index = 0; index < encoded.inputIdentifiers.count; ++index) {
        NSString *identifier = encoded.inputIdentifiers[index];
        NSArray<NSNumber *> *shape = encoded.inputShapes[index];
        NSString *shortName = [NSString stringWithFormat:@"i%lu",
            (unsigned long)index];
        [objectBindings addObject:objectBinding(identifier, shortName, shape,
            ANEElementTypeFP16, HWXObjectBindingRoleInput)];
        [runtimeBindings addObject:runtimeBinding(identifier, shape,
            ANEElementTypeFP16, ANESurfaceRoleInput, ioIndex++)];
    }
    if (encoded.outputResourceIndex > objectBindings.count) return nil;
    [objectBindings insertObject:objectBinding(encoded.outputIdentifier, @"o0",
        encoded.outputShape, ANEElementTypeFP16, HWXObjectBindingRoleOutput)
                          atIndex:encoded.outputResourceIndex];
    [runtimeBindings addObject:runtimeBinding(encoded.outputIdentifier,
        encoded.outputShape, ANEElementTypeFP16, ANESurfaceRoleOutput,
        ioIndex)];

    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:encoded.taskCount
        recordCount:encoded.tdProgram.programRecordCount
        formatCode:encoded.tdProgram.programFormatCode
        scratchByteLength:encoded.tdProgram.scratchByteLength
        scratchAllocationByteLength:encoded.scratchAllocationByteLength
        descriptorLayout:encoded.scratchBacked
            ? HWXProgramDescriptorLayoutScratchBackedMixed
            : HWXProgramDescriptorLayoutLinear];
    NSError *error = nil;
    NSData *image = [HWXObjectWriter
        buildObjectWithTaskDescriptor:encoded.tdProgram.data
        constantRegion:encoded.constantRegion bindings:objectBindings
        kernelRelocationOffsets:encoded.tdProgram.kernelRelocationOffsets
        programInfo:info error:&error];
    if (!image) {
        if (diagnostics)
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                code:@"h16g.assemble.object"
                message:error.localizedDescription ?: @"cannot write task object"
                range:range];
        return nil;
    }
    NSError *parseError = nil;
    if (![HWXImage imageWithData:image error:&parseError]) {
        if (diagnostics)
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                code:@"h16g.assemble.invalid-object"
                message:parseError.localizedDescription ?:
                    @"generated task object does not reparse"
                range:range];
        return nil;
    }
    return [[ANEHWXArtifact alloc] initWithImage:image
                                        bindings:runtimeBindings
                                      operations:operations];
}

static H16GEncodedTask *encodedTaskForTask(ANEOperationGraph *graph,
                                           ANEScheduledTask *task,
                                           H16GTarget *target) {
    NSArray<NSString *> *inputs = boundaryInputs(graph, task);
    NSArray<NSArray<NSNumber *> *> *inputShapes =
        shapesForIdentifiers(graph, inputs);
    ANEValueType *outputType = typeForIdentifier(graph, task.nodeIdentifier);
    H16GTaskEncodingRequest *request = inputShapes && outputType
        ? [[H16GTaskEncodingRequest alloc]
            initWithStageOperations:stageOperations(task)
            inputIdentifiers:inputs inputShapes:inputShapes
            outputIdentifier:task.nodeIdentifier outputShape:outputType.shape
            numericMode:task.numericMode
            outputStorage:ANEScheduledBridgeStorageExternal
            scalarOperand:scalarOperandForTask(graph, task, inputs)]
        : nil;
    NSError *error = nil;
    H16GEncodedTask *encoded = request
        ? [H16GTaskEncoder encodeRequest:request target:target error:&error]
        : nil;
    return encoded;
}

static H16GProgramCompositionCapability *compositionCapability(
    H16GTarget *target, H16GEncodedTask *producer,
    H16GEncodedTask *consumer) {
    return [target programCompositionCapabilityFrom:producer.packetFamily
        to:consumer.packetFamily
        producerOperationName:producer.compositionOperationName
        consumerOperationName:consumer.compositionOperationName
        producerGeometry:producer.geometry
        consumerGeometry:consumer.geometry
        bridgeStorage:ANEScheduledBridgeStorageSRAM];
}

static NSString *compositionOperationName(
    H16GTaskCapability *capability) {
    return capability.packetFamily == H16GTaskPacketFamilySquareMatmul
        ? capability.stageOperations.firstObject
        : capability.stageOperations.lastObject;
}

static H16GProgramCompositionCapability *compositionCapability(
    H16GTarget *target, H16GTaskCapability *producer,
    H16GTaskCapability *consumer) {
    return [target programCompositionCapabilityFrom:producer.packetFamily
        to:consumer.packetFamily
        producerOperationName:compositionOperationName(producer)
        consumerOperationName:compositionOperationName(consumer)
        producerGeometry:producer.geometry
        consumerGeometry:consumer.geometry
        bridgeStorage:ANEScheduledBridgeStorageSRAM];
}

static NSString *packetFamilyName(H16GTaskPacketFamily family) {
    switch (family) {
        case H16GTaskPacketFamilySquareMatmul: return @"square-matmul";
        case H16GTaskPacketFamilyUnaryLUT: return @"unary-lut";
        case H16GTaskPacketFamilyScalarScale: return @"scalar-scale";
        case H16GTaskPacketFamilyMatrixRowALU: return @"matrix-row-alu";
        case H16GTaskPacketFamilyRowStateALU: return @"row-state-alu";
        case H16GTaskPacketFamilyReduction: return @"reduction";
        case H16GTaskPacketFamilyMatrixRowDivision:
            return @"matrix-row-division";
        case H16GTaskPacketFamilySquareALU: return @"square-alu";
        case H16GTaskPacketFamilyPrimitiveFallback:
            return @"primitive-fallback";
    }
}

static NSString *capabilitySummary(H16GTaskCapability *capability) {
    return [NSString stringWithFormat:@"%@[%@ n=%lu]",
        packetFamilyName(capability.packetFamily),
        [capability.stageOperations componentsJoinedByString:@","],
        (unsigned long)capability.geometry];
}

static ANEGraphArgument *singleArgumentValue(ANEGraphArgument *argument) {
    while (argument.kind == ANEGraphArgumentKindCall &&
           argument.callArguments.count == 1)
        argument = argument.callArguments.firstObject.value;
    return argument;
}

static NSString *constantText(ANEGraphValue *value) {
    ANEGraphArgument *argument = value.producer
        ? singleArgumentValue(value.producer.attributes[@"val"]) : nil;
    return argument.text;
}

static BOOL falseArgument(ANEGraphArgument *argument) {
    if (!argument) return NO;
    if (argument.kind == ANEGraphArgumentKindValue)
        return [constantText(argument.value) isEqualToString:@"false"];
    return [singleArgumentValue(argument).text isEqualToString:@"false"];
}

static BOOL parseFiniteDouble(NSString *text, double *value) {
    if (!text) return NO;
    char *end = nullptr;
    double parsed = std::strtod(text.UTF8String, &end);
    if (!end || *end != '\0' || !std::isfinite(parsed)) return NO;
    *value = parsed;
    return YES;
}

static BOOL parseIntegerArgument(ANEGraphArgument *argument,
                                 NSInteger *value) {
    if (!argument) return NO;
    NSString *text = argument.kind == ANEGraphArgumentKindValue
        ? constantText(argument.value) : singleArgumentValue(argument).text;
    if (!text) return NO;
    char *end = nullptr;
    long long parsed = std::strtoll(text.UTF8String, &end, 10);
    if (!end || *end != '\0') return NO;
    *value = (NSInteger)parsed;
    return YES;
}

static ANEOperationNode *stageNodeNamed(ANEOperationGraph *graph,
                                        ANEScheduledTask *task,
                                        NSString *operationName) {
    for (ANEScheduledStage *stage in task.stages) {
        if (![stage.operationName isEqualToString:operationName]) continue;
        ANEOperationNode *node = [graph
            nodeForValueName:stage.sourceNodeIdentifier];
        if (node) return node;
    }
    return nil;
}

static BOOL validateMatmulSemantics(ANEOperationGraph *graph,
                                    ANEScheduledTask *task,
                                    NSArray<NSString *> *tensorInputs) {
    ANEOperationNode *node = stageNodeNamed(graph, task, @"matmul");
    ANEGraphOperation *operation = node.sourceOperation;
    if (!operation ||
        !falseArgument(operation.arguments[@"transpose_x"]) ||
        !falseArgument(operation.arguments[@"transpose_y"])) return NO;
    NSString *x = operation.operands[@"x"].value.name;
    NSString *y = operation.operands[@"y"].value.name;
    return x && y && tensorInputs.count >= 2 &&
        [tensorInputs[0] isEqualToString:x] &&
        [tensorInputs[1] isEqualToString:y];
}

/// Finds the constant scalar operand of a binary stage whose other operand
/// is the task's single tensor input. Returns nil when the stage is not of
/// that form or the scalar is not a finite constant.
static NSNumber *scalarOperandForStage(ANEOperationGraph *graph,
                                       ANEScheduledStage *stage,
                                       NSArray<NSString *> *tensorInputs) {
    ANEOperationNode *node = [graph nodeForValueName:stage.sourceNodeIdentifier];
    ANEGraphOperation *operation = node.sourceOperation;
    if (!operation) return nil;
    ANEGraphArgument *x = operation.operands[@"x"];
    ANEGraphArgument *y = operation.operands[@"y"];
    if (!x || !y) return nil;
    ANEGraphValue *tensor = x.value.type.kind == ANEValueTypeKindTensor
        ? x.value : (y.value.type.kind == ANEValueTypeKindTensor ? y.value : nil);
    ANEGraphValue *scalar = x.value.type.kind == ANEValueTypeKindScalar
        ? x.value : (y.value.type.kind == ANEValueTypeKindScalar ? y.value : nil);
    double value = 0.0;
    if (!tensor || !scalar || tensorInputs.count != 1 ||
        ![tensorInputs.firstObject isEqualToString:tensor.name] ||
        !parseFiniteDouble(constantText(scalar), &value)) return nil;
    return @(value);
}

static NSNumber *scalarOperandForTask(ANEOperationGraph *graph,
                                      ANEScheduledTask *task,
                                      NSArray<NSString *> *tensorInputs) {
    for (ANEScheduledStage *stage in task.stages) {
        NSNumber *scalar = scalarOperandForStage(graph, stage, tensorInputs);
        if (scalar) return scalar;
    }
    return nil;
}

static BOOL validateScalarOperandSemantics(ANEOperationGraph *graph,
                                           ANEScheduledTask *task,
                                           H16GTaskCapability *capability,
                                           NSArray<NSString *> *tensorInputs) {
    return capability.geometry != 0 &&
        scalarOperandForTask(graph, task, tensorInputs) != nil;
}

static BOOL validateLastAxisSemantics(ANEOperationGraph *graph,
                                      ANEScheduledTask *task) {
    ANEOperationNode *node = [graph
        nodeForValueName:task.stages.firstObject.sourceNodeIdentifier];
    ANEGraphOperation *operation = node.sourceOperation;
    ANEGraphArgument *input = operation.operands[@"x"] ?:
                              operation.operands[@"input"];
    ANEGraphArgument *axis = operation.arguments[@"axis"] ?:
                            operation.arguments[@"axes"];
    NSInteger value = 0;
    if (!operation || !input.value || input.value.type.shape.count == 0)
        return NO;
    if (axis.kind == ANEGraphArgumentKindList && axis.elements.count == 1)
        axis = axis.elements.firstObject;
    if (!parseIntegerArgument(axis, &value)) return NO;
    NSInteger rank = (NSInteger)input.value.type.shape.count;
    if (value < 0) value += rank;
    return value == rank - 1;
}

static BOOL validateTaskSemantics(ANEOperationGraph *graph,
                                  ANEScheduledTask *task,
                                  H16GTaskCapability *capability,
                                  NSArray<NSString *> *tensorInputs) {
    switch (capability.semanticConstraint) {
        case H16GTaskSemanticConstraintNone:
            return YES;
        case H16GTaskSemanticConstraintNonTransposedMatmul:
            return validateMatmulSemantics(graph, task, tensorInputs);
        case H16GTaskSemanticConstraintScalarOperand:
            return validateScalarOperandSemantics(graph, task, capability,
                                                  tensorInputs);
        case H16GTaskSemanticConstraintLastAxis:
            return validateLastAxisSemantics(graph, task);
    }
}

static H16GTaskCapability *taskCapabilityForTask(
    ANEOperationGraph *graph, ANEScheduledTask *task, H16GTarget *target,
    NSArray<NSString *> **inputsOut,
    NSArray<NSArray<NSNumber *> *> **inputShapesOut) {
    NSArray<NSString *> *inputs = boundaryInputs(graph, task);
    NSArray<NSArray<NSNumber *> *> *inputShapes =
        shapesForIdentifiers(graph, inputs);
    ANEValueType *outputType = typeForIdentifier(graph, task.nodeIdentifier);
    ANEScheduledBridgeStorage outputStorage =
        [graph.outputValueNames containsObject:task.nodeIdentifier]
            ? ANEScheduledBridgeStorageExternal
            : ANEScheduledBridgeStorageSRAM;
    H16GTaskCapability *capability = inputShapes && outputType
        ? [target taskCapabilityForStageOperations:stageOperations(task)
            inputShapes:inputShapes outputShape:outputType.shape
            numericMode:task.numericMode
            outputStorage:outputStorage]
        : nil;
    if (!capability && inputShapes && outputType &&
        outputStorage != ANEScheduledBridgeStorageExternal)
        capability = [target
            taskCapabilityForStageOperations:stageOperations(task)
            inputShapes:inputShapes outputShape:outputType.shape
            numericMode:task.numericMode
            outputStorage:ANEScheduledBridgeStorageExternal];
    if (capability && !validateTaskSemantics(graph, task, capability, inputs))
        capability = nil;
    if (inputsOut) *inputsOut = inputs;
    if (inputShapesOut) *inputShapesOut = inputShapes;
    return capability;
}

static NSArray<NSString *> *tensorInputsForIdentifiers(
    ANEOperationGraph *graph, NSArray<NSString *> *identifiers) {
    NSMutableArray<NSString *> *result = [NSMutableArray array];
    for (NSString *identifier in identifiers) {
        ANEOperationNode *node = [graph nodeForValueName:identifier];
        if (node && node.kind == ANEOperationKindConstant) continue;
        ANEValueType *type = typeForIdentifier(graph, identifier);
        if (type && type.kind == ANEValueTypeKindTensor)
            [result addObject:identifier];
    }
    return [result copy];
}

static NSArray<NSNumber *> *externalShapeForIdentifier(
    ANEOperationGraph *graph, NSString *identifier) {
    ANEValueType *type = typeForIdentifier(graph, identifier);
    if (!type) return nil;
    ANEOperationNode *node = [graph nodeForValueName:identifier];
    BOOL rowReduction =
        [node.operationName isEqualToString:@"reduce_max"] ||
        [node.operationName isEqualToString:@"reduce_sum"];
    if (rowReduction && type.shape.count == 4 &&
        type.shape[0].unsignedIntegerValue == 1 &&
        type.shape[1].unsignedIntegerValue == 1)
        return @[@1, @1, type.shape[2], @1];
    return type.shape;
}

static H16GEncodedTask *encodeExternalPrimitive(
    ANEOperationGraph *graph, H16GTarget *target,
    NSArray<NSString *> *operations, NSArray<NSString *> *inputs,
    NSString *output, NSNumber *scalarOperand) {
    NSMutableArray<NSArray<NSNumber *> *> *inputShapes =
        [NSMutableArray array];
    for (NSString *identifier in inputs) {
        NSArray<NSNumber *> *shape = externalShapeForIdentifier(
            graph, identifier);
        if (!shape) return nil;
        [inputShapes addObject:shape];
    }
    ANEValueType *outputType = typeForIdentifier(graph, output);
    if (!inputShapes || !outputType) return nil;
    NSArray<NSNumber *> *outputShape = externalShapeForIdentifier(
        graph, output);
    if (!outputShape) return nil;
    H16GTaskEncodingRequest *request = [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:operations inputIdentifiers:inputs
        inputShapes:inputShapes outputIdentifier:output
        outputShape:outputShape numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal
        scalarOperand:scalarOperand];
    NSError *error = nil;
    return [H16GTaskEncoder encodeRequest:request target:target error:&error];
}

static NSArray<ANEHWXArtifact *> *externalArtifactsForPartition(
    ANEOperationGraph *graph, ANEProgramPartition *partition,
    NSDictionary<NSNumber *, ANEScheduledTask *> *scheduleByIndex,
    H16GTarget *target, ANEDiagnosticEngine *diagnostics) {
    NSMutableArray<H16GEncodedTask *> *taskEncodings = [NSMutableArray array];
    BOOL allTasksEncode = YES;
    for (NSNumber *taskIndex in partition.scheduledTaskIndexes) {
        ANEScheduledTask *task = scheduleByIndex[taskIndex];
        H16GEncodedTask *encoded = task
            ? encodedTaskForTask(graph, task, target) : nil;
        if (!encoded) {
            allTasksEncode = NO;
            break;
        }
        [taskEncodings addObject:encoded];
    }
    if (allTasksEncode) {
        NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
        for (NSUInteger index = 0; index < taskEncodings.count; ++index) {
            ANEScheduledTask *task = scheduleByIndex[
                partition.scheduledTaskIndexes[index]];
            ANEHWXArtifact *artifact = artifactForEncodedTask(
                taskEncodings[index], rangeForTask(graph, task), diagnostics);
            if (!artifact) return nil;
            [artifacts addObject:artifact];
        }
        return [artifacts copy];
    }

    NSMutableArray<ANEScheduledStage *> *stages = [NSMutableArray array];
    for (NSNumber *taskIndex in partition.scheduledTaskIndexes) {
        ANEScheduledTask *task = scheduleByIndex[taskIndex];
        if (!task) return nil;
        [stages addObjectsFromArray:task.stages];
    }
    NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
    for (NSUInteger index = 0; index < stages.count; ++index) {
        ANEScheduledStage *stage = stages[index];
        H16GEncodedTask *encoded = nil;
        ANEScheduledStage *rangeStage = stage;
        if ([stage.operationName isEqualToString:@"reciprocal"]) {
            if (index + 1 >= stages.count) return nil;
            ANEScheduledStage *multiply = stages[index + 1];
            NSArray<NSString *> *multiplyInputs = tensorInputsForIdentifiers(
                graph, multiply.inputIdentifiers);
            if (![multiply.operationName isEqualToString:@"mul"] ||
                ![multiplyInputs containsObject:stage.outputIdentifier])
                return nil;
            NSString *numerator = nil;
            for (NSString *identifier in multiplyInputs)
                if (![identifier isEqualToString:stage.outputIdentifier]) {
                    numerator = identifier;
                    break;
                }
            NSArray<NSString *> *denominators = tensorInputsForIdentifiers(
                graph, stage.inputIdentifiers);
            if (!numerator || denominators.count != 1) return nil;
            encoded = encodeExternalPrimitive(graph, target, @[@"real_div"],
                @[numerator, denominators.firstObject],
                multiply.outputIdentifier, nil);
            rangeStage = multiply;
            ++index;
        } else {
            NSArray<NSString *> *inputs = tensorInputsForIdentifiers(
                graph, stage.inputIdentifiers);
            encoded = encodeExternalPrimitive(graph, target,
                @[stage.operationName], inputs, stage.outputIdentifier,
                scalarOperandForStage(graph, stage, inputs));
        }
        ANEOperationNode *node = [graph
            nodeForValueName:rangeStage.sourceNodeIdentifier];
        ANESourceRange range = node.sourceOperation
            ? node.sourceOperation.range
            : rangeForTask(graph, scheduleByIndex[
                partition.scheduledTaskIndexes.firstObject]);
        ANEHWXArtifact *artifact = encoded
            ? artifactForEncodedTask(encoded, range, diagnostics) : nil;
        if (!artifact) return nil;
        [artifacts addObject:artifact];
    }
    return [artifacts copy];
}

static H16GEncodedTask *encodedSRAMProgramForPartition(
    ANEOperationGraph *graph, ANEProgramPartition *partition,
    NSDictionary<NSNumber *, ANEScheduledTask *> *scheduleByIndex) {
    NSMutableArray<ANEScheduledStage *> *stages = [NSMutableArray array];
    for (NSNumber *taskIndex in partition.scheduledTaskIndexes) {
        ANEScheduledTask *task = scheduleByIndex[taskIndex];
        if (!task) return nil;
        [stages addObjectsFromArray:task.stages];
    }
    NSMutableArray<NSString *> *operations = [NSMutableArray array];
    for (ANEScheduledStage *stage in stages)
        [operations addObject:stage.operationName];
    NSUInteger taskCount = [H16GSRAMChainEncoder
        taskCountForStageOperations:operations];
    if (taskCount == 0 || partition.boundaryOutputIdentifiers.count != 1)
        return nil;

    NSMutableArray<NSArray<NSNumber *> *> *inputShapes =
        [NSMutableArray array];
    for (NSString *identifier in partition.boundaryInputIdentifiers) {
        NSArray<NSNumber *> *shape = externalShapeForIdentifier(
            graph, identifier);
        if (!shape) return nil;
        [inputShapes addObject:shape];
    }
    if (inputShapes.count == 0) return nil;
    NSArray<NSNumber *> *matrix = inputShapes.lastObject;
    if (matrix.count != 4 || matrix[0].unsignedIntegerValue != 1 ||
        matrix[1].unsignedIntegerValue != 1 ||
        matrix[2].unsignedIntegerValue != matrix[3].unsignedIntegerValue)
        return nil;
    NSUInteger geometry = matrix[2].unsignedIntegerValue;
    NSString *output = partition.boundaryOutputIdentifiers.firstObject;
    NSArray<NSNumber *> *outputShape = externalShapeForIdentifier(
        graph, output);
    if (!outputShape) return nil;

    NSError *error = nil;
    H16GEncodedTDProgram *program = [H16GSRAMChainEncoder
        encodeStageOperations:operations rows:geometry columns:geometry
        error:&error];
    NSData *constants = program ? [H16GSRAMChainEncoder
        constantRegionForStageOperations:operations error:&error] : nil;
    if (!program || !constants) return nil;
    return [[H16GEncodedTask alloc]
        initWithTDProgram:program constantRegion:constants
        stageOperations:operations
        inputIdentifiers:partition.boundaryInputIdentifiers
        inputShapes:inputShapes outputIdentifier:output
        outputShape:outputShape taskCount:taskCount
        scratchBacked:NO scratchAllocationByteLength:0
        outputResourceIndex:partition.boundaryInputIdentifiers.count
        packetFamily:H16GTaskPacketFamilyPrimitiveFallback
        geometry:geometry numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal
        compositionOperationName:operations.lastObject];
}

static H16GAssembledProgram *assemblePartitionedTasks(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    H16GTarget *target, ANEDiagnosticEngine *diagnostics) {
    NSMutableArray<ANEProgramTaskDescriptor *> *descriptors =
        [NSMutableArray array];
    NSMutableDictionary<NSNumber *, H16GEncodedTask *> *tasksByIndex =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSNumber *, H16GTaskCapability *> *capabilitiesByIndex =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSNumber *, ANEScheduledTask *> *scheduleByIndex =
        [NSMutableDictionary dictionary];
    H16GTaskCapability *previousCapability = nil;
    for (ANEScheduledTask *task in scheduled.tasks) {
        NSArray<NSString *> *inputs = nil;
        NSArray<NSArray<NSNumber *> *> *inputShapes = nil;
        H16GTaskCapability *capability = taskCapabilityForTask(
            graph, task, target, &inputs, &inputShapes);
        if (!capability) return nil;
        H16GEncodedTask *encoded = encodedTaskForTask(graph, task, target);
        ANEValueType *outputType = typeForIdentifier(
            graph, task.nodeIdentifier);
        if (!outputType) return nil;
        if (encoded) tasksByIndex[@(task.index)] = encoded;
        capabilitiesByIndex[@(task.index)] = capability;
        scheduleByIndex[@(task.index)] = task;
        NSUInteger composedContribution = capability.programTaskCount;
        H16GProgramCompositionCapability *transition = previousCapability
            ? compositionCapability(target, previousCapability, capability)
            : nil;
        if (transition)
            composedContribution = transition.consumerTaskCountContribution;
        [descriptors addObject:[[ANEProgramTaskDescriptor alloc]
            initWithScheduledTaskIndex:task.index
            inputIdentifiers:inputs
            outputIdentifier:task.nodeIdentifier
            encodedTaskCount:capability.standaloneTaskCount
            composedTaskCountContribution:composedContribution
            outputByteLength:elementCount(outputType) *
                elementBytes(outputType.elementType)]];
        previousCapability = capability;
    }
    NSUInteger maximumInputs = target.maximumProgramInputCount.available
        ? target.maximumProgramInputCount.value : NSUIntegerMax;
    NSNumber *maximumTasks = target.maximumProgramTaskCount.available
        ? @(target.maximumProgramTaskCount.value) : nil;
    // The forced standalone path proves every program is usable on its own.
    // It is a runtime switch, not a graph property, so it lives here.
    const char *disableComposition = getenv("ANE_DISABLE_PROGRAM_COMPOSITION");
    BOOL compositionDisabled = disableComposition && disableComposition[0] &&
        strcmp(disableComposition, "0") != 0;
    NSMutableArray<NSString *> *trace = [NSMutableArray array];
    if (compositionDisabled)
        [trace addObject:@"composition disabled by ANE_DISABLE_PROGRAM_COMPOSITION; "
                          "only decoded primitive pairs stay together"];
    NSMutableArray<ANEProgramTransitionRecord *> *transitions =
        [NSMutableArray array];
    NSArray<ANEProgramPartition *> *partitions = [ANEProgramPartitionPlanner
        partitionsForTasks:descriptors
        finalOutputIdentifiers:graph.outputValueNames
        maximumInputCount:maximumInputs maximumTaskCount:maximumTasks
        workingSetBytes:target.workingSetBytes
        reasonedCanCompose:^BOOL(ANEProgramTaskDescriptor *producerDescriptor,
                                 ANEProgramTaskDescriptor *consumerDescriptor,
                                 NSString **reason) {
            H16GTaskCapability *producer = capabilitiesByIndex[
                @(producerDescriptor.scheduledTaskIndex)];
            H16GTaskCapability *consumer = capabilitiesByIndex[
                @(consumerDescriptor.scheduledTaskIndex)];
            H16GProgramCompositionCapability *capability =
                compositionCapability(target, producer, consumer);
            if (!capability) {
                *reason = [NSString stringWithFormat:
                    @"no H16G composition capability row for %@ -> %@ "
                     "with an SRAM bridge",
                    capabilitySummary(producer), capabilitySummary(consumer)];
                return NO;
            }
            if (![H16GTaskComposer supportsCapability:capability] &&
                capability.action !=
                    H16GProgramCompositionActionPrimitiveFallback) {
                *reason = [NSString stringWithFormat:
                    @"composition action for %@ -> %@ has no decoded task "
                     "composer", capabilitySummary(producer),
                    capabilitySummary(consumer)];
                return NO;
            }
            // The forced standalone path keeps decoded primitive pairs (one
            // standalone program by definition) and declines everything else.
            if (compositionDisabled &&
                capability.action !=
                    H16GProgramCompositionActionPrimitiveFallback) {
                *reason = @"composition disabled by ANE_DISABLE_PROGRAM_COMPOSITION";
                return NO;
            }
            return YES;
        }
        transitions:transitions];
    for (ANEProgramTransitionRecord *transition in transitions) {
        H16GTaskCapability *producer =
            capabilitiesByIndex[@(transition.producerTaskIndex)];
        H16GTaskCapability *consumer =
            capabilitiesByIndex[@(transition.consumerTaskIndex)];
        [trace addObject:[NSString stringWithFormat:
            @"transition producer=%lu:%@ consumer=%lu:%@ result=%@ reason=%@",
            (unsigned long)transition.producerTaskIndex,
            capabilitySummary(producer),
            (unsigned long)transition.consumerTaskIndex,
            capabilitySummary(consumer),
            transition.accepted ? @"composed" : @"declined",
            transition.reason]];
    }
    NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
    NSMutableArray<NSNumber *> *dispatch = [NSMutableArray array];
    for (ANEProgramPartition *partition in partitions) {
        [trace addObject:[NSString stringWithFormat:@"partition %@",
                          partition.textualDescription]];
        H16GEncodedTask *region = compositionDisabled ? nil :
            encodedSRAMProgramForPartition(graph, partition, scheduleByIndex);
        if (region) {
            ANEScheduledTask *firstScheduled = scheduleByIndex[
                partition.scheduledTaskIndexes.firstObject];
            ANEHWXArtifact *artifact = artifactForEncodedTask(
                region, rangeForTask(graph, firstScheduled), diagnostics);
            if (!artifact) return nil;
            [artifacts addObject:artifact];
            [dispatch addObject:@(artifacts.count - 1)];
            [trace addObject:[NSString stringWithFormat:
                @"program %lu composed tasks=%@ operations=%@",
                (unsigned long)(artifacts.count - 1),
                [partition.scheduledTaskIndexes componentsJoinedByString:@","],
                [region.stageOperations componentsJoinedByString:@","]]];
            continue;
        }
        BOOL everyLeafEncodes = YES;
        for (NSNumber *taskIndex in partition.scheduledTaskIndexes)
            everyLeafEncodes = everyLeafEncodes &&
                tasksByIndex[taskIndex] && scheduleByIndex[taskIndex];
        if (!everyLeafEncodes) {
            NSArray<ANEHWXArtifact *> *fallback = externalArtifactsForPartition(
                graph, partition, scheduleByIndex, target, diagnostics);
            if (!fallback) return nil;
            for (ANEHWXArtifact *artifact in fallback) {
                [artifacts addObject:artifact];
                [dispatch addObject:@(artifacts.count - 1)];
            }
            [trace addObject:[NSString stringWithFormat:
                @"programs %lu..%lu standalone tasks=%@",
                (unsigned long)(artifacts.count - fallback.count),
                (unsigned long)(artifacts.count - 1),
                [partition.scheduledTaskIndexes componentsJoinedByString:@","]]];
            continue;
        }
        // Compose leaves left to right. When a transition the planner
        // accepted cannot be assembled, the program built so far is emitted
        // as-is and assembly restarts from the failing leaf, so the value at
        // the split becomes an ordinary shared surface.
        __block H16GEncodedTask *combined = nil;
        H16GEncodedTask *previousLeaf = nil;
        __block ANEScheduledTask *firstScheduled = nil;
        __block NSMutableArray<NSNumber *> *programTasks =
            [NSMutableArray array];
        BOOL (^emitCombined)(void) = ^BOOL(void) {
            ANEHWXArtifact *artifact = artifactForEncodedTask(
                combined, rangeForTask(graph, firstScheduled), diagnostics);
            if (!artifact) return NO;
            [artifacts addObject:artifact];
            [dispatch addObject:@(artifacts.count - 1)];
            [trace addObject:[NSString stringWithFormat:
                @"program %lu %@ tasks=%@ operations=%@",
                (unsigned long)(artifacts.count - 1),
                programTasks.count > 1 ? @"composed" : @"single",
                [programTasks componentsJoinedByString:@","],
                [combined.stageOperations componentsJoinedByString:@","]]];
            return YES;
        };
        for (NSNumber *taskIndex in partition.scheduledTaskIndexes) {
            H16GEncodedTask *leaf = tasksByIndex[taskIndex];
            ANEScheduledTask *scheduledTask = scheduleByIndex[taskIndex];
            if (combined) {
                H16GProgramCompositionCapability *capability =
                    compositionCapability(target, previousLeaf, leaf);
                NSError *error = nil;
                H16GEncodedTask *next = [H16GTaskComposer
                    composeProducer:combined consumer:leaf
                    capability:capability error:&error];
                if (next) {
                    combined = next;
                    previousLeaf = leaf;
                    [programTasks addObject:taskIndex];
                    continue;
                }
                [trace addObject:[NSString stringWithFormat:
                    @"assembly split before task %@: %@", taskIndex,
                    error.localizedDescription ?: @"composer declined"]];
                if (!emitCombined()) return nil;
            }
            combined = leaf;
            previousLeaf = leaf;
            firstScheduled = scheduledTask;
            programTasks = [NSMutableArray arrayWithObject:taskIndex];
        }
        if (combined && !emitCombined()) return nil;
    }
    return [[H16GAssembledProgram alloc] initWithArtifacts:artifacts
                                              dispatchPlan:dispatch
                                          compositionTrace:trace];
}

@implementation H16GAssembledProgram
- (instancetype)initWithArtifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                      dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan {
    return [self initWithArtifacts:artifacts dispatchPlan:dispatchPlan
                  compositionTrace:@[]];
}
- (instancetype)initWithArtifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                      dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan
                  compositionTrace:(NSArray<NSString *> *)compositionTrace {
    self = [super init];
    if (self) {
        _artifacts = [artifacts copy];
        _dispatchPlan = [dispatchPlan copy];
        _compositionTrace = [compositionTrace copy];
    }
    return self;
}
@end

@implementation H16GProgramAssembler
+ (H16GAssembledProgram *)assembleGraph:(ANEOperationGraph *)graph
                               scheduled:(ANEScheduledGraph *)scheduled
                                  target:(H16GTarget *)target
                             diagnostics:(ANEDiagnosticEngine *)diagnostics {
    return assemblePartitionedTasks(graph, scheduled, target, diagnostics);
}
@end
