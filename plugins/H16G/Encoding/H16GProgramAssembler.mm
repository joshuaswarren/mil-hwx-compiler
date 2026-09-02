#import "H16GProgramAssembler.h"

#import "ANEHWXArtifact.h"
#import "H16GEncodedTask.h"
#import "H16GTarget.h"
#import "H16GTaskEncoder.h"
#import "HWXObjectWriter.h"

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
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
            code:@"h16g.assemble.object"
            message:error.localizedDescription ?: @"cannot write task object"
            range:range];
        return nil;
    }
    return [[ANEHWXArtifact alloc] initWithImage:image
                                        bindings:runtimeBindings];
}

static ANEHWXArtifact *artifactForTask(ANEOperationGraph *graph,
                                       ANEScheduledTask *task,
                                       H16GTarget *target,
                                       ANEDiagnosticEngine *diagnostics) {
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
            outputStorage:ANEScheduledBridgeStorageExternal]
        : nil;
    NSError *error = nil;
    H16GEncodedTask *encoded = request
        ? [H16GTaskEncoder encodeRequest:request target:target error:&error]
        : nil;
    if (!encoded) return nil;

    return artifactForEncodedTask(encoded, rangeForTask(graph, task),
                                  diagnostics);
}

static BOOL planCoversAllTasks(ANEScheduledRegionPlan *plan,
                               ANEScheduledGraph *scheduled) {
    if (plan.taskIndexes.count != scheduled.tasks.count) return NO;
    for (NSUInteger index = 0; index < scheduled.tasks.count; ++index)
        if (![plan.taskIndexes containsObject:@(index)]) return NO;
    return YES;
}

static H16GTaskEncodingRequest *primitiveRequest(
    NSArray<NSString *> *operations, NSArray<NSString *> *inputs,
    NSArray<NSArray<NSNumber *> *> *inputShapes, NSString *output,
    NSArray<NSNumber *> *outputShape) {
    return [[H16GTaskEncodingRequest alloc]
        initWithStageOperations:operations inputIdentifiers:inputs
        inputShapes:inputShapes outputIdentifier:output outputShape:outputShape
        numericMode:ANELegalNumericModeFP16
        outputStorage:ANEScheduledBridgeStorageExternal];
}

static H16GAssembledProgram *assembleSingleTileOnlineReduction(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEScheduledRegionPlan *plan, H16GTarget *target,
    ANEDiagnosticEngine *diagnostics) {
    if (plan.queryTileCount != 1 || plan.keyValueTileCount != 1 ||
        plan.tileRows != 128 || plan.stageIdentifiers.count != 9 ||
        plan.boundaryInputIdentifiers.count != 3 ||
        !planCoversAllTasks(plan, scheduled)) return nil;
    NSArray<NSNumber *> *matrix = @[@1,@1,@128,@128];
    NSArray<NSNumber *> *row = @[@1,@1,@128,@1];
    for (NSString *identifier in plan.boundaryInputIdentifiers) {
        ANEValueType *type = typeForIdentifier(graph, identifier);
        if (type.elementType != ANEElementTypeFP16 ||
            ![type.shape isEqualToArray:matrix]) return nil;
    }
    ANEValueType *outputType = typeForIdentifier(graph, plan.outputIdentifier);
    if (outputType.elementType != ANEElementTypeFP16 ||
        ![outputType.shape isEqualToArray:matrix]) return nil;

    NSString *score = plan.stageIdentifiers[0];
    NSString *scaled = plan.stageIdentifiers[1];
    NSString *maximum = plan.stageIdentifiers[2];
    NSString *centered = plan.stageIdentifiers[3];
    NSString *exponential = plan.stageIdentifiers[4];
    NSString *sum = plan.stageIdentifiers[5];
    NSString *probabilities = plan.stageIdentifiers[7];
    NSString *query = plan.boundaryInputIdentifiers[0];
    NSString *key = plan.boundaryInputIdentifiers[1];
    NSString *value = plan.boundaryInputIdentifiers[2];
    NSArray<H16GTaskEncodingRequest *> *requests = @[
        primitiveRequest(@[@"matmul"], @[query,key], @[matrix,matrix],
                         score, matrix),
        primitiveRequest(@[@"mul"], @[score], @[matrix], scaled, matrix),
        primitiveRequest(@[@"reduce_max"], @[scaled], @[matrix], maximum, row),
        primitiveRequest(@[@"sub"], @[scaled,maximum], @[matrix,row],
                         centered, matrix),
        primitiveRequest(@[@"exp"], @[centered], @[matrix], exponential,
                         matrix),
        primitiveRequest(@[@"reduce_sum"], @[exponential], @[matrix], sum, row),
        primitiveRequest(@[@"real_div"], @[exponential,sum], @[matrix,row],
                         probabilities, matrix),
        primitiveRequest(@[@"matmul"], @[probabilities,value], @[matrix,matrix],
                         plan.outputIdentifier, matrix),
    ];
    NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
    NSMutableArray<NSNumber *> *dispatch = [NSMutableArray array];
    for (NSUInteger index = 0; index < requests.count; ++index) {
        NSError *error = nil;
        H16GEncodedTask *encoded = [H16GTaskEncoder
            encodeRequest:requests[index] target:target error:&error];
        ANEOperationNode *node = [graph
            nodeForValueName:requests[index].outputIdentifier];
        ANESourceRange range = node.sourceOperation
            ? node.sourceOperation.range : rangeForTask(graph,
                scheduled.tasks.firstObject);
        ANEHWXArtifact *artifact = encoded
            ? artifactForEncodedTask(encoded, range, diagnostics) : nil;
        if (!artifact) return nil;
        [artifacts addObject:artifact];
        [dispatch addObject:@(index)];
    }
    return [[H16GAssembledProgram alloc] initWithArtifacts:artifacts
                                              dispatchPlan:dispatch];
}

static H16GAssembledProgram *assembleScheduledPlan(
    ANEOperationGraph *graph, ANEScheduledGraph *scheduled,
    ANEScheduledRegionPlan *plan, H16GTarget *target,
    ANEDiagnosticEngine *diagnostics) {
    if (!planCoversAllTasks(plan, scheduled)) return nil;
    NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
    NSMutableArray<NSNumber *> *dispatch = [NSMutableArray array];
    for (NSNumber *rawIndex in plan.taskIndexes) {
        NSUInteger taskIndex = rawIndex.unsignedIntegerValue;
        if (taskIndex >= scheduled.tasks.count) return nil;
        ANEHWXArtifact *artifact = artifactForTask(
            graph, scheduled.tasks[taskIndex], target, diagnostics);
        if (!artifact) return nil;
        [artifacts addObject:artifact];
        [dispatch addObject:@(artifacts.count - 1)];
    }
    return [[H16GAssembledProgram alloc] initWithArtifacts:artifacts
                                              dispatchPlan:dispatch];
}

@implementation H16GAssembledProgram
- (instancetype)initWithArtifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                      dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan {
    self = [super init];
    if (self) {
        _artifacts = [artifacts copy];
        _dispatchPlan = [dispatchPlan copy];
    }
    return self;
}
@end

@implementation H16GProgramAssembler
+ (H16GAssembledProgram *)assembleGraph:(ANEOperationGraph *)graph
                               scheduled:(ANEScheduledGraph *)scheduled
                                  target:(H16GTarget *)target
                             diagnostics:(ANEDiagnosticEngine *)diagnostics {
    for (ANEScheduledRegionPlan *plan in scheduled.composedPlans) {
        if (plan.topology != ANEScheduledTopologyOnlineReduction) continue;
        H16GAssembledProgram *program = assembleSingleTileOnlineReduction(
            graph, scheduled, plan, target, diagnostics);
        if (program) return program;
    }
    for (ANEScheduledRegionPlan *plan in scheduled.composedPlans) {
        if (plan.topology != ANEScheduledTopologyAssociativeScan) continue;
        H16GAssembledProgram *program = assembleScheduledPlan(
            graph, scheduled, plan, target, diagnostics);
        if (program) return program;
    }
    NSMutableIndexSet *candidateIndexes = [NSMutableIndexSet indexSet];
    for (ANEScheduledRegionPlan *plan in scheduled.composedPlans) {
        if (plan.topology != ANEScheduledTopologyDirect ||
            plan.taskIndexes.count < 2) continue;
        for (NSNumber *index in plan.taskIndexes)
            [candidateIndexes addIndex:index.unsignedIntegerValue];
    }
    if (candidateIndexes.count != scheduled.tasks.count ||
        candidateIndexes.firstIndex != 0 ||
        candidateIndexes.lastIndex != scheduled.tasks.count - 1) return nil;

    NSMutableArray<ANEHWXArtifact *> *artifacts = [NSMutableArray array];
    NSMutableArray<NSNumber *> *dispatchPlan = [NSMutableArray array];
    for (NSUInteger index = 0; index < scheduled.tasks.count; ++index) {
        ANEHWXArtifact *artifact = artifactForTask(
            graph, scheduled.tasks[index], target, diagnostics);
        if (!artifact) return nil;
        [artifacts addObject:artifact];
        [dispatchPlan addObject:@(index)];
    }
    return [[H16GAssembledProgram alloc] initWithArtifacts:artifacts
                                              dispatchPlan:dispatchPlan];
}
@end
