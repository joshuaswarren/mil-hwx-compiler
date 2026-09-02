#import "ANEComposedRegionPlanner.h"

#include <cmath>
#include <cstdlib>

static NSDictionary<NSString *, ANEScheduledTask *> *taskByNode(
    NSArray<ANEScheduledTask *> *tasks) {
    NSMutableDictionary<NSString *, ANEScheduledTask *> *result =
        [NSMutableDictionary dictionary];
    for (ANEScheduledTask *task in tasks)
        for (NSString *identifier in task.sourceNodeIdentifiers)
            result[identifier] = task;
    return [result copy];
}

static NSArray<NSNumber *> *taskIndexes(
    NSArray<ANEOperationNode *> *nodes,
    NSDictionary<NSString *, ANEScheduledTask *> *tasksByNode) {
    NSMutableSet<NSNumber *> *indexes = [NSMutableSet set];
    for (ANEOperationNode *node in nodes) {
        ANEScheduledTask *task = tasksByNode[node.identifier];
        if (!task) return nil;
        [indexes addObject:@(task.index)];
    }
    return [[indexes allObjects] sortedArrayUsingSelector:@selector(compare:)];
}

static NSString *commonRegion(
    NSArray<NSNumber *> *indexes, NSArray<ANEScheduledTask *> *tasks) {
    NSString *region = nil;
    for (NSNumber *index in indexes) {
        if (index.unsignedIntegerValue >= tasks.count) return nil;
        NSString *candidate = tasks[index.unsignedIntegerValue].regionIdentifier;
        if (!region) region = candidate;
        else if (![region isEqualToString:candidate]) return nil;
    }
    return region;
}

static NSArray<NSString *> *identifiers(NSArray<ANEOperationNode *> *nodes) {
    NSMutableArray<NSString *> *result = [NSMutableArray array];
    for (ANEOperationNode *node in nodes) [result addObject:node.identifier];
    return [result copy];
}

static BOOL allFP16(NSArray<ANEOperationNode *> *nodes) {
    for (ANEOperationNode *node in nodes)
        if (node.outputType.elementType != ANEElementTypeFP16) return NO;
    return YES;
}

static ANEOperationNode *singleInputNamed(ANEOperationNode *node,
                                          NSString *name) {
    ANEOperationNode *found = nil;
    for (ANEOperationNode *input in node.inputs) {
        if (![input.operationName isEqualToString:name]) continue;
        if (found) return nil;
        found = input;
    }
    return found;
}

static ANEOperationNode *singleInputKind(ANEOperationNode *node,
                                         ANEOperationKind kind) {
    ANEOperationNode *found = nil;
    for (ANEOperationNode *input in node.inputs) {
        if (input.kind != kind) continue;
        if (found) return nil;
        found = input;
    }
    return found;
}

static ANEGraphArgument *singleValue(ANEGraphArgument *argument) {
    while (argument.kind == ANEGraphArgumentKindCall &&
           argument.callArguments.count == 1)
        argument = argument.callArguments.firstObject.value;
    return argument;
}

static BOOL hasAttentionScale(ANEOperationNode *scaled,
                              ANEOperationNode *scoreMatmul) {
    if (![scaled.operationName isEqualToString:@"mul"] ||
        !scoreMatmul.sourceOperation) return NO;
    ANEOperationNode *constant = nil;
    for (ANEOperationNode *input in scaled.inputs)
        if (input.kind == ANEOperationKindConstant) constant = input;
    NSString *text = constant.sourceOperation
        ? singleValue(constant.sourceOperation.attributes[@"val"]).text : nil;
    ANEGraphValue *query = scoreMatmul.sourceOperation.operands[@"x"].value;
    NSUInteger headWidth = query.type.shape.lastObject.unsignedIntegerValue;
    if (!text || headWidth == 0) return NO;
    char *end = nullptr;
    double actual = std::strtod(text.UTF8String, &end);
    double expected = 1.0 / std::sqrt((double)headWidth);
    return end && *end == '\0' && std::isfinite(actual) &&
        std::fabs(actual - expected) <= expected * 0.001;
}

static NSString *operandIdentifier(ANEOperationNode *node, NSString *name) {
    return node.sourceOperation.operands[name].value.name;
}

static BOOL falseMatmulFlag(ANEOperationNode *node, NSString *name) {
    ANEGraphValue *value = node.sourceOperation.operands[name].value;
    NSString *text = value.producer
        ? singleValue(value.producer.attributes[@"val"]).text : nil;
    return [text isEqualToString:@"false"];
}

static ANEScheduledRegionPlan *matmulGELUPlan(
    ANEOperationNode *gelu, NSArray<ANEScheduledTask *> *tasks,
    NSDictionary<NSString *, ANEScheduledTask *> *tasksByNode) {
    if (![gelu.operationName isEqualToString:@"gelu"] || gelu.inputs.count != 1)
        return nil;
    ANEOperationNode *bridge = gelu.inputs.firstObject;
    ANEOperationNode *matmul = bridge.kind == ANEOperationKindMatmul
        ? bridge : ([bridge.operationName isEqualToString:@"reshape"]
                    ? singleInputKind(bridge, ANEOperationKindMatmul) : nil);
    if (matmul.kind != ANEOperationKindMatmul || matmul.users.count != 1 ||
        bridge.users.count != 1) return nil;
    NSMutableArray<ANEOperationNode *> *nodes =
        [NSMutableArray arrayWithObject:matmul];
    if (bridge != matmul) [nodes addObject:bridge];
    [nodes addObject:gelu];
    if (!allFP16(nodes)) return nil;
    NSArray<NSNumber *> *indexes = taskIndexes(nodes, tasksByNode);
    NSString *region = indexes ? commonRegion(indexes, tasks) : nil;
    return region ? [[ANEScheduledRegionPlan alloc]
        initWithRegionIdentifier:region topology:ANEScheduledTopologyDirect
        taskIndexes:indexes stageIdentifiers:identifiers(nodes)] : nil;
}

static ANEScheduledRegionPlan *onlineAttentionPlan(
    ANEOperationNode *valueMatmul, NSArray<ANEScheduledTask *> *tasks,
    NSDictionary<NSString *, ANEScheduledTask *> *tasksByNode,
    NSSet<NSString *> *graphOutputs) {
    if (valueMatmul.kind != ANEOperationKindMatmul) return nil;
    ANEOperationNode *probabilities = singleInputNamed(valueMatmul, @"mul");
    if (!probabilities || probabilities.users.count != 1 ||
        [graphOutputs containsObject:probabilities.identifier]) return nil;
    ANEOperationNode *exponential = singleInputNamed(probabilities, @"exp");
    ANEOperationNode *inverse = singleInputNamed(probabilities, @"reciprocal");
    ANEOperationNode *sum = inverse ? singleInputNamed(inverse, @"reduce_sum") : nil;
    ANEOperationNode *centered = exponential
        ? singleInputNamed(exponential, @"sub") : nil;
    ANEOperationNode *maximum = centered
        ? singleInputNamed(centered, @"reduce_max") : nil;
    ANEOperationNode *scaled = maximum
        ? singleInputNamed(maximum, @"mul") : nil;
    ANEOperationNode *scoreMatmul = scaled
        ? singleInputKind(scaled, ANEOperationKindMatmul) : nil;
    if (!exponential || !inverse || !sum || !centered || !maximum ||
        !scaled || !scoreMatmul || !hasAttentionScale(scaled, scoreMatmul) ||
        !falseMatmulFlag(scoreMatmul, @"transpose_x") ||
        !falseMatmulFlag(scoreMatmul, @"transpose_y") ||
        !falseMatmulFlag(valueMatmul, @"transpose_x") ||
        !falseMatmulFlag(valueMatmul, @"transpose_y") ||
        sum.inputs.count != 1 ||
        sum.inputs.firstObject != exponential ||
        ![centered.inputs containsObject:scaled] ||
        ![probabilities.inputs containsObject:exponential] ||
        ![probabilities.inputs containsObject:inverse]) return nil;
    NSArray<ANEOperationNode *> *nodes = @[
        scoreMatmul, scaled, maximum, centered, exponential, sum, inverse,
        probabilities, valueMatmul,
    ];
    if (!allFP16(nodes)) return nil;
    NSArray<NSNumber *> *indexes = taskIndexes(nodes, tasksByNode);
    NSString *region = indexes ? commonRegion(indexes, tasks) : nil;
    if (!region) return nil;
    NSArray<NSNumber *> *scoreShape = scoreMatmul.outputType.shape;
    NSArray<NSNumber *> *outputShape = valueMatmul.outputType.shape;
    if (scoreShape.count < 2 || outputShape.count < 2) return nil;
    NSUInteger queryRows = scoreShape[scoreShape.count - 2].unsignedIntegerValue;
    NSUInteger keyValueRows = scoreShape.lastObject.unsignedIntegerValue;
    NSUInteger tileRows = 128;
    NSString *query = operandIdentifier(scoreMatmul, @"x");
    NSString *key = operandIdentifier(scoreMatmul, @"y");
    NSString *valueX = operandIdentifier(valueMatmul, @"x");
    NSString *valueY = operandIdentifier(valueMatmul, @"y");
    NSString *value = [valueX isEqualToString:probabilities.identifier]
        ? valueY : ([valueY isEqualToString:probabilities.identifier]
                    ? valueX : nil);
    if (!query || !key || !value) return nil;
    NSArray<NSString *> *carry = @[
        [valueMatmul.identifier stringByAppendingString:@".online_max"],
        [valueMatmul.identifier stringByAppendingString:@".online_sum"],
        [valueMatmul.identifier stringByAppendingString:@".online_output"],
    ];
    NSMutableArray<NSString *> *elided = [NSMutableArray array];
    for (ANEOperationNode *node in nodes)
        if (node != valueMatmul) [elided addObject:node.identifier];
    return [[ANEScheduledRegionPlan alloc]
        initWithRegionIdentifier:region
        topology:ANEScheduledTopologyOnlineReduction taskIndexes:indexes
        stageIdentifiers:identifiers(nodes)
        queryTileCount:(queryRows + tileRows - 1) / tileRows
        keyValueTileCount:(keyValueRows + tileRows - 1) / tileRows
        tileRows:tileRows carriedSurfaceIdentifiers:carry
        elidedSurfaceIdentifiers:elided
        boundaryInputIdentifiers:@[query, key, value]
        outputIdentifier:valueMatmul.identifier];
}

static ANEOperationNode *transitionMultiply(ANEOperationNode *add) {
    if (![add.operationName isEqualToString:@"add"] ||
        add.inputs.count + add.externalValueNames.count != 2)
        return nil;
    ANEOperationNode *multiply = singleInputNamed(add, @"mul");
    return multiply.users.count == 1 ? multiply : nil;
}

static ANEOperationNode *previousTransition(ANEOperationNode *multiply) {
    for (ANEOperationNode *input in multiply.inputs)
        if ([input.operationName isEqualToString:@"add"] &&
            transitionMultiply(input)) return input;
    return nil;
}

static ANEOperationNode *nextTransition(ANEOperationNode *add) {
    if (add.users.count != 1) return nil;
    ANEOperationNode *multiply = add.users.firstObject;
    if (![multiply.operationName isEqualToString:@"mul"] ||
        multiply.users.count != 1) return nil;
    ANEOperationNode *next = multiply.users.firstObject;
    return transitionMultiply(next) == multiply ? next : nil;
}

static ANEScheduledRegionPlan *affineScanPlan(
    ANEOperationNode *first, NSArray<ANEScheduledTask *> *tasks,
    NSDictionary<NSString *, ANEScheduledTask *> *tasksByNode,
    NSSet<NSString *> *graphOutputs) {
    ANEOperationNode *firstMultiply = transitionMultiply(first);
    if (!firstMultiply || previousTransition(firstMultiply)) return nil;
    NSMutableArray<ANEOperationNode *> *nodes = [NSMutableArray array];
    ANEOperationNode *transition = first;
    NSUInteger count = 0;
    while (transition) {
        if (count > 0 && [graphOutputs containsObject:
                          nodes[nodes.count - 1].identifier]) return nil;
        [nodes addObject:transitionMultiply(transition)];
        [nodes addObject:transition];
        count++;
        transition = nextTransition(transition);
    }
    if (count < 2 || !allFP16(nodes)) return nil;
    for (NSUInteger index = 1; index + 1 < nodes.count; index += 2)
        if ([graphOutputs containsObject:nodes[index].identifier]) return nil;
    NSArray<NSNumber *> *indexes = taskIndexes(nodes, tasksByNode);
    NSString *region = indexes ? commonRegion(indexes, tasks) : nil;
    return region ? [[ANEScheduledRegionPlan alloc]
        initWithRegionIdentifier:region
        topology:ANEScheduledTopologyAssociativeScan taskIndexes:indexes
        stageIdentifiers:identifiers(nodes)] : nil;
}

@implementation ANEComposedRegionPlanner
+ (NSArray<ANEScheduledRegionPlan *> *)plansForGraph:(ANEOperationGraph *)graph
                                               tasks:(NSArray<ANEScheduledTask *> *)tasks {
    NSDictionary<NSString *, ANEScheduledTask *> *tasksByNode = taskByNode(tasks);
    NSSet<NSString *> *outputs = [NSSet setWithArray:graph.outputValueNames];
    NSMutableArray<ANEScheduledRegionPlan *> *plans = [NSMutableArray array];
    NSMutableSet<NSString *> *claimedStages = [NSMutableSet set];
    for (ANEOperationNode *node in graph.nodes) {
        ANEScheduledRegionPlan *plan = matmulGELUPlan(node, tasks, tasksByNode);
        if (!plan) plan = onlineAttentionPlan(node, tasks, tasksByNode, outputs);
        if (!plan) plan = affineScanPlan(node, tasks, tasksByNode, outputs);
        if (!plan) continue;
        BOOL overlaps = NO;
        for (NSString *identifier in plan.stageIdentifiers)
            if ([claimedStages containsObject:identifier]) overlaps = YES;
        if (overlaps) continue;
        [plans addObject:plan];
        [claimedStages addObjectsFromArray:plan.stageIdentifiers];
    }
    return [plans copy];
}
@end
