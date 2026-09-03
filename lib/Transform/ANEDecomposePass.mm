#import "ANEDecomposePass.h"

static ANEOperationNode *synthetic(ANEOperationNode *source,
                                   NSString *identifier,
                                   NSString *name,
                                   ANEOperationKind kind,
                                   NSArray<ANEOperationNode *> *inputs,
                                   NSArray<NSString *> *external) {
    return [[ANEOperationNode alloc] initWithIdentifier:identifier
        operationName:name kind:kind outputType:source.outputType inputs:inputs
        externalValueNames:external sourceOperation:source.sourceOperation];
}

static NSArray<ANEOperationNode *> *softmaxNodes(ANEOperationNode *node) {
    NSString *base = node.identifier;
    ANEOperationNode *maximum = synthetic(node, [base stringByAppendingString:@".max"],
        @"reduce_max", ANEOperationKindReduce, node.inputs, node.externalValueNames);
    NSMutableArray *subInputs = [node.inputs mutableCopy];
    [subInputs addObject:maximum];
    ANEOperationNode *centered = synthetic(node, [base stringByAppendingString:@".centered"],
        @"sub", ANEOperationKindALU, subInputs, node.externalValueNames);
    ANEOperationNode *exponential = synthetic(node, [base stringByAppendingString:@".exp"],
        @"exp", ANEOperationKindLUT, @[centered], @[]);
    ANEOperationNode *sum = synthetic(node, [base stringByAppendingString:@".sum"],
        @"reduce_sum", ANEOperationKindReduce, @[exponential], @[]);
    ANEOperationNode *inverse = synthetic(node, [base stringByAppendingString:@".reciprocal"],
        @"reciprocal", ANEOperationKindALU, @[sum], @[]);
    ANEOperationNode *result = synthetic(node, base, @"mul", ANEOperationKindALU,
        @[exponential, inverse], @[]);
    return @[maximum, centered, exponential, sum, inverse, result];
}

static NSArray<ANEOperationNode *> *layerNormNodes(ANEOperationNode *node) {
    NSString *base = node.identifier;
    ANEOperationNode *mean = synthetic(node, [base stringByAppendingString:@".mean"],
        @"reduce_mean", ANEOperationKindReduce, node.inputs, node.externalValueNames);
    NSMutableArray *subInputs = [node.inputs mutableCopy];
    [subInputs addObject:mean];
    ANEOperationNode *centered = synthetic(node, [base stringByAppendingString:@".centered"],
        @"sub", ANEOperationKindALU, subInputs, node.externalValueNames);
    ANEOperationNode *square = synthetic(node, [base stringByAppendingString:@".square"],
        @"square", ANEOperationKindALU, @[centered], @[]);
    ANEOperationNode *variance = synthetic(node, [base stringByAppendingString:@".variance"],
        @"reduce_mean", ANEOperationKindReduce, @[square], @[]);
    ANEOperationNode *inverse = synthetic(node, [base stringByAppendingString:@".rsqrt"],
        @"rsqrt", ANEOperationKindALU, @[variance], @[]);
    ANEOperationNode *scaled = synthetic(node, [base stringByAppendingString:@".scaled"],
        @"mul", ANEOperationKindALU, @[centered, inverse], @[]);
    ANEOperationNode *result = synthetic(node, base, @"add", ANEOperationKindALU,
        @[scaled], @[]);
    return @[mean, centered, square, variance, inverse, scaled, result];
}

@implementation ANEDecomposePass
+ (BOOL)runOnGraph:(ANEOperationGraph *)graph
       diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSArray<ANEOperationNode *> *original = [graph.nodes copy];
    for (ANEOperationNode *node in original) {
        NSArray<ANEOperationNode *> *replacement = nil;
        if ([node.operationName isEqualToString:@"softmax"])
            replacement = softmaxNodes(node);
        else if ([node.operationName isEqualToString:@"layer_norm"])
            replacement = layerNormNodes(node);
        if (replacement && ![graph replaceNode:node withNodes:replacement
            replacementNode:replacement.lastObject diagnostics:diagnostics]) return NO;
    }
    return YES;
}
@end
