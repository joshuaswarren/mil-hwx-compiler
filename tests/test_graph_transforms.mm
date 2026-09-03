#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEDecomposePass.h"
#import "ANEGraphIR.h"
#import "ANENormalizePass.h"
#import "ANEOperationGraph.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}
static ANESourceRange zeroRange(void) {
    ANESourceLocation p = ANESourceLocationMake(0, 1, 1);
    return ANESourceRangeMake(p, p);
}
static ANEGraphArgument *valueArgument(ANEGraphValue *value) {
    return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindValue
        text:nil value:value calleeName:nil calleeValueType:nil
        callArguments:@[] elements:@[] range:zeroRange()];
}
static ANEGraphOperation *unary(NSString *name, NSString *resultName,
                                ANEValueType *type, ANEGraphValue *input) {
    ANEGraphValue *result = [[ANEGraphValue alloc] initWithName:resultName type:type];
    return [[ANEGraphOperation alloc] initWithOperationName:name result:result
        arguments:@{@"x": valueArgument(input)} attributes:@{} range:zeroRange()];
}
static ANEOperationGraph *makeGraph(NSArray<NSString *> *names,
                                    NSArray<NSString *> *results,
                                    NSArray<NSNumber *> *inputIndices,
                                    NSArray<NSNumber *> *returns) {
    ANEValueType *type = [[ANEValueType alloc] initWithKind:ANEValueTypeKindTensor
        elementType:ANEElementTypeFP16 shape:@[@1, @4, @64, @64]];
    ANEGraphValue *input = [[ANEGraphValue alloc] initWithName:@"x" type:type];
    NSMutableArray<ANEGraphOperation *> *ops = [NSMutableArray array];
    for (NSUInteger i = 0; i < names.count; ++i) {
        NSInteger source = inputIndices[i].integerValue;
        ANEGraphValue *operand = source < 0 ? input : ops[(NSUInteger)source].result;
        [ops addObject:unary(names[i], results[i], type, operand)];
    }
    NSMutableArray<ANEGraphValue *> *returnValues = [NSMutableArray array];
    for (NSNumber *index in returns) [returnValues addObject:ops[index.unsignedIntegerValue].result];
    ANEGraphFunction *function = [[ANEGraphFunction alloc] initWithName:@"f"
        inputs:@[input] operations:ops returnValues:returnValues];
    return [[ANEOperationGraph alloc] initWithFunction:function
        diagnostics:[[ANEDiagnosticEngine alloc] init]];
}

static void testDeadCodeElimination(void) {
    ANEOperationGraph *graph = makeGraph(
        @[@"relu", @"tanh", @"sigmoid"], @[@"root", @"dead", @"live"],
        @[@-1, @0, @0], @[@2]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANENormalizePass runOnGraph:graph diagnostics:diagnostics],
           @"normalization succeeds");
    expect(graph.nodes.count == 2, @"unreachable operation is removed");
    expect([graph.nodes[0].identifier isEqualToString:@"root"] &&
           [graph.nodes[1].identifier isEqualToString:@"live"],
           @"live graph order is stable");
    expect(graph.nodes[0].users.count == 1 &&
           graph.nodes[0].users[0] == graph.nodes[1],
           @"use lists are rebuilt after normalization");
}

static void testSoftmaxDecomposition(void) {
    ANEOperationGraph *graph = makeGraph(@[@"relu", @"softmax", @"tanh"],
        @[@"input_ready", @"prob", @"out"], @[@-1, @0, @1], @[@2]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEDecomposePass runOnGraph:graph diagnostics:diagnostics],
           @"softmax decomposition succeeds");
    NSArray<NSString *> *expected = @[@"relu", @"reduce_max", @"sub", @"exp",
        @"reduce_sum", @"reciprocal", @"mul", @"tanh"];
    NSMutableArray<NSString *> *actual = [NSMutableArray array];
    for (ANEOperationNode *node in graph.nodes) [actual addObject:node.operationName];
    expect([actual isEqualToArray:expected],
           @"softmax becomes six generic hardware primitives");
    expect([graph nodeForValueName:@"prob"].kind == ANEOperationKindALU,
           @"the replacement keeps the original SSA result name");
    expect([graph nodeForValueName:@"out"].inputs[0] ==
           [graph nodeForValueName:@"prob"],
           @"downstream users consume the decomposition result");
    for (ANEOperationNode *node in graph.nodes)
        expect(node.kind != ANEOperationKindHighLevel,
               @"no high-level softmax node survives");
}

static void testLayerNormDecomposition(void) {
    ANEOperationGraph *graph = makeGraph(@[@"layer_norm"], @[@"y"], @[@-1], @[@0]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEDecomposePass runOnGraph:graph diagnostics:diagnostics],
           @"layer normalization decomposition succeeds");
    NSArray<NSString *> *expected = @[@"reduce_mean", @"sub", @"square",
        @"reduce_mean", @"rsqrt", @"mul", @"add"];
    NSMutableArray<NSString *> *actual = [NSMutableArray array];
    for (ANEOperationNode *node in graph.nodes) [actual addObject:node.operationName];
    expect([actual isEqualToArray:expected],
           @"layer normalization becomes seven generic primitives");
    expect([graph.nodes.lastObject.identifier isEqualToString:@"y"],
           @"final primitive retains the original result");
}

int main(void) {
    @autoreleasepool {
        testDeadCodeElimination();
        testSoftmaxDecomposition();
        testLayerNormDecomposition();
        printf("graph transforms: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
