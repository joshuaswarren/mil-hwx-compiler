#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEDecomposePass.h"
#import "ANEFusionPass.h"
#import "ANEGraphIR.h"
#import "ANEOperationGraph.h"
#import "H16GTarget.h"

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
static ANEGraphArgument *arg(ANEGraphValue *value) {
    return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindValue
        text:nil value:value calleeName:nil calleeValueType:nil
        callArguments:@[] elements:@[] range:zeroRange()];
}
static ANEGraphOperation *op(NSString *name, NSString *resultName,
                             ANEValueType *type,
                             NSArray<ANEGraphValue *> *inputs) {
    ANEGraphValue *result = [[ANEGraphValue alloc] initWithName:resultName type:type];
    NSMutableDictionary *arguments = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < inputs.count; ++i)
        arguments[[NSString stringWithFormat:@"in%lu", (unsigned long)i]] = arg(inputs[i]);
    return [[ANEGraphOperation alloc] initWithOperationName:name result:result
        arguments:arguments attributes:@{} range:zeroRange()];
}
static ANEOperationGraph *graphWithOps(NSArray<NSString *> *names,
                                       NSArray<NSArray<NSNumber *> *> *edges,
                                       NSArray<NSNumber *> *returns) {
    ANEValueType *type = [[ANEValueType alloc] initWithKind:ANEValueTypeKindTensor
        elementType:ANEElementTypeFP16 shape:@[@1, @64, @64, @64]];
    ANEGraphValue *input = [[ANEGraphValue alloc] initWithName:@"x" type:type];
    NSMutableArray<ANEGraphOperation *> *ops = [NSMutableArray array];
    for (NSUInteger i = 0; i < names.count; ++i) {
        NSMutableArray<ANEGraphValue *> *inputs = [NSMutableArray array];
        for (NSNumber *edge in edges[i])
            [inputs addObject:edge.integerValue < 0 ? input
                : ops[edge.unsignedIntegerValue].result];
        [ops addObject:op(names[i], [NSString stringWithFormat:@"v%lu",
            (unsigned long)i], type, inputs)];
    }
    NSMutableArray<ANEGraphValue *> *outputs = [NSMutableArray array];
    for (NSNumber *index in returns) [outputs addObject:ops[index.unsignedIntegerValue].result];
    ANEGraphFunction *function = [[ANEGraphFunction alloc] initWithName:@"f"
        inputs:@[input] operations:ops returnValues:outputs];
    return [[ANEOperationGraph alloc] initWithFunction:function
        diagnostics:[[ANEDiagnosticEngine alloc] init]];
}

static void testConvReluUsesStructuralEdge(void) {
    ANEOperationGraph *graph = graphWithOps(@[@"conv", @"relu"],
        @[@[@-1], @[@0]], @[@1]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEFusionPass runOnGraph:graph target:[H16GTarget currentTarget]
        diagnostics:diagnostics], @"fusion succeeds");
    expect(graph.regions.count == 1 && graph.regions[0].nodes.count == 2,
           @"Conv and ReLU fuse from their legal data edge");
}

static void testFanoutPreventsUnsafeFusion(void) {
    ANEOperationGraph *graph = graphWithOps(@[@"conv", @"relu", @"tanh"],
        @[@[@-1], @[@0], @[@0]], @[@1, @2]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEFusionPass runOnGraph:graph target:[H16GTarget currentTarget]
        diagnostics:diagnostics], @"fanout graph is handled");
    expect(graph.regions.count == 3,
           @"a producer with two external consumers remains materialized");
}

static void testDecomposedSoftmaxFusesWithoutName(void) {
    ANEOperationGraph *graph = graphWithOps(@[@"relu", @"softmax", @"tanh"],
        @[@[@-1], @[@0], @[@1]], @[@2]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEDecomposePass runOnGraph:graph diagnostics:diagnostics],
           @"softmax is decomposed before fusion");
    expect([ANEFusionPass runOnGraph:graph target:[H16GTarget currentTarget]
        diagnostics:diagnostics], @"primitive graph fuses");
    expect(graph.regions.count == 1 && graph.regions[0].nodes.count == 8,
           @"the primitive chain forms one region without an Attention kind");
    for (ANEOperationNode *node in graph.regions[0].nodes)
        expect(![node.operationName containsString:@"attention"],
               @"region contains only primitive operation names");
}

static void testUnsupportedOperationIsBarrier(void) {
    ANEOperationGraph *graph = graphWithOps(@[@"relu", @"future_op", @"tanh"],
        @[@[@-1], @[@0], @[@1]], @[@2]);
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    expect([ANEFusionPass runOnGraph:graph target:[H16GTarget currentTarget]
        diagnostics:diagnostics], @"unsupported operation remains inspectable");
    expect(graph.regions.count == 3,
           @"unsupported operation prevents fusion across itself");
}

int main(void) {
    @autoreleasepool {
        testConvReluUsesStructuralEdge();
        testFanoutPreventsUnsafeFusion();
        testDecomposedSoftmaxFusesWithoutName();
        testUnsupportedOperationIsBarrier();
        printf("structural fusion: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
