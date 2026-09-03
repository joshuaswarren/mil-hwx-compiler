#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEGraphIR.h"
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
    return [[ANEGraphArgument alloc]
        initWithKind:ANEGraphArgumentKindValue
                 text:nil
                value:value
           calleeName:nil
      calleeValueType:nil
        callArguments:@[]
             elements:@[]
                range:zeroRange()];
}

static ANEGraphOperation *unary(NSString *name, NSString *resultName,
                                ANEValueType *type, ANEGraphValue *input) {
    ANEGraphValue *result = [[ANEGraphValue alloc] initWithName:resultName
                                                           type:type];
    return [[ANEGraphOperation alloc]
        initWithOperationName:name
                       result:result
                    arguments:@{@"x": valueArgument(input)}
                   attributes:@{}
                        range:zeroRange()];
}

static void testArbitraryDataflowAndUsers(void) {
    ANEValueType *type = [[ANEValueType alloc]
        initWithKind:ANEValueTypeKindTensor
         elementType:ANEElementTypeFP16
               shape:@[@1, @64, @64, @64]];
    ANEGraphValue *input = [[ANEGraphValue alloc] initWithName:@"x" type:type];
    ANEGraphOperation *producer = unary(@"relu", @"r", type, input);
    ANEGraphOperation *left = unary(@"sigmoid", @"a", type, producer.result);
    ANEGraphOperation *right = unary(@"tanh", @"b", type, producer.result);
    ANEGraphFunction *function = [[ANEGraphFunction alloc]
        initWithName:@"branched"
              inputs:@[input]
          operations:@[producer, left, right]
        returnValues:@[left.result, right.result]];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *graph = [[ANEOperationGraph alloc]
        initWithFunction:function diagnostics:diagnostics];

    expect(graph != nil && diagnostics.errorCount == 0,
           @"arbitrary typed DAG imports without a workload selector");
    expect(graph.nodes.count == 3, @"one operation node is created per SSA op");
    ANEOperationNode *root = graph.nodes[0];
    expect(root.kind == ANEOperationKindALU, @"relu maps to the generic ALU kind");
    expect(root.users.count == 2, @"use lists preserve fan-out");
    expect([root.users[0].identifier isEqualToString:@"a"] &&
           [root.users[1].identifier isEqualToString:@"b"],
           @"users remain in deterministic graph order");
    expect([graph textualDescription].length > 0,
           @"graph has a deterministic pass-dump representation");
}

static void testUnknownOperationRemainsExplicit(void) {
    ANEValueType *type = [[ANEValueType alloc]
        initWithKind:ANEValueTypeKindTensor
         elementType:ANEElementTypeFP16 shape:@[@1, @32]];
    ANEGraphValue *input = [[ANEGraphValue alloc] initWithName:@"x" type:type];
    ANEGraphOperation *mystery = unary(@"future_op", @"y", type, input);
    ANEGraphFunction *function = [[ANEGraphFunction alloc]
        initWithName:@"unknown" inputs:@[input] operations:@[mystery]
        returnValues:@[mystery.result]];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *graph = [[ANEOperationGraph alloc]
        initWithFunction:function diagnostics:diagnostics];
    expect(graph.nodes[0].kind == ANEOperationKindUnsupported,
           @"unknown operations do not silently select another primitive");
    expect([graph.nodes[0].operationName isEqualToString:@"future_op"],
           @"unknown spelling survives for legalization diagnostics");
}

static void testSpaceDepthOperationsAreFirstClassLayoutNodes(void) {
    ANEValueType *inputType = [[ANEValueType alloc]
        initWithKind:ANEValueTypeKindTensor
         elementType:ANEElementTypeFP16 shape:@[@1, @32, @64, @64]];
    ANEValueType *packedType = [[ANEValueType alloc]
        initWithKind:ANEValueTypeKindTensor
         elementType:ANEElementTypeFP16 shape:@[@1, @512, @16, @16]];
    ANEGraphValue *input = [[ANEGraphValue alloc] initWithName:@"x"
                                                           type:inputType];
    ANEGraphOperation *pack = unary(@"space_to_depth", @"packed",
                                    packedType, input);
    ANEGraphOperation *unpack = unary(@"depth_to_space", @"y",
                                      inputType, pack.result);
    ANEGraphFunction *function = [[ANEGraphFunction alloc]
        initWithName:@"layout_round_trip" inputs:@[input]
        operations:@[pack, unpack] returnValues:@[unpack.result]];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *graph = [[ANEOperationGraph alloc]
        initWithFunction:function diagnostics:diagnostics];

    expect(graph != nil && diagnostics.errorCount == 0,
           @"space/depth graph imports as an ordinary typed DAG");
    expect(graph.nodes[0].kind == ANEOperationKindLayout &&
           graph.nodes[1].kind == ANEOperationKindLayout,
           @"space_to_depth and depth_to_space map to the neutral layout kind");
    expect([graph.nodes[0].outputType.shape isEqualToArray:
            @[@1, @512, @16, @16]] &&
           [graph.nodes[1].outputType.shape isEqualToArray:
            @[@1, @32, @64, @64]],
           @"layout nodes preserve symbolic tensor geometry");
}

int main(void) {
    @autoreleasepool {
        testArbitraryDataflowAndUsers();
        testSpaceDepthOperationsAreFirstClassLayoutNodes();
        testUnknownOperationRemainsExplicit();
        printf("operation graph: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
