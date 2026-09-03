#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEDecomposePass.h"
#import "ANEGraphIR.h"
#import "ANEGraphVerifier.h"
#import "ANEH16GLegalizePass.h"
#import "ANEOperationGraph.h"
#import "H16GTarget.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static ANEOperationGraph *fixture(NSString *name,
                                  ANEDiagnosticEngine *diagnostics) {
    NSString *path = [@"tests/fixtures" stringByAppendingPathComponent:name];
    MILLexer *lexer = [[MILLexer alloc]
        initWithData:[NSData dataWithContentsOfFile:path] diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
        diagnostics:diagnostics];
    MILProgramSyntax *syntax = [parser parseProgram];
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return nil;
    return [[ANEOperationGraph alloc] initWithFunction:module.functions[0]
        diagnostics:diagnostics];
}

static ANEOperationGraph *graphFromSource(NSString *source,
                                          ANEDiagnosticEngine *diagnostics) {
    MILLexer *lexer = [[MILLexer alloc]
        initWithData:[source dataUsingEncoding:NSUTF8StringEncoding]
        diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
        diagnostics:diagnostics];
    MILProgramSyntax *syntax = [parser parseProgram];
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return nil;
    return [[ANEOperationGraph alloc] initWithFunction:module.functions[0]
        diagnostics:diagnostics];
}

static NSString *layoutMIL(NSString *outputShape) {
    return [NSString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 32, 128, 128]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    tensor<fp16, %@> y = space_to_depth(x = x, block_size = b)"
         "[name = string(\"pack\")];\n"
         "  } -> (y);\n}\n", outputShape];
}

static void testW8A8ModesComeFromEdges(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *graph = fixture(@"w8a8_conv_chain.mil", diagnostics);
    expect([ANEH16GLegalizePass runOnGraph:graph
        target:[H16GTarget currentTarget] diagnostics:diagnostics],
        @"W8A8 graph legalizes");
    NSMutableArray<ANEOperationNode *> *convs = [NSMutableArray array];
    for (ANEOperationNode *node in graph.nodes)
        if (node.kind == ANEOperationKindConv) [convs addObject:node];
    expect(convs.count == 4, @"fixture has four ordinary Conv nodes");
    NSArray<NSNumber *> *expected = @[
        @(ANELegalNumericModeW8A8InputBoundary),
        @(ANELegalNumericModeW8A8Packed),
        @(ANELegalNumericModeW8A8Packed),
        @(ANELegalNumericModeW8A8OutputBoundary),
    ];
    for (NSUInteger i = 0; i < convs.count; ++i) {
        expect(convs[i].numericMode == expected[i].unsignedIntegerValue,
               @"numeric mode follows quantized graph boundaries");
        expect(convs[i].physicalLayout == ANEPhysicalLayoutNCHW,
               @"convolution layout is selected by legalization");
    }
    NSUInteger folded = 0;
    for (ANEOperationNode *node in graph.nodes)
        if (node.foldedIntoNumericBoundary) folded++;
    expect(folded == 10,
           @"four weight dequants and three activation Q/DQ pairs are folded");
}

static void testAttentionHasNoWorkloadMode(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *graph = fixture(@"attention.mil", diagnostics);
    expect([ANEDecomposePass runOnGraph:graph diagnostics:diagnostics],
           @"attention fixture softmax decomposes");
    expect([ANEH16GLegalizePass runOnGraph:graph
        target:[H16GTarget currentTarget] diagnostics:diagnostics],
        @"decomposed attention legalizes as primitives");
    for (ANEOperationNode *node in graph.nodes) {
        if (node.kind == ANEOperationKindConstant) continue;
        expect(node.numericMode == ANELegalNumericModeFP16,
               @"attention primitives use ordinary fp16 mode");
        expect(node.physicalLayout != ANEPhysicalLayoutUnknown,
               @"every compute primitive receives a physical layout");
    }
}

static void testLayoutLegalityComesFromShapeRelations(void) {
    ANEDiagnosticEngine *validDiagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *valid = graphFromSource(
        layoutMIL(@"[1, 512, 32, 32]"), validDiagnostics);
    expect(valid != nil && [ANEH16GLegalizePass runOnGraph:valid
        target:[H16GTarget currentTarget] diagnostics:validDiagnostics],
        @"S2D legality is derived from C32/S128/B4 shape relations");
    expect(valid.nodes.lastObject.kind == ANEOperationKindLayout &&
           valid.nodes.lastObject.physicalLayout == ANEPhysicalLayoutNCHW,
           @"legal S2D receives the tensor layout used by its DMA plan");

    ANEDiagnosticEngine *invalidDiagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEOperationGraph *invalid = graphFromSource(
        layoutMIL(@"[1, 512, 64, 32]"), invalidDiagnostics);
    expect(invalid != nil && ![ANEH16GLegalizePass runOnGraph:invalid
        target:[H16GTarget currentTarget] diagnostics:invalidDiagnostics] &&
        invalidDiagnostics.errorCount > 0,
        @"S2D rejects an output shape inconsistent with its input and block");

    NSString *narrowSource=
        @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
         "  func main<ios18>(tensor<fp16, [1, 32, 64, 64]> x) {\n"
         "    int32 b = const()[name = string(\"b\"), val = int32(4)];\n"
         "    tensor<fp16, [1, 512, 16, 16]> y = space_to_depth"
         "(x = x, block_size = b)[name = string(\"pack\")];\n"
         "  } -> (y);\n}\n";
    ANEDiagnosticEngine *narrowDiagnostics=[[ANEDiagnosticEngine alloc]init];
    ANEOperationGraph *narrow=graphFromSource(narrowSource,narrowDiagnostics);
    expect(narrow != nil && [ANEH16GLegalizePass runOnGraph:narrow
        target:[H16GTarget currentTarget] diagnostics:narrowDiagnostics],
        @"logical rows below 64 bytes remain legal when the surface plan pads them");
}

int main(void) {
    @autoreleasepool {
        testW8A8ModesComeFromEdges();
        testAttentionHasNoWorkloadMode();
        testLayoutLegalityComesFromShapeRelations();
        printf("H16G legalization: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
