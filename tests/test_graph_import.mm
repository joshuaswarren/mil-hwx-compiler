#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEGraphIR.h"
#import "ANEGraphVerifier.h"
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

static ANEGraphModule *importData(NSData *data,
                                  ANEDiagnosticEngine **outDiagnostics) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    MILLexer *lexer = [[MILLexer alloc] initWithData:data
                                         diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
                                              diagnostics:diagnostics];
    MILProgramSyntax *syntax = [parser parseProgram];
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (module && ![ANEGraphVerifier verifyModule:module
                                      diagnostics:diagnostics])
        module = nil;
    if (outDiagnostics) *outDiagnostics = diagnostics;
    return module;
}

static ANEGraphModule *importFixture(NSString *name,
                                     ANEDiagnosticEngine **diagnostics) {
    NSString *path = [@"tests/fixtures" stringByAppendingPathComponent:name];
    return importData([NSData dataWithContentsOfFile:path], diagnostics);
}

static ANEGraphOperation *operationNamed(ANEGraphFunction *function,
                                         NSString *resultName) {
    for (ANEGraphOperation *operation in function.operations) {
        if ([operation.result.name isEqualToString:resultName]) return operation;
    }
    return nil;
}

static void testConvGraph(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    ANEGraphModule *module = importFixture(@"conv_relu.mil", &diagnostics);
    expect(module != nil && diagnostics.errorCount == 0,
           @"Conv+ReLU imports and verifies");
    ANEGraphFunction *function = module.functions[0];
    expect(function.inputs.count == 1, @"function input is an SSA value");
    ANEGraphValue *input = function.inputs[0];
    expect(input.producer == nil, @"input has no producer");
    expect(input.type.kind == ANEValueTypeKindTensor &&
           input.type.elementType == ANEElementTypeFP16 &&
           [input.type.shape isEqualToArray:@[@1, @64, @64, @64]],
           @"input has a typed tensor shape");

    ANEGraphOperation *conv = operationNamed(function, @"c0");
    ANEGraphOperation *relu = operationNamed(function, @"y");
    expect([conv.operationName isEqualToString:@"conv"],
           @"conv operation identity is retained");
    expect([relu.operationName isEqualToString:@"relu"],
           @"relu operation identity is retained");
    expect(conv.operands[@"x"].value == input,
           @"conv input resolves to the function value");
    expect(relu.operands[@"x"].value == conv.result,
           @"relu input resolves to the conv result");
    expect(function.returnValues[0] == relu.result,
           @"return resolves to the relu result");

    ANEGraphOperation *weight = operationNamed(function, @"W");
    ANEGraphArgument *constant = weight.attributes[@"val"];
    expect(constant.kind == ANEGraphArgumentKindCall &&
           [constant.calleeName isEqualToString:@"tensor"],
           @"typed constant constructor is retained");
    ANEGraphArgument *blob = constant.callArguments[0].value;
    expect(blob.kind == ANEGraphArgumentKindCall &&
           [blob.calleeName isEqualToString:@"BLOBFILE"],
           @"blob payload is structural");
    expect([blob.namedArguments[@"path"].value.callArguments[0].value.text
            isEqualToString:@"@model_path/weights/weight.bin"],
           @"blob path survives semantic import");
}

static void testAllArticleGraphs(void) {
    for (NSString *name in @[@"attention.mil", @"w8a8_conv_chain.mil"]) {
        ANEDiagnosticEngine *diagnostics = nil;
        ANEGraphModule *module = importFixture(name, &diagnostics);
        if (!module) {
            for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
                fprintf(stderr, "DIAG %s %s: %s\n", name.UTF8String,
                        diagnostic.code.UTF8String,
                        diagnostic.message.UTF8String);
        }
        expect(module != nil && diagnostics.errorCount == 0,
               [NSString stringWithFormat:@"%@ imports and verifies", name]);
    }
}

static void testSemanticFailures(void) {
    NSArray<NSDictionary<NSString *, NSString *> *> *cases = @[
        @{
            @"label": @"duplicate value",
            @"code": @"mil.import.duplicate-value",
            @"source": @"program(1.3) { func main<ios18>(fp16 x) { "
                        "fp16 x = relu(x = x); } -> (x); }"
        },
        @{
            @"label": @"unknown operand",
            @"code": @"mil.import.unknown-value",
            @"source": @"program(1.3) { func main<ios18>(fp16 x) { "
                        "fp16 y = relu(x = missing); } -> (y); }"
        },
        @{
            @"label": @"unknown return",
            @"code": @"mil.import.unknown-return",
            @"source": @"program(1.3) { func main<ios18>(fp16 x) { "
                        "fp16 y = relu(x = x); } -> (missing); }"
        },
        @{
            @"label": @"relu type mismatch",
            @"code": @"ane.verify.relu-type",
            @"source": @"program(1.3) { func main<ios18>(fp16 x) { "
                        "int8 y = relu(x = x); } -> (y); }"
        },
    ];
    for (NSDictionary<NSString *, NSString *> *testCase in cases) {
        ANEDiagnosticEngine *diagnostics = nil;
        ANEGraphModule *module = importData(
            [testCase[@"source"] dataUsingEncoding:NSUTF8StringEncoding],
            &diagnostics);
        expect(module == nil, [testCase[@"label"]
                               stringByAppendingString:@" is rejected"]);
        BOOL found = NO;
        for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
            if ([diagnostic.code isEqualToString:testCase[@"code"]]) found = YES;
        expect(found, [testCase[@"label"]
                       stringByAppendingString:@" has stable diagnostic"]);
    }
}

int main(void) {
    @autoreleasepool {
        testConvGraph();
        testAllArticleGraphs();
        testSemanticFailures();
        printf("graph import: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
