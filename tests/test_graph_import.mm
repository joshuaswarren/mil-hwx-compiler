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
    for (ANEGraphOperation *operation in function.operations)
        for (ANEGraphValue *result in operation.results)
            if ([result.name isEqualToString:resultName]) return operation;
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
    expect(relu.operands[@"x"].value == conv.results[0],
           @"relu input resolves to the conv result");
    expect(function.returnValues[0] == relu.results[0],
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

static NSString *splitProgram(NSString *version, NSString *opset,
                              NSString *results) {
    return [NSString stringWithFormat:
        @"program(%@) { func main<%@>(tensor<fp16, [1, 2048, 375]> x) { "
         "tensor<int32, []> count = const()[val = tensor<int32, []>(2)]; "
         "tensor<int32, []> axis = const()[val = tensor<int32, []>(1)]; "
         "%@ = split(axis = axis, num_splits = count, x = x); "
         "tensor<fp16, [1, 1024, 375]> gate = sigmoid(x = second); "
         "tensor<fp16, [1, 1024, 375]> y = mul(x = first, y = gate); "
         "} -> (y); }", version, opset, results];
}

static void testOrderedResultImport(void) {
    NSString *results =
        @"(tensor<fp16, [1, 1024, 375]> first, "
         "tensor<fp16, [1, 1024, 375]> second)";
    ANEDiagnosticEngine *diagnostics = nil;
    ANEGraphModule *module = importData(
        [splitProgram(@"1", @"CoreML8", results)
         dataUsingEncoding:NSUTF8StringEncoding], &diagnostics);
    expect(module != nil && diagnostics.errorCount == 0,
           @"two-output split imports and verifies");
    ANEGraphFunction *function = module.functions[0];
    expect([module.version isEqualToString:@"1"] &&
           [function.opset isEqualToString:@"CoreML8"],
           @"semantic graph retains program version and function opset");
    ANEGraphOperation *split = operationNamed(function, @"first");
    expect(split.results.count == 2 &&
           [split.results[0].name isEqualToString:@"first"] &&
           [split.results[1].name isEqualToString:@"second"],
           @"semantic import preserves result order");
    expect(split.results[0].producer == split &&
           split.results[1].producer == split,
           @"every imported result records the same producer");
    ANEGraphOperation *sigmoid = operationNamed(function, @"gate");
    ANEGraphOperation *mul = operationNamed(function, @"y");
    expect(sigmoid.operands[@"x"].value == split.results[1] &&
           mul.operands[@"x"].value == split.results[0],
           @"consumers bind the intended ordered split results");
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
        @{
            @"label": @"duplicate operation result",
            @"code": @"mil.import.duplicate-value",
            @"source": @"program(1) { func main<CoreML8>(fp16 x) { "
                        "(fp16 y, fp16 y) = split(x = x); } -> (y); }"
        },
        @{
            @"label": @"mismatched split constant type",
            @"code": @"ane.verify.split-constant-parameters",
            @"source": @"program(1) { func main<CoreML8>(tensor<fp16, [1, 2048, 375]> x) { "
                        "tensor<int32, []> count = const()[val = tensor<int32, []>(2)]; "
                        "tensor<fp16, []> axis = const()[val = tensor<int32, []>(1)]; "
                        "(tensor<fp16, [1, 1024, 375]> first, tensor<fp16, [1, 1024, 375]> second) = split(axis = axis, num_splits = count, x = x); "
                        "} -> (first, second); }"
        },
        @{
            @"label": @"split result arity",
            @"code": @"ane.verify.split-result-arity",
            @"source": @"program(1) { func main<CoreML8>(tensor<fp16, [1, 2048, 375]> x) { "
                        "tensor<int32, []> count = const()[val = tensor<int32, []>(2)]; "
                        "tensor<int32, []> axis = const()[val = tensor<int32, []>(1)]; "
                        "tensor<fp16, [1, 1024, 375]> first = split(axis = axis, num_splits = count, x = x); "
                        "} -> (first); }"
        },
        @{
            @"label": @"unsupported program version",
            @"code": @"ane.verify.unsupported-program-version",
            @"source": @"program(2) { func main<CoreML8>(fp16 x) { "
                        "fp16 y = relu(x = x); } -> (y); }"
        },
        @{
            @"label": @"unsupported function opset",
            @"code": @"ane.verify.unsupported-opset",
            @"source": @"program(1) { func main<CoreML9>(fp16 x) { "
                        "fp16 y = relu(x = x); } -> (y); }"
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
        testOrderedResultImport();
        testSemanticFailures();
        printf("graph import: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
