#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "MILLexer.h"
#import "MILParser.h"
#import "MILPrinter.h"

#include <stdio.h>

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static MILProgramSyntax *parseData(NSData *data,
                                   ANEDiagnosticEngine **outDiagnostics) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    MILLexer *lexer = [[MILLexer alloc] initWithData:data
                                         diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
                                              diagnostics:diagnostics];
    MILProgramSyntax *program = [parser parseProgram];
    if (outDiagnostics) *outDiagnostics = diagnostics;
    return program;
}

static MILProgramSyntax *parseFixture(NSString *name,
                                      ANEDiagnosticEngine **diagnostics) {
    NSString *path = [@"tests/fixtures" stringByAppendingPathComponent:name];
    NSData *data = [NSData dataWithContentsOfFile:path];
    expect(data != nil, [NSString stringWithFormat:@"fixture %@ exists", name]);
    return parseData(data, diagnostics);
}

static NSUInteger countOperation(MILFunctionSyntax *function,
                                 NSString *operationName) {
    NSUInteger count = 0;
    for (MILOperationSyntax *operation in function.operations) {
        if ([operation.operationName isEqualToString:operationName]) count++;
    }
    return count;
}

static NSArray<NSString *> *operationSequence(MILFunctionSyntax *function) {
    NSMutableArray<NSString *> *names = [NSMutableArray array];
    for (MILOperationSyntax *operation in function.operations)
        [names addObject:operation.operationName];
    return names;
}

static void assertRoundTrip(MILProgramSyntax *program,
                            NSArray<NSString *> *expectedSequence,
                            NSString *label) {
    NSString *printed = [MILPrinter stringForProgram:program];
    ANEDiagnosticEngine *diagnostics = nil;
    MILProgramSyntax *reparsed = parseData(
        [printed dataUsingEncoding:NSUTF8StringEncoding], &diagnostics);
    expect(reparsed != nil && diagnostics.errorCount == 0,
           [label stringByAppendingString:@" printer output reparses"]);
    expect([operationSequence(reparsed.functions[0])
            isEqualToArray:expectedSequence],
           [label stringByAppendingString:@" operation order round-trips"]);
}

static void testConvRelu(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    MILProgramSyntax *program = parseFixture(@"conv_relu.mil", &diagnostics);
    expect(program != nil && diagnostics.errorCount == 0,
           @"Conv+ReLU fixture parses");
    expect([program.version isEqualToString:@"1.3"],
           @"program version is parsed");
    expect(program.functions.count == 1, @"one function is parsed");
    MILFunctionSyntax *function = program.functions[0];
    expect([function.name isEqualToString:@"main"], @"function name is main");
    expect([function.opset isEqualToString:@"ios18"], @"opset is parsed");
    expect(function.parameters.count == 1, @"one input parameter is parsed");
    expect([function.parameters[0].type.name isEqualToString:@"tensor"] &&
           [function.parameters[0].type.dimensions
            isEqualToArray:@[@1, @64, @64, @64]],
           @"input tensor shape is structural");
    expect(countOperation(function, @"conv") == 1, @"one conv is present");
    expect(countOperation(function, @"relu") == 1, @"one relu is present");
    expect([function.returnNames isEqualToArray:@[@"y"]],
           @"return value is parsed");
    assertRoundTrip(program, operationSequence(function), @"Conv+ReLU");
}

static void testAttention(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    MILProgramSyntax *program = parseFixture(@"attention.mil", &diagnostics);
    expect(program != nil && diagnostics.errorCount == 0,
           @"attention fixture parses");
    MILFunctionSyntax *function = program.functions[0];
    expect(countOperation(function, @"slice_by_size") == 3,
           @"attention has Q, K and V slices");
    expect(countOperation(function, @"transpose") == 4,
           @"attention has four layout transposes");
    expect(countOperation(function, @"matmul") == 2,
           @"attention has QK and PV matmuls");
    expect(countOperation(function, @"softmax") == 1,
           @"attention has one softmax");
    assertRoundTrip(program, operationSequence(function), @"attention");
}

static void testW8A8(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    MILProgramSyntax *program = parseFixture(@"w8a8_conv_chain.mil",
                                             &diagnostics);
    expect(program != nil && diagnostics.errorCount == 0,
           @"W8A8 fixture parses");
    MILFunctionSyntax *function = program.functions[0];
    expect(countOperation(function, @"constexpr_affine_dequantize") == 4,
           @"W8A8 has four quantized weights");
    expect(countOperation(function, @"conv") == 4,
           @"W8A8 has four convolutions");
    expect(countOperation(function, @"quantize") == 3,
           @"W8A8 has three activation quantizers");
    expect(countOperation(function, @"dequantize") == 3,
           @"W8A8 has three activation dequantizers");
    expect([function.returnNames isEqualToArray:@[@"c3"]],
           @"W8A8 returns the fourth convolution");
    assertRoundTrip(program, operationSequence(function), @"W8A8");
}

static void testMalformedProgram(void) {
    NSString *source = @"program(1.3) { func main<ios18>(fp16 x) { "
                        "fp16 y = relu(x = x) } -> (y); }";
    ANEDiagnosticEngine *diagnostics = nil;
    MILProgramSyntax *program = parseData(
        [source dataUsingEncoding:NSUTF8StringEncoding], &diagnostics);
    expect(program == nil, @"missing operation semicolon rejects program");
    expect(diagnostics.errorCount > 0, @"malformed program is diagnosed");
    expect([diagnostics.diagnostics[0].code hasPrefix:@"mil.parse."],
           @"parser diagnostic has stable namespace");
}

int main(void) {
    @autoreleasepool {
        testConvRelu();
        testAttention();
        testW8A8();
        testMalformedProgram();
        printf("mil parser: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
