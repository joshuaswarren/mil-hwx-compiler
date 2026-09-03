#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "MILLexer.h"

#include <stdio.h>

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static NSArray<MILToken *> *lex(NSString *source,
                                ANEDiagnosticEngine **outDiagnostics) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    MILLexer *lexer = [[MILLexer alloc]
        initWithData:[source dataUsingEncoding:NSUTF8StringEncoding]
        diagnostics:diagnostics];
    NSArray<MILToken *> *tokens = [lexer lexAllTokens];
    if (outDiagnostics) *outDiagnostics = diagnostics;
    return tokens;
}

static void testConvFragment(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    NSArray<MILToken *> *tokens = lex(
        @"tensor<fp16, [1, 64, 64, 64]> y = relu(x = c0);", &diagnostics);
    NSArray<NSNumber *> *expected = @[
        @(MILTokenKindIdentifier), @(MILTokenKindLess),
        @(MILTokenKindIdentifier), @(MILTokenKindComma),
        @(MILTokenKindLBracket), @(MILTokenKindInteger),
        @(MILTokenKindComma), @(MILTokenKindInteger), @(MILTokenKindComma),
        @(MILTokenKindInteger), @(MILTokenKindComma), @(MILTokenKindInteger),
        @(MILTokenKindRBracket), @(MILTokenKindGreater),
        @(MILTokenKindIdentifier), @(MILTokenKindEqual),
        @(MILTokenKindIdentifier), @(MILTokenKindLParen),
        @(MILTokenKindIdentifier), @(MILTokenKindEqual),
        @(MILTokenKindIdentifier), @(MILTokenKindRParen),
        @(MILTokenKindSemicolon), @(MILTokenKindEndOfFile)
    ];
    expect(tokens.count == expected.count, @"conv fragment token count");
    NSUInteger count = MIN(tokens.count, expected.count);
    for (NSUInteger index = 0; index < count; ++index) {
        expect(tokens[index].kind == expected[index].unsignedIntegerValue,
               [NSString stringWithFormat:@"conv token kind at %lu",
                                          (unsigned long)index]);
    }
    expect(diagnostics.errorCount == 0, @"valid conv fragment has no errors");
}

static void testAttentionAndQuantLiterals(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    NSArray<MILToken *> *tokens = lex(
        @"// attention scale\nfp16 scale = const()[val = fp16(0.125)];\n"
         "int32 axis = const()[val = int32(-1)];\n"
         "fp16 qs = const()[val = fp16(0x1p-3)];\n"
         "string p = const()[val = string(\"@model_path/weights/weight.bin\")];",
        &diagnostics);
    NSMutableArray<NSString *> *numbers = [NSMutableArray array];
    NSString *path = nil;
    for (MILToken *token in tokens) {
        if (token.kind == MILTokenKindInteger ||
            token.kind == MILTokenKindFloatingPoint)
            [numbers addObject:token.spelling];
        if (token.kind == MILTokenKindString) path = token.spelling;
    }
    expect([numbers containsObject:@"0.125"], @"decimal float is one token");
    expect([numbers containsObject:@"-1"], @"negative integer is one token");
    expect([numbers containsObject:@"0x1p-3"], @"hex float is one token");
    expect([path isEqualToString:@"@model_path/weights/weight.bin"],
           @"blob path string is decoded");
    expect(diagnostics.errorCount == 0,
           @"attention and quant literals have no errors");
}

static void testLocationsAndMalformedInput(void) {
    ANEDiagnosticEngine *diagnostics = nil;
    NSArray<MILToken *> *tokens = lex(@"program\n  main", &diagnostics);
    expect(tokens[1].range.start.line == 2 &&
           tokens[1].range.start.column == 3,
           @"line and column are one-based");

    lex(@"string(\"unterminated)", &diagnostics);
    expect(diagnostics.errorCount == 1,
           @"unterminated string emits one error");
    expect([diagnostics.diagnostics[0].code isEqualToString:
            @"mil.lex.unterminated-string"],
           @"unterminated string diagnostic is stable");

    lex(@"/* unterminated", &diagnostics);
    expect(diagnostics.errorCount == 1,
           @"unterminated block comment emits one error");
}

int main(void) {
    @autoreleasepool {
        testConvFragment();
        testAttentionAndQuantLiterals();
        testLocationsAndMalformedInput();
        printf("mil lexer: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
