#import <Foundation/Foundation.h>

#import "ANECompiler.h"
#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"

#include <stdio.h>

// Parakeet TDT encoder coverage, failing-first (SUP-30).
// Encoder MIL: parakeet-tdt-0.6b-v3 encoder, first compute op in program
// order is expand_dims ([3000,128] -> [1,1,3000,128]). Today the H16G
// staged pipeline rejects it at h16g.legalize with
// "H16G has no decoded primitive for 'expand_dims'". This test asserts
// the desired end state; it goes green when a decoded primitive lands.

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

static NSString *expandDimsMIL(void) {
    return @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
            "  func main<ios18>(tensor<fp16, [3000, 128]> x) {\n"
            "    tensor<int32, [2]> ax = const()[name = string(\"ax\"),"
            " val = tensor<int32, [2]>([1, 2])];\n"
            "    tensor<fp16, [1, 1, 3000, 128]> y ="
            " expand_dims(axes = ax, x = x)[name = string(\"y\")];\n"
            "  } -> (y);\n}\n";
}

static void testEncoderFirstOpExpandDimsCompiles(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    NSURL *modelRoot = [NSURL fileURLWithPath:@"tests/models/parakeet"
                                  isDirectory:YES];
    ANEExecutableBundle *bundle = [compiler
        compileMILData:[expandDimsMIL() dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:modelRoot target:@"H16G" diagnostics:diagnostics];
    if (bundle == nil)
        for (ANEDiagnostic *entry in diagnostics.diagnostics)
            fprintf(stderr, "diag [%s]: %s\n", entry.code.UTF8String, entry.message.UTF8String);
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"encoder op expand_dims lowers through the staged pipeline");
}

// ---- SUP-86, class 5: matmul and conv shape variants ----
// w7X SUP-30 measured: matmul lowers square-[1,N,N]-only; conv decodes in
// exactly one form. The Parakeet encoder drives matmul rectangular
// ([3000,128] @ [128,384] qkv projection) and a depthwise-then-pointwise
// conv frontend ([1,128,3000], kernel [256,128,5,1]) - neither is a
// decoded form, so both cases below are expected RED on main until a
// decoded primitive lands. The square attention matmul
// ([1,3000,128] @ [1,128,3000]) is the decoded form and is expected GREEN.

static NSString *matmulRectEncoderMIL(void) {
    return @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
            "  func main<ios18>(tensor<fp16, [3000, 128]> x,\n"
            "                  tensor<fp16, [128, 384]> w) {\n"
            "    tensor<fp16, [3000, 384]> y = matmul(x = x,\n"
            "        weight = w)[name = string(\"y\")];\n"
            "  } -> (y);\n}\n";
}

static NSString *matmulSquareAttentionMIL(void) {
    return @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
            "  func main<ios18>(tensor<fp16, [1, 3000, 128]> q,\n"
            "                  tensor<fp16, [1, 128, 3000]> k) {\n"
            "    tensor<fp16, [1, 3000, 3000]> s = matmul(x = q,\n"
            "        weight = k)[name = string(\"s\")];\n"
            "  } -> (s);\n}\n";
}

static NSString *convEncoderFrontendMIL(void) {
    return @"program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n"
            "  func main<ios18>(tensor<fp16, [1, 128, 3000]> x,\n"
            "                  tensor<fp16, [256, 128, 5, 1]> wt) {\n"
            "    tensor<fp16, [1, 256, 3000]> y = conv(x = x,\n"
            "        weight = wt, pads = [0, 0, 2, 2])[name = string(\"y\")];\n"
            "  } -> (y);\n}\n";
}

static BOOL compileMIL(NSString *mil) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    NSURL *modelRoot = [NSURL fileURLWithPath:@"tests/models/parakeet"
                                  isDirectory:YES];
    ANEExecutableBundle *bundle = [compiler
        compileMILData:[mil dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:modelRoot target:@"H16G" diagnostics:diagnostics];
    if (bundle == nil)
        for (ANEDiagnostic *entry in diagnostics.diagnostics)
            fprintf(stderr, "diag [%s]: %s\n", entry.code.UTF8String, entry.message.UTF8String);
    return bundle != nil && diagnostics.errorCount == 0;
}

static void testClass5MatmulRectEncoder(void) {
    BOOL ok = compileMIL(matmulRectEncoderMIL());
    expect(ok, @"SUP-86 class5: rectangular encoder matmul [3000,128]@[128,384] "
               "lowers (w7X measured square-[1,N,N]-only; this should be RED)");
}

static void testClass5MatmulSquareAttention(void) {
    BOOL ok = compileMIL(matmulSquareAttentionMIL());
    expect(ok, @"SUP-86 class5: square attention matmul [1,3000,128]@[1,128,3000] "
               "lowers (w7X measured square-[1,N,N] as the decoded form; expected GREEN)");
}

static void testClass5ConvEncoderFrontend(void) {
    BOOL ok = compileMIL(convEncoderFrontendMIL());
    expect(ok, @"SUP-86 class5: encoder conv frontend [1,128,3000] conv [256,128,5,1] "
               "lowers (w7X measured decoded-form-only; expected RED)");
}

int main(void) {
    @autoreleasepool {
        testEncoderFirstOpExpandDimsCompiles();
        testClass5MatmulRectEncoder();
        testClass5MatmulSquareAttention();
        testClass5ConvEncoderFrontend();
        printf("Parakeet encoder ops: %s\n", failures == 0 ? "PASS" : "FAIL");
        printf("SUP-86 class5 green list: square [1,N,N] matmul; conv decoded-form-only "
               "(per w7X SUP-30, receipts/2026-11-26-w7X-sup30-parakeet-encoder-lowering-gaps.md). "
               "Red list: rectangular encoder matmul, encoder conv frontend.\n");
    }
    return failures == 0 ? 0 : 1;
}

