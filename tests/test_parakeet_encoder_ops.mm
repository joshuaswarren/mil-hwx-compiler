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

int main(void) {
    @autoreleasepool {
        testEncoderFirstOpExpandDimsCompiles();
        printf("Parakeet encoder ops: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
