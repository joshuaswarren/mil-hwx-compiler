#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

#include <stdio.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) {
            fprintf(stderr, "usage: %s OUTPUT_DIR\n", argv[0]);
            return 64;
        }
        NSData *mil = [NSData dataWithContentsOfFile:
            @"tests/fixtures/attention.mil"];
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = [ANEStagedCompiler compileMILData:mil
            modelRoot:[NSURL fileURLWithPath:@"tests/models/attention"
                                 isDirectory:YES]
            target:@"H16G" diagnostics:diagnostics];
        for (ANEDiagnostic *diagnostic in diagnostics.diagnostics)
            fprintf(stderr, "%s: %s\n", diagnostic.code.UTF8String,
                    diagnostic.message.UTF8String);
        if (!bundle) return 2;
        NSError *error = nil;
        NSURL *output = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]] isDirectory:YES];
        if (![bundle writeToDirectory:output error:&error]) {
            fprintf(stderr, "bundle write failed: %s\n",
                    error.description.UTF8String);
            return 3;
        }
        printf("staged mixed bundle passes=%s bytes=%lu\n",
               [bundle.passTrace componentsJoinedByString:@","].UTF8String,
               (unsigned long)bundle.artifacts[0].image.length);
        return 0;
    }
}
