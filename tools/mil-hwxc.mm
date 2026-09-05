#import <Foundation/Foundation.h>

#import "ANECompiler.h"
#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEH13Compiler.h"

#include <stdio.h>

static void printUsage(const char *program) {
    fprintf(stderr,
        "usage: %s --mil FILE --model-root DIR "
        "--output DIR [--target H16G|H13]\n", program);
}

static NSString *severityName(ANEDiagnosticSeverity severity) {
    switch (severity) {
        case ANEDiagnosticSeverityNote: return @"note";
        case ANEDiagnosticSeverityWarning: return @"warning";
        case ANEDiagnosticSeverityError: return @"error";
        case ANEDiagnosticSeverityFatal: return @"fatal";
    }
}

static void printDiagnostics(ANEDiagnosticEngine *engine) {
    for (ANEDiagnostic *diagnostic in engine.diagnostics) {
        fprintf(stderr, "%lu:%lu: %s [%s]: %s\n",
            (unsigned long)diagnostic.range.start.line,
            (unsigned long)diagnostic.range.start.column,
            severityName(diagnostic.severity).UTF8String,
            diagnostic.code.UTF8String, diagnostic.message.UTF8String);
    }
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSMutableDictionary<NSString *, NSString *> *arguments =
            [NSMutableDictionary dictionary];
        for (int index = 1; index < argc; ++index) {
            NSString *key = [NSString stringWithUTF8String:argv[index]];
            if ([key isEqualToString:@"--help"]) {
                printUsage(argv[0]);
                return 0;
            }
            if (![key hasPrefix:@"--"] || index + 1 >= argc) {
                printUsage(argv[0]);
                return 64;
            }
            arguments[key] = [NSString stringWithUTF8String:argv[++index]];
        }

        NSString *milPath = arguments[@"--mil"];
        NSString *modelRoot = arguments[@"--model-root"];
        NSString *output = arguments[@"--output"];
        NSString *target = arguments[@"--target"] ?: @"H16G";
        NSSet<NSString *> *accepted = [NSSet setWithArray:@[
            @"--mil", @"--model-root", @"--output", @"--target",
        ]];
        for (NSString *key in arguments) {
            if (![accepted containsObject:key]) {
                fprintf(stderr, "unknown option: %s\n", key.UTF8String);
                printUsage(argv[0]);
                return 64;
            }
        }
        if (!milPath || !modelRoot || !output) {
            printUsage(argv[0]);
            return 64;
        }

        NSData *milData = [NSData dataWithContentsOfFile:milPath];
        if (!milData) {
            fprintf(stderr, "cannot read MIL file: %s\n", milPath.UTF8String);
            return 66;
        }
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        if ([target isEqualToString:@"H13"]) {
            NSError *error = nil;
            BOOL success = [ANEH13Compiler compileMILData:milData
                modelRoot:[NSURL fileURLWithPath:modelRoot isDirectory:YES]
                outputDirectory:[NSURL fileURLWithPath:output isDirectory:YES]
                diagnostics:diagnostics error:&error];
            printDiagnostics(diagnostics);
            if (!success) {
                if (error) fprintf(stderr, "cannot write H13 package: %s\n",
                    error.description.UTF8String);
                return error ? 74 : 65;
            }
            printf("compiled target=H13 artifacts=1 format=anec output=%s\n",
                output.UTF8String);
            return 0;
        }
        ANECompiler *compiler = [[ANECompiler alloc] init];
        ANEExecutableBundle *bundle = [compiler compileMILData:milData
            modelRoot:[NSURL fileURLWithPath:modelRoot isDirectory:YES]
            target:target diagnostics:diagnostics];
        printDiagnostics(diagnostics);
        if (!bundle) return 65;

        NSError *error = nil;
        if (![bundle writeToDirectory:
                [NSURL fileURLWithPath:output isDirectory:YES] error:&error]) {
            fprintf(stderr, "cannot write bundle: %s\n",
                    error.description.UTF8String);
            return 74;
        }
        printf("compiled target=%s artifacts=%lu output=%s\n",
               bundle.target.UTF8String, (unsigned long)bundle.artifacts.count,
               output.UTF8String);
        return 0;
    }
}
