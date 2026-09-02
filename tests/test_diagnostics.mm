#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"

#include <stdio.h>

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

int main(void) {
    @autoreleasepool {
        ANEDiagnosticEngine *engine = [[ANEDiagnosticEngine alloc] init];
        ANESourceRange first = ANESourceRangeMake(
            ANESourceLocationMake(8, 2, 3),
            ANESourceLocationMake(12, 2, 7));
        ANESourceRange second = ANESourceRangeMake(
            ANESourceLocationMake(21, 4, 1),
            ANESourceLocationMake(22, 4, 2));

        [engine emitSeverity:ANEDiagnosticSeverityWarning
                        code:@"mil.warning"
                     message:@"first"
                       range:first];
        [engine emitSeverity:ANEDiagnosticSeverityError
                        code:@"mil.error"
                     message:@"second"
                       range:second];

        NSArray<ANEDiagnostic *> *items = engine.diagnostics;
        expect(items.count == 2, @"diagnostics retain emission order");
        expect(engine.errorCount == 1, @"only errors increase errorCount");
        expect(!engine.hasFatalError, @"ordinary errors are not fatal");
        expect([items[0].code isEqualToString:@"mil.warning"],
               @"first code is preserved");
        expect(items[1].range.start.line == 4 &&
               items[1].range.start.column == 1,
               @"source range is preserved");

        [engine emitSeverity:ANEDiagnosticSeverityFatal
                        code:@"mil.fatal"
                     message:@"stop"
                       range:second];
        expect(engine.errorCount == 2, @"fatal diagnostics count as errors");
        expect(engine.hasFatalError, @"fatal state is observable");

        printf("diagnostics: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
