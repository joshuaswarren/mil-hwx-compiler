#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEPassManager.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

@interface RecordingPass : NSObject <ANEPass>
- (instancetype)initWithIdentifier:(NSString *)identifier
                             input:(ANEIRLevel)input
                            output:(ANEIRLevel)output
                            suffix:(NSString *)suffix
                           succeed:(BOOL)succeed;
@end

@implementation RecordingPass {
    NSString *_identifier;
    ANEIRLevel _input;
    ANEIRLevel _output;
    NSString *_suffix;
    BOOL _succeed;
}
- (instancetype)initWithIdentifier:(NSString *)identifier
                             input:(ANEIRLevel)input
                            output:(ANEIRLevel)output
                            suffix:(NSString *)suffix
                           succeed:(BOOL)succeed {
    self = [super init];
    if (self) {
        _identifier = [identifier copy];
        _input = input;
        _output = output;
        _suffix = [suffix copy];
        _succeed = succeed;
    }
    return self;
}
- (NSString *)identifier { return _identifier; }
- (ANEIRLevel)inputLevel { return _input; }
- (ANEIRLevel)outputLevel { return _output; }
- (BOOL)runOnState:(ANECompilationState *)state
            context:(ANEPassContext *)context {
    if (!_succeed) {
        [context.diagnostics emitSeverity:ANEDiagnosticSeverityError
                                     code:@"test.pass.failed"
                                  message:_identifier
                                    range:ANESourceRangeMake(
                                        ANESourceLocationMake(0, 1, 1),
                                        ANESourceLocationMake(0, 1, 1))];
        return NO;
    }
    state.payload = [NSString stringWithFormat:@"%@%@", state.payload, _suffix];
    return YES;
}
@end

static void testOrderedPipeline(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEPassContext *context = [[ANEPassContext alloc]
        initWithDiagnostics:diagnostics];
    ANEPassManager *manager = [[ANEPassManager alloc] init];
    [manager addPass:[[RecordingPass alloc] initWithIdentifier:@"parse"
        input:ANEIRLevelSyntax output:ANEIRLevelGraph suffix:@"G" succeed:YES]];
    [manager addPass:[[RecordingPass alloc] initWithIdentifier:@"legalize"
        input:ANEIRLevelGraph output:ANEIRLevelPrimitive suffix:@"P" succeed:YES]];
    ANECompilationState *state = [[ANECompilationState alloc]
        initWithLevel:ANEIRLevelSyntax payload:@"S"];
    expect([manager runState:state context:context], @"ordered pipeline succeeds");
    expect(state.level == ANEIRLevelPrimitive, @"pipeline advances IR level");
    expect([state.payload isEqual:@"SGP"], @"passes run in insertion order");
    expect([state.passTrace isEqualToArray:@[@"parse", @"legalize"]],
           @"pass trace records successful passes");
}

static void testLevelAndFailureGates(void) {
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEPassContext *context = [[ANEPassContext alloc]
        initWithDiagnostics:diagnostics];
    ANEPassManager *wrongLevel = [[ANEPassManager alloc] init];
    [wrongLevel addPass:[[RecordingPass alloc] initWithIdentifier:@"machine"
        input:ANEIRLevelMachine output:ANEIRLevelExecutable suffix:@"X"
        succeed:YES]];
    ANECompilationState *state = [[ANECompilationState alloc]
        initWithLevel:ANEIRLevelGraph payload:@"G"];
    expect(![wrongLevel runState:state context:context],
           @"pass cannot consume the wrong IR level");
    expect([diagnostics.diagnostics[0].code isEqualToString:
            @"ane.pass.level-mismatch"], @"level mismatch is diagnosed");

    diagnostics = [[ANEDiagnosticEngine alloc] init];
    context = [[ANEPassContext alloc] initWithDiagnostics:diagnostics];
    ANEPassManager *failure = [[ANEPassManager alloc] init];
    [failure addPass:[[RecordingPass alloc] initWithIdentifier:@"bad"
        input:ANEIRLevelGraph output:ANEIRLevelPrimitive suffix:@"P"
        succeed:NO]];
    [failure addPass:[[RecordingPass alloc] initWithIdentifier:@"unreachable"
        input:ANEIRLevelPrimitive output:ANEIRLevelMachine suffix:@"M"
        succeed:YES]];
    state = [[ANECompilationState alloc] initWithLevel:ANEIRLevelGraph payload:@"G"];
    expect(![failure runState:state context:context], @"failed pass stops pipeline");
    expect(state.level == ANEIRLevelGraph && state.passTrace.count == 0,
           @"failed pass does not commit level or trace");
}

int main(void) {
    @autoreleasepool {
        testOrderedPipeline();
        testLevelAndFailureGates();
        printf("pass manager: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}

