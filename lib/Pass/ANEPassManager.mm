#import "ANEPassManager.h"

@interface ANECompilationState ()
@property(nonatomic) NSMutableArray<NSString *> *mutablePassTrace;
@end

@implementation ANECompilationState
- (instancetype)initWithLevel:(ANEIRLevel)level payload:(id)payload {
    self = [super init];
    if (self) {
        _level = level;
        _payload = payload;
        _mutablePassTrace = [NSMutableArray array];
    }
    return self;
}
- (NSArray<NSString *> *)passTrace { return [_mutablePassTrace copy]; }
- (void)recordCompletedPass:(NSString *)identifier {
    [_mutablePassTrace addObject:identifier];
}
@end

@interface ANEPassContext ()
@property(nonatomic) NSMutableDictionary<NSNumber *, ANEIRVerifier> *verifiers;
@end

@implementation ANEPassContext
- (instancetype)initWithDiagnostics:(ANEDiagnosticEngine *)diagnostics {
    self = [super init];
    if (self) {
        _diagnostics = diagnostics;
        _verifiers = [NSMutableDictionary dictionary];
    }
    return self;
}
- (void)setVerifier:(ANEIRVerifier)verifier forLevel:(ANEIRLevel)level {
    _verifiers[@(level)] = [verifier copy];
}
- (BOOL)verifyState:(ANECompilationState *)state {
    ANEIRVerifier verifier = _verifiers[@(state.level)];
    return verifier ? verifier(state.payload, _diagnostics) : YES;
}
@end

@interface ANEPassManager ()
@property(nonatomic) NSMutableArray<id<ANEPass>> *passes;
@end

@implementation ANEPassManager
- (instancetype)init {
    self = [super init];
    if (self) _passes = [NSMutableArray array];
    return self;
}
- (void)addPass:(id<ANEPass>)pass {
    NSParameterAssert(pass != nil);
    [_passes addObject:pass];
}
- (BOOL)runState:(ANECompilationState *)state context:(ANEPassContext *)context {
    for (id<ANEPass> pass in _passes) {
        if (state.level != pass.inputLevel) {
            NSString *message = [NSString stringWithFormat:
                @"pass '%@' requires IR level %lu, current level is %lu",
                pass.identifier, (unsigned long)pass.inputLevel,
                (unsigned long)state.level];
            [context.diagnostics emitSeverity:ANEDiagnosticSeverityError
                                         code:@"ane.pass.level-mismatch"
                                      message:message
                                        range:ANESourceRangeMake(
                                            ANESourceLocationMake(0, 1, 1),
                                            ANESourceLocationMake(0, 1, 1))];
            return NO;
        }
        if (![context verifyState:state]) return NO;
        if (![pass runOnState:state context:context]) return NO;
        state.level = pass.outputLevel;
        if (![context verifyState:state]) return NO;
        [state recordCompletedPass:pass.identifier];
    }
    return YES;
}
@end

