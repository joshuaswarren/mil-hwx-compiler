#import "ANEDiagnostic.h"

@interface ANEDiagnostic ()
@property(nonatomic, readwrite) ANEDiagnosticSeverity severity;
@property(nonatomic, readwrite, copy) NSString *code;
@property(nonatomic, readwrite, copy) NSString *message;
@property(nonatomic, readwrite) ANESourceRange range;
@end

@implementation ANEDiagnostic
@end

@interface ANEDiagnosticEngine ()
@property(nonatomic) NSMutableArray<ANEDiagnostic *> *mutableDiagnostics;
@property(nonatomic, readwrite) NSUInteger errorCount;
@property(nonatomic, readwrite) BOOL hasFatalError;
@end

@implementation ANEDiagnosticEngine

- (instancetype)init {
    self = [super init];
    if (self) _mutableDiagnostics = [NSMutableArray array];
    return self;
}

- (NSArray<ANEDiagnostic *> *)diagnostics {
    return [_mutableDiagnostics copy];
}

- (void)emitSeverity:(ANEDiagnosticSeverity)severity
                 code:(NSString *)code
              message:(NSString *)message
                range:(ANESourceRange)range {
    NSParameterAssert(code != nil);
    NSParameterAssert(message != nil);
    ANEDiagnostic *diagnostic = [[ANEDiagnostic alloc] init];
    diagnostic.severity = severity;
    diagnostic.code = code;
    diagnostic.message = message;
    diagnostic.range = range;
    [_mutableDiagnostics addObject:diagnostic];
    if (severity >= ANEDiagnosticSeverityError) _errorCount++;
    if (severity == ANEDiagnosticSeverityFatal) _hasFatalError = YES;
}

@end

