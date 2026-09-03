#import <Foundation/Foundation.h>

typedef struct {
    NSUInteger offset;
    NSUInteger line;
    NSUInteger column;
} ANESourceLocation;

typedef struct {
    ANESourceLocation start;
    ANESourceLocation end;
} ANESourceRange;

NS_INLINE ANESourceLocation ANESourceLocationMake(NSUInteger offset,
                                                   NSUInteger line,
                                                   NSUInteger column) {
    return (ANESourceLocation){offset, line, column};
}

NS_INLINE ANESourceRange ANESourceRangeMake(ANESourceLocation start,
                                             ANESourceLocation end) {
    return (ANESourceRange){start, end};
}

typedef NS_ENUM(NSUInteger, ANEDiagnosticSeverity) {
    ANEDiagnosticSeverityNote,
    ANEDiagnosticSeverityWarning,
    ANEDiagnosticSeverityError,
    ANEDiagnosticSeverityFatal,
};

@interface ANEDiagnostic : NSObject
@property(nonatomic, readonly) ANEDiagnosticSeverity severity;
@property(nonatomic, readonly, copy) NSString *code;
@property(nonatomic, readonly, copy) NSString *message;
@property(nonatomic, readonly) ANESourceRange range;
@end

@interface ANEDiagnosticEngine : NSObject
@property(nonatomic, readonly, copy) NSArray<ANEDiagnostic *> *diagnostics;
@property(nonatomic, readonly) NSUInteger errorCount;
@property(nonatomic, readonly) BOOL hasFatalError;

- (void)emitSeverity:(ANEDiagnosticSeverity)severity
                 code:(NSString *)code
              message:(NSString *)message
                range:(ANESourceRange)range;
@end

