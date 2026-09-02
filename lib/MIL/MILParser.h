#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "MILSyntax.h"
#import "MILToken.h"

NS_ASSUME_NONNULL_BEGIN

@interface MILParser : NSObject
- (instancetype)initWithTokens:(NSArray<MILToken *> *)tokens
                    diagnostics:(ANEDiagnosticEngine *)diagnostics;
- (nullable MILProgramSyntax *)parseProgram;
@end

NS_ASSUME_NONNULL_END
