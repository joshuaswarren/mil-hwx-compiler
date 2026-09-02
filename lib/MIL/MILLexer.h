#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "MILToken.h"

@interface MILLexer : NSObject
- (instancetype)initWithData:(NSData *)data
                  diagnostics:(ANEDiagnosticEngine *)diagnostics;
- (NSArray<MILToken *> *)lexAllTokens;
@end

