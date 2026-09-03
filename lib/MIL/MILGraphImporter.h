#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEGraphIR.h"
#import "MILSyntax.h"

NS_ASSUME_NONNULL_BEGIN

@interface MILGraphImporter : NSObject
+ (nullable ANEGraphModule *)importProgram:(MILProgramSyntax *)program
                               diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END

