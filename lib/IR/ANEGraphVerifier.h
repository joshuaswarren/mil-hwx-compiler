#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEGraphIR.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEGraphVerifier : NSObject
+ (BOOL)verifyModule:(ANEGraphModule *)module
          diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END

