#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"

@class ANEExecutableBundle;

NS_ASSUME_NONNULL_BEGIN

@interface ANECompiler : NSObject
- (nullable ANEExecutableBundle *)compileMILData:(NSData *)milData
                                       modelRoot:(NSURL *)modelRoot
                                          target:(NSString *)target
                                     diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END
