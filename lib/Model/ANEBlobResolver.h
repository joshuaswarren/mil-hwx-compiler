#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEGraphIR.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEBlobResolver : NSObject
+ (nullable NSData *)loadConstantForOperation:(ANEGraphOperation *)operation
                                expectedBytes:(NSUInteger)expectedBytes
                                     modelRoot:(NSURL *)modelRoot
                                   diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END
