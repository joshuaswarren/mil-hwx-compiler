#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEH13Compiler : NSObject
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
