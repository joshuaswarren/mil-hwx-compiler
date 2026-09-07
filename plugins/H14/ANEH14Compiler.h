#pragma once

#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEH14Compiler : NSObject
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
