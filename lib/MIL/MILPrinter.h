#import <Foundation/Foundation.h>

#import "MILSyntax.h"

NS_ASSUME_NONNULL_BEGIN

@interface MILPrinter : NSObject
+ (NSString *)stringForProgram:(MILProgramSyntax *)program;
@end

NS_ASSUME_NONNULL_END
