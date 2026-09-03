#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GALUEncoder : NSObject
+ (BOOL)supportsOperationName:(NSString *)operationName squareSize:(NSUInteger)size;
+ (nullable H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                            squareSize:(NSUInteger)size
                                                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
