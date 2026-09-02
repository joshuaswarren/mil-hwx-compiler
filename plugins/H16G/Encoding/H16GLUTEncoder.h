#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GLUTEncoder : NSObject
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape;
+ (nullable H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
                                             inputShape:(NSArray<NSNumber *> *)inputShape
                                                  error:(NSError **)error;
+ (nullable NSData *)constantRegionForOperationName:(NSString *)operationName
                                               error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
