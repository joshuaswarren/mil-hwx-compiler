#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface H16GDepthwiseEncoder : NSObject
+ (BOOL)supportsChannels:(NSUInteger)channels spatial:(NSUInteger)spatial;
+ (nullable NSData *)encode3x3WithChannels:(NSUInteger)channels
                                    spatial:(NSUInteger)spatial
                                      error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
