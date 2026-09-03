#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface H16GRegularConvEncoder : NSObject
+ (BOOL)supportsChannels:(NSUInteger)channels
                  spatial:(NSUInteger)spatial
                   kernel:(NSUInteger)kernel;
+ (nullable NSData *)encodeWithChannels:(NSUInteger)channels
                                 spatial:(NSUInteger)spatial
                                  kernel:(NSUInteger)kernel
                                   error:(NSError **)error;
+ (NSUInteger)kernelRelocationOffsetForChannels:(NSUInteger)channels;
@end

NS_ASSUME_NONNULL_END
