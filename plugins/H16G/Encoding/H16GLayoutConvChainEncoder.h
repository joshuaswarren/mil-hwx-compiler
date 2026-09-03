#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GLayoutConvChainEncoder : NSObject
+ (nullable H16GEncodedTDProgram *)encodeNaturalChannels:(NSUInteger)channels
                                                 spatial:(NSUInteger)spatial
                                               blockSize:(NSUInteger)blockSize
                                                   error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END

