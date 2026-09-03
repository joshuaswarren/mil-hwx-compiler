#import <Foundation/Foundation.h>

#import "ANEScheduledGraph.h"
#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GLayoutEncoder : NSObject
+ (nullable H16GEncodedTDProgram *)encodeOperationName:(NSString *)operationName
    inputShape:(NSArray<NSNumber *> *)inputShape
    outputShape:(NSArray<NSNumber *> *)outputShape
    blockSize:(NSUInteger)blockSize
    strategy:(ANETileStrategy)strategy
    error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
