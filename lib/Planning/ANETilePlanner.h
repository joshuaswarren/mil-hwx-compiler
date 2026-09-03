#import <Foundation/Foundation.h>
#import "ANEScheduledGraph.h"
#import "H16GTarget.h"
NS_ASSUME_NONNULL_BEGIN
@interface ANETilePlanner : NSObject
+ (ANETilePlan *)tilePlanForNode:(ANEOperationNode *)node
                          target:(H16GTarget *)target;
@end
NS_ASSUME_NONNULL_END
