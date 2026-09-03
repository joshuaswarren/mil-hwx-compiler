#import <Foundation/Foundation.h>

#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEComposedRegionPlanner : NSObject
+ (NSArray<ANEScheduledRegionPlan *> *)plansForGraph:(ANEOperationGraph *)graph
                                               tasks:(NSArray<ANEScheduledTask *> *)tasks;
@end

NS_ASSUME_NONNULL_END
