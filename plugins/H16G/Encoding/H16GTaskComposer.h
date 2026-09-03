#import <Foundation/Foundation.h>

#import "H16GEncodedTask.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GTaskComposer : NSObject
+ (BOOL)supportsCapability:(H16GProgramCompositionCapability *)capability;
+ (nullable H16GEncodedTask *)composeProducer:(H16GEncodedTask *)producer
    consumer:(H16GEncodedTask *)consumer
    capability:(nullable H16GProgramCompositionCapability *)capability
    error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
