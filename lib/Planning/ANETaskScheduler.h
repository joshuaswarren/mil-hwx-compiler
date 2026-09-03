#import <Foundation/Foundation.h>
#import "ANEDiagnostic.h"
#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"
#import "H16GTarget.h"
NS_ASSUME_NONNULL_BEGIN
@interface ANETaskScheduler : NSObject
+ (nullable ANEScheduledGraph *)scheduleGraph:(ANEOperationGraph *)graph
                                       target:(H16GTarget *)target
                                  diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end
NS_ASSUME_NONNULL_END
