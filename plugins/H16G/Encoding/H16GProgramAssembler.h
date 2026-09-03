#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"

@class ANEHWXArtifact;
@class H16GTarget;

NS_ASSUME_NONNULL_BEGIN

@interface H16GAssembledProgram : NSObject
@property(nonatomic, readonly, copy) NSArray<ANEHWXArtifact *> *artifacts;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *dispatchPlan;
/// One line per planner transition decision plus one line per emitted
/// partition. Every declined transition names its exact reason.
@property(nonatomic, readonly, copy) NSArray<NSString *> *compositionTrace;
- (instancetype)initWithArtifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                      dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan;
- (instancetype)initWithArtifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                      dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan
                  compositionTrace:(NSArray<NSString *> *)compositionTrace;
@end

@interface H16GProgramAssembler : NSObject
+ (nullable H16GAssembledProgram *)assembleGraph:(ANEOperationGraph *)graph
                                       scheduled:(ANEScheduledGraph *)scheduled
                                          target:(H16GTarget *)target
                                     diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END
