#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"

@class ANEHWXArtifact;

NS_ASSUME_NONNULL_BEGIN

@interface H16GProgramEncoder : NSObject
+ (nullable ANEHWXArtifact *)encodeGraph:(ANEOperationGraph *)graph
                                scheduled:(ANEScheduledGraph *)scheduled
                                modelRoot:(NSURL *)modelRoot
                              diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END
