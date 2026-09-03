#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEOperationGraph.h"

NS_ASSUME_NONNULL_BEGIN
@interface ANEDecomposePass : NSObject
+ (BOOL)runOnGraph:(ANEOperationGraph *)graph
       diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end
NS_ASSUME_NONNULL_END
