#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEPlugin.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEPluginRegistry : NSObject
- (BOOL)registerPlugin:(id<ANECompilerPlugin>)plugin
            diagnostics:(ANEDiagnosticEngine *)diagnostics;
- (nullable id<ANEPatternPlugin>)resolveCapability:(NSString *)capability
                                             object:(id)object
                                             target:(NSString *)target
                                        diagnostics:(ANEDiagnosticEngine *)diagnostics;
@end

NS_ASSUME_NONNULL_END

