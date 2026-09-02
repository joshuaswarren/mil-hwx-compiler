#import "ANECompiler.h"

#import "ANEExecutableBundle.h"
#import "ANEStagedCompiler.h"

@implementation ANECompiler
- (ANEExecutableBundle *)compileMILData:(NSData *)milData
                              modelRoot:(NSURL *)modelRoot
                                 target:(NSString *)target
                            diagnostics:(ANEDiagnosticEngine *)diagnostics {
    return [ANEStagedCompiler compileMILData:milData modelRoot:modelRoot
                                      target:target diagnostics:diagnostics];
}
@end
