#import "ANENormalizePass.h"

@implementation ANENormalizePass
+ (BOOL)runOnGraph:(ANEOperationGraph *)graph
       diagnostics:(ANEDiagnosticEngine *)diagnostics {
    (void)diagnostics;
    NSMutableSet<ANEOperationNode *> *live = [NSMutableSet set];
    NSMutableArray<ANEOperationNode *> *work = [NSMutableArray array];
    for (NSString *name in graph.outputValueNames) {
        ANEOperationNode *node = [graph nodeForValueName:name];
        if (node) [work addObject:node];
    }
    while (work.count) {
        ANEOperationNode *node = work.lastObject;
        [work removeLastObject];
        if ([live containsObject:node]) continue;
        [live addObject:node];
        [work addObjectsFromArray:node.inputs];
    }
    NSMutableArray<ANEOperationNode *> *dead = [NSMutableArray array];
    for (ANEOperationNode *node in graph.nodes)
        if (![live containsObject:node]) [dead addObject:node];
    [graph removeNodes:dead];
    return YES;
}
@end
