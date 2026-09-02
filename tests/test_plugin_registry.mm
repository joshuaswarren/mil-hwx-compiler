#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEPlugin.h"
#import "ANEPluginRegistry.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    failures++;
}

@interface TestPattern : NSObject <ANEPatternPlugin>
- (instancetype)initWithIdentifier:(NSString *)identifier
                           priority:(NSInteger)priority
                               kind:(ANEPluginMatchKind)kind;
@end

@implementation TestPattern {
    NSString *_identifier;
    NSInteger _priority;
    ANEPluginMatchKind _kind;
}
- (instancetype)initWithIdentifier:(NSString *)identifier
                           priority:(NSInteger)priority
                               kind:(ANEPluginMatchKind)kind {
    self = [super init];
    if (self) {
        _identifier = [identifier copy];
        _priority = priority;
        _kind = kind;
    }
    return self;
}
- (NSString *)identifier { return _identifier; }
- (NSUInteger)pluginVersion { return 1; }
- (NSSet<NSString *> *)capabilities { return [NSSet setWithObject:@"conv"]; }
- (NSInteger)priority { return _priority; }
- (ANEPluginMatch *)matchObject:(id)object target:(NSString *)target {
    (void)object;
    (void)target;
    return [[ANEPluginMatch alloc] initWithKind:_kind detail:_identifier];
}
@end

static ANEDiagnosticEngine *newDiagnostics(void) {
    return [[ANEDiagnosticEngine alloc] init];
}

static void testSelectionAndDecline(void) {
    ANEPluginRegistry *registry = [[ANEPluginRegistry alloc] init];
    ANEDiagnosticEngine *diagnostics = newDiagnostics();
    id<ANEPatternPlugin> low = [[TestPattern alloc] initWithIdentifier:@"low"
        priority:10 kind:ANEPluginMatchKindMatch];
    id<ANEPatternPlugin> high = [[TestPattern alloc] initWithIdentifier:@"high"
        priority:20 kind:ANEPluginMatchKindMatch];
    id<ANEPatternPlugin> decline = [[TestPattern alloc]
        initWithIdentifier:@"decline" priority:100
        kind:ANEPluginMatchKindDecline];
    expect([registry registerPlugin:low diagnostics:diagnostics],
           @"first plugin registers");
    expect([registry registerPlugin:high diagnostics:diagnostics],
           @"second plugin registers");
    expect([registry registerPlugin:decline diagnostics:diagnostics],
           @"declining plugin registers");
    id<ANEPatternPlugin> selected = [registry resolveCapability:@"conv"
        object:@"graph" target:@"H16G" diagnostics:diagnostics];
    expect([selected.identifier isEqualToString:@"high"],
           @"highest-priority matching plugin wins");
}

static void testDuplicateAmbiguityAndInvalid(void) {
    ANEPluginRegistry *registry = [[ANEPluginRegistry alloc] init];
    ANEDiagnosticEngine *diagnostics = newDiagnostics();
    id<ANEPatternPlugin> one = [[TestPattern alloc] initWithIdentifier:@"one"
        priority:7 kind:ANEPluginMatchKindMatch];
    expect([registry registerPlugin:one diagnostics:diagnostics],
           @"plugin registers");
    expect(![registry registerPlugin:one diagnostics:diagnostics],
           @"duplicate identifier is rejected");

    registry = [[ANEPluginRegistry alloc] init];
    diagnostics = newDiagnostics();
    [registry registerPlugin:[[TestPattern alloc] initWithIdentifier:@"a"
        priority:7 kind:ANEPluginMatchKindMatch] diagnostics:diagnostics];
    [registry registerPlugin:[[TestPattern alloc] initWithIdentifier:@"b"
        priority:7 kind:ANEPluginMatchKindMatch] diagnostics:diagnostics];
    expect([registry resolveCapability:@"conv" object:@"g" target:@"H16G"
                           diagnostics:diagnostics] == nil,
           @"equal-priority matches are ambiguous");
    expect([diagnostics.diagnostics.lastObject.code isEqualToString:
            @"ane.plugin.ambiguous"], @"ambiguity is diagnosed");

    registry = [[ANEPluginRegistry alloc] init];
    diagnostics = newDiagnostics();
    [registry registerPlugin:[[TestPattern alloc] initWithIdentifier:@"invalid"
        priority:1 kind:ANEPluginMatchKindInvalid] diagnostics:diagnostics];
    expect([registry resolveCapability:@"conv" object:@"g" target:@"H16G"
                           diagnostics:diagnostics] == nil,
           @"invalid match stops selection");
    expect([diagnostics.diagnostics.lastObject.code isEqualToString:
            @"ane.plugin.invalid-input"], @"invalid match is diagnosed");
}

int main(void) {
    @autoreleasepool {
        testSelectionAndDecline();
        testDuplicateAmbiguityAndInvalid();
        printf("plugin registry: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
