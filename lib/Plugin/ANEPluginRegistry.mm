#import "ANEPluginRegistry.h"

@implementation ANEPluginMatch
- (instancetype)initWithKind:(ANEPluginMatchKind)kind
                        detail:(NSString *)detail {
    self = [super init];
    if (self) {
        _kind = kind;
        _detail = [detail copy];
    }
    return self;
}
@end

@interface ANEPluginRegistry ()
@property(nonatomic) NSMutableDictionary<NSString *, id<ANECompilerPlugin>> *plugins;
@end

@implementation ANEPluginRegistry
- (instancetype)init {
    self = [super init];
    if (self) _plugins = [NSMutableDictionary dictionary];
    return self;
}

+ (ANESourceRange)syntheticRange {
    ANESourceLocation location = ANESourceLocationMake(0, 1, 1);
    return ANESourceRangeMake(location, location);
}

- (BOOL)registerPlugin:(id<ANECompilerPlugin>)plugin
            diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (_plugins[plugin.identifier]) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
                            code:@"ane.plugin.duplicate-identifier"
                         message:[NSString stringWithFormat:
                            @"plugin '%@' is already registered",
                            plugin.identifier]
                           range:[ANEPluginRegistry syntheticRange]];
        return NO;
    }
    _plugins[plugin.identifier] = plugin;
    return YES;
}

- (id<ANEPatternPlugin>)resolveCapability:(NSString *)capability
                                    object:(id)object
                                    target:(NSString *)target
                               diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSMutableArray<id<ANEPatternPlugin>> *matches = [NSMutableArray array];
    for (id<ANECompilerPlugin> plugin in _plugins.allValues) {
        if (![plugin conformsToProtocol:@protocol(ANEPatternPlugin)] ||
            ![plugin.capabilities containsObject:capability])
            continue;
        id<ANEPatternPlugin> pattern = (id<ANEPatternPlugin>)plugin;
        ANEPluginMatch *match = [pattern matchObject:object target:target];
        if (match.kind == ANEPluginMatchKindInvalid) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"ane.plugin.invalid-input"
                             message:[NSString stringWithFormat:
                                @"plugin '%@' rejected invalid input: %@",
                                pattern.identifier, match.detail]
                               range:[ANEPluginRegistry syntheticRange]];
            return nil;
        }
        if (match.kind == ANEPluginMatchKindMatch) [matches addObject:pattern];
    }
    if (matches.count == 0) return nil;
    [matches sortUsingComparator:^NSComparisonResult(
        id<ANEPatternPlugin> left, id<ANEPatternPlugin> right) {
        if (left.priority > right.priority) return NSOrderedAscending;
        if (left.priority < right.priority) return NSOrderedDescending;
        return [left.identifier compare:right.identifier];
    }];
    if (matches.count > 1 && matches[0].priority == matches[1].priority) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
                            code:@"ane.plugin.ambiguous"
                         message:[NSString stringWithFormat:
                            @"plugins '%@' and '%@' match capability '%@' at priority %ld",
                            matches[0].identifier, matches[1].identifier,
                            capability, (long)matches[0].priority]
                           range:[ANEPluginRegistry syntheticRange]];
        return nil;
    }
    return matches[0];
}
@end

