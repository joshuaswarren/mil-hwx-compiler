#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, ANEPluginMatchKind) {
    ANEPluginMatchKindDecline,
    ANEPluginMatchKindMatch,
    ANEPluginMatchKindInvalid,
};

@interface ANEPluginMatch : NSObject
@property(nonatomic, readonly) ANEPluginMatchKind kind;
@property(nonatomic, readonly, copy) NSString *detail;
- (instancetype)initWithKind:(ANEPluginMatchKind)kind
                        detail:(NSString *)detail;
@end

@protocol ANECompilerPlugin <NSObject>
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly) NSUInteger pluginVersion;
@property(nonatomic, readonly, copy) NSSet<NSString *> *capabilities;
@end

@protocol ANEPatternPlugin <ANECompilerPlugin>
@property(nonatomic, readonly) NSInteger priority;
- (ANEPluginMatch *)matchObject:(id)object target:(NSString *)target;
@end

NS_ASSUME_NONNULL_END

