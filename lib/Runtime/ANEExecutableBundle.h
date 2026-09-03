#import <Foundation/Foundation.h>

#import "ANEHWXArtifact.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEExecutableBundle : NSObject
@property(nonatomic, readonly, copy) NSString *target;
@property(nonatomic, readonly, copy) NSArray<ANEHWXArtifact *> *artifacts;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *dispatchPlan;
@property(nonatomic, readonly, copy) NSArray<NSString *> *sharedSurfaceIdentifiers;
@property(nonatomic, readonly, copy) NSArray<NSString *> *passTrace;
/// One line per adjacent scheduled-task transition the program planner
/// evaluated, stating whether it composed and, if not, the exact reason.
@property(nonatomic, readonly, copy) NSArray<NSString *> *compositionTrace;
- (instancetype)initWithTarget:(NSString *)target
                      artifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                   dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan
                      passTrace:(NSArray<NSString *> *)passTrace;
- (instancetype)initWithTarget:(NSString *)target
                      artifacts:(NSArray<ANEHWXArtifact *> *)artifacts
                   dispatchPlan:(NSArray<NSNumber *> *)dispatchPlan
                      passTrace:(NSArray<NSString *> *)passTrace
               compositionTrace:(NSArray<NSString *> *)compositionTrace;
- (BOOL)writeToDirectory:(NSURL *)directory error:(NSError **)error;
+ (nullable instancetype)bundleWithContentsOfDirectory:(NSURL *)directory
                                                  error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
