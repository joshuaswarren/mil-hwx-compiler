#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

@class ANEExecutableBundle;

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const ANERuntimeErrorDomain;

typedef NS_ERROR_ENUM(ANERuntimeErrorDomain, ANERuntimeError) {
    ANERuntimeErrorUnsupportedBundle = 1,
    ANERuntimeErrorSurfaceAllocation = 2,
    ANERuntimeErrorFrameworkUnavailable = 3,
    ANERuntimeErrorCacheMiss = 4,
    ANERuntimeErrorLoadFailed = 5,
    ANERuntimeErrorNotLoaded = 6,
    ANERuntimeErrorBindingMismatch = 7,
    ANERuntimeErrorEvaluationFailed = 8,
    ANERuntimeErrorUnloadFailed = 9,
    ANERuntimeErrorPrivateAPIException = 10,
};

@interface ANEIOSurfaceBuffer : NSObject
@property(nonatomic, readonly) IOSurfaceRef ioSurface;
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly) NSUInteger logicalByteLength;
@property(nonatomic, readonly) NSUInteger allocationByteLength;
@end

/// Timing for one program submission inside one graph evaluation.
///
/// `surfaceWrapMicroseconds` covers wrapping the program's IOSurfaces into
/// runtime objects, `requestBuildMicroseconds` covers request construction,
/// and `submitMicroseconds` is the wall time of the synchronous evaluate
/// call, which includes queueing, hardware execution, and completion wait.
/// `hardwareExecutionNanoseconds` is the accelerator-side execution time the
/// runtime reports when performance statistics are enabled; it is 0 when the
/// runtime returns none. The host-side wait is then submit time minus
/// hardware time.
@interface ANEDispatchProfileEntry : NSObject
@property(nonatomic, readonly) NSUInteger artifactIndex;
@property(nonatomic, readonly) NSUInteger inputSurfaceCount;
@property(nonatomic, readonly) NSUInteger outputSurfaceCount;
@property(nonatomic, readonly) double surfaceWrapMicroseconds;
@property(nonatomic, readonly) double requestBuildMicroseconds;
@property(nonatomic, readonly) double submitMicroseconds;
@property(nonatomic, readonly) uint64_t hardwareExecutionNanoseconds;
@end

@interface ANEEvaluationProfile : NSObject
/// One entry per dispatched program, in dispatch order.
@property(nonatomic, readonly, copy) NSArray<ANEDispatchProfileEntry *> *entries;
/// Wall time of the whole evaluation including every submission.
@property(nonatomic, readonly) double totalMicroseconds;
/// Performance-statistics mask requested from the runtime. 0 disables
/// hardware statistics and leaves `hardwareExecutionNanoseconds` at 0.
@property(nonatomic) unsigned int performanceStatisticsMask;
@end

@interface ANEProvisionedRuntime : NSObject
@property(nonatomic, readonly) ANEExecutableBundle *bundle;
@property(nonatomic, readonly, copy) NSString *modelHash;
@property(nonatomic, readonly, copy) NSArray<NSString *> *modelHashes;
@property(nonatomic, readonly) unsigned int qos;
@property(nonatomic, readonly, getter=isLoaded) BOOL loaded;

- (nullable instancetype)initWithBundle:(ANEExecutableBundle *)bundle
                               modelHash:(NSString *)modelHash
                                     qos:(unsigned int)qos
                                   error:(NSError * _Nullable * _Nullable)error;
- (nullable instancetype)initWithBundle:(ANEExecutableBundle *)bundle
                             modelHashes:(NSArray<NSString *> *)modelHashes
                                     qos:(unsigned int)qos
                                   error:(NSError * _Nullable * _Nullable)error;
- (nullable NSArray<ANEIOSurfaceBuffer *> *)createInputBuffersWithError:
    (NSError * _Nullable * _Nullable)error;
- (nullable NSArray<ANEIOSurfaceBuffer *> *)createOutputBuffersWithError:
    (NSError * _Nullable * _Nullable)error;
- (nullable NSArray<ANEIOSurfaceBuffer *> *)surfaceBuffersForArtifactAtIndex:
    (NSUInteger)artifactIndex
    error:(NSError * _Nullable * _Nullable)error;
- (BOOL)loadWithError:(NSError * _Nullable * _Nullable)error;
- (BOOL)evaluateInputs:(NSArray<ANEIOSurfaceBuffer *> *)inputs
                outputs:(NSArray<ANEIOSurfaceBuffer *> *)outputs
                  error:(NSError * _Nullable * _Nullable)error;
/// Evaluates like `evaluateInputs:outputs:error:` and records per-program
/// timing into `profile`. The profile's previous entries are replaced.
- (BOOL)evaluateInputs:(NSArray<ANEIOSurfaceBuffer *> *)inputs
                outputs:(NSArray<ANEIOSurfaceBuffer *> *)outputs
                profile:(nullable ANEEvaluationProfile *)profile
                  error:(NSError * _Nullable * _Nullable)error;
- (BOOL)unloadWithError:(NSError * _Nullable * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
