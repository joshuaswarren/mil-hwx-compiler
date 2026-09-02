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
- (BOOL)unloadWithError:(NSError * _Nullable * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
