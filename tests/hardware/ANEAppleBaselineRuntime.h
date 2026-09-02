#import <Foundation/Foundation.h>

@class ANEIOSurfaceBuffer;

NS_ASSUME_NONNULL_BEGIN

@interface ANEAppleBaselineRuntime : NSObject
@property(nonatomic, readonly) unsigned int qos;
@property(nonatomic, readonly) double compileMicroseconds;
@property(nonatomic, readonly) double loadMicroseconds;
@property(nonatomic, readonly, getter=isLoaded) BOOL loaded;

- (nullable instancetype)initWithMILData:(NSData *)milData
                                     qos:(unsigned int)qos
                                   error:(NSError * _Nullable * _Nullable)error;
- (BOOL)loadWithError:(NSError * _Nullable * _Nullable)error;
- (BOOL)evaluateInputs:(NSArray<ANEIOSurfaceBuffer *> *)inputs
                outputs:(NSArray<ANEIOSurfaceBuffer *> *)outputs
                  error:(NSError * _Nullable * _Nullable)error;
- (BOOL)unloadWithError:(NSError * _Nullable * _Nullable)error;
@end

NS_ASSUME_NONNULL_END
