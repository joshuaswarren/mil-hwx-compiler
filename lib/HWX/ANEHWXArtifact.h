#import <Foundation/Foundation.h>

#import "ANESurfaceRole.h"

NS_ASSUME_NONNULL_BEGIN

@interface ANEHWXBinding : NSObject
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly) ANESurfaceRole role;
@property(nonatomic, readonly) NSUInteger logicalByteLength;
@property(nonatomic, readonly) NSUInteger allocationByteLength;
@property(nonatomic, readonly) NSInteger ioSurfaceIndex;
@property(nonatomic, readonly) NSUInteger rowStrideBytes;
@property(nonatomic, readonly) NSUInteger planeStrideBytes;
@property(nonatomic, readonly) NSUInteger batchStrideBytes;
- (instancetype)initWithIdentifier:(NSString *)identifier
                               role:(ANESurfaceRole)role
                  logicalByteLength:(NSUInteger)logicalByteLength
               allocationByteLength:(NSUInteger)allocationByteLength
                     ioSurfaceIndex:(NSInteger)ioSurfaceIndex;
- (instancetype)initWithIdentifier:(NSString *)identifier
                               role:(ANESurfaceRole)role
                  logicalByteLength:(NSUInteger)logicalByteLength
               allocationByteLength:(NSUInteger)allocationByteLength
                     ioSurfaceIndex:(NSInteger)ioSurfaceIndex
                     rowStrideBytes:(NSUInteger)rowStrideBytes
                   planeStrideBytes:(NSUInteger)planeStrideBytes
                   batchStrideBytes:(NSUInteger)batchStrideBytes;
@end

@interface ANEHWXArtifact : NSObject
@property(nonatomic, readonly, copy) NSData *image;
@property(nonatomic, readonly, copy) NSArray<ANEHWXBinding *> *bindings;
- (instancetype)initWithImage:(NSData *)image
                      bindings:(NSArray<ANEHWXBinding *> *)bindings;
@end

NS_ASSUME_NONNULL_END
