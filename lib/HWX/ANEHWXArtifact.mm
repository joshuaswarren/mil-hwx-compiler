#import "ANEHWXArtifact.h"

@implementation ANEHWXBinding
- (instancetype)initWithIdentifier:(NSString *)identifier
                               role:(ANESurfaceRole)role
                  logicalByteLength:(NSUInteger)logicalByteLength
               allocationByteLength:(NSUInteger)allocationByteLength
                     ioSurfaceIndex:(NSInteger)ioSurfaceIndex {
    return [self initWithIdentifier:identifier role:role
                  logicalByteLength:logicalByteLength
               allocationByteLength:allocationByteLength
                     ioSurfaceIndex:ioSurfaceIndex rowStrideBytes:0
                   planeStrideBytes:0 batchStrideBytes:logicalByteLength];
}

- (instancetype)initWithIdentifier:(NSString *)identifier
                               role:(ANESurfaceRole)role
                  logicalByteLength:(NSUInteger)logicalByteLength
               allocationByteLength:(NSUInteger)allocationByteLength
                     ioSurfaceIndex:(NSInteger)ioSurfaceIndex
                     rowStrideBytes:(NSUInteger)rowStrideBytes
                   planeStrideBytes:(NSUInteger)planeStrideBytes
                   batchStrideBytes:(NSUInteger)batchStrideBytes {
    self = [super init];
    if (self) {
        _identifier = [identifier copy];
        _role = role;
        _logicalByteLength = logicalByteLength;
        _allocationByteLength = allocationByteLength;
        _ioSurfaceIndex = ioSurfaceIndex;
        _rowStrideBytes = rowStrideBytes;
        _planeStrideBytes = planeStrideBytes;
        _batchStrideBytes = batchStrideBytes;
    }
    return self;
}
@end

@implementation ANEHWXArtifact
- (instancetype)initWithImage:(NSData *)image
                      bindings:(NSArray<ANEHWXBinding *> *)bindings {
    self = [super init];
    if (self) {
        _image = [image copy];
        _bindings = [bindings copy];
    }
    return self;
}
@end
