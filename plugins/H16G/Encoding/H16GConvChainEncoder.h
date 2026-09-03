#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface H16GEncodedTDProgram : NSObject
@property(nonatomic, readonly, copy) NSData *data;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *kernelRelocationOffsets;
@property(nonatomic, readonly) NSUInteger programRecordCount;
@property(nonatomic, readonly) uint32_t programFormatCode;
@property(nonatomic, readonly) NSUInteger scratchByteLength;
- (instancetype)initWithData:(NSData *)data
      kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
           programRecordCount:(NSUInteger)programRecordCount
            programFormatCode:(uint32_t)programFormatCode;
- (instancetype)initWithData:(NSData *)data
      kernelRelocationOffsets:(NSArray<NSNumber *> *)kernelRelocationOffsets
           programRecordCount:(NSUInteger)programRecordCount
            programFormatCode:(uint32_t)programFormatCode
            scratchByteLength:(NSUInteger)scratchByteLength;
@end

@interface H16GConvChainEncoder : NSObject
+ (nullable H16GEncodedTDProgram *)encodeW8A8C64S64WithDepth:(NSUInteger)depth
                                                       error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
