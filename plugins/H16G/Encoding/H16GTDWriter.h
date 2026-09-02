#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN
FOUNDATION_EXPORT NSString *const H16GTDWriterErrorDomain;
@interface H16GTDWriter : NSObject
@property(nonatomic, readonly, copy) NSData *data;
- (instancetype)initWithByteLength:(NSUInteger)byteLength;
- (BOOL)writeUInt32:(uint32_t)value atOffset:(NSUInteger)offset
              field:(NSString *)field error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END
