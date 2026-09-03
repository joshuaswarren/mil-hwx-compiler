#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSInteger, HWXImageErrorCode) {
    HWXImageErrorTruncated = 1,
    HWXImageErrorBadMagic,
    HWXImageErrorMalformedCommands,
    HWXImageErrorRange,
};

FOUNDATION_EXPORT NSString *const HWXImageErrorDomain;

@interface HWXSection : NSObject
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly, copy) NSString *segmentName;
@property(nonatomic, readonly) uint64_t size;
@property(nonatomic, readonly) uint32_t fileOffset;
@property(nonatomic, readonly, copy) NSData *data;
@end

@interface HWXImage : NSObject
@property(nonatomic, readonly, copy) NSData *data;
@property(nonatomic, readonly) uint32_t magic;
@property(nonatomic, readonly) int32_t cpuType;
@property(nonatomic, readonly) int32_t cpuSubtype;
@property(nonatomic, readonly, copy) NSArray<HWXSection *> *sections;
+ (nullable instancetype)imageWithData:(NSData *)data
                                 error:(NSError **)error;
- (nullable HWXSection *)firstSectionNamed:(NSString *)name
                                 inSegment:(NSString *)segmentName;
@end

NS_ASSUME_NONNULL_END
