#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GReduceEncoding : NSObject
@property(nonatomic, readonly, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readonly) NSUInteger taskCount;
@property(nonatomic, readonly) NSUInteger inputRowStrideBytes;
@property(nonatomic, readonly) NSUInteger inputPlaneStrideBytes;
@property(nonatomic, readonly) NSUInteger inputBatchStrideBytes;
@property(nonatomic, readonly) NSUInteger inputStorageByteLength;
@property(nonatomic, readonly) NSUInteger outputRowStrideBytes;
@property(nonatomic, readonly) NSUInteger outputPlaneStrideBytes;
@property(nonatomic, readonly) NSUInteger outputBatchStrideBytes;
@property(nonatomic, readonly) NSUInteger outputStorageByteLength;
@end

@interface H16GReduceEncoder : NSObject
+ (BOOL)supportsOperationName:(NSString *)operationName
                   inputShape:(NSArray<NSNumber *> *)inputShape
                         axis:(NSUInteger)axis;
+ (nullable H16GReduceEncoding *)encodeOperationName:(NSString *)operationName
                                            inputShape:(NSArray<NSNumber *> *)inputShape
                                                  axis:(NSUInteger)axis
                                                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
