#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GMatrixRowDivisionEncoding : NSObject
@property(nonatomic, readonly, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readonly, copy) NSData *constantRegion;
@property(nonatomic, readonly) NSUInteger taskCount;
@end

@interface H16GMatrixRowDivisionEncoder : NSObject
+ (nullable H16GMatrixRowDivisionEncoding *)encodeRows:(NSUInteger)rows
                                               columns:(NSUInteger)columns
                                                 error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
