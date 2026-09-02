#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, H16GMeasuredScaleKind) {
    H16GMeasuredScaleInverseSqrt128,
};

@interface H16GBroadcastALUEncoder : NSObject
+ (nullable H16GEncodedTDProgram *)encodeScalarScaleForMatrixRows:
    (NSUInteger)rows
    columns:(NSUInteger)columns
    measuredScaleKind:(H16GMeasuredScaleKind)scaleKind
    error:(NSError **)error;
+ (nullable H16GEncodedTDProgram *)encodeMatrixRowOperation:
    (NSString *)operationName
    rows:(NSUInteger)rows
    columns:(NSUInteger)columns
    error:(NSError **)error;
+ (nullable H16GEncodedTDProgram *)encodeRowOperation:(NSString *)operationName
                                                 rows:(NSUInteger)rows
                                                error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
