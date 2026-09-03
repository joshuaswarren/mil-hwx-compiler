#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

/// The 32-bit H16G ALU scalar operand word for `value`: IEEE single
/// precision rounded to nearest even at ten mantissa bits, low bits zero.
/// Measured against Apple-compiled scalar multiply oracles: 0.0884 ->
/// 0x3db50000, 0.25 -> 0x3e800000, 0.7 -> 0x3f334000.
uint32_t H16GALUScalarWord(double value);

@interface H16GBroadcastALUEncoder : NSObject
+ (nullable H16GEncodedTDProgram *)encodeScalarScaleForMatrixRows:
    (NSUInteger)rows
    columns:(NSUInteger)columns
    scalar:(double)scalar
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
