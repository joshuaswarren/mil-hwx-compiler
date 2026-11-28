#import <Foundation/Foundation.h>

#import "ANEGraphIR.h"
#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GBoundaryConversionEncoder : NSObject
/// Encode a standalone dtype-conversion task (SUP-89 class 8): quantize
/// fp16->int8, dequantize int8->fp16, or cast fp16->bool/fp32 and
/// fp32->fp16 at the decoded shapes. Apple rejects every int32-surface
/// form on h16, so no int32 direction exists.
+ (nullable H16GEncodedTDProgram *)
    encodeOperationName:(NSString *)operationName
             sourceType:(ANEElementType)sourceType
        destinationType:(ANEElementType)destinationType
                  shape:(NSArray<NSNumber *> *)shape
                  error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
