#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GMatmulPostOperationEncoding : NSObject
@property(nonatomic, readonly, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readonly, copy) NSData *kernelRegion;
- (instancetype)initWithTDProgram:(H16GEncodedTDProgram *)tdProgram
                     kernelRegion:(NSData *)kernelRegion;
@end

@interface H16GMatmulEncoder : NSObject
+ (BOOL)supportsSquareSize:(NSUInteger)size;
+ (nullable H16GEncodedTDProgram *)encodeSquareSize:(NSUInteger)size
                                              error:(NSError **)error;
+ (NSUInteger)tileCountForSquareSize:(NSUInteger)size;
+ (NSUInteger)rowsPerTileForSquareSize:(NSUInteger)size;
/// Word index of the decoded fp16 output-scale field in the plain square
/// matmul task stream, or NSNotFound for geometries without a decoded row.
+ (NSUInteger)outputScaleWordIndexForSquareSize:(NSUInteger)size;
+ (nullable H16GMatmulPostOperationEncoding *)
    encodeSquareSize:(NSUInteger)size
    postOperationName:(NSString *)postOperationName
    kernelRegion:(NSData *)kernelRegion
    error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
