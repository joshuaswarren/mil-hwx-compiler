#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GMatmulEncoder : NSObject
+ (BOOL)supportsSquareSize:(NSUInteger)size;
+ (nullable H16GEncodedTDProgram *)encodeSquareSize:(NSUInteger)size
                                              error:(NSError **)error;
+ (NSUInteger)tileCountForSquareSize:(NSUInteger)size;
+ (NSUInteger)rowsPerTileForSquareSize:(NSUInteger)size;
@end

NS_ASSUME_NONNULL_END
