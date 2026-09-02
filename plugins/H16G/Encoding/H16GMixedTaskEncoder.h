#import <Foundation/Foundation.h>

#import "ANEOperationGraph.h"
#import "ANEScheduledGraph.h"
#import "H16GConvChainEncoder.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GMixedTaskEncoding : NSObject
@property(nonatomic, readonly, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readonly, copy) NSData *constantRegion;
@property(nonatomic, readonly) NSUInteger scratchByteLength;
- (instancetype)initWithTDProgram:(H16GEncodedTDProgram *)tdProgram
                   constantRegion:(NSData *)constantRegion
                scratchByteLength:(NSUInteger)scratchByteLength;
@end

// Encodes the decoded mixed layout/matmul/ALU/reduce/LUT packet grammar.
// Selection is based only on scheduled primitive groups and tensor geometry.
@interface H16GMixedTaskEncoder : NSObject
+ (nullable H16GMixedTaskEncoding *)encodeGraph:(ANEOperationGraph *)graph
                                       scheduled:(ANEScheduledGraph *)scheduled
                                           error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
