#import <Foundation/Foundation.h>
#import "ANEOperationGraph.h"

NS_ASSUME_NONNULL_BEGIN
@interface H16GConvEncoder : NSObject
+ (BOOL)supportsConv1x1WithInputChannels:(NSUInteger)inputChannels
                          outputChannels:(NSUInteger)outputChannels
                                  spatial:(NSUInteger)spatial;
+ (nullable NSData *)encodeConv1x1WithInputChannels:(NSUInteger)inputChannels
                                    outputChannels:(NSUInteger)outputChannels
                                            spatial:(NSUInteger)spatial
                                     bytesPerWeight:(NSUInteger)bytesPerWeight
                                        numericMode:(ANELegalNumericMode)numericMode
                                       reluEpilogue:(BOOL)reluEpilogue
                                               error:(NSError **)error;
@end
NS_ASSUME_NONNULL_END
