#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, H16GConvWeightPackingFormat) {
    H16GConvWeightPackingFormatDense4x8,
    H16GConvWeightPackingFormatW8A8,
    H16GConvWeightPackingFormatLayoutConv,
};

@interface H16GConstantPacker : NSObject
+ (nullable NSData *)packConv1x1Weights:(NSData *)weights
                          inputChannels:(NSUInteger)inputChannels
                         outputChannels:(NSUInteger)outputChannels
                         bytesPerWeight:(NSUInteger)bytesPerWeight
                                  error:(NSError **)error;
+ (nullable NSData *)packConv1x1Weights:(NSData *)weights
                          inputChannels:(NSUInteger)inputChannels
                         outputChannels:(NSUInteger)outputChannels
                         bytesPerWeight:(NSUInteger)bytesPerWeight
                          packingFormat:(H16GConvWeightPackingFormat)packingFormat
                                  error:(NSError **)error;
+ (nullable NSData *)packDepthwise3x3Weights:(NSData *)weights
                                     channels:(NSUInteger)channels
                                        error:(NSError **)error;
+ (nullable NSData *)packRegularConvWeights:(NSData *)weights
                               inputChannels:(NSUInteger)inputChannels
                              outputChannels:(NSUInteger)outputChannels
                                kernelHeight:(NSUInteger)kernelHeight
                                 kernelWidth:(NSUInteger)kernelWidth
                                       error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
