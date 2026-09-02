#import "H16GConstantPacker.h"

static NSString *const H16GConstantPackerErrorDomain =
    @"ANE.H16G.ConstantPacker";

@implementation H16GConstantPacker
+ (NSData *)packRegularConvWeights:(NSData *)weights
                      inputChannels:(NSUInteger)inputChannels
                     outputChannels:(NSUInteger)outputChannels
                       kernelHeight:(NSUInteger)kernelHeight
                        kernelWidth:(NSUInteger)kernelWidth
                              error:(NSError **)error {
    BOOL measuredChannels = inputChannels == outputChannels &&
        (inputChannels == 64 || inputChannels == 128);
    BOOL measuredKernel = kernelHeight == kernelWidth &&
        (kernelHeight == 3 || kernelHeight == 5);
    NSUInteger taps = kernelHeight * kernelWidth;
    NSUInteger expectedBytes = inputChannels * outputChannels * taps * 2;
    if (!measuredChannels || !measuredKernel || weights.length != expectedBytes) {
        if (error) *error = [NSError errorWithDomain:H16GConstantPackerErrorDomain
            code:4 userInfo:@{NSLocalizedDescriptionKey:
                @"regular Conv packing requires measured square fp16 C64/C128 3x3 or 5x5 weights"}];
        return nil;
    }
    NSUInteger outputLanes = outputChannels / 16;
    NSMutableData *packed = [NSMutableData dataWithLength:weights.length];
    const uint8_t *source = (const uint8_t *)weights.bytes;
    uint8_t *destination = (uint8_t *)packed.mutableBytes;
    NSUInteger cursor = 0;
    for (NSUInteger packet = 0; packet < 16; ++packet)
        for (NSUInteger input = 0; input < inputChannels; ++input)
            for (NSUInteger tap = 0; tap < taps; ++tap)
                for (NSUInteger lane = 0; lane < outputLanes; ++lane) {
                    NSUInteger output = packet * outputLanes + lane;
                    NSUInteger sourceIndex =
                        (output * inputChannels + input) * taps + tap;
                    memcpy(destination + cursor, source + sourceIndex * 2, 2);
                    cursor += 2;
                }
    return packed;
}

+ (NSData *)packDepthwise3x3Weights:(NSData *)weights
                            channels:(NSUInteger)channels
                               error:(NSError **)error {
    BOOL measuredChannels = channels == 64 || channels == 128 ||
        channels == 256 || channels == 512;
    NSUInteger rawBytes = channels * 9 * sizeof(_Float16);
    if (!measuredChannels || weights.length != rawBytes) {
        if (error) *error = [NSError errorWithDomain:H16GConstantPackerErrorDomain
            code:3 userInfo:@{NSLocalizedDescriptionKey:
                @"depthwise 3x3 packing requires fp16 [C,1,3,3] for measured C64/C128/C256/C512"}];
        return nil;
    }
    NSUInteger channelsPerPacket = channels / 16;
    NSUInteger packetBytes = channels + 64;
    NSMutableData *packed = [NSMutableData dataWithLength:16 * packetBytes];
    const uint8_t *source = (const uint8_t *)weights.bytes;
    uint8_t *destination = (uint8_t *)packed.mutableBytes;
    for (NSUInteger packet = 0; packet < 16; ++packet) {
        NSUInteger cursor = packet * packetBytes;
        for (NSUInteger lane = 0; lane < channelsPerPacket; ++lane) {
            NSUInteger channel = packet + lane * 16;
            memcpy(destination + cursor, source + channel * 9 * 2, 9 * 2);
            cursor += 9 * 2;
        }
    }
    return packed;
}

+ (NSData *)packConv1x1Weights:(NSData *)weights
                 inputChannels:(NSUInteger)inputChannels
                outputChannels:(NSUInteger)outputChannels
                bytesPerWeight:(NSUInteger)bytesPerWeight
                         error:(NSError **)error {
    return [self packConv1x1Weights:weights inputChannels:inputChannels
        outputChannels:outputChannels bytesPerWeight:bytesPerWeight
        packingFormat:H16GConvWeightPackingFormatDense4x8 error:error];
}

+ (NSData *)packConv1x1Weights:(NSData *)weights
                 inputChannels:(NSUInteger)inputChannels
                outputChannels:(NSUInteger)outputChannels
                bytesPerWeight:(NSUInteger)bytesPerWeight
                 packingFormat:(H16GConvWeightPackingFormat)packingFormat
                         error:(NSError **)error {
    NSUInteger inputGroup = 0;
    NSUInteger outputGroup = 0;
    BOOL inputLaneFirst = NO;
    BOOL layoutConv = NO;
    switch (packingFormat) {
        case H16GConvWeightPackingFormatDense4x8:
            inputGroup = 4;
            outputGroup = 8;
            break;
        case H16GConvWeightPackingFormatW8A8:
            inputGroup = 4;
            outputGroup = 4;
            inputLaneFirst = YES;
            break;
        case H16GConvWeightPackingFormatLayoutConv:
            inputGroup = 64;
            outputGroup = 128;
            layoutConv = YES;
            break;
        default:
            if (error) {
                *error = [NSError errorWithDomain:H16GConstantPackerErrorDomain
                    code:2 userInfo:@{NSLocalizedDescriptionKey:
                        @"unknown H16G Conv weight-packing format"}];
            }
            return nil;
    }
    BOOL dimensionsAreLegal = inputChannels != 0 && outputChannels != 0 &&
        inputChannels % inputGroup == 0 &&
        outputChannels % outputGroup == 0;
    BOOL elementWidthIsLegal = bytesPerWeight == 1 || bytesPerWeight == 2;
    if (packingFormat == H16GConvWeightPackingFormatW8A8)
        elementWidthIsLegal = bytesPerWeight == 1;
    if (packingFormat == H16GConvWeightPackingFormatLayoutConv) {
        dimensionsAreLegal = dimensionsAreLegal &&
            inputChannels == outputChannels;
        elementWidthIsLegal = bytesPerWeight == 2;
    }
    BOOL sizeFits = inputChannels <= NSUIntegerMax / outputChannels &&
        inputChannels * outputChannels <= NSUIntegerMax / bytesPerWeight;
    NSUInteger expectedBytes = sizeFits
        ? inputChannels * outputChannels * bytesPerWeight : 0;
    if (!dimensionsAreLegal || !elementWidthIsLegal || !sizeFits ||
        weights.length != expectedBytes) {
        if (error) {
            *error = [NSError errorWithDomain:H16GConstantPackerErrorDomain
                code:1 userInfo:@{NSLocalizedDescriptionKey:
                    packingFormat == H16GConvWeightPackingFormatW8A8
                        ? @"W8A8 Conv1x1 weights require complete 4-input by 4-output groups"
                        : (packingFormat == H16GConvWeightPackingFormatLayoutConv
                            ? @"fused layout Conv weights require a square fp16 matrix in 128-channel groups"
                            : @"fp16 Conv1x1 weights require complete 4-input by 8-output groups")}];
        }
        return nil;
    }

    NSMutableData *packed = [NSMutableData dataWithLength:weights.length];
    const uint8_t *source = (const uint8_t *)weights.bytes;
    uint8_t *destination = (uint8_t *)packed.mutableBytes;
    NSUInteger cursor = 0;
    if (layoutConv) {
        NSUInteger tileChannels = outputChannels / 4;
        NSUInteger minorChannels = outputChannels / 16;
        NSUInteger outputLanes = outputChannels / 64;
        for (NSUInteger tile = 0; tile < 4; ++tile) {
            NSUInteger tileBase = tile * tileChannels;
            for (NSUInteger chunk = 0; chunk < tileChannels;
                 chunk += outputLanes) {
                for (NSUInteger input = 0; input < inputChannels; ++input) {
                    for (NSUInteger lane = 0; lane < outputLanes; ++lane) {
                        NSUInteger sequence = chunk + lane;
                        NSUInteger minor = sequence / 4;
                        NSUInteger group = sequence % 4;
                        NSUInteger output = tileBase + minor +
                            group * minorChannels;
                        NSUInteger sourceOffset =
                            (output * inputChannels + input) * 2;
                        memcpy(destination + cursor, source + sourceOffset, 2);
                        cursor += 2;
                    }
                }
            }
        }
        return packed;
    }
    for (NSUInteger outerBlock = 0;
         outerBlock < (inputLaneFirst ? outputChannels : inputChannels);
         outerBlock += (inputLaneFirst ? outputGroup : inputGroup)) {
        for (NSUInteger innerBlock = 0;
             innerBlock < (inputLaneFirst ? inputChannels : outputChannels);
             innerBlock += (inputLaneFirst ? inputGroup : outputGroup)) {
            for (NSUInteger outerLane = 0;
                 outerLane < (inputLaneFirst ? inputGroup : outputGroup);
                 ++outerLane) {
                for (NSUInteger innerLane = 0;
                     innerLane < (inputLaneFirst ? outputGroup : inputGroup);
                     ++innerLane) {
                    NSUInteger output = inputLaneFirst
                        ? outerBlock + innerLane : innerBlock + outerLane;
                    NSUInteger input = inputLaneFirst
                        ? innerBlock + outerLane : outerBlock + innerLane;
                    NSUInteger sourceOffset =
                        (output * inputChannels + input) * bytesPerWeight;
                    memcpy(destination + cursor, source + sourceOffset,
                           bytesPerWeight);
                    cursor += bytesPerWeight;
                }
            }
        }
    }
    return packed;
}
@end
