#import "H16GTaskComposer.h"

#import "H16GLUTEncoder.h"
#import "H16GMatmulEncoder.h"
#import "H16GSRAMChainEncoder.h"

#include <cmath>
#include <cstring>
#include <limits>

static NSString *const H16GTaskComposerErrorDomain =
    @"ANE.H16G.TaskComposer";

static H16GEncodedTask *fail(NSError **error, NSInteger code,
                             NSString *message) {
    if (error)
        *error = [NSError errorWithDomain:H16GTaskComposerErrorDomain code:code
            userInfo:@{NSLocalizedDescriptionKey:message}];
    return nil;
}

static BOOL consumerUsesProducer(H16GEncodedTask *producer,
                                 H16GEncodedTask *consumer) {
    NSUInteger index = [consumer.inputIdentifiers
        indexOfObject:producer.outputIdentifier];
    return index != NSNotFound && index < consumer.inputShapes.count &&
        [consumer.inputShapes[index] isEqualToArray:producer.outputShape];
}

static uint64_t roundToNearestEven(double value) {
    double lower = std::floor(value);
    double fraction = value - lower;
    return (uint64_t)lower +
        (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0));
}

static uint16_t fp16Bits(double value) {
    uint16_t sign = std::signbit(value) ? 0x8000u : 0;
    if (std::isnan(value)) return sign | 0x7e00u;
    if (std::isinf(value)) return sign | 0x7c00u;
    double magnitude = std::fabs(value);
    if (magnitude == 0.0) return sign;
    if (magnitude < std::ldexp(1.0, -14)) {
        uint64_t mantissa = roundToNearestEven(std::ldexp(magnitude, 24));
        return sign | (uint16_t)mantissa;
    }
    int exponent = 0;
    double fraction = std::frexp(magnitude, &exponent);
    int unbiasedExponent = exponent - 1;
    uint64_t significand = roundToNearestEven(std::ldexp(fraction, 11));
    if (significand == 2048) {
        significand = 1024;
        ++unbiasedExponent;
    }
    if (unbiasedExponent > 15) return sign | 0x7c00u;
    return sign | (uint16_t)((unbiasedExponent + 15) << 10) |
        (uint16_t)(significand - 1024);
}

static double fp16Value(uint16_t bits) {
    double sign = (bits & 0x8000u) ? -1.0 : 1.0;
    uint16_t exponent = (bits >> 10) & 0x1fu;
    uint16_t mantissa = bits & 0x03ffu;
    if (exponent == 0x1fu)
        return mantissa ? std::numeric_limits<double>::quiet_NaN()
                        : sign * std::numeric_limits<double>::infinity();
    if (exponent == 0)
        return sign * std::ldexp((double)mantissa, -24);
    return sign * std::ldexp((double)(1024 + mantissa), exponent - 25);
}

static BOOL readWord(NSData *data, NSUInteger byteOffset, uint32_t *value) {
    if (byteOffset % sizeof(uint32_t) != 0 || byteOffset > data.length ||
        sizeof(uint32_t) > data.length - byteOffset) return NO;
    [data getBytes:value range:NSMakeRange(byteOffset, sizeof(uint32_t))];
    return YES;
}

static H16GEncodedTDProgram *programWithWord(H16GEncodedTDProgram *program,
                                             NSUInteger byteOffset,
                                             uint32_t value) {
    NSMutableData *data = [program.data mutableCopy];
    [data replaceBytesInRange:NSMakeRange(byteOffset, sizeof(value))
                    withBytes:&value];
    return [[H16GEncodedTDProgram alloc] initWithData:data
        kernelRelocationOffsets:program.kernelRelocationOffsets
        programRecordCount:program.programRecordCount
        programFormatCode:program.programFormatCode
        scratchByteLength:program.scratchByteLength];
}

/// The consumer's scalar multiplies the producer's output-scale word.
static H16GEncodedTask *composeMatmulOutputScale(H16GEncodedTask *producer,
                                                 H16GEncodedTask *consumer,
                                                 NSError **error) {
    NSNumber *scalar = consumer.scalarOperand;
    if (!scalar || !std::isfinite(scalar.doubleValue))
        return fail(error, 10, @"scalar scale consumer has no finite operand");
    NSUInteger wordIndex = [H16GMatmulEncoder
        outputScaleWordIndexForSquareSize:producer.geometry];
    if (wordIndex == NSNotFound)
        return fail(error, 11,
                    @"matmul geometry has no decoded output-scale field");
    NSUInteger byteOffset = wordIndex * sizeof(uint32_t);
    uint32_t current = 0;
    if (!readWord(producer.tdProgram.data, byteOffset, &current) ||
        current != 0x00003c00u)
        return fail(error, 12,
                    @"matmul output-scale field does not hold the plain unit scale");
    uint16_t folded = fp16Bits(scalar.doubleValue);
    if (!std::isfinite(fp16Value(folded)))
        return fail(error, 13, @"folded output scale is not representable in fp16");
    NSArray<NSString *> *operations = [producer.stageOperations
        arrayByAddingObjectsFromArray:consumer.stageOperations];
    return [[H16GEncodedTask alloc]
        initWithTDProgram:programWithWord(producer.tdProgram, byteOffset,
                                          (uint32_t)folded)
        constantRegion:producer.constantRegion
        stageOperations:operations
        inputIdentifiers:producer.inputIdentifiers
        inputShapes:producer.inputShapes
        outputIdentifier:consumer.outputIdentifier
        outputShape:consumer.outputShape
        taskCount:producer.taskCount
        scratchBacked:producer.scratchBacked
        scratchAllocationByteLength:producer.scratchAllocationByteLength
        outputResourceIndex:producer.outputResourceIndex
        packetFamily:producer.packetFamily
        geometry:producer.geometry
        numericMode:producer.numericMode
        outputStorage:consumer.outputStorage
        compositionOperationName:producer.compositionOperationName
        scalarOperand:nil];
}

/// The producer's scalar multiplies the consumer's table input-scale word.
static H16GEncodedTask *composeLUTInputScale(H16GEncodedTask *producer,
                                             H16GEncodedTask *consumer,
                                             NSError **error) {
    NSNumber *scalar = producer.scalarOperand;
    if (!scalar || !std::isfinite(scalar.doubleValue))
        return fail(error, 20, @"scalar scale producer has no finite operand");
    if (consumer.inputShapes.count != 1)
        return fail(error, 21, @"table unary consumer must have one input");
    NSUInteger byteOffset = [H16GLUTEncoder
        inputScaleByteOffsetForOperationName:consumer.compositionOperationName
        inputShape:consumer.inputShapes.firstObject];
    if (byteOffset == NSNotFound)
        return fail(error, 22,
                    @"unary operation or geometry has no decoded input-scale field");
    uint32_t current = 0;
    double existing = [H16GLUTEncoder
        inputScaleValueForOperationName:consumer.compositionOperationName];
    if (!readWord(consumer.tdProgram.data, byteOffset, &current) ||
        !std::isfinite(existing) || current != fp16Bits(existing))
        return fail(error, 23,
                    @"table input-scale field does not hold the plain scale");
    // Apple folds in full precision and rounds once, so multiply the exact
    // scale rather than the fp16 word the plain program carries.
    double folded = existing * scalar.doubleValue;
    uint16_t bits = fp16Bits(folded);
    if (!std::isfinite(fp16Value(bits)))
        return fail(error, 24, @"folded input scale is not representable in fp16");
    NSArray<NSString *> *operations = [producer.stageOperations
        arrayByAddingObjectsFromArray:consumer.stageOperations];
    return [[H16GEncodedTask alloc]
        initWithTDProgram:programWithWord(consumer.tdProgram, byteOffset,
                                          (uint32_t)bits)
        constantRegion:consumer.constantRegion
        stageOperations:operations
        inputIdentifiers:producer.inputIdentifiers
        inputShapes:producer.inputShapes
        outputIdentifier:consumer.outputIdentifier
        outputShape:consumer.outputShape
        taskCount:consumer.taskCount
        scratchBacked:consumer.scratchBacked
        scratchAllocationByteLength:consumer.scratchAllocationByteLength
        outputResourceIndex:consumer.outputResourceIndex
        packetFamily:consumer.packetFamily
        geometry:consumer.geometry
        numericMode:consumer.numericMode
        outputStorage:consumer.outputStorage
        compositionOperationName:consumer.compositionOperationName
        scalarOperand:nil];
}

@implementation H16GTaskComposer
+ (BOOL)supportsCapability:(H16GProgramCompositionCapability *)capability {
    switch (capability.action) {
        case H16GProgramCompositionActionTiledPostOperation:
        case H16GProgramCompositionActionMatmulOutputScale:
        case H16GProgramCompositionActionLUTInputScale:
        case H16GProgramCompositionActionSRAMTaskChain:
            return YES;
        case H16GProgramCompositionActionPrimitiveFallback:
            return NO;
    }
    return NO;
}
+ (H16GEncodedTask *)composeProducer:(H16GEncodedTask *)producer
    consumer:(H16GEncodedTask *)consumer
    capability:(H16GProgramCompositionCapability *)capability
    error:(NSError **)error {
    if (!capability)
        return fail(error, 1, @"task transition has no composition capability");
    NSString *producerOperation = producer.compositionOperationName;
    NSString *consumerOperation = consumer.compositionOperationName;
    BOOL matches =
        producer.packetFamily == capability.producerPacketFamily &&
        consumer.packetFamily == capability.consumerPacketFamily &&
        producer.geometry == capability.producerGeometry &&
        consumer.geometry == capability.consumerGeometry &&
        [producerOperation isEqualToString:capability.producerOperationName] &&
        [consumerOperation isEqualToString:capability.consumerOperationName] &&
        producer.numericMode == consumer.numericMode &&
        capability.bridgeStorage == ANEScheduledBridgeStorageSRAM &&
        consumerUsesProducer(producer, consumer);
    if (!matches)
        return fail(error, 2,
                    @"task metadata does not match the composition capability");
    if (![self supportsCapability:capability])
        return fail(error, 3,
                    @"composition action has no decoded task composer");

    switch (capability.action) {
        case H16GProgramCompositionActionMatmulOutputScale:
            return composeMatmulOutputScale(producer, consumer, error);
        case H16GProgramCompositionActionLUTInputScale:
            return composeLUTInputScale(producer, consumer, error);
        case H16GProgramCompositionActionSRAMTaskChain: {
            H16GEncodedTDProgram *program = [H16GSRAMChainEncoder
                encodeProducerOperationName:producerOperation
                consumerOperationName:consumerOperation
                rows:producer.geometry columns:producer.geometry
                error:error];
            if (!program) return nil;
            NSArray<NSString *> *operations = [producer.stageOperations
                arrayByAddingObjectsFromArray:consumer.stageOperations];
            return [[H16GEncodedTask alloc]
                initWithTDProgram:program
                constantRegion:producer.constantRegion
                stageOperations:operations
                inputIdentifiers:producer.inputIdentifiers
                inputShapes:producer.inputShapes
                outputIdentifier:consumer.outputIdentifier
                outputShape:consumer.outputShape
                taskCount:producer.taskCount + consumer.taskCount
                scratchBacked:NO scratchAllocationByteLength:0
                outputResourceIndex:consumer.outputResourceIndex
                packetFamily:consumer.packetFamily
                geometry:consumer.geometry numericMode:consumer.numericMode
                outputStorage:consumer.outputStorage
                compositionOperationName:consumer.compositionOperationName
                scalarOperand:nil];
        }
        case H16GProgramCompositionActionPrimitiveFallback:
            return fail(error, 3,
                        @"composition action has no decoded task composer");
        case H16GProgramCompositionActionTiledPostOperation:
            break;
    }

    H16GMatmulPostOperationEncoding *encoding = [H16GMatmulEncoder
        encodeSquareSize:producer.geometry
        postOperationName:consumerOperation
        kernelRegion:consumer.constantRegion error:error];
    if (!encoding) return nil;
    NSArray<NSString *> *operations = [producer.stageOperations
        arrayByAddingObjectsFromArray:consumer.stageOperations];
    return [[H16GEncodedTask alloc]
        initWithTDProgram:encoding.tdProgram
        constantRegion:encoding.kernelRegion
        stageOperations:operations
        inputIdentifiers:producer.inputIdentifiers
        inputShapes:producer.inputShapes
        outputIdentifier:consumer.outputIdentifier
        outputShape:consumer.outputShape
        taskCount:producer.taskCount
        scratchBacked:producer.scratchBacked
        scratchAllocationByteLength:producer.scratchAllocationByteLength
        outputResourceIndex:producer.outputResourceIndex
        packetFamily:consumer.packetFamily
        geometry:consumer.geometry
        numericMode:consumer.numericMode
        outputStorage:consumer.outputStorage
        compositionOperationName:consumer.compositionOperationName];
}
@end
