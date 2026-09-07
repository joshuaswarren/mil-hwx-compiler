#import "ANEH13Compiler.h"

#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"
#import "HWXObjectWriter.h"
#include "H13Program.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
static BOOL reject(ANEDiagnosticEngine *diagnostics, NSString *message,
                   ANEGraphOperation *operation = nil,
                   NSString *code = @"h13.unsupported-program") {
    ANESourceLocation start = ANESourceLocationMake(0, 1, 1);
    [diagnostics emitSeverity:ANEDiagnosticSeverityError
        code:code message:message
        range:operation ? operation.range : ANESourceRangeMake(start, start)];
    return NO;
}

static BOOL fp16Tensor(ANEGraphValue *value) {
    return value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeFP16;
}

static BOOL tensor(ANEGraphValue *value, NSArray<NSNumber *> *shape) {
    return fp16Tensor(value) && [value.type.shape isEqualToArray:shape];
}

static BOOL tensorElementCount(ANEGraphValue *value, NSUInteger *count) {
    if (!fp16Tensor(value)) return NO;
    NSUInteger elements = 1;
    for (NSNumber *number in value.type.shape) {
        NSUInteger dimension = number.unsignedIntegerValue;
        if (!dimension || elements > NSUIntegerMax / dimension) return NO;
        elements *= dimension;
    }
    *count = elements;
    return YES;
}

static BOOL elementwiseShape(ANEGraphValue *value,
                             ane::h13::ElementwiseShape *shape) {
    if (!fp16Tensor(value) || value.type.shape.count != 4 ||
        value.type.shape[0].unsignedIntegerValue != 1) return NO;
    const uint64_t channels = value.type.shape[1].unsignedLongLongValue;
    const uint64_t height = value.type.shape[2].unsignedLongLongValue;
    const uint64_t width = value.type.shape[3].unsignedLongLongValue;
    if (!channels || !height || !width || channels > UINT32_MAX ||
        height > UINT32_MAX || width > UINT32_MAX) return NO;
    *shape = {static_cast<std::uint32_t>(channels),
              static_cast<std::uint32_t>(height),
              static_cast<std::uint32_t>(width)};
    return YES;
}


static BOOL binaryEncoding(NSString *name, ane::h13::BinaryOperation *encoding) {
    if ([name isEqualToString:@"add"]) *encoding = ane::h13::BinaryOperation::Add;
    else if ([name isEqualToString:@"mul"]) *encoding = ane::h13::BinaryOperation::Multiply;
    else if ([name isEqualToString:@"maximum"]) *encoding = ane::h13::BinaryOperation::Maximum;
    else if ([name isEqualToString:@"minimum"]) *encoding = ane::h13::BinaryOperation::Minimum;
    else if ([name isEqualToString:@"sub"]) *encoding = ane::h13::BinaryOperation::Subtract;
    else if ([name isEqualToString:@"real_div"]) *encoding = ane::h13::BinaryOperation::RealDivide;
    else return NO;
    return YES;
}

static BOOL unaryEncoding(NSString *name, ane::h13::UnaryOperation *encoding) {
    if ([name isEqualToString:@"abs"]) *encoding = ane::h13::UnaryOperation::Absolute;
    else if ([name isEqualToString:@"exp"]) *encoding = ane::h13::UnaryOperation::Exponential;
    else if ([name isEqualToString:@"gelu"]) *encoding = ane::h13::UnaryOperation::Gelu;
    else if ([name isEqualToString:@"leaky_relu"]) *encoding = ane::h13::UnaryOperation::LeakyRelu;
    else if ([name isEqualToString:@"relu"]) *encoding = ane::h13::UnaryOperation::Relu;
    else if ([name isEqualToString:@"rsqrt"]) *encoding = ane::h13::UnaryOperation::ReciprocalSquareRoot;
    else if ([name isEqualToString:@"sigmoid"]) *encoding = ane::h13::UnaryOperation::Sigmoid;
    else if ([name isEqualToString:@"silu"]) *encoding = ane::h13::UnaryOperation::Silu;
    else if ([name isEqualToString:@"sqrt"]) *encoding = ane::h13::UnaryOperation::SquareRoot;
    else if ([name isEqualToString:@"tanh"]) *encoding = ane::h13::UnaryOperation::Tanh;
    else return NO;
    return YES;
}

static BOOL normEncoding(NSString *name, ane::h13::NormOperation *encoding) {
    if ([name isEqualToString:@"softmax"]) *encoding = ane::h13::NormOperation::Softmax;
    else if ([name isEqualToString:@"layer_norm"]) *encoding = ane::h13::NormOperation::LayerNorm;
    else if ([name isEqualToString:@"reduce_sum"]) *encoding = ane::h13::NormOperation::ReduceSum;
    else if ([name isEqualToString:@"reduce_max"]) *encoding = ane::h13::NormOperation::ReduceMax;
    else if ([name isEqualToString:@"reduce_mean"]) *encoding = ane::h13::NormOperation::ReduceMean;
    else return NO;
    return YES;
}

/// The CHW surface H13 lays a logical MIL shape out as: leading unit
/// dimensions collapse into the batch, and a rank below three pads on the
/// left. Every decoded normalization and reduction surface follows this,
/// including the rank-reduced `keep_dims = false` results.
static BOOL normSurface(NSArray<NSNumber *> *shape,
                        ane::h13::ElementwiseShape *surface,
                        NSInteger *axisShift) {
    NSMutableArray<NSNumber *> *dimensions = [shape mutableCopy];
    NSInteger shift = 0;
    while (dimensions.count > 3 && dimensions[0].unsignedIntegerValue == 1) {
        [dimensions removeObjectAtIndex:0];
        --shift;
    }
    while (dimensions.count && dimensions.count < 3) {
        [dimensions insertObject:@1 atIndex:0];
        ++shift;
    }
    if (dimensions.count != 3) return NO;
    uint64_t extents[3];
    for (NSUInteger index = 0; index < 3; ++index) {
        extents[index] = dimensions[index].unsignedLongLongValue;
        if (!extents[index] || extents[index] > UINT32_MAX) return NO;
    }
    *surface = {static_cast<std::uint32_t>(extents[0]),
                static_cast<std::uint32_t>(extents[1]),
                static_cast<std::uint32_t>(extents[2])};
    if (axisShift) *axisShift = shift;
    return YES;
}

static BOOL int32Literal(ANEGraphArgument *argument, long long *value) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"int32"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    long long parsed = std::strtoll(text, &end, 10);
    if (end == text || *end) return NO;
    *value = parsed;
    return YES;
}

/// Resolves a constant axis operand — softmax's `int32` scalar or a rank-1
/// `int32` axes tensor — into the NCHW mask of the physical surface, with
/// `axisShift` mapping logical axes onto the canonical CHW the encoder keys
/// its templates on. Bit `i` of the mask is NCHW axis `i`.
static BOOL constantAxisMask(ANEGraphValue *value, NSUInteger rank,
                             NSInteger axisShift, std::uint32_t *mask) {
    if (!value || ![value.producer.operationName isEqualToString:@"const"] ||
        value.producer.arguments.count || !rank || rank > 4) return NO;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (!literal || value.type.elementType != ANEElementTypeInt32) return NO;
    NSArray<ANEGraphArgument *> *axes = nil;
    if (value.type.kind == ANEValueTypeKindScalar) {
        axes = @[literal];
    } else {
        if (value.type.kind != ANEValueTypeKindTensor ||
            value.type.shape.count != 1 ||
            literal.kind != ANEGraphArgumentKindCall ||
            ![literal.calleeValueType isEqualToValueType:value.type] ||
            literal.callArguments.count != 1) return NO;
        ANEGraphArgument *payload = literal.callArguments[0].value;
        if (payload.kind != ANEGraphArgumentKindList ||
            payload.elements.count != value.type.shape[0].unsignedIntegerValue)
            return NO;
        axes = payload.elements;
    }
    if (!axes.count) return NO;
    std::uint32_t resolved = 0;
    for (ANEGraphArgument *element in axes) {
        long long axis = 0;
        if (!int32Literal(element, &axis)) return NO;
        if (axis < 0) axis += (long long)rank;
        if (axis < 0 || axis >= (long long)rank) return NO;
        // Canonical CHW index 0 is NCHW axis 1, so the surface bit is shifted.
        const long long surfaceAxis = axis + axisShift + 1;
        if (surfaceAxis < 1 || surfaceAxis > 3) return NO;
        const std::uint32_t bit = 1u << surfaceAxis;
        if (resolved & bit) return NO;
        resolved |= bit;
    }
    *mask = resolved;
    return YES;
}

static uint64_t roundToNearestEven(double value) {
    double lower = std::floor(value);
    double fraction = value - lower;
    return (uint64_t)lower +
        (fraction > 0.5 || (fraction == 0.5 && std::fmod(lower, 2.0) != 0.0));
}

static uint16_t fp16Bits(double value) {
    uint16_t sign = std::signbit(value) ? 0x8000u : 0;
    double magnitude = std::fabs(value);
    if (magnitude == 0.0) return sign;
    if (magnitude < std::ldexp(1.0, -14))
        return sign | (uint16_t)roundToNearestEven(std::ldexp(magnitude, 24));
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

static BOOL fp16Scalar(ANEGraphArgument *argument, uint16_t *bits) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"fp16"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    double value = std::strtod(text, &end);
    if (end == text || *end || !std::isfinite(value)) return NO;
    *bits = fp16Bits(value);
    return (*bits & 0x7c00u) != 0x7c00u;
}

static BOOL exactFP16Attribute(ANEGraphArgument *argument, uint16_t *bits,
                               double *valueOut) {
    if (argument.kind != ANEGraphArgumentKindCall ||
        ![argument.calleeName isEqualToString:@"fp32"] ||
        argument.callArguments.count != 1) return NO;
    argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    float value = static_cast<float>(parsed);
    if (end == text || *end || !std::isfinite(value)) return NO;
    *bits = fp16Bits(value);
    uint16_t exponent = (*bits >> 10) & 0x1fu;
    uint16_t fraction = *bits & 0x03ffu;
    double magnitude = exponent
        ? std::ldexp(1024.0 + fraction, exponent - 25)
        : std::ldexp(static_cast<double>(fraction), -24);
    double decoded = (*bits & 0x8000u) ? -magnitude : magnitude;
    if (static_cast<double>(value) != decoded) return NO;
    *valueOut = value;
    return YES;
}

static BOOL fp32Attribute(ANEGraphArgument *argument, float *valueOut) {
    if (argument.kind != ANEGraphArgumentKindCall ||
        ![argument.calleeName isEqualToString:@"fp32"] ||
        argument.callArguments.count != 1) return NO;
    argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    double parsed = std::strtod(text, &end);
    if (end == text || *end || !std::isfinite(parsed)) return NO;
    *valueOut = static_cast<float>(parsed);
    return YES;
}

static NSData *splatFP16(uint16_t bits) {
    NSMutableData *data = [NSMutableData dataWithLength:128];
    uint16_t *words = static_cast<uint16_t *>(data.mutableBytes);
    for (NSUInteger index = 0; index < 64; ++index) words[index] = bits;
    return data;
}

static ANEGraphArgument *valueArgument(ANEGraphValue *value,
                                       ANESourceRange range) {
    return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindValue
        text:nil value:value calleeName:nil calleeValueType:nil
        callArguments:@[] elements:@[] range:range];
}

static ANEGraphArgument *booleanArgument(BOOL value, ANESourceRange range) {
    return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindBoolean
        text:value ? @"true" : @"false" value:nil calleeName:nil
        calleeValueType:nil callArguments:@[] elements:@[] range:range];
}

static ANEGraphOperation *binaryOperation(NSString *name, ANEGraphValue *x,
                                          ANEGraphValue *y,
                                          ANEGraphValue *result,
                                          ANESourceRange range) {
    return [[ANEGraphOperation alloc] initWithOperationName:name result:result
        arguments:@{@"x": valueArgument(x, range), @"y": valueArgument(y, range)}
        attributes:@{} range:range];
}

static NSString *hexData(NSData *data) {
    const uint8_t *bytes = static_cast<const uint8_t *>(data.bytes);
    NSMutableString *hex = [NSMutableString stringWithCapacity:data.length * 2];
    for (NSUInteger index = 0; index < data.length; ++index)
        [hex appendFormat:@"%02x", bytes[index]];
    return hex;
}

static BOOL boolean(ANEGraphArgument *argument, BOOL expected) {
    if (argument.kind == ANEGraphArgumentKindValue) {
        ANEGraphOperation *producer = argument.value.producer;
        if (![producer.operationName isEqualToString:@"const"] ||
            producer.result.type.kind != ANEValueTypeKindScalar ||
            producer.result.type.elementType != ANEElementTypeBool) return NO;
        argument = producer.attributes[@"val"];
    }
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"bool"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument.kind == ANEGraphArgumentKindBoolean &&
        [argument.text isEqualToString:expected ? @"true" : @"false"];
}

static BOOL matmulGeometry(ANEGraphValue *x, ANEGraphValue *result,
                           BOOL transposeX, NSUInteger *reduction,
                           NSUInteger *rows, NSUInteger *columns) {
    if (!fp16Tensor(x) || !fp16Tensor(result)) return NO;
    NSArray<NSNumber *> *inputShape = x.type.shape;
    NSArray<NSNumber *> *outputShape = result.type.shape;
    NSUInteger rank = inputShape.count;
    if (outputShape.count != rank || (!transposeX && rank == 0) ||
        (transposeX && rank < 2)) return NO;

    NSUInteger leading = rank - (transposeX ? 2 : 1);
    NSUInteger rowCount = 1;
    for (NSUInteger index = 0; index < leading; ++index) {
        NSUInteger dimension = inputShape[index].unsignedIntegerValue;
        if (!dimension || outputShape[index].unsignedIntegerValue != dimension ||
            rowCount > NSUIntegerMax / dimension) return NO;
        rowCount *= dimension;
    }
    NSUInteger candidate = inputShape[rank - (transposeX ? 2 : 1)].unsignedIntegerValue;
    NSUInteger outputColumns = outputShape[rank - 1].unsignedIntegerValue;
    if (!candidate || !outputColumns) return NO;
    if (transposeX) {
        NSUInteger rowDimension = inputShape[rank - 1].unsignedIntegerValue;
        if (!rowDimension || outputShape[rank - 2].unsignedIntegerValue != rowDimension ||
            rowCount > NSUIntegerMax / rowDimension) return NO;
        rowCount *= rowDimension;
    }
    if (rowCount > NSUIntegerMax / candidate ||
        rowCount > NSUIntegerMax / outputColumns) return NO;
    *reduction = candidate;
    *rows = rowCount;
    *columns = outputColumns;
    return YES;
}

/// The NCHW surface a rank-4 elementwise operand lays out as, batch included.
static BOOL batchedShape(ANEGraphValue *value, ane::h13::BatchedShape *shape) {
    if (!fp16Tensor(value) || value.type.shape.count != 4) return NO;
    uint64_t extents[4];
    for (NSUInteger index = 0; index < 4; ++index) {
        extents[index] = value.type.shape[index].unsignedLongLongValue;
        if (!extents[index] || extents[index] > UINT32_MAX) return NO;
    }
    *shape = {static_cast<std::uint32_t>(extents[0]),
              static_cast<std::uint32_t>(extents[1]),
              static_cast<std::uint32_t>(extents[2]),
              static_cast<std::uint32_t>(extents[3])};
    return YES;
}

/// Matches the matmul geometries whose whole H13 task streams are decoded
/// byte-for-byte from Apple oracles, so they encode as one whole-tensor
/// program with every row and column in place. The decoded stream depends on
/// both transpose flags and on whether the second operand is a runtime
/// surface, so all three are part of the key.
static BOOL matmulParityShape(ANEGraphValue *x, ANEGraphValue *result,
                              BOOL transposeX, BOOL transposeY,
                              BOOL runtimeWeight, BOOL preferNative,
                              ane::h13::MatmulShape *shape) {
    NSUInteger reduction = 0, rows = 0, columns = 0;
    if (!fp16Tensor(x) ||
        !matmulGeometry(x, result, transposeX, &reduction, &rows, &columns) ||
        rows > UINT32_MAX || reduction > UINT32_MAX || columns > UINT32_MAX)
        return NO;
    if (preferNative && rows == 1 && !runtimeWeight) return NO;
    const ane::h13::MatmulShape candidate{static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(reduction),
        static_cast<std::uint32_t>(columns), transposeX == YES,
        transposeY == YES, runtimeWeight == YES};
    if (!ane::h13::supportsMatmulParity(candidate)) return NO;
    if (shape) *shape = candidate;
    return YES;
}

/// True when the second matmul operand is a runtime surface of the shape the
/// transpose flag implies, with rank matching x and every batch axis at 1.
static BOOL runtimeMatmulOperand(ANEGraphValue *y, NSUInteger reduction,
                                 NSUInteger columns, BOOL transposeY,
                                 NSUInteger rank) {
    if (!fp16Tensor(y) || y.type.shape.count != rank || rank < 2) return NO;
    for (NSUInteger index = 0; index + 2 < rank; ++index)
        if (y.type.shape[index].unsignedIntegerValue != 1) return NO;
    const NSUInteger rows = transposeY ? columns : reduction;
    const NSUInteger width = transposeY ? reduction : columns;
    return y.type.shape[rank - 2].unsignedIntegerValue == rows &&
        y.type.shape[rank - 1].unsignedIntegerValue == width;
}

/// True when Apple's decoded corpus covers this geometry as one program, so
/// the planner must not slice its reduction, rows, or columns.
static BOOL matmulParityCovered(NSUInteger rows, NSUInteger reduction,
                                NSUInteger columns, BOOL transposeX,
                                BOOL transposeY, BOOL runtimeWeight, BOOL preferNative) {
    if (preferNative && rows == 1 && !runtimeWeight) return NO;
    if (rows > UINT32_MAX || reduction > UINT32_MAX || columns > UINT32_MAX)
        return NO;
    return ane::h13::supportsMatmulParity({static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(reduction),
        static_cast<std::uint32_t>(columns), transposeX == YES,
        transposeY == YES, runtimeWeight == YES});
}

static NSDictionary *binding(ANEGraphValue *value,
                             NSArray<NSNumber *> *logicalShape,
                             const ane::h13::TensorLayout &layout) {
    NSMutableArray *physical = [NSMutableArray arrayWithCapacity:6];
    for (std::uint64_t dimension : layout.nchw) [physical addObject:@(dimension)];
    NSUInteger elements = 1;
    for (NSNumber *dimension in logicalShape)
        elements *= dimension.unsignedIntegerValue;
    return @{@"name": value.name, @"dtype": @"float16",
        @"shape": logicalShape, @"logicalBytes": @(elements * 2),
        @"index": @(layout.index), @"nchw": physical,
        @"allocationBytes": @(layout.allocationBytes)};
}

static HWXObjectBinding *objectBinding(const ane::h13::TensorLayout &layout,
                                       HWXObjectBindingRole role,
                                       NSUInteger ordinal) {
    NSArray<NSNumber *> *shape = @[@(layout.nchw[0]), @(layout.nchw[1]),
        @(layout.nchw[2]), @(layout.nchw[3])];
    NSUInteger batchStride = (NSUInteger)(layout.nchw[1] * layout.nchw[4]);
    NSString *name = [NSString stringWithFormat:
        role == HWXObjectBindingRoleInput ? @"input%lu" : @"output%lu",
        (unsigned long)ordinal];
    // Apple's descriptor declares one batch element's span as the surface
    // size and spaces the surfaces by the whole allocation.
    HWXObjectBinding *binding = [[HWXObjectBinding alloc]
        initWithSymbol:name shortName:name role:role
        elementType:ANEElementTypeFP16 shape:shape
        rowStrideBytes:(NSUInteger)layout.nchw[5]
        planeStrideBytes:(NSUInteger)layout.nchw[4]
        batchStrideBytes:batchStride storageByteLength:batchStride];
    binding.allocationByteLength = (NSUInteger)layout.allocationBytes;
    return binding;
}

static NSArray<NSNumber *> *relocationOffsets(const ane::h13::Program &program) {
    NSMutableArray<NSNumber *> *offsets =
        [NSMutableArray arrayWithCapacity:program.kernelRelocations.size()];
    for (std::size_t offset : program.kernelRelocations)
        [offsets addObject:@(offset)];
    return offsets;
}

static NSData *encodeHWX(const ane::h13::Program &program, NSError **error) {
    // The writer walks this array to assign surface addresses. Apple's matmul
    // objects place the output surface below both operands, a broadcast puts
    // it between them, and every other decoded object places the inputs first.
    NSMutableArray<HWXObjectBinding *> *bindings = [NSMutableArray array];
    HWXObjectBinding *output = objectBinding(program.output,
        HWXObjectBindingRoleOutput, 0);
    const std::size_t outputIndex =
        std::min(program.outputBindingIndex, program.inputs.size());
    for (NSUInteger index = 0; index < program.inputs.size(); ++index) {
        if (index == outputIndex) [bindings addObject:output];
        [bindings addObject:objectBinding(program.inputs[index],
            HWXObjectBindingRoleInput, index)];
    }
    if (outputIndex >= program.inputs.size()) [bindings addObject:output];
    NSData *task = [NSData dataWithBytes:program.task.data()
                                  length:program.task.size()];
    NSData *constants = [NSData dataWithBytes:program.constants.data()
                                       length:program.constants.size()];
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:program.taskCount
        firstTaskByteLength:program.firstTaskBytes recordCount:1
        formatCode:0 scratchByteLength:0
        descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.scratchAllocationByteLength =
        static_cast<NSUInteger>(program.scratchAllocationBytes);
    return [HWXObjectWriter buildObjectForArchitecture:HWXObjectArchitectureH13
        taskDescriptor:task constantRegion:constants bindings:bindings
        kernelRelocationOffsets:relocationOffsets(program) programInfo:info
        error:error];
}

static void recordTensor(NSMutableDictionary<NSString *, NSDictionary *> *tensors,
                         ANEGraphValue *value, NSArray<NSNumber *> *shape,
                         NSString *role) {
    if (tensors[value.name]) return;
    NSUInteger elements = 1;
    for (NSNumber *dimension in shape) elements *= dimension.unsignedIntegerValue;
    tensors[value.name] = @{@"shape": shape, @"logicalBytes": @(elements * 2),
                            @"role": role};
}

static void addSlice(NSMutableDictionary *record, ANEGraphValue *value,
                     NSUInteger offset, NSUInteger count, NSUInteger physical) {
    NSMutableDictionary *slice = [@{@"tensor": value.name,
        @"elementOffset": @(offset), @"elementCount": @(count)} mutableCopy];
    if (physical != count) slice[@"physicalElements"] = @(physical);
    record[@"slice"] = slice;
}

static BOOL constantValue(ANEGraphValue *value) {
    return [value.producer.operationName isEqualToString:@"const"];
}

static NSString *stringArgument(ANEGraphArgument *argument) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"string"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument.kind == ANEGraphArgumentKindString ? argument.text : nil;
}

struct H13ParityPlan {
    BOOL unary;
    BOOL scalarConstant;
    ane::h13::UnaryOperation unaryOperation;
    ane::h13::BinaryOperation binaryOperation;
    uint16_t scalarBits;
    ane::h13::ElementwiseShape shape;
    NSUInteger elements;
    NSUInteger inputCount;
};

/// The decoded shapes a tensor may lower through: its literal NCHW geometry
/// when spatial, and its channel-flattened form, which shares the physical
/// 64-byte row layout the pipeline already packs logical tensors into.
static NSUInteger parityShapes(ANEGraphValue *value,
                               ane::h13::ElementwiseShape shapes[2]) {
    NSUInteger elements = 0;
    if (!tensorElementCount(value, &elements) || elements > UINT32_MAX) return 0;
    NSUInteger count = 0;
    ane::h13::ElementwiseShape literal{};
    if (elementwiseShape(value, &literal) &&
        (literal.height != 1 || literal.width != 1))
        shapes[count++] = literal;
    shapes[count++] = {static_cast<std::uint32_t>(elements), 1, 1};
    return count;
}

static BOOL nativeBinaryPlan(ANEGraphOperation *operation) {
    NSString *name = operation.operationName;
    BOOL multiply = [name isEqualToString:@"mul"];
    if (operation.arguments.count != 2 ||
        (!multiply && ![name isEqualToString:@"add"] &&
         ![name isEqualToString:@"maximum"] &&
         ![name isEqualToString:@"minimum"])) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    if (!tensor(x, operation.result.type.shape)) return NO;
    return tensor(y, x.type.shape) ||
        (multiply && constantValue(y) &&
         y.type.kind == ANEValueTypeKindScalar &&
         y.type.elementType == ANEElementTypeFP16);
}

/// Matches the operations whose H13 task streams are decoded byte-for-byte
/// from Apple oracles, so they encode as one whole-tensor program.
static BOOL parityPlan(ANEGraphOperation *operation,
                       NSDictionary<NSString *, NSData *> *synthesizedConstants,
                       BOOL preferNative, H13ParityPlan *plan) {
    if (preferNative && nativeBinaryPlan(operation)) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    H13ParityPlan candidate{};
    ane::h13::ElementwiseShape shapes[2];
    NSUInteger shapeCount = parityShapes(operation.result, shapes);
    if (!shapeCount || !tensor(x, operation.result.type.shape)) return NO;
    BOOL leaky = [name isEqualToString:@"leaky_relu"];
    BOOL gelu = [name isEqualToString:@"gelu"];
    BOOL rsqrt = [name isEqualToString:@"rsqrt"];
    if (unaryEncoding(name, &candidate.unaryOperation)) {
        uint16_t alphaBits = 0;
        double alpha = 0.0;
        float epsilon = 0.0f;
        if (operation.arguments.count != (leaky || gelu || rsqrt ? 2u : 1u) ||
            (leaky && (!exactFP16Attribute(operation.arguments[@"alpha"],
                                           &alphaBits, &alpha) ||
                       alphaBits != 0x3000)) ||
            (gelu && ![stringArgument(operation.arguments[@"mode"])
                          isEqualToString:@"EXACT"]) ||
            (rsqrt && (!fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
                       epsilon != 1e-6f)))
            return NO;
        for (NSUInteger index = 0; index < shapeCount; ++index) {
            if (!ane::h13::supportsElementwise(candidate.unaryOperation, shapes[index]))
                continue;
            candidate.shape = shapes[index];
            candidate.elements = (NSUInteger)shapes[index].channels *
                shapes[index].height * shapes[index].width;
            candidate.unary = YES;
            candidate.inputCount = 1;
            *plan = candidate;
            return YES;
        }
        return NO;
    }
    if (!binaryEncoding(name, &candidate.binaryOperation) ||
        operation.arguments.count != 2 || !y) return NO;
    if (synthesizedConstants[x.name] || constantValue(x)) return NO;
    BOOL runtime = !synthesizedConstants[y.name] && !constantValue(y);
    if (runtime) {
        if (!tensor(y, operation.result.type.shape)) return NO;
    } else if (synthesizedConstants[y.name] ||
               y.type.kind != ANEValueTypeKindScalar ||
               y.type.elementType != ANEElementTypeFP16 ||
               y.producer.arguments.count ||
               !fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits) ||
               candidate.scalarBits != 0x3800) {
        return NO;
    }
    for (NSUInteger index = 0; index < shapeCount; ++index) {
        if (!ane::h13::supportsElementwise(candidate.binaryOperation, shapes[index],
                                           !runtime))
            continue;
        candidate.shape = shapes[index];
        candidate.elements = (NSUInteger)shapes[index].channels *
            shapes[index].height * shapes[index].width;
        candidate.scalarConstant = !runtime;
        candidate.inputCount = runtime ? 2 : 1;
        *plan = candidate;
        return YES;
    }
    return NO;
}

struct H13BroadcastPlan {
    ane::h13::BinaryOperation operation;
    ane::h13::BroadcastOperand operand;
    ane::h13::BroadcastShape shape;
    ane::h13::BatchedShape result;
    uint16_t scalarBits;
    NSUInteger inputCount;
};

static BOOL sameBatchedShape(ane::h13::BatchedShape left,
                             ane::h13::BatchedShape right) {
    return left.batch == right.batch && left.channels == right.channels &&
        left.height == right.height && left.width == right.width;
}

/// Matches the broadcast forms whose whole H13 task streams are decoded from
/// Apple oracles: an NCHW runtime tensor against a second runtime tensor of
/// any broadcastable NCHW shape, an inline fp16 scalar, or a per-channel
/// constant Apple folds into the constant section's bias and scale blocks.
static BOOL broadcastPlan(ANEGraphOperation *operation,
                          NSDictionary<NSString *, NSData *> *synthesizedConstants,
                          BOOL preferNative, H13BroadcastPlan *plan,
                          ANEGraphValue *__autoreleasing *constantOut) {
    if (preferNative && nativeBinaryPlan(operation)) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    H13BroadcastPlan candidate{};
    if (!binaryEncoding(operation.operationName, &candidate.operation) ||
        operation.arguments.count != 2 || !x || !y) return NO;
    if (synthesizedConstants[x.name] || constantValue(x) ||
        synthesizedConstants[y.name]) return NO;
    if (!batchedShape(x, &candidate.shape.x) ||
        !batchedShape(operation.result, &candidate.result)) return NO;
    ANEGraphValue *constant = nil;
    if (constantValue(y)) {
        candidate.inputCount = 1;
        if (y.producer.arguments.count) return NO;
        if (y.type.kind == ANEValueTypeKindScalar &&
            y.type.elementType == ANEElementTypeFP16) {
            if (!fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits))
                return NO;
            candidate.operand = ane::h13::BroadcastOperand::Scalar;
        } else if (batchedShape(y, &candidate.shape.y) &&
                   candidate.shape.y.batch == 1 &&
                   candidate.shape.y.height == 1 &&
                   candidate.shape.y.width == 1 &&
                   candidate.shape.y.channels == candidate.shape.x.channels) {
            candidate.operand = ane::h13::BroadcastOperand::Constant;
            constant = y;
        } else {
            return NO;
        }
    } else if (batchedShape(y, &candidate.shape.y)) {
        candidate.operand = ane::h13::BroadcastOperand::Runtime;
        candidate.inputCount = 2;
    } else {
        return NO;
    }
    ane::h13::BatchedShape broadcast = candidate.shape.x;
    if (candidate.operand == ane::h13::BroadcastOperand::Runtime)
        broadcast = {std::max(candidate.shape.x.batch, candidate.shape.y.batch),
                     std::max(candidate.shape.x.channels, candidate.shape.y.channels),
                     std::max(candidate.shape.x.height, candidate.shape.y.height),
                     std::max(candidate.shape.x.width, candidate.shape.y.width)};
    if (!sameBatchedShape(broadcast, candidate.result)) return NO;
    if (!ane::h13::supportsBroadcast(candidate.operation, candidate.operand,
                                     candidate.shape)) return NO;
    *plan = candidate;
    *constantOut = constant;
    return YES;
}

/// The fp16 values of a per-channel `[1, C, 1, 1]` constant operand, from an
/// inline typed list or a BLOBFILE payload.
static NSData *perChannelConstantData(ANEGraphValue *value, NSUInteger channels,
                                      NSURL *modelRoot,
                                      ANEDiagnosticEngine *diagnostics,
                                      NSMutableDictionary<NSString *, NSData *> *resolved) {
    NSData *data = resolved[value.name];
    if (data) return data;
    ANEGraphOperation *producer = value.producer;
    ANEGraphArgument *literal = producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) {
        reject(diagnostics,
            @"H13 per-channel constants require a matching typed inline list or BLOBFILE payload",
            producer, @"h13.invalid-constant-payload");
        return nil;
    }
    ANEGraphArgument *payload = literal.callArguments[0].value;
    if (payload.kind == ANEGraphArgumentKindList &&
        payload.elements.count == channels) {
        NSMutableData *dense = [NSMutableData dataWithLength:channels * 2];
        uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
        for (NSUInteger index = 0; index < channels; ++index)
            if (!fp16Scalar(payload.elements[index], &words[index])) {
                reject(diagnostics,
                    @"H13 per-channel constants require finite fp16 elements",
                    producer, @"h13.invalid-constant-payload");
                return nil;
            }
        data = dense;
    } else if (payload.kind == ANEGraphArgumentKindCall &&
               [payload.calleeName isEqualToString:@"BLOBFILE"]) {
        data = [ANEBlobResolver loadConstantForOperation:producer
            expectedBytes:channels * 2 modelRoot:modelRoot diagnostics:diagnostics];
    }
    if (!data) {
        if (!diagnostics.errorCount)
            reject(diagnostics,
                @"H13 per-channel constants require C finite fp16 elements",
                producer, @"h13.invalid-constant-payload");
        return nil;
    }
    resolved[value.name] = data;
    return data;
}

struct H13NormPlan {
    ane::h13::NormOperation operation;
    ane::h13::NormShape shape;
    NSUInteger inputElements;
    NSUInteger outputElements;
};

/// Matches the softmax, layer_norm, and reduction programs whose whole H13
/// task streams are decoded from Apple oracles. `epsilon` must be the decoded
/// 1e-5 and layer_norm must carry no gamma or beta: Apple's compiler in this
/// harness rejects every affine form, so no oracle covers one.
static BOOL normParityPlan(ANEGraphOperation *operation, H13NormPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSString *name = operation.operationName;
    H13NormPlan candidate{};
    if (!normEncoding(name, &candidate.operation) || !x) return NO;
    const BOOL softmax = candidate.operation == ane::h13::NormOperation::Softmax;
    const BOOL layerNorm = candidate.operation == ane::h13::NormOperation::LayerNorm;
    NSInteger inputShift = 0;
    if (!fp16Tensor(x) || !fp16Tensor(operation.result) ||
        !normSurface(x.type.shape, &candidate.shape.input, &inputShift) ||
        !normSurface(operation.result.type.shape, &candidate.shape.output, nullptr))
        return NO;
    if (!constantAxisMask(operation.operands[softmax ? @"axis" : @"axes"].value,
                          x.type.shape.count, inputShift,
                          &candidate.shape.axisMask))
        return NO;
    float epsilon = 0.0f;
    if (softmax) {
        if (operation.arguments.count != 2) return NO;
    } else if (layerNorm) {
        if (operation.arguments.count != 3 ||
            !fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
            epsilon != 1e-5f) return NO;
    } else {
        if (operation.arguments.count != 3) return NO;
    }
    candidate.shape.keepDims = softmax || layerNorm ||
        boolean(operation.arguments[@"keep_dims"], YES);
    if (!candidate.shape.keepDims &&
        !boolean(operation.arguments[@"keep_dims"], NO)) return NO;
    if (!tensorElementCount(x, &candidate.inputElements) ||
        !tensorElementCount(operation.result, &candidate.outputElements))
        return NO;
    if (!ane::h13::supportsNormParity(candidate.operation, candidate.shape))
        return NO;
    *plan = candidate;
    return YES;
}

struct H13ConvPlan {
    ane::h13::ConvShape shape;
    ANEGraphValue *weight;
    ANEGraphValue *bias;
};

/// Resolves a rank-1 `int32` constant into its literal values.
static BOOL int32Vector(ANEGraphValue *value, NSUInteger count,
                        long long *values) {
    if (!value || !constantValue(value) || value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindTensor ||
        value.type.elementType != ANEElementTypeInt32 ||
        value.type.shape.count != 1 ||
        value.type.shape[0].unsignedIntegerValue != count) return NO;
    ANEGraphArgument *literal = value.producer.attributes[@"val"];
    if (literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:value.type] ||
        literal.callArguments.count != 1) return NO;
    ANEGraphArgument *payload = literal.callArguments[0].value;
    if (payload.kind != ANEGraphArgumentKindList ||
        payload.elements.count != count) return NO;
    for (NSUInteger index = 0; index < count; ++index)
        if (!int32Literal(payload.elements[index], &values[index])) return NO;
    return YES;
}

static BOOL int32Scalar(ANEGraphValue *value, long long *result) {
    if (!value || !constantValue(value) || value.producer.arguments.count ||
        value.type.kind != ANEValueTypeKindScalar ||
        value.type.elementType != ANEElementTypeInt32) return NO;
    return int32Literal(value.producer.attributes[@"val"], result);
}

static BOOL constantString(ANEGraphValue *value, NSString **text) {
    if (!value || !constantValue(value) || value.producer.arguments.count ||
        value.type.elementType != ANEElementTypeString) return NO;
    NSString *resolved = stringArgument(value.producer.attributes[@"val"]);
    if (!resolved) return NO;
    *text = resolved;
    return YES;
}

/// The CHW surface a convolution operand covers: a rank-4 NCHW tensor with a
/// batch of one, which is the only form the decoded corpus carries.
static BOOL convSurface(ANEGraphValue *value, ane::h13::ElementwiseShape *shape) {
    return elementwiseShape(value, shape);
}

/// Matches the convolutions whose whole H13 task stream and constant section
/// are decoded from Apple oracles.
///
/// coremltools 9d9de1aebd4f082fb9e7076c9799a1b5f29ba5e4 defines `conv` with a
/// `[Cout, Cin / groups, kh, kw]` weight, square `strides` and `dilations`
/// vectors, and either an explicit `pad` with `pad_type="custom"` or the
/// `same`, `same_lower` and `valid` spellings. The decoded corpus covers
/// square kernels, unit dilations, `same` and `valid`, and the explicit `pad`
/// vector only when it is all zeroes, which is what those two spellings emit.
static BOOL convParityPlan(ANEGraphOperation *operation, H13ConvPlan *plan) {
    if (![operation.operationName isEqualToString:@"conv"]) return NO;
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *weight = operation.operands[@"weight"].value;
    ANEGraphValue *bias = operation.operands[@"bias"].value;
    H13ConvPlan candidate{};
    candidate.weight = weight;
    candidate.bias = bias;
    if (!x || !weight || !fp16Tensor(weight) || weight.type.shape.count != 4 ||
        !convSurface(x, &candidate.shape.input) ||
        !convSurface(operation.result, &candidate.shape.output)) return NO;
    if (operation.arguments.count != (bias ? 8u : 7u)) return NO;
    NSString *padType = nil;
    long long strides[2] = {0, 0}, dilations[2] = {0, 0}, padding[4] = {0, 0, 0, 0};
    long long groups = 1;
    if (!constantString(operation.operands[@"pad_type"].value, &padType) ||
        !int32Vector(operation.operands[@"strides"].value, 2, strides) ||
        !int32Vector(operation.operands[@"dilations"].value, 2, dilations) ||
        !int32Vector(operation.operands[@"pad"].value, 4, padding) ||
        !int32Scalar(operation.operands[@"groups"].value, &groups)) return NO;
    if (![padType isEqualToString:@"same"] && ![padType isEqualToString:@"valid"])
        return NO;
    if (strides[0] != strides[1] || dilations[0] != 1 || dilations[1] != 1 ||
        strides[0] < 1 || groups < 1) return NO;
    for (NSUInteger index = 0; index < 4; ++index)
        if (padding[index]) return NO;
    const std::uint32_t kernel = weight.type.shape[2].unsignedIntValue;
    if (weight.type.shape[3].unsignedIntValue != kernel || !kernel) return NO;
    if (weight.type.shape[0].unsignedIntValue != candidate.shape.output.channels)
        return NO;
    if (!candidate.shape.input.channels ||
        candidate.shape.input.channels % groups ||
        weight.type.shape[1].unsignedIntValue !=
            candidate.shape.input.channels / groups) return NO;
    // The result surface must be what the pad type asks for: `same` covers
    // ceil(D / stride) and `valid` covers floor((D - kernel) / stride) + 1.
    const std::uint32_t extents[2] = {candidate.shape.input.height,
                                      candidate.shape.input.width};
    const std::uint32_t results[2] = {candidate.shape.output.height,
                                      candidate.shape.output.width};
    for (NSUInteger axis = 0; axis < 2; ++axis) {
        const std::uint32_t stride = static_cast<std::uint32_t>(strides[0]);
        if ([padType isEqualToString:@"same"]) {
            if (results[axis] != (extents[axis] + stride - 1) / stride) return NO;
        } else {
            if (extents[axis] < kernel ||
                results[axis] != (extents[axis] - kernel) / stride + 1) return NO;
        }
    }
    candidate.shape.kernel = kernel;
    candidate.shape.stride = static_cast<std::uint32_t>(strides[0]);
    candidate.shape.groups = static_cast<std::uint32_t>(groups);
    candidate.shape.bias = bias != nil;
    if (bias && (!fp16Tensor(bias) || bias.type.shape.count != 1 ||
                 bias.type.shape[0].unsignedIntValue !=
                     candidate.shape.output.channels)) return NO;
    if (!ane::h13::supportsConvParity(candidate.shape)) return NO;
    *plan = candidate;
    return YES;
}

static NSData *linearBiasData(ANEGraphValue *bias, NSUInteger columns,
                                NSURL *modelRoot,
                                ANEDiagnosticEngine *diagnostics,
                                NSMutableDictionary<NSString *, NSData *> *resolved) {
    if (!tensor(bias, @[@(columns)])) {
        reject(diagnostics,
            @"H13 linear bias must be a constant fp16 vector of N elements",
            bias.producer, @"h13.invalid-linear-bias");
        return nil;
    }
    ANEGraphOperation *producer = bias.producer;
    ANEGraphArgument *literal = producer.attributes[@"val"];
    if (producer.arguments.count || literal.kind != ANEGraphArgumentKindCall ||
        ![literal.calleeValueType isEqualToValueType:bias.type] ||
        literal.callArguments.count != 1) {
        reject(diagnostics,
            @"H13 linear bias requires a typed inline list or BLOBFILE payload",
            producer, @"h13.invalid-linear-bias");
        return nil;
    }
    ANEGraphArgument *payload = literal.callArguments[0].value;
    NSData *data = resolved[bias.name];
    if (!data && payload.kind == ANEGraphArgumentKindList &&
        payload.elements.count == columns) {
        NSMutableData *dense = [NSMutableData dataWithLength:columns * 2];
        uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
        for (NSUInteger index = 0; index < columns; ++index) {
            if (!fp16Scalar(payload.elements[index], &words[index])) {
                reject(diagnostics, @"H13 linear bias requires finite fp16 elements",
                    producer, @"h13.invalid-linear-bias");
                return nil;
            }
        }
        data = dense;
    } else if (!data && payload.kind == ANEGraphArgumentKindCall &&
               [payload.calleeName isEqualToString:@"BLOBFILE"]) {
        data = [ANEBlobResolver loadConstantForOperation:producer
            expectedBytes:columns * 2 modelRoot:modelRoot diagnostics:diagnostics];
    }
    if (!data) {
        if (!diagnostics.errorCount)
            reject(diagnostics, @"H13 linear bias requires N finite fp16 elements",
                producer, @"h13.invalid-linear-bias");
        return nil;
    }
    resolved[bias.name] = data;
    return data;
}

static BOOL lowerOperation(ANEGraphOperation *operation, NSURL *modelRoot,
                           ANEDiagnosticEngine *diagnostics, BOOL preferNative,
                           NSDictionary<NSString *, NSData *> *synthesizedConstants,
                           NSMutableDictionary<NSString *, NSData *> *resolvedConstants,
                           NSUInteger elementOffset, NSUInteger inputElementCount,
                           NSUInteger outputElementOffset, ane::h13::Program &program,
                           NSArray<ANEGraphValue *> *__autoreleasing *inputsOut,
                           ANEGraphValue *__autoreleasing *constantInputOut,
                           NSData *__autoreleasing *constantDataOut,
                           NSString *__autoreleasing *manifestOperationOut) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    NSArray<NSString *> *binaryNames =
        @[@"add", @"mul", @"maximum", @"minimum", @"sub", @"real_div"];
    NSUInteger binaryIndex = [binaryNames indexOfObject:name];
    NSArray<ANEGraphValue *> *inputs = nil;
    ANEGraphValue *constantInput = nil;
    NSData *constantData = nil;
    NSString *manifestOperation = name;

    H13BroadcastPlan broadcast{};
    ANEGraphValue *broadcastConstant = nil;
    if (broadcastPlan(operation, synthesizedConstants, preferNative, &broadcast,
                      &broadcastConstant)) {
        NSData *values = nil;
        if (broadcast.operand == ane::h13::BroadcastOperand::Constant) {
            values = perChannelConstantData(broadcastConstant,
                broadcast.shape.x.channels, modelRoot, diagnostics,
                resolvedConstants);
            if (!values) return NO;
        }
        program = ane::h13::encodeBroadcast(broadcast.operation,
            broadcast.operand, broadcast.shape,
            static_cast<const std::uint8_t *>(values.bytes), values.length,
            broadcast.scalarBits);
        *inputsOut = broadcast.inputCount == 2 ? @[x, y] : @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    H13ParityPlan plan{};
    if (parityPlan(operation, synthesizedConstants, preferNative, &plan)) {
        program = plan.unary
            ? ane::h13::encodeElementwise(plan.unaryOperation, plan.shape)
            : ane::h13::encodeElementwise(plan.binaryOperation, plan.shape,
                                          plan.scalarConstant, plan.scalarBits);
        *inputsOut = plan.inputCount == 2 ? @[x, y] : @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    H13NormPlan normalization{};
    if (normParityPlan(operation, &normalization)) {
        program = ane::h13::encodeNormParity(normalization.operation,
                                             normalization.shape);
        *inputsOut = @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    H13ConvPlan convolution{};
    if (convParityPlan(operation, &convolution)) {
        const std::uint32_t reduction = convolution.shape.input.channels /
            convolution.shape.groups * convolution.shape.kernel *
            convolution.shape.kernel;
        NSData *weights = resolvedConstants[convolution.weight.name];
        if (!weights) {
            weights = [ANEBlobResolver loadConstantForOperation:convolution.weight.producer
                expectedBytes:convolution.shape.output.channels * reduction * 2
                modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            resolvedConstants[convolution.weight.name] = weights;
        }
        NSData *biasData = nil;
        if (convolution.bias) {
            biasData = perChannelConstantData(convolution.bias,
                convolution.shape.output.channels, modelRoot, diagnostics,
                resolvedConstants);
            if (!biasData) return NO;
        }
        // The MIL weight is already `[Cout, Cin / groups, kh, kw]` row-major,
        // which is the order the packing consumes, so no host transform runs.
        program = ane::h13::encodeConvParity(convolution.shape,
            static_cast<const std::uint8_t *>(weights.bytes), weights.length,
            static_cast<const std::uint8_t *>(biasData.bytes), biasData.length);
        *inputsOut = @[x];
        *constantInputOut = nil;
        *constantDataOut = nil;
        *manifestOperationOut = name;
        return YES;
    }

    if (binaryIndex != NSNotFound) {
        if (operation.arguments.count != 2 || !x || !y)
            return reject(diagnostics,
                @"H13 binary operations require x and y value operands", operation);
        BOOL xIsConstant = synthesizedConstants[x.name] || constantValue(x);
        BOOL yIsConstant = synthesizedConstants[y.name] || constantValue(y);
        NSUInteger elements = 0;
        if (!xIsConstant && !yIsConstant) {
            if (binaryIndex >= 4)
                return reject(diagnostics, [NSString stringWithFormat:
                    @"H13 '%@' with two runtime inputs cannot lower exactly through the verified binary modes", name],
                    operation, @"h13.nonfoldable-binary");
            if (!tensorElementCount(x, &elements) || !tensor(y, x.type.shape) ||
                !tensor(operation.result, x.type.shape))
                return reject(diagnostics,
                    @"H13 binary operations require fp16 inputs with the same positive static shape",
                    operation);
            inputs = @[x, y];
        } else {
            if (xIsConstant == yIsConstant || (binaryIndex >= 4 && xIsConstant)) {
                NSString *code = binaryIndex >= 4
                    ? @"h13.nonfoldable-binary" : @"h13.invalid-constant-input";
                return reject(diagnostics,
                    @"H13 binary folding requires one runtime fp16 tensor and one eligible const operand",
                    operation, code);
            }
            ANEGraphValue *runtimeInput = xIsConstant ? y : x;
            constantInput = xIsConstant ? x : y;
            if (!tensorElementCount(runtimeInput, &elements) ||
                !tensor(operation.result, runtimeInput.type.shape))
                return reject(diagnostics,
                    @"H13 folded binary operations require one fp16 input and output with the same positive static shape",
                    operation, @"h13.invalid-constant-input");
            if (elementOffset >= elements)
                return reject(diagnostics, @"H13 elementwise slice exceeds its tensor",
                    operation, @"h13.invalid-slice");
            NSUInteger sliceElements = MIN((NSUInteger)64, elements - elementOffset);

            NSData *wholeData = synthesizedConstants[constantInput.name];
            BOOL repeatedConstant = wholeData.length == 128;
            if (wholeData) {
                if (!tensor(constantInput, runtimeInput.type.shape) ||
                    (!repeatedConstant && wholeData.length != elements * 2))
                    return reject(diagnostics,
                        @"H13 synthesized constants must match the runtime fp16 tensor shape",
                        operation, @"h13.invalid-constant-input");
            } else {
                ANEGraphOperation *producer = constantInput.producer;
                BOOL scalar = constantInput.type.kind == ANEValueTypeKindScalar &&
                    constantInput.type.elementType == ANEElementTypeFP16;
                if ((!tensor(constantInput, runtimeInput.type.shape) && !scalar) ||
                    (scalar && binaryIndex != 1) || producer.arguments.count)
                    return reject(diagnostics,
                        @"H13 constants must be a matching fp16 tensor; only mul accepts an inline fp16 scalar broadcast",
                        producer, @"h13.invalid-constant-input");

                ANEGraphArgument *literal = producer.attributes[@"val"];
                if (scalar) {
                    uint16_t bits = 0;
                    if (!fp16Scalar(literal, &bits))
                        return reject(diagnostics,
                            @"H13 scalar mul requires one finite inline fp16 value",
                            producer, @"h13.invalid-constant-payload");
                    wholeData = splatFP16(bits);
                    repeatedConstant = YES;
                } else {
                    wholeData = resolvedConstants[constantInput.name];
                    if (!wholeData) {
                        if (literal.kind != ANEGraphArgumentKindCall ||
                            ![literal.calleeValueType isEqualToValueType:constantInput.type] ||
                            literal.callArguments.count != 1)
                            return reject(diagnostics,
                                @"H13 tensor constants require a matching typed inline list or BLOBFILE payload",
                                producer, @"h13.invalid-constant-payload");
                        ANEGraphArgument *payload = literal.callArguments[0].value;
                        if (payload.kind == ANEGraphArgumentKindList &&
                            payload.elements.count == elements) {
                            NSMutableData *dense = [NSMutableData dataWithLength:elements * 2];
                            uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
                            for (NSUInteger index = 0; index < elements; ++index)
                                if (!fp16Scalar(payload.elements[index], &words[index]))
                                    return reject(diagnostics,
                                        @"H13 inline tensor constants require finite fp16 elements matching the tensor shape",
                                        producer, @"h13.invalid-constant-payload");
                            wholeData = dense;
                        } else if (payload.kind == ANEGraphArgumentKindCall &&
                                   [payload.calleeName isEqualToString:@"BLOBFILE"]) {
                            wholeData = [ANEBlobResolver loadConstantForOperation:producer
                                expectedBytes:elements * 2 modelRoot:modelRoot
                                diagnostics:diagnostics];
                            if (!wholeData)
                                return reject(diagnostics,
                                    @"H13 could not load the constant BLOBFILE payload",
                                    producer, @"h13.invalid-constant-payload");
                        } else {
                            return reject(diagnostics,
                                @"H13 inline tensor constants require finite fp16 elements matching the tensor shape",
                                producer, @"h13.invalid-constant-payload");
                        }
                        resolvedConstants[constantInput.name] = wholeData;
                    }
                }
            }

            if (!repeatedConstant && wholeData.length != elements * 2)
                return reject(diagnostics, @"H13 constant payload has the wrong size",
                    operation, @"h13.invalid-constant-payload");
            NSMutableData *padded = [NSMutableData dataWithLength:128];
            NSRange sourceRange = NSMakeRange(
                repeatedConstant ? 0 : elementOffset * 2, sliceElements * 2);
            [wholeData getBytes:padded.mutableBytes range:sourceRange];
            constantData = padded;
            NSMutableData *folded = [constantData mutableCopy];
            uint16_t *words = static_cast<uint16_t *>(folded.mutableBytes);
            if (binaryIndex == 4) {
                for (NSUInteger index = 0; index < sliceElements; ++index) {
                    if ((words[index] & 0x7c00u) == 0x7c00u)
                        return reject(diagnostics,
                            @"H13 sub constants must be finite for exact add lowering",
                            operation, @"h13.nonfinite-constant");
                    words[index] ^= 0x8000u;
                }
                manifestOperation = @"add";
            } else if (binaryIndex == 5) {
                for (NSUInteger index = 0; index < sliceElements; ++index) {
                    uint16_t exponent = (words[index] >> 10) & 0x1fu;
                    if (exponent == 0 || exponent == 0x1fu || (words[index] & 0x03ffu))
                        return reject(diagnostics,
                            @"H13 real_div constants require finite nonzero powers of two whose reciprocals are exactly representable in fp16",
                            operation, @"h13.inexact-reciprocal");
                    uint16_t sign = words[index] & 0x8000u;
                    words[index] = exponent == 30 ? sign | 0x0200u
                                                 : sign | ((30 - exponent) << 10);
                }
                manifestOperation = @"mul";
            }
            constantData = folded;
            inputs = @[runtimeInput, constantInput];
        }
        const ane::h13::BinaryOperation operations[] = {
            ane::h13::BinaryOperation::Add, ane::h13::BinaryOperation::Multiply,
            ane::h13::BinaryOperation::Maximum, ane::h13::BinaryOperation::Minimum,
            ane::h13::BinaryOperation::Add, ane::h13::BinaryOperation::Multiply};
        program = ane::h13::encodeBinary(operations[binaryIndex]);
    } else if ([name isEqualToString:@"matmul"]) {
        NSUInteger reduction = 0, rows = 0, columns = 0;
        BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
        BOOL transposeY = boolean(operation.arguments[@"transpose_y"], YES);
        BOOL geometry = matmulGeometry(x, operation.result, transposeX,
                                       &reduction, &rows, &columns);
        if (operation.arguments.count != 4 || constantValue(x) ||
            (!transposeX && !boolean(operation.arguments[@"transpose_x"], NO)) ||
            (!transposeY && !boolean(operation.arguments[@"transpose_y"], NO)) ||
            !geometry || !y)
            return reject(diagnostics,
                @"H13 matmul requires positive fp16 x rows, matching explicit transpose flags, and a matching positive output shape",
                operation);
        const BOOL runtimeWeight =
            !constantValue(y) && !synthesizedConstants[y.name];
        ane::h13::MatmulShape parityShape{};
        if (runtimeWeight) {
            // Both operands runtime: attention's QK^T and PV. Apple stages the
            // second operand as its own surface, so there is nothing to pack
            // and nothing to slice.
            if (!runtimeMatmulOperand(y, reduction, columns, transposeY,
                                      x.type.shape.count) ||
                !matmulParityShape(x, operation.result, transposeX, transposeY,
                                   YES, preferNative, &parityShape) ||
                inputElementCount != rows * reduction)
                return reject(diagnostics,
                    @"H13 matmul with two runtime operands needs a decoded geometry: fp16 static shapes of matching rank, the second operand shaped by transpose_y, and rows, reduction, and columns inside the oracle parity envelope",
                    operation, @"h13.matmul-outside-envelope");
            program = ane::h13::encodeMatmulParity(parityShape, nullptr, 0);
            // Apple declares the second operand first and lays the output out
            // below both, so the package's channel 5 carries y and 6 carries x.
            *inputsOut = @[y, x];
            *constantInputOut = nil;
            *constantDataOut = nil;
            *manifestOperationOut = name;
            return YES;
        }
        if (!tensor(y, transposeY ? @[@(columns), @(reduction)]
                                  : @[@(reduction), @(columns)]))
            return reject(diagnostics,
                @"H13 matmul requires a constant rank-2 W shaped by transpose_y",
                operation);
        ANEGraphArgument *value = y.producer.attributes[@"val"];
        if (y.producer.arguments.count || value.kind != ANEGraphArgumentKindCall ||
            ![value.calleeValueType isEqualToValueType:y.type] ||
            value.callArguments.count != 1 ||
            ![value.callArguments[0].value.calleeName isEqualToString:@"BLOBFILE"])
            return reject(diagnostics,
                @"H13 weights require a matching tensor value with one BLOBFILE payload", y.producer);
        if (columns > NSUIntegerMax / reduction ||
            columns * reduction > NSUIntegerMax / 2)
            return reject(diagnostics, @"H13 matmul weight size overflows",
                operation, @"h13.invalid-constant-payload");
        NSUInteger count = columns * reduction;
        NSData *weights = resolvedConstants[y.name];
        if (!weights) {
            weights = [ANEBlobResolver loadConstantForOperation:y.producer
                expectedBytes:count * 2 modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            resolvedConstants[y.name] = weights;
        }
        if (transposeX && rows > 1 &&
            !matmulParityShape(x, operation.result, YES, YES, NO, preferNative, nullptr))
            return reject(diagnostics,
                @"H13 transpose_x=true matmul supports exactly one logical row outside the decoded parity envelope",
                operation, @"h13.transpose-x-multirow");
        // Apple refuses a transpose_y=false constant weight, so the host
        // transpose below feeds the transpose_y=true program it accepts.
        if (matmulParityShape(x, operation.result, transposeX, YES, NO,
                              preferNative, &parityShape) &&
            inputElementCount == rows * reduction) {
            // transpose_y=false weights are [K, N]; Apple rejects that form, so
            // the exact host transpose feeds the same encoder.
            NSData *rowMajor = weights;
            if (!transposeY) {
                NSMutableData *transposed =
                    [NSMutableData dataWithLength:count * 2];
                const uint16_t *source =
                    static_cast<const uint16_t *>(weights.bytes);
                uint16_t *destination =
                    static_cast<uint16_t *>(transposed.mutableBytes);
                for (NSUInteger column = 0; column < columns; ++column)
                    for (NSUInteger index = 0; index < reduction; ++index)
                        destination[column * reduction + index] =
                            source[index * columns + column];
                rowMajor = transposed;
            }
            program = ane::h13::encodeMatmulParity(parityShape,
                static_cast<const std::uint8_t *>(rowMajor.bytes),
                rowMajor.length);
            *inputsOut = @[x];
            *constantInputOut = nil;
            *constantDataOut = nil;
            *manifestOperationOut = name;
            return YES;
        }
        if (transposeX && rows > 1)
            return reject(diagnostics,
                @"H13 transpose_x=true matmul supports exactly one logical row",
                operation, @"h13.transpose-x-multirow");
        NSUInteger reductionStart = elementOffset % reduction;
        NSUInteger chunkReduction = inputElementCount;
        if (!chunkReduction || chunkReduction > 512 ||
            reductionStart > reduction - chunkReduction)
            return reject(diagnostics, @"H13 matmul reduction slice is invalid",
                operation, @"h13.invalid-slice");
        NSUInteger physicalReduction = chunkReduction <= 256 ? 256 : 512;
        NSUInteger outputColumn = outputElementOffset % columns;
        NSUInteger outputElements = MIN((NSUInteger)512, columns - outputColumn);
        NSMutableData *paddedWeights =
            [NSMutableData dataWithLength:512 * physicalReduction * 2];
        const uint8_t *source = static_cast<const uint8_t *>(weights.bytes);
        uint8_t *destination = static_cast<uint8_t *>(paddedWeights.mutableBytes);
        if (transposeY) {
            for (NSUInteger row = 0; row < outputElements; ++row)
                std::memcpy(destination + row * physicalReduction * 2,
                    source + ((outputColumn + row) * reduction + reductionStart) * 2,
                    chunkReduction * 2);
        } else {
            for (NSUInteger row = 0; row < chunkReduction; ++row)
                std::memcpy(destination + row * 512 * 2,
                    source + ((reductionStart + row) * columns + outputColumn) * 2,
                    outputElements * 2);
        }
        program = ane::h13::encodeMatvec(
            static_cast<std::uint32_t>(physicalReduction),
            static_cast<const std::uint8_t *>(paddedWeights.bytes),
            paddedWeights.length, transposeY);
        inputs = @[x];
    } else {
        ane::h13::NormOperation normOperation{};
        if (normEncoding(name, &normOperation))
            return reject(diagnostics, [NSString stringWithFormat:
                @"H13 '%@' needs a decoded geometry: fp16 static shapes, "
                 "constant axes, no gamma or beta, epsilon 1e-5, and an input "
                 "and output surface inside the oracle parity envelope", name],
                operation, @"h13.norm-outside-envelope");
        if ([name isEqualToString:@"conv"])
            return reject(diagnostics,
                @"H13 conv needs a decoded geometry: an fp16 rank-4 input and "
                 "result with a batch of one, a constant [Cout, Cin/groups, k, k] "
                 "weight, unit dilations, zero explicit padding, pad_type 'same' "
                 "or 'valid', and a kernel, stride, group count and surface pair "
                 "inside the oracle parity envelope",
                operation, @"h13.conv-outside-envelope");
        return reject(diagnostics, [NSString stringWithFormat:
            @"H13 has no source-qualified encoder for '%@'", name], operation);
    }

    *inputsOut = inputs;
    *constantInputOut = constantInput;
    *constantDataOut = constantData;
    *manifestOperationOut = manifestOperation;
    return YES;
}

@implementation ANEH13Compiler
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
              schedule:(NSString *)schedule
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    BOOL hwx = [format isEqualToString:@"hwx"];
    if (!hwx && ![format isEqualToString:@"anec"]) {
        reject(diagnostics, @"H13 artifact format must be 'anec' or 'hwx'",
               nil, @"h13.unsupported-format");
        if (error) *error = [NSError errorWithDomain:@"dev.maderix.H13" code:2
            userInfo:@{NSLocalizedDescriptionKey:
                @"H13 artifact format must be 'anec' or 'hwx'"}];
        return NO;
    }
    MILLexer *lexer = [[MILLexer alloc] initWithData:milData
                                         diagnostics:diagnostics];
    if (![schedule isEqualToString:@"per-op"] &&
        ![schedule isEqualToString:@"chain"])
        return reject(diagnostics,
            @"H13 schedule must be 'per-op' or 'chain'", nil,
            @"h13.unsupported-schedule");
    NSArray<MILToken *> *tokens = lexer.lexAllTokens;
    if (diagnostics.errorCount) return NO;
    MILParser *parser = [[MILParser alloc] initWithTokens:tokens
        diagnostics:diagnostics];
    MILProgramSyntax *syntax = parser.parseProgram;
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || diagnostics.errorCount ||
        ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return NO;
    if (module.functions.count != 1)
        return reject(diagnostics, @"H13 requires exactly one function");
    ANEGraphFunction *function = module.functions[0];
    NSMutableArray<ANEGraphOperation *> *sourceOperations = [NSMutableArray array];
    for (ANEGraphOperation *candidate in function.operations)
        if (![candidate.operationName isEqualToString:@"const"])
            [sourceOperations addObject:candidate];
    if (!sourceOperations.count)
        return reject(diagnostics, @"H13 requires at least one operation");

    BOOL chain = sourceOperations.count > 1;
    NSString *chainCode = @"h13.unsupported-chain";
    ANEGraphOperation *lastSourceOperation = sourceOperations.lastObject;
    if (function.returnValues.count != 1 ||
        function.returnValues[0] != lastSourceOperation.result)
        return reject(diagnostics,
            chain ? @"H13 chains must return only the last operation result"
                  : @"H13 requires one operation with its result returned",
            lastSourceOperation, chain ? chainCode : @"h13.unsupported-program");

    for (ANEGraphValue *input in function.inputs) {
        BOOL used = NO;
        for (ANEGraphOperation *candidate in sourceOperations)
            for (ANEGraphArgument *operand in candidate.operands.allValues)
                if (operand.value == input) used = YES;
        if (!used)
            return reject(diagnostics, @"H13 function inputs must all be used",
                sourceOperations[0], chain ? chainCode : @"h13.unsupported-program");
    }
    for (NSUInteger index = 0; index + 1 < sourceOperations.count; ++index) {
        ANEGraphValue *value = sourceOperations[index].result;
        BOOL used = NO;
        for (NSUInteger consumer = index + 1;
             consumer < sourceOperations.count; ++consumer)
            for (ANEGraphArgument *operand in sourceOperations[consumer].operands.allValues)
                if (operand.value == value) used = YES;
        if (!used)
            return reject(diagnostics,
                @"H13 operation results not returned must be consumed by a later operation",
                sourceOperations[index], chainCode);
    }

    BOOL chainSchedule = [schedule isEqualToString:@"chain"];
    if (chainSchedule) {
        if (sourceOperations.count < 2 || function.inputs.count > 2)
            return reject(diagnostics,
                @"H13 composed scheduling needs at least two operations and at most two boundary inputs",
                sourceOperations[0], @"h13.chain-outside-envelope");
        ANEGraphOperation *first = sourceOperations[0];
        if (sourceOperations.count != 2)
            return reject(diagnostics,
                @"h13.chain-unrepresentable-edge: composed scheduling supports exactly one producer followed by relu",
                sourceOperations[0], @"h13.chain-unrepresentable-edge");
        if (![lastSourceOperation.operationName isEqualToString:@"relu"])
            return reject(diagnostics,
                @"h13.chain-unrepresentable-edge: only a final relu has a decoded single-kernel fusion",
                lastSourceOperation, @"h13.chain-unrepresentable-edge");
        for (ANEGraphValue *input in function.inputs) {
            BOOL firstUse = NO;
            for (ANEGraphArgument *operand in first.operands.allValues)
                firstUse = firstUse || operand.value == input;
            if (!firstUse)
                return reject(diagnostics,
                    @"H13 composed scheduling needs every boundary input resident in the first operation",
                    first, @"h13.chain-outside-envelope");
        }
        for (NSUInteger index = 0; index + 1 < sourceOperations.count; ++index) {
            ANEGraphValue *value = sourceOperations[index].result;
            BOOL nextUse = NO, laterUse = NO;
            for (ANEGraphArgument *operand in
                    sourceOperations[index + 1].operands.allValues)
                nextUse = nextUse || operand.value == value;
            for (NSUInteger later = index + 2; later < sourceOperations.count;
                 ++later)
                for (ANEGraphArgument *operand in
                        sourceOperations[later].operands.allValues)
                    laterUse = laterUse || operand.value == value;
            if (!nextUse || laterUse)
                return reject(diagnostics,
                    @"H13 composed scheduling needs a straight-line chain with one live intermediate",
                    sourceOperations[index], @"h13.chain-outside-envelope");
        }
    }

    NSMutableArray<ANEGraphOperation *> *operations = [NSMutableArray array];
    NSMutableArray<ANEGraphValue *> *manifestValues = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *aliases =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *synthesizedConstants =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *resolvedConstants =
        [NSMutableDictionary dictionary];
    NSMapTable<ANEGraphOperation *, NSNumber *> *reductionOffsets =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMapTable<ANEGraphOperation *, NSNumber *> *reductionCounts =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMapTable<ANEGraphOperation *, NSNumber *> *outputBaseOffsets =
        [NSMapTable strongToStrongObjectsMapTable];
    NSMutableSet<NSString *> *chunkedAccumulations = [NSMutableSet set];
    NSMapTable<ANEGraphValue *, ANEGraphValue *> *loweredValues =
        [NSMapTable strongToStrongObjectsMapTable];
    for (ANEGraphOperation *candidate in sourceOperations) {
        NSMutableDictionary<NSString *, ANEGraphArgument *> *arguments =
            [candidate.arguments mutableCopy];
        for (NSString *key in candidate.operands) {
            ANEGraphValue *value = candidate.operands[key].value;
            ANEGraphValue *lowered = [loweredValues objectForKey:value];
            if (lowered) arguments[key] = valueArgument(lowered, candidate.range);
        }
        NSString *name = candidate.operationName;
        ANEGraphValue *x = arguments[@"x"].value;
        BOOL reshape = [name isEqualToString:@"reshape"];
        BOOL squeeze = [name isEqualToString:@"squeeze"];
        BOOL expand = [name isEqualToString:@"expand_dims"];
        if (reshape || squeeze || expand) {
            NSString *parameterName = reshape ? @"shape" : @"axes";
            ANEGraphArgument *parameter = arguments[parameterName];
            BOOL validCount = squeeze ? (candidate.arguments.count == 1 ||
                                         candidate.arguments.count == 2)
                                      : candidate.arguments.count == 2;
            NSUInteger inputElements = 0, resultElements = 0;
            if (!validCount || !x || (!squeeze && !parameter) ||
                (parameter && (parameter.kind != ANEGraphArgumentKindValue ||
                               !constantValue(parameter.value))) ||
                !tensorElementCount(x, &inputElements) ||
                !tensorElementCount(candidate.result, &resultElements) ||
                inputElements != resultElements)
                return reject(diagnostics,
                    @"H13 shape aliases require static fp16 input and result shapes with equal element counts and constant shape parameters",
                    candidate, @"h13.invalid-shape-alias");
            ANEGraphValue *alias = [[ANEGraphValue alloc]
                initWithName:x.name type:candidate.result.type];
            aliases[candidate.result.name] = @{
                @"aliasOf": x.name, @"shape": candidate.result.type.shape};
            [manifestValues addObject:candidate.result];
            [loweredValues setObject:alias forKey:candidate.result];
            continue;
        }

        ANEGraphValue *result = [[ANEGraphValue alloc]
            initWithName:candidate.result.name type:candidate.result.type];
        if ([name isEqualToString:@"relu"]) {
            if (candidate.arguments.count != 1 || !x)
                return reject(diagnostics, @"H13 relu requires one x value operand",
                    candidate);
            ane::h13::ElementwiseShape shapes[2];
            NSUInteger shapeCount = parityShapes(result, shapes);
            BOOL nativeRelu = NO;
            for (NSUInteger index = 0; index < shapeCount; ++index)
                nativeRelu = nativeRelu ||
                    ane::h13::supportsElementwise(ane::h13::UnaryOperation::Relu,
                                                  shapes[index]);
            if (nativeRelu) {
                [operations addObject:[[ANEGraphOperation alloc]
                    initWithOperationName:name result:result arguments:arguments
                    attributes:candidate.attributes range:candidate.range]];
            } else {
                NSString *zeroName = [NSString stringWithFormat:@"$h13.%@.zero",
                    result.name];
                ANEGraphValue *zero = [[ANEGraphValue alloc]
                    initWithName:zeroName type:x.type];
                synthesizedConstants[zeroName] = splatFP16(0);
                [operations addObject:binaryOperation(@"maximum", x, zero,
                    result, candidate.range)];
            }
            [manifestValues addObject:result];
        } else if ([name isEqualToString:@"clip"]) {
            if (candidate.arguments.count != 3 || !x)
                return reject(diagnostics,
                    @"H13 clip requires x, alpha, and beta arguments", candidate);
            uint16_t alphaBits = 0, betaBits = 0;
            double alpha = 0.0, beta = 0.0;
            if (!exactFP16Attribute(arguments[@"alpha"], &alphaBits, &alpha) ||
                !exactFP16Attribute(arguments[@"beta"], &betaBits, &beta))
                return reject(diagnostics,
                    @"H13 clip alpha and beta must be finite fp32 scalars exactly representable in fp16",
                    candidate, @"h13.inexact-constant");
            if (alpha > beta)
                return reject(diagnostics,
                    @"H13 clip requires alpha less than or equal to beta",
                    candidate, @"h13.invalid-clip-range");
            NSString *prefix = [NSString stringWithFormat:@"$h13.%@", result.name];
            NSString *alphaName = [prefix stringByAppendingString:@".alpha"];
            NSString *betaName = [prefix stringByAppendingString:@".beta"];
            NSString *lowName = [prefix stringByAppendingString:@".clipped-low"];
            ANEGraphValue *alphaValue = [[ANEGraphValue alloc]
                initWithName:alphaName type:x.type];
            ANEGraphValue *low = [[ANEGraphValue alloc]
                initWithName:lowName type:result.type];
            ANEGraphValue *betaValue = [[ANEGraphValue alloc]
                initWithName:betaName type:result.type];
            synthesizedConstants[alphaName] = splatFP16(alphaBits);
            synthesizedConstants[betaName] = splatFP16(betaBits);
            [operations addObject:binaryOperation(@"maximum", x, alphaValue,
                low, candidate.range)];
            [operations addObject:binaryOperation(@"minimum", low, betaValue,
                result, candidate.range)];
            [manifestValues addObject:low];
            [manifestValues addObject:result];
        } else if ([name isEqualToString:@"matmul"] ||
                   [name isEqualToString:@"linear"]) {
            BOOL linear = [name isEqualToString:@"linear"];
            ANEGraphValue *weight = linear ? arguments[@"weight"].value
                                           : arguments[@"y"].value;
            ANEGraphValue *bias = linear ? arguments[@"bias"].value : nil;
            if (linear && (!weight || !constantValue(weight)))
                return reject(diagnostics,
                    @"H13 linear requires a constant weight tensor", candidate,
                    @"h13.linear-nonconstant-weight");
            if (linear && bias && !constantValue(bias))
                return reject(diagnostics,
                    @"H13 linear requires a constant bias tensor", candidate,
                    @"h13.linear-nonconstant-bias");

            NSUInteger reduction = 0, rows = 0, columns = 0;
            BOOL transposeX = linear ? NO : boolean(arguments[@"transpose_x"], YES);
            BOOL geometry = matmulGeometry(x, result, transposeX,
                                           &reduction, &rows, &columns);
            if (linear && (candidate.arguments.count < 2 ||
                           candidate.arguments.count > 3 || !geometry ||
                           !tensor(weight, @[@(columns), @(reduction)])))
                return reject(diagnostics,
                    @"H13 linear requires positive fp16 x rows, constant [N,K] weight, optional constant [N] bias, and a matching output shape",
                    candidate, @"h13.invalid-linear");

            if (linear && bias && rows > 1) {
                NSData *biasData = linearBiasData(bias, columns, modelRoot,
                    diagnostics, resolvedConstants);
                if (!biasData) return NO;
                ANEValueType *rowInputType = [[ANEValueType alloc]
                    initWithKind:ANEValueTypeKindTensor
                    elementType:ANEElementTypeFP16 shape:@[@1, @(reduction)]];
                ANEValueType *rowOutputType = [[ANEValueType alloc]
                    initWithKind:ANEValueTypeKindTensor
                    elementType:ANEElementTypeFP16 shape:@[@1, @(columns)]];
                for (NSUInteger row = 0; row < rows; ++row) {
                    ANEGraphValue *rowInput = [[ANEGraphValue alloc]
                        initWithName:x.name type:rowInputType];
                    NSDictionary *matmulArguments = @{
                        @"x": valueArgument(rowInput, candidate.range),
                        @"y": valueArgument(weight, candidate.range),
                        @"transpose_x": booleanArgument(NO, candidate.range),
                        @"transpose_y": booleanArgument(YES, candidate.range),
                    };
                    NSString *prefix = [NSString stringWithFormat:@"%@.row%lu",
                        result.name, (unsigned long)row];
                    NSString *matrixName = [NSString stringWithFormat:@"$h13.%@.linear",
                        prefix];
                    ANEGraphValue *matrix = [[ANEGraphValue alloc]
                        initWithName:matrixName type:rowOutputType];
                    if (reduction > 512 &&
                        !matmulParityCovered(1, reduction, columns, NO, YES, NO, !chainSchedule)) {
                        NSUInteger chunks = (reduction - 1) / 512 + 1;
                        NSMutableArray<ANEGraphValue *> *partials =
                            [NSMutableArray arrayWithCapacity:chunks];
                        for (NSUInteger chunk = 0; chunk < chunks; ++chunk) {
                            NSString *partialName = [NSString stringWithFormat:
                                @"$h13.%@.partial%lu", prefix,
                                (unsigned long)chunk];
                            ANEGraphValue *partial = [[ANEGraphValue alloc]
                                initWithName:partialName type:rowOutputType];
                            ANEGraphOperation *partialOperation = [[ANEGraphOperation alloc]
                                initWithOperationName:@"matmul" result:partial
                                arguments:matmulArguments attributes:@{}
                                range:candidate.range];
                            [reductionOffsets setObject:@(row * reduction + chunk * 512)
                                                 forKey:partialOperation];
                            [reductionCounts setObject:@(MIN((NSUInteger)512,
                                reduction - chunk * 512)) forKey:partialOperation];
                            [operations addObject:partialOperation];
                            [manifestValues addObject:partial];
                            [partials addObject:partial];
                        }
                        ANEGraphValue *accumulator = partials[0];
                        for (NSUInteger chunk = 1; chunk < chunks; ++chunk) {
                            BOOL final = chunk + 1 == chunks;
                            NSString *sumName = [NSString stringWithFormat:
                                @"$h13.%@.accum%lu", prefix,
                                (unsigned long)chunk];
                            ANEGraphValue *sum = final ? matrix : [[ANEGraphValue alloc]
                                initWithName:sumName type:rowOutputType];
                            [operations addObject:binaryOperation(@"add", accumulator,
                                partials[chunk], sum, candidate.range)];
                            [manifestValues addObject:sum];
                            accumulator = sum;
                        }
                        [chunkedAccumulations addObject:matrix.name];
                    } else {
                        ANEGraphOperation *matmulOperation = [[ANEGraphOperation alloc]
                            initWithOperationName:@"matmul" result:matrix
                            arguments:matmulArguments attributes:@{}
                            range:candidate.range];
                        [reductionOffsets setObject:@(row * reduction)
                                             forKey:matmulOperation];
                        [operations addObject:matmulOperation];
                        [manifestValues addObject:matrix];
                    }
                    NSString *biasName = [NSString stringWithFormat:@"$h13.%@.bias",
                        prefix];
                    ANEGraphValue *expandedBias = [[ANEGraphValue alloc]
                        initWithName:biasName type:rowOutputType];
                    synthesizedConstants[biasName] = biasData;
                    ANEGraphValue *rowOutput = [[ANEGraphValue alloc]
                        initWithName:result.name type:rowOutputType];
                    ANEGraphOperation *add = binaryOperation(@"add", matrix,
                        expandedBias, rowOutput, candidate.range);
                    [outputBaseOffsets setObject:@(row * columns) forKey:add];
                    [operations addObject:add];
                }
                [manifestValues addObject:result];
                [loweredValues setObject:result forKey:candidate.result];
                continue;
            }

            ANEGraphValue *matmulResult = result;
            if (linear && bias) {
                NSString *name = [NSString stringWithFormat:@"$h13.%@.linear",
                    result.name];
                matmulResult = [[ANEGraphValue alloc]
                    initWithName:name type:result.type];
            }
            NSDictionary<NSString *, ANEGraphArgument *> *matmulArguments = arguments;
            if (linear)
                matmulArguments = @{
                    @"x": valueArgument(x, candidate.range),
                    @"y": valueArgument(weight, candidate.range),
                    @"transpose_x": booleanArgument(NO, candidate.range),
                    @"transpose_y": booleanArgument(YES, candidate.range),
                };
            const BOOL runtimeWeight = !linear && weight &&
                !constantValue(weight) && !synthesizedConstants[weight.name];
            if (geometry && !runtimeWeight && reduction > 512 &&
                !matmulParityCovered(rows, reduction, columns, transposeX, YES,
                                     NO, !chainSchedule)) {
                NSUInteger chunks = (reduction - 1) / 512 + 1;
                NSMutableArray<ANEGraphValue *> *partials =
                    [NSMutableArray arrayWithCapacity:chunks];
                for (NSUInteger chunk = 0; chunk < chunks; ++chunk) {
                    NSString *partialName = [NSString stringWithFormat:
                        @"$h13.%@.partial%lu", result.name, (unsigned long)chunk];
                    ANEGraphValue *partial = [[ANEGraphValue alloc]
                        initWithName:partialName type:matmulResult.type];
                    ANEGraphOperation *partialOperation = [[ANEGraphOperation alloc]
                        initWithOperationName:@"matmul" result:partial
                        arguments:matmulArguments attributes:@{} range:candidate.range];
                    [reductionOffsets setObject:@(chunk * 512)
                                         forKey:partialOperation];
                    [reductionCounts setObject:@(MIN((NSUInteger)512,
                        reduction - chunk * 512)) forKey:partialOperation];
                    [operations addObject:partialOperation];
                    [manifestValues addObject:partial];
                    [partials addObject:partial];
                }
                ANEGraphValue *accumulator = partials[0];
                for (NSUInteger chunk = 1; chunk < chunks; ++chunk) {
                    BOOL final = chunk + 1 == chunks;
                    NSString *sumName = [NSString stringWithFormat:
                        @"$h13.%@.accum%lu", result.name, (unsigned long)chunk];
                    ANEGraphValue *sum = final ? matmulResult : [[ANEGraphValue alloc]
                        initWithName:sumName type:matmulResult.type];
                    [operations addObject:binaryOperation(@"add", accumulator,
                        partials[chunk], sum, candidate.range)];
                    [manifestValues addObject:sum];
                    accumulator = sum;
                }
                [chunkedAccumulations addObject:matmulResult.name];
            } else {
                ANEGraphOperation *matmulOperation = [[ANEGraphOperation alloc]
                    initWithOperationName:@"matmul" result:matmulResult
                    arguments:matmulArguments attributes:@{} range:candidate.range];
                [operations addObject:matmulOperation];
                [manifestValues addObject:matmulResult];
            }

            if (linear && bias) {
                NSData *biasData = linearBiasData(bias, columns, modelRoot,
                    diagnostics, resolvedConstants);
                if (!biasData) return NO;
                if (rows * columns > NSUIntegerMax / 2)
                    return reject(diagnostics, @"H13 linear output size overflows",
                        candidate, @"h13.invalid-linear-bias");
                NSMutableData *expanded =
                    [NSMutableData dataWithLength:rows * columns * 2];
                for (NSUInteger row = 0; row < rows; ++row)
                    std::memcpy(static_cast<uint8_t *>(expanded.mutableBytes) +
                        row * columns * 2, biasData.bytes, columns * 2);
                NSString *biasName = [NSString stringWithFormat:@"$h13.%@.bias",
                    result.name];
                ANEGraphValue *expandedBias = [[ANEGraphValue alloc]
                    initWithName:biasName type:result.type];
                synthesizedConstants[biasName] = expanded;
                [operations addObject:binaryOperation(@"add", matmulResult,
                    expandedBias, result, candidate.range)];
                [manifestValues addObject:result];
            }
        } else {
            [operations addObject:[[ANEGraphOperation alloc]
                initWithOperationName:name result:result arguments:arguments
                attributes:candidate.attributes range:candidate.range]];
            [manifestValues addObject:result];
        }
        [loweredValues setObject:result forKey:candidate.result];
    }

    ANEGraphValue *returned = [loweredValues objectForKey:function.returnValues[0]];
    NSMutableSet<NSString *> *inputNames = [NSMutableSet set];
    for (ANEGraphValue *input in function.inputs) [inputNames addObject:input.name];
    if (!operations.count) {
        if (returned && [inputNames containsObject:returned.name])
            return reject(diagnostics,
                @"H13 cannot return an alias of a function input because no program produces it",
                lastSourceOperation, @"h13.returned-input-alias");
        return reject(diagnostics, @"H13 requires at least one encoded operation");
    }
    ANEGraphOperation *lastOperation = operations.lastObject;
    if (!returned || ![returned.name isEqualToString:lastOperation.result.name])
        return reject(diagnostics,
            @"H13 returned aliases must refer to the last encoded operation result",
            lastSourceOperation, chainCode);

    NSString *returnedSourceName = function.returnValues[0].name;
    NSString *returnedStorageName = returned.name;
    NSMutableSet<NSString *> *intermediateStorageNames = [NSMutableSet set];
    NSMutableArray<NSString *> *intermediateNames = [NSMutableArray array];
    for (ANEGraphValue *value in manifestValues) {
        BOOL alias = aliases[value.name] != nil;
        if ([value.name isEqualToString:returnedSourceName] ||
            (!alias && [value.name isEqualToString:returnedStorageName]))
            continue;
        [intermediateNames addObject:value.name];
        if (!alias) [intermediateStorageNames addObject:value.name];
    }
    NSDictionary<NSString *, NSArray<NSNumber *> *> *outputShapes =
        @{returnedStorageName: function.returnValues[0].type.shape};

    NSMutableArray<NSDictionary *> *programRecords = [NSMutableArray array];
    NSMutableArray<NSData *> *payloads = [NSMutableArray array];
    std::vector<ane::h13::Program> chainPrograms;
    NSMutableArray<NSNumber *> *dispatchPlan = [NSMutableArray array];
    NSMutableArray<NSDictionary *> *taskRecords = nil;
    NSDictionary *scratch = nil;
    NSMutableDictionary<NSString *, NSDictionary *> *tensors =
        [NSMutableDictionary dictionary];
    for (ANEGraphValue *input in function.inputs)
        recordTensor(tensors, input, input.type.shape, @"input");
    NSArray<NSString *> *binaryNames =
        @[@"add", @"mul", @"maximum", @"minimum", @"sub", @"real_div"];
    ANEGraphOperation *operation = nil;
    try {
        for (operation in operations) {
            NSUInteger sliceCount = 1, inputSliceElements = 0,
                inputPhysicalElements = 0, outputSliceElements = 0,
                outputPhysicalElements = 0, outputChunks = 1;
            BOOL matmul = [operation.operationName isEqualToString:@"matmul"];
            ANEGraphValue *secondOperand = operation.operands[@"y"].value;
            BOOL runtimeWeight = matmul && secondOperand &&
                !constantValue(secondOperand) &&
                !synthesizedConstants[secondOperand.name];
            BOOL matvecParity = matmul &&
                matmulParityShape(operation.operands[@"x"].value, operation.result,
                    boolean(operation.arguments[@"transpose_x"], YES),
                    runtimeWeight
                        ? boolean(operation.arguments[@"transpose_y"], YES) : YES,
                    runtimeWeight, !chainSchedule, nullptr);
            H13ParityPlan operationPlan{};
            H13NormPlan normalizationPlan{};
            H13BroadcastPlan broadcastPlanned{};
            ANEGraphValue *broadcastConstant = nil;
            BOOL broadcast = broadcastPlan(operation, synthesizedConstants,
                                           !chainSchedule, &broadcastPlanned, &broadcastConstant);
            BOOL parity = !broadcast &&
                parityPlan(operation, synthesizedConstants, !chainSchedule, &operationPlan);
            H13ConvPlan convolutionPlan{};
            BOOL normalization = !parity && !broadcast &&
                normParityPlan(operation, &normalizationPlan);
            BOOL convolution = !parity && !broadcast && !normalization &&
                convParityPlan(operation, &convolutionPlan);
            if (parity) {
                inputSliceElements = outputSliceElements =
                    inputPhysicalElements = outputPhysicalElements =
                        operationPlan.elements;
            } else if (broadcast) {
                // Each operand covers its own whole tensor: a broadcast reads
                // fewer elements from y than it writes to the result.
                NSUInteger elements = 0;
                if (!tensorElementCount(operation.result, &elements))
                    return reject(diagnostics,
                        @"H13 broadcast result must have a positive static shape",
                        operation);
                outputSliceElements = outputPhysicalElements = elements;
                if (broadcastConstant)
                    recordTensor(tensors, broadcastConstant,
                                 broadcastConstant.type.shape, @"constant");
            } else if (normalization) {
                inputSliceElements = inputPhysicalElements =
                    normalizationPlan.inputElements;
                outputSliceElements = outputPhysicalElements =
                    normalizationPlan.outputElements;
            } else if (convolution) {
                NSUInteger elements = 0;
                if (!tensorElementCount(operation.operands[@"x"].value, &elements))
                    return reject(diagnostics,
                        @"H13 convolution input must have a positive static shape",
                        operation);
                inputSliceElements = inputPhysicalElements = elements;
                if (!tensorElementCount(operation.result, &elements))
                    return reject(diagnostics,
                        @"H13 convolution result must have a positive static shape",
                        operation);
                outputSliceElements = outputPhysicalElements = elements;
                recordTensor(tensors, convolutionPlan.weight,
                             convolutionPlan.weight.type.shape, @"constant");
                if (convolutionPlan.bias)
                    recordTensor(tensors, convolutionPlan.bias,
                                 convolutionPlan.bias.type.shape, @"constant");
            } else if ([binaryNames containsObject:operation.operationName]) {
                NSUInteger elements = 0;
                if (tensorElementCount(operation.result, &elements)) {
                    sliceCount = (elements - 1) / 64 + 1;
                    inputPhysicalElements = outputPhysicalElements = 64;
                }
            } else if (matmul) {
                NSUInteger reduction = 0, rows = 0, columns = 0;
                BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
                if (matmulGeometry(operation.operands[@"x"].value, operation.result,
                                   transposeX, &reduction, &rows, &columns)) {
                    if (matvecParity) {
                        inputSliceElements = inputPhysicalElements =
                            rows * reduction;
                        outputSliceElements = outputPhysicalElements =
                            rows * columns;
                    } else {
                        outputChunks = (columns - 1) / 512 + 1;
                        if (rows > NSUIntegerMax / outputChunks)
                            return reject(diagnostics,
                                @"H13 matmul program count overflows", operation);
                        sliceCount = rows * outputChunks;
                        NSNumber *count = [reductionCounts objectForKey:operation];
                        inputSliceElements =
                            count ? count.unsignedIntegerValue : reduction;
                        inputPhysicalElements =
                            inputSliceElements <= 256 ? 256 : 512;
                        outputPhysicalElements = 512;
                    }
                }
                if (fp16Tensor(secondOperand) && !runtimeWeight)
                    recordTensor(tensors, secondOperand,
                                 secondOperand.type.shape, @"constant");
            }

            for (NSUInteger sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
                NSUInteger inputOffset = 0, outputOffset = 0;
                if (matmul) {
                    NSUInteger reduction = 0, rows = 0, geometryColumns = 0;
                    BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
                    matmulGeometry(operation.operands[@"x"].value, operation.result,
                                   transposeX, &reduction, &rows, &geometryColumns);
                    NSUInteger columns = operation.result.type.shape.lastObject.unsignedIntegerValue;
                    NSNumber *reductionOffset = [reductionOffsets objectForKey:operation];
                    if (matvecParity) {
                        inputOffset = reductionOffset.unsignedIntegerValue;
                    } else {
                        NSUInteger row = sliceIndex / outputChunks;
                        NSUInteger chunk = sliceIndex % outputChunks;
                        inputOffset =
                            row * reduction + reductionOffset.unsignedIntegerValue;
                        outputOffset = row * columns + chunk * 512;
                        outputSliceElements =
                            MIN((NSUInteger)512, columns - chunk * 512);
                    }
                } else if (parity || broadcast || normalization || convolution) {
                    outputOffset =
                        [[outputBaseOffsets objectForKey:operation] unsignedIntegerValue];
                } else {
                    inputOffset = sliceIndex * 64;
                    outputOffset = inputOffset +
                        [[outputBaseOffsets objectForKey:operation] unsignedIntegerValue];
                    NSUInteger elements = 0;
                    if (tensorElementCount(operation.result, &elements))
                        inputSliceElements = outputSliceElements =
                            MIN((NSUInteger)64, elements - inputOffset);
                }
                ane::h13::Program program;
                NSArray<ANEGraphValue *> *inputs = nil;
                ANEGraphValue *constantInput = nil;
                NSData *constantData = nil;
                NSString *manifestOperation = nil;
                if (!lowerOperation(operation, modelRoot, diagnostics, !chainSchedule,
                                    synthesizedConstants, resolvedConstants,
                                    inputOffset, inputSliceElements, outputOffset,
                                    program, &inputs, &constantInput, &constantData,
                                    &manifestOperation))
                    return NO;
                std::vector<std::uint8_t> anec = ane::h13::encodeANEC(program);
                NSMutableArray *inputRecords =
                    [NSMutableArray arrayWithCapacity:inputs.count];
                NSMutableDictionary *constantInputs = [NSMutableDictionary dictionary];
                for (NSUInteger index = 0; index < inputs.count; ++index) {
                    ANEGraphValue *input = inputs[index];
                    NSArray<NSNumber *> *fullShape = input == constantInput
                        ? operation.result.type.shape : input.type.shape;
                    BOOL intermediate = input != constantInput &&
                        [intermediateStorageNames containsObject:input.name];
                    NSString *role = input == constantInput ? @"constant"
                        : (intermediate ? @"intermediate" : @"input");
                    recordTensor(tensors, input, fullShape, role);
                    NSUInteger fullElements = 1;
                    for (NSNumber *dimension in fullShape)
                        fullElements *= dimension.unsignedIntegerValue;
                    // A broadcast and a runtime-operand matmul read each
                    // operand whole, and the operands differ in size, so the
                    // program-wide slice counts do not apply to them.
                    const BOOL wholeOperand = broadcast || runtimeWeight;
                    NSUInteger sliceElements =
                        wholeOperand ? fullElements : inputSliceElements;
                    NSUInteger physicalElements =
                        wholeOperand ? fullElements : inputPhysicalElements;
                    NSArray<NSNumber *> *logicalShape =
                        inputOffset == 0 && sliceElements == fullElements
                            ? fullShape : @[@(sliceElements)];
                    NSMutableDictionary *record =
                        [binding(input, logicalShape, program.inputs.at(index)) mutableCopy];
                    BOOL aliasShape =
                        ![fullShape isEqualToArray:tensors[input.name][@"shape"]];
                    if (inputOffset || sliceElements != fullElements ||
                        physicalElements != sliceElements || aliasShape)
                        addSlice(record, input, inputOffset, sliceElements,
                                 physicalElements);
                    if (input == constantInput) {
                        record[@"binding"] = @"constant";
                        constantInputs[input.name] = hexData(constantData);
                    } else if (intermediate) {
                        record[@"role"] = @"intermediate";
                    }
                    [inputRecords addObject:record];
                }
                BOOL intermediateOutput =
                    [intermediateStorageNames containsObject:operation.result.name];
                NSString *outputRole = intermediateOutput ? @"intermediate" : @"output";
                NSArray<NSNumber *> *fullOutputShape = outputShapes[operation.result.name]
                    ?: operation.result.type.shape;
                recordTensor(tensors, operation.result, fullOutputShape, outputRole);
                NSUInteger fullOutputElements = 1;
                for (NSNumber *dimension in fullOutputShape)
                    fullOutputElements *= dimension.unsignedIntegerValue;
                NSArray<NSNumber *> *outputShape =
                    outputOffset == 0 && outputSliceElements == fullOutputElements
                        ? fullOutputShape : @[@(outputSliceElements)];
                NSMutableDictionary *outputRecord =
                    [binding(operation.result, outputShape, program.output) mutableCopy];
                if (outputOffset || outputSliceElements != fullOutputElements ||
                    outputPhysicalElements != outputSliceElements)
                    addSlice(outputRecord, operation.result, outputOffset,
                             outputSliceElements, outputPhysicalElements);
                if (intermediateOutput) outputRecord[@"role"] = @"intermediate";
                NSUInteger programIndex = programRecords.count;
                NSData *payload = nil;
                if (hwx) {
                    payload = encodeHWX(program, error);
                    if (!payload) return NO;
                } else {
                    payload = [NSData dataWithBytes:anec.data() length:anec.size()];
                }
                NSString *file = [NSString stringWithFormat:@"program-%lu.%@",
                    (unsigned long)programIndex, format];
                NSDictionary *record = @{
                    @"file": file, @"bytes": @(payload.length),
                    @"taskDescriptors": @(program.taskCount),
                    @"encoder": matvecParity
                        ? (runtimeWeight ? @"apple-parity-matmul"
                                         : @"apple-parity-matvec")
                        : (broadcast ? @"apple-parity-broadcast"
                        : (normalization ? @"apple-parity-norm"
                        : (convolution ? @"apple-parity-conv"
                        : (parity ? @"h13-oracle-parity" : @"h13-source-qualified")))),
                    @"operation": manifestOperation,
                    @"inputs": inputRecords, @"constantInputs": constantInputs,
                    @"outputs": @[outputRecord],
                    @"constantOffset": @(program.constantOffsetBytes),
                    @"constantBytes": @(program.constants.size()),
                };
                [programRecords addObject:record];
                [dispatchPlan addObject:@(programIndex)];
                [payloads addObject:payload];
                if (chainSchedule)
                    chainPrograms.push_back(std::move(program));
            }
        }
        if (chainSchedule) {
            if (chainPrograms.size() != programRecords.count ||
                chainPrograms.empty())
                throw std::logic_error(
                    "h13.chain-outside-envelope: chain program bookkeeping is inconsistent");
            NSDictionary *firstRecord = programRecords.firstObject;
            NSString *producerResult = sourceOperations[0].result.name;
            NSUInteger producerPrograms = 0;
            for (NSDictionary *record in programRecords)
                for (NSDictionary *item in record[@"outputs"])
                    producerPrograms += [item[@"name"] isEqualToString:producerResult];
            NSArray *firstOutputs = firstRecord[@"outputs"];
            if (producerPrograms != 1 || firstOutputs.count != 1 ||
                ![firstOutputs[0][@"name"] isEqualToString:producerResult])
                throw std::invalid_argument(
                    "h13.chain-outside-envelope: the producer must encode as one whole-tensor program");
            NSArray *boundaryInputs = firstRecord[@"inputs"];
            if (boundaryInputs.count != function.inputs.count)
                throw std::invalid_argument(
                    "h13.chain-outside-envelope: first operation bindings differ from the function inputs");
            NSMutableSet<NSString *> *boundaryNames = [NSMutableSet set];
            for (NSDictionary *item in boundaryInputs) {
                if (item[@"slice"] || item[@"binding"])
                    throw std::invalid_argument(
                        "h13.chain-outside-envelope: boundary inputs must be whole runtime tensors");
                [boundaryNames addObject:item[@"name"]];
            }
            if (![boundaryNames isEqualToSet:inputNames] || firstOutputs[0][@"slice"])
                throw std::invalid_argument(
                    "h13.chain-outside-envelope: producer bindings must be whole boundary tensors");

            const NSString *encoder = firstRecord[@"encoder"];
            if ([encoder isEqualToString:@"apple-parity-matvec"])
                ane::h13::fuseMatmulPostOperation(chainPrograms[0],
                    ane::h13::PostOperation::Relu);
            else if ([encoder isEqualToString:@"h13-oracle-parity"])
                ane::h13::fuseElementwisePostOperation(chainPrograms[0],
                    ane::h13::PostOperation::Relu);
            else
                throw std::invalid_argument(
                    "h13.chain-unrepresentable-edge: this producer encoding carries no decoded post-operation field");
            chainPrograms.resize(1);
            ane::h13::Program combined = ane::h13::composePrograms(chainPrograms);
            NSData *combinedPayload = nil;
            if (hwx) {
                combinedPayload = encodeHWX(combined, error);
                if (!combinedPayload) return NO;
            } else {
                std::vector<std::uint8_t> bytes = ane::h13::encodeANEC(combined);
                combinedPayload = [NSData dataWithBytes:bytes.data()
                                                 length:bytes.size()];
            }
            taskRecords = [NSMutableArray arrayWithCapacity:combined.taskCount];
            NSString *producerOperation = sourceOperations[0].operationName;
            for (NSUInteger index = 0; index < combined.taskCount; ++index)
                [taskRecords addObject:@{@"index": @(index),
                    @"operation": producerOperation}];
            scratch = @{@"bytes": @(combined.scratchAllocationBytes),
                        @"regions": @{}};
            [tensors removeObjectForKey:producerResult];
            [intermediateNames removeObject:producerResult];
            [intermediateStorageNames removeObject:producerResult];
            NSString *finalResult = lastSourceOperation.result.name;
            NSMutableDictionary *fusedOutput = [firstOutputs[0] mutableCopy];
            fusedOutput[@"name"] = finalResult;
            [fusedOutput removeObjectForKey:@"role"];
            [fusedOutput removeObjectForKey:@"slice"];
            NSString *file = [NSString stringWithFormat:@"program-0.%@", format];
            NSDictionary *combinedRecord = @{
                @"file": file, @"bytes": @(combinedPayload.length),
                @"taskDescriptors": @(combined.taskCount),
                @"encoder": @"composed-chain", @"operation": @"chain",
                @"inputs": boundaryInputs, @"constantInputs": @{},
                @"outputs": @[fusedOutput],
                @"constantOffset": @(combined.constantOffsetBytes),
                @"constantBytes": @(combined.constants.size()),
                @"fused": @[finalResult],
            };
            [programRecords removeAllObjects];
            [programRecords addObject:combinedRecord];
            [payloads removeAllObjects];
            [payloads addObject:combinedPayload];
            [dispatchPlan removeAllObjects];
            [dispatchPlan addObject:@0];
        }
    } catch (const std::exception &exception) {
        NSString *message = [NSString stringWithUTF8String:exception.what()];
        NSString *code = chainSchedule ? @"h13.chain-outside-envelope"
                                       : @"h13.unsupported-program";
        NSRange codeEnd = [message rangeOfString:@": "];
        if ([message hasPrefix:@"h13."] && codeEnd.location != NSNotFound &&
            codeEnd.location <= 48)
            code = [message substringToIndex:codeEnd.location];
        return reject(diagnostics, message, operation, code);
    }
    for (NSString *name in aliases) {
        NSDictionary *alias = aliases[name];
        NSArray<NSNumber *> *shape = alias[@"shape"];
        NSUInteger elements = 1;
        for (NSNumber *dimension in shape)
            elements *= dimension.unsignedIntegerValue;
        NSString *role = [name isEqualToString:returnedSourceName]
            ? @"output" : @"intermediate";
        tensors[name] = @{@"shape": shape, @"logicalBytes": @(elements * 2),
                          @"role": role, @"aliasOf": alias[@"aliasOf"]};
    }
    for (NSString *name in intermediateStorageNames) {
        NSMutableArray<NSArray<NSNumber *> *> *produced = [NSMutableArray array];
        NSMutableArray<NSArray<NSNumber *> *> *consumed = [NSMutableArray array];
        for (NSDictionary *record in programRecords) {
            for (NSString *direction in @[@"outputs", @"inputs"]) {
                for (NSDictionary *item in record[direction]) {
                    if (![item[@"name"] isEqualToString:name]) continue;
                    NSDictionary *slice = item[@"slice"];
                    NSUInteger offset = [slice[@"elementOffset"] unsignedIntegerValue];
                    NSUInteger count = slice
                        ? [slice[@"elementCount"] unsignedIntegerValue]
                        : [item[@"logicalBytes"] unsignedIntegerValue] / 2;
                    NSUInteger physical = slice[@"physicalElements"]
                        ? [slice[@"physicalElements"] unsignedIntegerValue] : count;
                    [([direction isEqualToString:@"outputs"] ? produced : consumed)
                        addObject:@[@(offset), @(offset + physical)]];
                }
            }
        }
        [produced sortUsingComparator:^NSComparisonResult(
            NSArray<NSNumber *> *left, NSArray<NSNumber *> *right) {
            return [left[0] compare:right[0]];
        }];
        NSUInteger previousEnd = 0;
        for (NSArray<NSNumber *> *range in produced) {
            NSUInteger start = [range[0] unsignedIntegerValue];
            if (start < previousEnd)
                return reject(diagnostics,
                    @"H13 intermediate physical writes must not overlap",
                    lastOperation, chainCode);
            previousEnd = [range[1] unsignedIntegerValue];
        }
        for (NSArray<NSNumber *> *consumer in consumed) {
            NSUInteger cursor = [consumer[0] unsignedIntegerValue];
            NSUInteger end = [consumer[1] unsignedIntegerValue];
            for (NSArray<NSNumber *> *producer in produced) {
                NSUInteger producerStart = [producer[0] unsignedIntegerValue];
                NSUInteger producerEnd = [producer[1] unsignedIntegerValue];
                if (producerEnd <= cursor) continue;
                if (producerStart > cursor) break;
                cursor = MAX(cursor, producerEnd);
                if (cursor >= end) break;
            }
            if (cursor < end)
                return reject(diagnostics,
                    @"H13 intermediate consumer physical range exceeds producer writes",
                    lastOperation, chainCode);
        }
    }

    for (NSString *name in chunkedAccumulations) {
        NSMutableDictionary *record = [tensors[name] mutableCopy];
        if (record) {
            record[@"accumulation"] = @"chunked-fp16";
            tensors[name] = record;
        }
    }

    NSMutableDictionary *manifest = [@{
        @"schema": @"mil-hwxc.h13-anec-package.v1",
        @"target": @"H13", @"artifactFormat": format,
        @"programs": programRecords, @"dispatchPlan": dispatchPlan,
        @"intermediates": intermediateNames, @"tensors": tensors,
    } mutableCopy];
    if (chainSchedule)
        [manifest addEntriesFromDictionary:@{@"schedule": @"chain",
            @"tasks": taskRecords, @"scratch": scratch}];
    if (programRecords.count == 1)
        [manifest addEntriesFromDictionary:programRecords[0]];
    NSData *metadata = [NSJSONSerialization dataWithJSONObject:manifest
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:error];
    if (!metadata) return NO;
    NSFileManager *manager = NSFileManager.defaultManager;
    if ([manager fileExistsAtPath:directory.path]) {
        NSArray *existing = [manager contentsOfDirectoryAtPath:directory.path error:error];
        if (!existing) return NO;
        if (existing.count != 0) {
            if (error) *error = [NSError errorWithDomain:@"dev.maderix.H13"
                code:1 userInfo:@{NSLocalizedDescriptionKey:
                    @"H13 output directory must be empty"}];
            return NO;
        }
    } else if (![manager createDirectoryAtURL:directory withIntermediateDirectories:YES
        attributes:nil error:error]) return NO;
    for (NSUInteger index = 0; index < payloads.count; ++index)
        if (![payloads[index] writeToURL:
                [directory URLByAppendingPathComponent:programRecords[index][@"file"]]
                options:NSDataWritingAtomic error:error])
            return NO;
    return [metadata writeToURL:[directory URLByAppendingPathComponent:@"manifest.json"]
        options:NSDataWritingAtomic error:error];
}
@end
