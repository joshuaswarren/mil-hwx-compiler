#import "ANEH14Compiler.h"

#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"
#import "HWXObjectWriter.h"
#include "H14Program.h"

#include <cmath>
#include <cstdlib>
#include <exception>

static BOOL reject(ANEDiagnosticEngine *diagnostics, NSString *message,
                   ANEGraphOperation *operation = nil,
                   NSString *code = @"h14.unsupported-program") {
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
                             ane::h14::ElementwiseShape *shape) {
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

static BOOL binaryEncoding(NSString *name, ane::h14::BinaryOperation *encoding) {
    if ([name isEqualToString:@"add"]) *encoding = ane::h14::BinaryOperation::Add;
    else if ([name isEqualToString:@"mul"]) *encoding = ane::h14::BinaryOperation::Multiply;
    else if ([name isEqualToString:@"maximum"]) *encoding = ane::h14::BinaryOperation::Maximum;
    else if ([name isEqualToString:@"minimum"]) *encoding = ane::h14::BinaryOperation::Minimum;
    else if ([name isEqualToString:@"sub"]) *encoding = ane::h14::BinaryOperation::Subtract;
    else if ([name isEqualToString:@"real_div"]) *encoding = ane::h14::BinaryOperation::RealDivide;
    else return NO;
    return YES;
}

static BOOL unaryEncoding(NSString *name, ane::h14::UnaryOperation *encoding) {
    if ([name isEqualToString:@"abs"]) *encoding = ane::h14::UnaryOperation::Absolute;
    else if ([name isEqualToString:@"exp"]) *encoding = ane::h14::UnaryOperation::Exponential;
    else if ([name isEqualToString:@"gelu"]) *encoding = ane::h14::UnaryOperation::Gelu;
    else if ([name isEqualToString:@"leaky_relu"]) *encoding = ane::h14::UnaryOperation::LeakyRelu;
    else if ([name isEqualToString:@"relu"]) *encoding = ane::h14::UnaryOperation::Relu;
    else if ([name isEqualToString:@"rsqrt"]) *encoding = ane::h14::UnaryOperation::ReciprocalSquareRoot;
    else if ([name isEqualToString:@"sigmoid"]) *encoding = ane::h14::UnaryOperation::Sigmoid;
    else if ([name isEqualToString:@"silu"]) *encoding = ane::h14::UnaryOperation::Silu;
    else if ([name isEqualToString:@"sqrt"]) *encoding = ane::h14::UnaryOperation::SquareRoot;
    else if ([name isEqualToString:@"tanh"]) *encoding = ane::h14::UnaryOperation::Tanh;
    else return NO;
    return YES;
}

static BOOL normEncoding(NSString *name, ane::h14::NormOperation *encoding) {
    if ([name isEqualToString:@"softmax"]) *encoding = ane::h14::NormOperation::Softmax;
    else if ([name isEqualToString:@"layer_norm"]) *encoding = ane::h14::NormOperation::LayerNorm;
    else if ([name isEqualToString:@"reduce_sum"]) *encoding = ane::h14::NormOperation::ReduceSum;
    else if ([name isEqualToString:@"reduce_max"]) *encoding = ane::h14::NormOperation::ReduceMax;
    else if ([name isEqualToString:@"reduce_mean"]) *encoding = ane::h14::NormOperation::ReduceMean;
    else return NO;
    return YES;
}

/// The CHW surface H14 lays a logical MIL shape out as: leading unit
/// dimensions collapse into the batch, and a rank below three pads on the
/// left. Every decoded normalization and reduction surface follows this,
/// including the rank-reduced `keep_dims = false` results.
static BOOL normSurface(NSArray<NSNumber *> *shape,
                        ane::h14::ElementwiseShape *surface,
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

static BOOL floatingText(ANEGraphArgument *argument, double *value) {
    if (argument.kind != ANEGraphArgumentKindFloatingPoint &&
        argument.kind != ANEGraphArgumentKindInteger) return NO;
    const char *text = argument.text.UTF8String;
    if (!text) return NO;
    char *end = nullptr;
    *value = std::strtod(text, &end);
    return end != text && !*end && std::isfinite(*value);
}

static BOOL fp16Scalar(ANEGraphArgument *argument, uint16_t *bits) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"fp16"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    double value = 0.0;
    if (!floatingText(argument, &value)) return NO;
    *bits = fp16Bits(value);
    return (*bits & 0x7c00u) != 0x7c00u;
}

static BOOL fp32Attribute(ANEGraphArgument *argument, float *valueOut) {
    if (argument.kind != ANEGraphArgumentKindCall ||
        ![argument.calleeName isEqualToString:@"fp32"] ||
        argument.callArguments.count != 1) return NO;
    double parsed = 0.0;
    if (!floatingText(argument.callArguments[0].value, &parsed)) return NO;
    *valueOut = static_cast<float>(parsed);
    return YES;
}

static NSString *stringArgument(ANEGraphArgument *argument) {
    if (argument.kind == ANEGraphArgumentKindCall &&
        [argument.calleeName isEqualToString:@"string"] &&
        argument.callArguments.count == 1)
        argument = argument.callArguments[0].value;
    return argument.kind == ANEGraphArgumentKindString ? argument.text : nil;
}

static BOOL constantValue(ANEGraphValue *value) {
    return [value.producer.operationName isEqualToString:@"const"];
}

static NSUInteger logicalBytes(ANEGraphValue *value) {
    NSUInteger elements = 1;
    for (NSNumber *dimension in value.type.shape)
        elements *= dimension.unsignedIntegerValue;
    return elements * 2;
}

static NSDictionary *binding(ANEGraphValue *value,
                             const ane::h14::TensorLayout &layout) {
    NSMutableArray *physical = [NSMutableArray arrayWithCapacity:6];
    for (std::uint64_t dimension : layout.nchw) [physical addObject:@(dimension)];
    return @{@"name": value.name, @"dtype": @"float16",
        @"shape": value.type.shape, @"logicalBytes": @(logicalBytes(value)),
        @"index": @(layout.index), @"nchw": physical,
        @"allocationBytes": @(layout.allocationBytes)};
}

static HWXObjectBinding *objectBinding(const ane::h14::TensorLayout &layout,
                                       HWXObjectBindingRole role,
                                       NSUInteger ordinal) {
    NSArray<NSNumber *> *shape = @[@(layout.nchw[0]), @(layout.nchw[1]),
        @(layout.nchw[2]), @(layout.nchw[3])];
    NSUInteger batchStride = (NSUInteger)(layout.nchw[1] * layout.nchw[4]);
    NSUInteger storageBytes = (NSUInteger)(layout.nchw[0] * batchStride);
    NSString *name = [NSString stringWithFormat:
        role == HWXObjectBindingRoleInput ? @"input%lu" : @"output%lu",
        (unsigned long)ordinal];
    return [[HWXObjectBinding alloc] initWithSymbol:name shortName:name role:role
        elementType:ANEElementTypeFP16 shape:shape
        rowStrideBytes:(NSUInteger)layout.nchw[5]
        planeStrideBytes:(NSUInteger)layout.nchw[4]
        batchStrideBytes:batchStride storageByteLength:storageBytes];
}

static NSData *encodeHWX(const ane::h14::Program &program, NSError **error) {
    NSMutableArray<HWXObjectBinding *> *bindings = [NSMutableArray array];
    for (NSUInteger index = 0; index < program.inputs.size(); ++index)
        [bindings addObject:objectBinding(program.inputs[index],
            HWXObjectBindingRoleInput, index)];
    [bindings addObject:objectBinding(program.output,
        HWXObjectBindingRoleOutput, 0)];
    NSData *task = [NSData dataWithBytes:program.taskStream.data()
                                  length:program.taskStream.size()];
    NSData *constants = [NSData dataWithBytes:program.constants.data()
                                       length:program.constants.size()];
    // firstTaskByteLength stays zero: the H14 descriptor records the whole
    // text word count at 0x824, not the first task's size.
    HWXObjectProgramInfo *info = [[HWXObjectProgramInfo alloc]
        initWithTaskCount:program.taskCount
        recordCount:program.programRecordCount formatCode:0
        scratchByteLength:0 descriptorLayout:HWXProgramDescriptorLayoutLinear];
    info.unresolvedDescriptorWord = program.unresolvedDescriptorWord;
    info.h14ScratchDescriptorWord = program.scratchDescriptorWord;
    return [HWXObjectWriter buildObjectForArchitecture:HWXObjectArchitectureH14
        taskDescriptor:task constantRegion:constants bindings:bindings
        kernelRelocationOffsets:@[] programInfo:info error:error];
}

struct H14ParityPlan {
    BOOL unary;
    BOOL scalarConstant;
    ane::h14::UnaryOperation unaryOperation;
    ane::h14::BinaryOperation binaryOperation;
    uint16_t scalarBits;
    ane::h14::ElementwiseShape shape;
    ane::h14::ElementwiseShape operand;
};

/// Apple emits a different program per gelu mode, so each approximation is its
/// own encoder operation.
static BOOL geluEncoding(NSString *mode, ane::h14::UnaryOperation *encoding) {
    if ([mode isEqualToString:@"EXACT"]) *encoding = ane::h14::UnaryOperation::Gelu;
    else if ([mode isEqualToString:@"SIGMOID_APPROXIMATION"])
        *encoding = ane::h14::UnaryOperation::GeluSigmoidApproximation;
    else if ([mode isEqualToString:@"TANH_APPROXIMATION"])
        *encoding = ane::h14::UnaryOperation::GeluTanhApproximation;
    else return NO;
    return YES;
}

/// The decoded shapes a tensor may lower through: its literal NCHW geometry
/// when spatial, and its channel-flattened form, which shares the physical
/// 64-byte row layout the pipeline packs logical tensors into.
static NSUInteger parityShapes(ANEGraphValue *value,
                               ane::h14::ElementwiseShape shapes[2]) {
    NSUInteger elements = 0;
    if (!tensorElementCount(value, &elements) || elements > UINT32_MAX) return 0;
    NSUInteger count = 0;
    ane::h14::ElementwiseShape literal{};
    if (elementwiseShape(value, &literal) &&
        (literal.height != 1 || literal.width != 1))
        shapes[count++] = literal;
    shapes[count++] = {static_cast<std::uint32_t>(elements), 1, 1};
    return count;
}

/// Matches the operations whose H14 task streams are decoded word-for-word
/// from Apple oracles, so they encode as one whole-tensor program.
static BOOL parityPlan(ANEGraphOperation *operation, H14ParityPlan *plan) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    H14ParityPlan candidate{};
    ane::h14::ElementwiseShape shapes[2];
    NSUInteger shapeCount = parityShapes(operation.result, shapes);
    if (!shapeCount || !tensor(x, operation.result.type.shape)) return NO;
    BOOL leaky = [name isEqualToString:@"leaky_relu"];
    BOOL gelu = [name isEqualToString:@"gelu"];
    BOOL rsqrt = [name isEqualToString:@"rsqrt"];
    if (unaryEncoding(name, &candidate.unaryOperation)) {
        uint16_t alphaBits = 0;
        float epsilon = 0.0f, alpha = 0.0f;
        if (operation.arguments.count != (leaky || gelu || rsqrt ? 2u : 1u) ||
            (leaky && (!fp32Attribute(operation.arguments[@"alpha"], &alpha) ||
                       (alphaBits = fp16Bits(alpha)) != 0x3000)) ||
            (gelu && !geluEncoding(stringArgument(operation.arguments[@"mode"]),
                                   &candidate.unaryOperation)) ||
            (rsqrt && (!fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
                       epsilon != 1e-6f)))
            return NO;
        for (NSUInteger index = 0; index < shapeCount; ++index) {
            if (!ane::h14::supportsElementwise(candidate.unaryOperation,
                                               shapes[index])) continue;
            candidate.shape = shapes[index];
            candidate.unary = YES;
            *plan = candidate;
            return YES;
        }
        return NO;
    }
    if (!binaryEncoding(name, &candidate.binaryOperation) ||
        operation.arguments.count != 2 || !y || constantValue(x)) return NO;
    BOOL runtime = !constantValue(y);
    // A runtime operand may be any decoded broadcast surface, not just the
    // result's own; a constant operand is the decoded fp16 0.5 scalar.
    ane::h14::ElementwiseShape operands[2];
    NSUInteger operandCount = 0;
    if (runtime) {
        operandCount = parityShapes(y, operands);
        if (!operandCount) return NO;
    } else if (y.type.kind != ANEValueTypeKindScalar ||
               y.type.elementType != ANEElementTypeFP16 ||
               y.producer.arguments.count ||
               !fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits) ||
               candidate.scalarBits != 0x3800) {
        return NO;
    }
    for (NSUInteger index = 0; index < shapeCount; ++index)
        for (NSUInteger operandIndex = 0;
             operandIndex < (runtime ? operandCount : 1u); ++operandIndex) {
            const ane::h14::ElementwiseShape operand =
                runtime ? operands[operandIndex] : shapes[index];
            if (!ane::h14::supportsElementwise(candidate.binaryOperation,
                                               shapes[index], operand, !runtime))
                continue;
            candidate.shape = shapes[index];
            candidate.operand = operand;
            candidate.scalarConstant = !runtime;
            *plan = candidate;
            return YES;
        }
    return NO;
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

/// Matches the softmax, layer_norm, and reduction programs whose whole H14
/// task streams are decoded from Apple oracles. `epsilon` must be the decoded
/// 1e-5 and layer_norm must carry no gamma or beta: Apple's compiler in this
/// harness rejects every affine form on H14 exactly as it does on H13, so no
/// oracle covers one (17 rejected `h14norm_*_affine` cases).
static BOOL normParityPlan(ANEGraphOperation *operation,
                           ane::h14::NormOperation *operationOut,
                           ane::h14::NormShape *shapeOut) {
    ANEGraphValue *x = operation.operands[@"x"].value;
    NSString *name = operation.operationName;
    ane::h14::NormOperation encoding{};
    ane::h14::NormShape shape{};
    if (!normEncoding(name, &encoding) || !x) return NO;
    const BOOL softmax = encoding == ane::h14::NormOperation::Softmax;
    const BOOL layerNorm = encoding == ane::h14::NormOperation::LayerNorm;
    NSInteger inputShift = 0;
    if (!fp16Tensor(x) || !fp16Tensor(operation.result) ||
        !normSurface(x.type.shape, &shape.input, &inputShift) ||
        !normSurface(operation.result.type.shape, &shape.output, nullptr))
        return NO;
    if (!constantAxisMask(operation.operands[softmax ? @"axis" : @"axes"].value,
                          x.type.shape.count, inputShift, &shape.axisMask))
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
    shape.keepDims = softmax || layerNorm ||
        boolean(operation.arguments[@"keep_dims"], YES);
    if (!shape.keepDims && !boolean(operation.arguments[@"keep_dims"], NO))
        return NO;
    if (!ane::h14::supportsNormParity(encoding, shape)) return NO;
    *operationOut = encoding;
    *shapeOut = shape;
    return YES;
}

/// The logical geometry of a rank-two or higher fp16 matmul. Every leading
/// dimension collapses into `rows`, because Apple's x surface is one dense
/// [1, 1, rows, reduction] plane however the MIL shape spells the batch.
static BOOL matvecGeometry(ANEGraphValue *x, ANEGraphValue *result,
                           NSUInteger *rows, NSUInteger *reduction,
                           NSUInteger *columns) {
    if (!fp16Tensor(x) || !fp16Tensor(result)) return NO;
    NSArray<NSNumber *> *inputShape = x.type.shape;
    NSArray<NSNumber *> *outputShape = result.type.shape;
    NSUInteger rank = inputShape.count;
    if (rank < 2 || outputShape.count != rank) return NO;
    NSUInteger rowCount = 1;
    for (NSUInteger index = 0; index + 1 < rank; ++index) {
        NSUInteger dimension = inputShape[index].unsignedIntegerValue;
        if (!dimension || outputShape[index].unsignedIntegerValue != dimension ||
            rowCount > NSUIntegerMax / dimension) return NO;
        rowCount *= dimension;
    }
    NSUInteger inner = inputShape[rank - 1].unsignedIntegerValue;
    NSUInteger outer = outputShape[rank - 1].unsignedIntegerValue;
    if (!inner || !outer || inner > UINT32_MAX || outer > UINT32_MAX ||
        rowCount > UINT32_MAX || inner > NSUIntegerMax / 2 / outer) return NO;
    *rows = rowCount;
    *reduction = inner;
    *columns = outer;
    return YES;
}

/// Matches the matmul and linear operations whose two-task H14 streams are
/// decoded word-for-word from Apple oracles, and resolves their weight to the
/// [columns, reduction] row-major fp16 form the encoder packs.
static BOOL matvecPlan(ANEGraphOperation *operation, NSURL *modelRoot,
                       ANEDiagnosticEngine *diagnostics,
                       ane::h14::MatvecShape *shape,
                       NSData *__autoreleasing *weightsOut) {
    BOOL linear = [operation.operationName isEqualToString:@"linear"];
    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = linear ? operation.operands[@"weight"].value
                              : operation.operands[@"y"].value;
    if (linear && operation.operands[@"bias"])
        return reject(diagnostics,
            @"H14 emits one program per artifact, so a linear bias has no encoder",
            operation, @"h14.linear-bias-unsupported");
    if (!linear && (operation.arguments.count != 4 ||
                    !boolean(operation.arguments[@"transpose_x"], NO)))
        return reject(diagnostics,
            @"H14 matmul requires an explicit transpose_x = false, an explicit transpose_y, x, and y",
            operation, @"h14.transpose-x-unsupported");
    BOOL transposeY = linear || boolean(operation.arguments[@"transpose_y"], YES);
    if (!linear && !transposeY &&
        !boolean(operation.arguments[@"transpose_y"], NO))
        return reject(diagnostics, @"H14 matmul requires an explicit transpose_y flag",
                      operation);
    NSUInteger rows = 0, reduction = 0, columns = 0;
    if (!x || !y || constantValue(x) || !constantValue(y) ||
        (linear && operation.arguments.count != 2) ||
        !matvecGeometry(x, operation.result, &rows, &reduction, &columns) ||
        !tensor(y, transposeY ? @[@(columns), @(reduction)]
                              : @[@(reduction), @(columns)]))
        return reject(diagnostics,
            @"H14 matmul requires a runtime fp16 x, a constant rank-2 weight matching its transpose flag, and a matching output shape",
            operation);
    ane::h14::MatvecShape candidate{static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(reduction),
        static_cast<std::uint32_t>(columns)};
    if (!ane::h14::supportsMatvecParity(candidate))
        return reject(diagnostics,
            @"H14 matmul supports only the decoded geometries: rows 1, 2, 8, or 64 with reduction and columns each 256, 512, or 1024",
            operation, @"h14.outside-parity-envelope");
    NSUInteger count = reduction * columns;
    NSData *weights = [ANEBlobResolver loadConstantForOperation:y.producer
        expectedBytes:count * 2 modelRoot:modelRoot diagnostics:diagnostics];
    if (!weights) return NO;
    if (!transposeY) {
        // Apple rejects transpose_y=false, so the exact host transpose of the
        // [reduction, columns] weight feeds the same decoded encoder.
        NSMutableData *rowMajor = [NSMutableData dataWithLength:count * 2];
        const uint16_t *source = static_cast<const uint16_t *>(weights.bytes);
        uint16_t *destination = static_cast<uint16_t *>(rowMajor.mutableBytes);
        for (NSUInteger column = 0; column < columns; ++column)
            for (NSUInteger index = 0; index < reduction; ++index)
                destination[column * reduction + index] =
                    source[index * columns + column];
        weights = rowMajor;
    }
    *shape = candidate;
    *weightsOut = weights;
    return YES;
}

@implementation ANEH14Compiler
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    BOOL hwx = [format isEqualToString:@"hwx"];
    if (!hwx && ![format isEqualToString:@"anec"]) {
        reject(diagnostics, @"H14 artifact format must be 'anec' or 'hwx'",
               nil, @"h14.unsupported-format");
        if (error) *error = [NSError errorWithDomain:@"dev.maderix.H14" code:2
            userInfo:@{NSLocalizedDescriptionKey:
                @"H14 artifact format must be 'anec' or 'hwx'"}];
        return NO;
    }
    MILLexer *lexer = [[MILLexer alloc] initWithData:milData
                                         diagnostics:diagnostics];
    MILParser *parser = [[MILParser alloc] initWithTokens:lexer.lexAllTokens
        diagnostics:diagnostics];
    if (diagnostics.errorCount) return NO;
    MILProgramSyntax *syntax = parser.parseProgram;
    ANEGraphModule *module = syntax
        ? [MILGraphImporter importProgram:syntax diagnostics:diagnostics] : nil;
    if (!module || diagnostics.errorCount ||
        ![ANEGraphVerifier verifyModule:module diagnostics:diagnostics])
        return NO;
    if (module.functions.count != 1)
        return reject(diagnostics, @"H14 requires exactly one function");
    ANEGraphFunction *function = module.functions[0];
    NSMutableArray<ANEGraphOperation *> *sourceOperations = [NSMutableArray array];
    for (ANEGraphOperation *candidate in function.operations)
        if (![candidate.operationName isEqualToString:@"const"])
            [sourceOperations addObject:candidate];
    if (sourceOperations.count != 1)
        return reject(diagnostics,
            @"H14 parity encodes exactly one elementwise, unary, matmul, "
             "normalization, or reduction operation",
            sourceOperations.lastObject);
    ANEGraphOperation *operation = sourceOperations[0];
    if (function.returnValues.count != 1 ||
        function.returnValues[0] != operation.result)
        return reject(diagnostics,
            @"H14 requires one operation with its result returned", operation);
    for (ANEGraphValue *input in function.inputs) {
        BOOL used = NO;
        for (ANEGraphArgument *operand in operation.operands.allValues)
            if (operand.value == input) used = YES;
        if (!used)
            return reject(diagnostics, @"H14 function inputs must all be used",
                          operation);
    }

    NSString *name = operation.operationName;
    BOOL matvec = [name isEqualToString:@"matmul"] ||
        [name isEqualToString:@"linear"];
    H14ParityPlan plan{};
    ane::h14::MatvecShape matvecShape{};
    ane::h14::NormOperation normOperation{};
    ane::h14::NormShape normShape{};
    NSData *matvecWeights = nil;
    BOOL normalization = NO;
    if (matvec) {
        if (!matvecPlan(operation, modelRoot, diagnostics, &matvecShape,
                        &matvecWeights)) return NO;
    } else if (normParityPlan(operation, &normOperation, &normShape)) {
        normalization = YES;
    } else if (normEncoding(name, &normOperation)) {
        return reject(diagnostics, [NSString stringWithFormat:
            @"H14 '%@' needs a decoded geometry: fp16 static shapes, constant "
             "axes, no gamma or beta, epsilon 1e-5, and an input and output "
             "surface inside the oracle parity envelope", name],
            operation, @"h14.norm-outside-envelope");
    } else if (!parityPlan(operation, &plan)) {
        return reject(diagnostics,
            @"H14 supports only the decoded fp16 elementwise, scalar-constant, and unary parity envelope",
            operation, @"h14.outside-parity-envelope");
    }

    ane::h14::Program program;
    NSData *payload = nil;
    NSUInteger inputCount =
        matvec || normalization || plan.unary || plan.scalarConstant ? 1 : 2;
    try {
        program = matvec
            ? ane::h14::encodeMatvecParity(matvecShape,
                static_cast<const std::uint8_t *>(matvecWeights.bytes),
                matvecWeights.length)
            : (normalization
                ? ane::h14::encodeNormParity(normOperation, normShape)
                : (plan.unary
                    ? ane::h14::encodeElementwise(plan.unaryOperation, plan.shape)
                    : ane::h14::encodeElementwise(plan.binaryOperation, plan.shape,
                                                  plan.operand,
                                                  plan.scalarConstant,
                                                  plan.scalarBits)));
        if (program.inputs.size() != inputCount)
            return reject(diagnostics, @"H14 encoder returned unexpected inputs",
                          operation);
        if (hwx) {
            payload = encodeHWX(program, error);
            if (!payload) return NO;
        } else {
            std::vector<std::uint8_t> anec = ane::h14::encodeANEC(program);
            payload = [NSData dataWithBytes:anec.data() length:anec.size()];
        }
    } catch (const std::exception &exception) {
        return reject(diagnostics,
            [NSString stringWithUTF8String:exception.what()], operation);
    }

    NSMutableArray<NSDictionary *> *inputRecords = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *tensors =
        [NSMutableDictionary dictionary];
    NSMutableArray<ANEGraphValue *> *inputValues = [NSMutableArray array];
    [inputValues addObject:operation.operands[@"x"].value];
    if (inputCount == 2) [inputValues addObject:operation.operands[@"y"].value];
    for (NSUInteger index = 0; index < inputCount; ++index) {
        ANEGraphValue *value = inputValues[index];
        [inputRecords addObject:binding(value, program.inputs[index])];
        tensors[value.name] = @{@"shape": value.type.shape,
            @"logicalBytes": @(logicalBytes(value)), @"role": @"input"};
    }
    tensors[operation.result.name] = @{@"shape": operation.result.type.shape,
        @"logicalBytes": @(logicalBytes(operation.result)), @"role": @"output"};
    NSDictionary *record = @{
        @"file": [@"program-0." stringByAppendingString:format],
        @"bytes": @(payload.length),
        @"taskDescriptors": @(program.taskCount),
        @"encoder": matvec ? @"apple-parity-matvec"
            : (normalization ? @"apple-parity-norm" : @"h14-oracle-parity"),
        @"operation": operation.operationName,
        @"inputs": inputRecords,
        @"constantInputs": @{},
        @"outputs": @[binding(operation.result, program.output)],
        @"constantOffset": @(program.constantOffsetBytes),
        @"constantBytes": @(program.constants.size()),
        @"firstTaskBytes": @(program.firstTaskBytes),
    };
    NSMutableDictionary *manifest = [@{
        @"schema": @"mil-hwxc.h14-anec-package.v1",
        @"target": @"H14", @"artifactFormat": format,
        @"programs": @[record], @"dispatchPlan": @[@0],
        @"intermediates": @[], @"tensors": tensors,
    } mutableCopy];
    [manifest addEntriesFromDictionary:record];
    NSData *metadata = [NSJSONSerialization dataWithJSONObject:manifest
        options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:error];
    if (!metadata) return NO;
    NSFileManager *manager = NSFileManager.defaultManager;
    if ([manager fileExistsAtPath:directory.path]) {
        NSArray *existing = [manager contentsOfDirectoryAtPath:directory.path
                                                         error:error];
        if (!existing) return NO;
        if (existing.count != 0) {
            if (error) *error = [NSError errorWithDomain:@"dev.maderix.H14"
                code:1 userInfo:@{NSLocalizedDescriptionKey:
                    @"H14 output directory must be empty"}];
            return NO;
        }
    } else if (![manager createDirectoryAtURL:directory
        withIntermediateDirectories:YES attributes:nil error:error]) return NO;
    if (![payload writeToURL:[directory URLByAppendingPathComponent:record[@"file"]]
            options:NSDataWritingAtomic error:error])
        return NO;
    return [metadata writeToURL:
        [directory URLByAppendingPathComponent:@"manifest.json"]
        options:NSDataWritingAtomic error:error];
}
@end
