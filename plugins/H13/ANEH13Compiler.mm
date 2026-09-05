#import "ANEH13Compiler.h"

#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"
#include "H13Program.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>

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

static BOOL lowerOperation(ANEGraphOperation *operation, NSURL *modelRoot,
                           ANEDiagnosticEngine *diagnostics,
                           NSDictionary<NSString *, NSData *> *synthesizedConstants,
                           NSMutableDictionary<NSString *, NSData *> *resolvedConstants,
                           NSUInteger elementOffset, NSUInteger outputElementOffset,
                           ane::h13::Program &program,
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
            BOOL repeatedConstant = wholeData != nil;
            if (wholeData) {
                if (!tensor(constantInput, runtimeInput.type.shape) || wholeData.length != 128)
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
            !geometry ||
            !tensor(y, transposeY ? @[@(columns), @(reduction)]
                                  : @[@(reduction), @(columns)]) ||
            !constantValue(y))
            return reject(diagnostics,
                @"H13 matmul requires positive fp16 x rows, constant rank-2 W, matching explicit transpose flags, and a matching positive output shape",
                operation);
        if (reduction > 512)
            return reject(diagnostics,
                @"H13 matmul reduction exceeds the largest bit-exact descriptor (K=512)",
                operation, @"h13.reduction-too-large");
        if (transposeX && rows > 1)
            return reject(diagnostics,
                @"H13 transpose_x=true matmul supports exactly one logical row",
                operation, @"h13.transpose-x-multirow");
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
        NSUInteger physicalReduction = reduction <= 256 ? 256 : 512;
        NSUInteger outputColumn = outputElementOffset % columns;
        NSUInteger outputElements = MIN((NSUInteger)512, columns - outputColumn);
        NSMutableData *paddedWeights =
            [NSMutableData dataWithLength:512 * physicalReduction * 2];
        const uint8_t *source = static_cast<const uint8_t *>(weights.bytes);
        uint8_t *destination = static_cast<uint8_t *>(paddedWeights.mutableBytes);
        if (transposeY) {
            for (NSUInteger row = 0; row < outputElements; ++row)
                std::memcpy(destination + row * physicalReduction * 2,
                    source + ((outputColumn + row) * reduction) * 2,
                    reduction * 2);
        } else {
            for (NSUInteger row = 0; row < reduction; ++row)
                std::memcpy(destination + row * 512 * 2,
                    source + (row * columns + outputColumn) * 2,
                    outputElements * 2);
        }
        program = ane::h13::encodeMatvec(
            static_cast<std::uint32_t>(physicalReduction),
            static_cast<const std::uint8_t *>(paddedWeights.bytes),
            paddedWeights.length, transposeY);
        inputs = @[x];
    } else {
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
       outputDirectory:(NSURL *)directory
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    MILLexer *lexer = [[MILLexer alloc] initWithData:milData diagnostics:diagnostics];
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

    NSMutableArray<ANEGraphOperation *> *operations = [NSMutableArray array];
    NSMutableArray<ANEGraphValue *> *manifestValues = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *aliases =
        [NSMutableDictionary dictionary];
    NSMutableDictionary<NSString *, NSData *> *synthesizedConstants =
        [NSMutableDictionary dictionary];
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
            NSString *zeroName = [NSString stringWithFormat:@"$h13.%@.zero",
                result.name];
            ANEGraphValue *zero = [[ANEGraphValue alloc]
                initWithName:zeroName type:x.type];
            synthesizedConstants[zeroName] = splatFP16(0);
            [operations addObject:binaryOperation(@"maximum", x, zero,
                result, candidate.range)];
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
    NSMutableArray<NSNumber *> *dispatchPlan = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *tensors =
        [NSMutableDictionary dictionary];
    for (ANEGraphValue *input in function.inputs)
        recordTensor(tensors, input, input.type.shape, @"input");
    NSMutableDictionary<NSString *, NSData *> *resolvedConstants =
        [NSMutableDictionary dictionary];
    NSArray<NSString *> *binaryNames =
        @[@"add", @"mul", @"maximum", @"minimum", @"sub", @"real_div"];
    ANEGraphOperation *operation = nil;
    try {
        for (operation in operations) {
            NSUInteger sliceCount = 1, inputSliceElements = 0,
                inputPhysicalElements = 0, outputSliceElements = 0,
                outputPhysicalElements = 0, outputChunks = 1;
            BOOL matmul = [operation.operationName isEqualToString:@"matmul"];
            if ([binaryNames containsObject:operation.operationName]) {
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
                    outputChunks = (columns - 1) / 512 + 1;
                    if (rows > NSUIntegerMax / outputChunks)
                        return reject(diagnostics, @"H13 matmul program count overflows",
                            operation);
                    sliceCount = rows * outputChunks;
                    inputSliceElements = reduction;
                    inputPhysicalElements = reduction <= 256 ? 256 : 512;
                    outputPhysicalElements = 512;
                }
                ANEGraphValue *weights = operation.operands[@"y"].value;
                if (fp16Tensor(weights))
                    recordTensor(tensors, weights, weights.type.shape, @"constant");
            }

            for (NSUInteger sliceIndex = 0; sliceIndex < sliceCount; ++sliceIndex) {
                NSUInteger inputOffset = 0, outputOffset = 0;
                if (matmul) {
                    NSUInteger reduction = inputSliceElements;
                    NSUInteger columns = operation.result.type.shape.lastObject.unsignedIntegerValue;
                    NSUInteger row = sliceIndex / outputChunks;
                    NSUInteger chunk = sliceIndex % outputChunks;
                    inputOffset = row * reduction;
                    outputOffset = row * columns + chunk * 512;
                    outputSliceElements = MIN((NSUInteger)512, columns - chunk * 512);
                } else {
                    inputOffset = outputOffset = sliceIndex * 64;
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
                if (!lowerOperation(operation, modelRoot, diagnostics,
                                    synthesizedConstants, resolvedConstants,
                                    inputOffset, outputOffset, program, &inputs,
                                    &constantInput, &constantData, &manifestOperation))
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
                    NSArray<NSNumber *> *logicalShape =
                        inputOffset == 0 && inputSliceElements == fullElements
                            ? fullShape : @[@(inputSliceElements)];
                    NSMutableDictionary *record =
                        [binding(input, logicalShape, program.inputs.at(index)) mutableCopy];
                    BOOL aliasShape =
                        ![fullShape isEqualToArray:tensors[input.name][@"shape"]];
                    if (inputOffset || inputSliceElements != fullElements ||
                        inputPhysicalElements != inputSliceElements || aliasShape)
                        addSlice(record, input, inputOffset, inputSliceElements,
                                 inputPhysicalElements);
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
                NSString *file = [NSString stringWithFormat:@"program-%lu.anec",
                    (unsigned long)programIndex];
                NSDictionary *record = @{
                    @"file": file, @"bytes": @(anec.size()),
                    @"taskDescriptors": @1, @"operation": manifestOperation,
                    @"inputs": inputRecords, @"constantInputs": constantInputs,
                    @"outputs": @[outputRecord],
                    @"constantOffset": @(ane::h13::constantOffset),
                    @"constantBytes": @(program.constants.size()),
                };
                [programRecords addObject:record];
                [payloads addObject:[NSData dataWithBytes:anec.data() length:anec.size()]];
                [dispatchPlan addObject:@(programIndex)];
            }
        }
    } catch (const std::exception &exception) {
        return reject(diagnostics, [NSString stringWithUTF8String:exception.what()], operation);
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

    NSMutableDictionary *manifest = [@{
        @"schema": @"mil-hwxc.h13-anec-package.v1",
        @"target": @"H13", @"artifactFormat": @"anec",
        @"programs": programRecords, @"dispatchPlan": dispatchPlan,
        @"intermediates": intermediateNames, @"tensors": tensors,
    } mutableCopy];
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
