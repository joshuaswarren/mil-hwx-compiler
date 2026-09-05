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

static BOOL tensorElements(ANEGraphValue *value, NSUInteger expected) {
    if (!fp16Tensor(value)) return NO;
    NSUInteger elements = 1;
    for (NSNumber *number in value.type.shape) {
        NSUInteger dimension = number.unsignedIntegerValue;
        if (!dimension || dimension > expected / elements) return NO;
        elements *= dimension;
    }
    return elements == expected;
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

static BOOL singleVectorMatmul(ANEGraphValue *x, ANEGraphValue *result,
                               BOOL transposeX, NSUInteger *reduction) {
    NSArray<NSNumber *> *inputShape = x.type.shape;
    NSArray<NSNumber *> *outputShape = result.type.shape;
    NSUInteger rank = inputShape.count;
    if (!fp16Tensor(x) || !fp16Tensor(result) || outputShape.count != rank ||
        (!transposeX && rank == 0) || (transposeX && rank < 2)) return NO;

    NSUInteger leading = rank - (transposeX ? 2 : 1);
    for (NSUInteger index = 0; index < leading; ++index)
        if (inputShape[index].unsignedIntegerValue != 1 ||
            outputShape[index].unsignedIntegerValue != 1) return NO;

    NSUInteger candidate = inputShape[rank - (transposeX ? 2 : 1)].unsignedIntegerValue;
    if ((candidate != 256 && candidate != 512) ||
        outputShape[rank - (transposeX ? 2 : 1)].unsignedIntegerValue !=
            (transposeX ? 1 : 512)) return NO;
    if (transposeX && (inputShape[rank - 1].unsignedIntegerValue != 1 ||
                       outputShape[rank - 1].unsignedIntegerValue != 512)) return NO;
    *reduction = candidate;
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
    ANEGraphOperation *operation = nil;
    for (ANEGraphOperation *candidate in function.operations) {
        if ([candidate.operationName isEqualToString:@"const"]) continue;
        if (operation)
            return reject(diagnostics, @"H13 multi-operation scheduling is not qualified", candidate);
        operation = candidate;
    }
    if (!operation || function.returnValues.count != 1 ||
        function.returnValues[0] != operation.result)
        return reject(diagnostics, @"H13 requires one operation with its result returned", operation);

    ANEGraphValue *x = operation.operands[@"x"].value;
    ANEGraphValue *y = operation.operands[@"y"].value;
    NSString *name = operation.operationName;
    NSArray<NSString *> *binaryNames =
        @[@"add", @"mul", @"maximum", @"minimum", @"sub", @"real_div"];
    NSUInteger binaryIndex = [binaryNames indexOfObject:name];
    ane::h13::Program program;
    NSArray<ANEGraphValue *> *inputs;
    ANEGraphValue *constantInput = nil;
    NSData *constantData = nil;
    NSString *manifestOperation = name;
    try {
        if (binaryIndex != NSNotFound) {
            if (operation.arguments.count != 2 || !x || !y)
                return reject(diagnostics,
                    @"H13 binary operations require x and y value operands", operation);
            BOOL xIsInput = [function.inputs containsObject:x];
            BOOL yIsInput = [function.inputs containsObject:y];
            if (function.inputs.count == 2 && xIsInput && yIsInput) {
                if (binaryIndex >= 4)
                    return reject(diagnostics, [NSString stringWithFormat:
                        @"H13 '%@' with two runtime inputs cannot lower exactly through the verified binary modes", name],
                        operation, @"h13.nonfoldable-binary");
                if (x == y || !tensorElements(x, 64) || !tensor(y, x.type.shape) ||
                    !tensor(operation.result, x.type.shape))
                    return reject(diagnostics,
                        @"H13 binary operations require two distinct fp16 inputs with the same positive static shape containing exactly 64 elements", operation);
                inputs = @[x, y];
            } else {
                ANEGraphValue *runtimeInput = nil;
                if (function.inputs.count == 1 && xIsInput &&
                    [y.producer.operationName isEqualToString:@"const"]) {
                    runtimeInput = x;
                    constantInput = y;
                } else if (function.inputs.count == 1 && binaryIndex < 4 && yIsInput &&
                           [x.producer.operationName isEqualToString:@"const"]) {
                    runtimeInput = y;
                    constantInput = x;
                } else {
                    NSString *code = binaryIndex >= 4
                        ? @"h13.nonfoldable-binary" : @"h13.invalid-constant-input";
                    return reject(diagnostics,
                        @"H13 binary folding requires one runtime fp16 tensor and one eligible const operand",
                        operation, code);
                }
                if (function.inputs[0] != runtimeInput || !tensorElements(runtimeInput, 64) ||
                    !tensor(operation.result, runtimeInput.type.shape))
                    return reject(diagnostics,
                        @"H13 folded binary operations require one fp16 input and output with the same positive static shape containing exactly 64 elements",
                        operation, @"h13.invalid-constant-input");

                ANEGraphOperation *producer = constantInput.producer;
                BOOL scalar = constantInput.type.kind == ANEValueTypeKindScalar &&
                    constantInput.type.elementType == ANEElementTypeFP16;
                if ((!tensor(constantInput, runtimeInput.type.shape) && !scalar) ||
                    (scalar && binaryIndex != 1) || producer.arguments.count)
                    return reject(diagnostics,
                        @"H13 constants must be a matching 64-element fp16 tensor; only mul accepts an inline fp16 scalar broadcast",
                        producer, @"h13.invalid-constant-input");

                ANEGraphArgument *literal = producer.attributes[@"val"];
                if (scalar) {
                    uint16_t bits = 0;
                    if (!fp16Scalar(literal, &bits))
                        return reject(diagnostics, @"H13 scalar mul requires one finite inline fp16 value",
                            producer, @"h13.invalid-constant-payload");
                    NSMutableData *expanded = [NSMutableData dataWithLength:128];
                    uint16_t *words = static_cast<uint16_t *>(expanded.mutableBytes);
                    for (NSUInteger index = 0; index < 64; ++index) words[index] = bits;
                    constantData = expanded;
                } else {
                    if (literal.kind != ANEGraphArgumentKindCall ||
                        ![literal.calleeValueType isEqualToValueType:constantInput.type] ||
                        literal.callArguments.count != 1)
                        return reject(diagnostics,
                            @"H13 tensor constants require a matching typed inline list or BLOBFILE payload",
                            producer, @"h13.invalid-constant-payload");
                    ANEGraphArgument *payload = literal.callArguments[0].value;
                    if (payload.kind == ANEGraphArgumentKindList && payload.elements.count == 64) {
                        NSMutableData *dense = [NSMutableData dataWithLength:128];
                        uint16_t *words = static_cast<uint16_t *>(dense.mutableBytes);
                        for (NSUInteger index = 0; index < 64; ++index)
                            if (!fp16Scalar(payload.elements[index], &words[index]))
                                return reject(diagnostics,
                                    @"H13 inline tensor constants require exactly 64 finite fp16 elements",
                                    producer, @"h13.invalid-constant-payload");
                        constantData = dense;
                    } else if (payload.kind == ANEGraphArgumentKindCall &&
                               [payload.calleeName isEqualToString:@"BLOBFILE"]) {
                        constantData = [ANEBlobResolver loadConstantForOperation:producer
                            expectedBytes:128 modelRoot:modelRoot diagnostics:diagnostics];
                        if (!constantData)
                            return reject(diagnostics, @"H13 could not load the constant BLOBFILE payload",
                                producer, @"h13.invalid-constant-payload");
                    } else {
                        return reject(diagnostics,
                            @"H13 inline tensor constants require exactly 64 finite fp16 elements",
                            producer, @"h13.invalid-constant-payload");
                    }
                }

                NSMutableData *folded = [constantData mutableCopy];
                uint16_t *words = static_cast<uint16_t *>(folded.mutableBytes);
                if (binaryIndex == 4) {
                    for (NSUInteger index = 0; index < 64; ++index) {
                        if ((words[index] & 0x7c00u) == 0x7c00u)
                            return reject(diagnostics,
                                @"H13 sub constants must be finite for exact add lowering",
                                operation, @"h13.nonfinite-constant");
                        words[index] ^= 0x8000u;
                    }
                    manifestOperation = @"add";
                } else if (binaryIndex == 5) {
                    for (NSUInteger index = 0; index < 64; ++index) {
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
            NSUInteger reduction = 0;
            BOOL transposeX = boolean(operation.arguments[@"transpose_x"], YES);
            BOOL transposeY = boolean(operation.arguments[@"transpose_y"], YES);
            if (operation.arguments.count != 4 || function.inputs.count != 1 ||
                function.inputs[0] != x ||
                (!transposeX && !boolean(operation.arguments[@"transpose_x"], NO)) ||
                (!transposeY && !boolean(operation.arguments[@"transpose_y"], NO)) ||
                !singleVectorMatmul(x, operation.result, transposeX, &reduction) ||
                !tensor(y, transposeY ? @[@512, @(reduction)] : @[@(reduction), @512]) ||
                ![y.producer.operationName isEqualToString:@"const"])
                return reject(diagnostics,
                    @"H13 matmul requires one fp16 logical vector with K=256 or 512, constant rank-2 W, and matching explicit transpose flags and output shape", operation);
            ANEGraphArgument *value = y.producer.attributes[@"val"];
            if (y.producer.arguments.count || value.kind != ANEGraphArgumentKindCall ||
                ![value.calleeValueType isEqualToValueType:y.type] ||
                value.callArguments.count != 1 ||
                ![value.callArguments[0].value.calleeName isEqualToString:@"BLOBFILE"])
                return reject(diagnostics,
                    @"H13 weights require a matching tensor value with one BLOBFILE payload", y.producer);
            NSUInteger count = 512 * reduction;
            NSData *weights = [ANEBlobResolver loadConstantForOperation:y.producer
                expectedBytes:count * 2 modelRoot:modelRoot diagnostics:diagnostics];
            if (!weights) return NO;
            program = ane::h13::encodeMatvec(static_cast<std::uint32_t>(reduction),
                static_cast<const std::uint8_t *>(weights.bytes), weights.length, transposeY);
            inputs = @[x];
        } else {
            return reject(diagnostics, [NSString stringWithFormat:
                @"H13 has no source-qualified encoder for '%@'", name], operation);
        }
        std::vector<std::uint8_t> anec = ane::h13::encodeANEC(program);
        NSMutableArray *inputRecords = [NSMutableArray arrayWithCapacity:inputs.count];
        NSMutableDictionary *constantInputs = [NSMutableDictionary dictionary];
        for (NSUInteger index = 0; index < inputs.count; ++index) {
            ANEGraphValue *input = inputs[index];
            NSArray<NSNumber *> *shape = input == constantInput
                ? operation.result.type.shape : input.type.shape;
            NSMutableDictionary *record =
                [binding(input, shape, program.inputs.at(index)) mutableCopy];
            if (input == constantInput) {
                record[@"binding"] = @"constant";
                constantInputs[input.name] = hexData(constantData);
            }
            [inputRecords addObject:record];
        }
        NSDictionary *manifest = @{
            @"schema": @"mil-hwxc.h13-anec-package.v1",
            @"target": @"H13", @"artifactFormat": @"anec",
            @"file": @"program-0.anec", @"bytes": @(anec.size()),
            @"taskDescriptors": @1, @"operation": manifestOperation,
            @"inputs": inputRecords, @"constantInputs": constantInputs,
            @"outputs": @[binding(operation.result, operation.result.type.shape,
                                    program.output)],
            @"constantOffset": @(ane::h13::constantOffset),
            @"constantBytes": @(program.constants.size()),
        };
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
        NSData *payload = [NSData dataWithBytesNoCopy:anec.data() length:anec.size() freeWhenDone:NO];
        return [payload writeToURL:[directory URLByAppendingPathComponent:@"program-0.anec"]
            options:NSDataWritingAtomic error:error] &&
            [metadata writeToURL:[directory URLByAppendingPathComponent:@"manifest.json"]
                options:NSDataWritingAtomic error:error];
    } catch (const std::exception &exception) {
        return reject(diagnostics, [NSString stringWithUTF8String:exception.what()], operation);
    }
}
@end
