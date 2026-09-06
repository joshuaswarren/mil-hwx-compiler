#import "ANEH14Compiler.h"

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

static NSDictionary *binding(ANEGraphValue *value,
                             NSArray<NSNumber *> *logicalShape,
                             const ane::h14::TensorLayout &layout) {
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
    NSUInteger elements;
};

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
            (gelu && ![stringArgument(operation.arguments[@"mode"])
                          isEqualToString:@"EXACT"]) ||
            (rsqrt && (!fp32Attribute(operation.arguments[@"epsilon"], &epsilon) ||
                       epsilon != 1e-6f)))
            return NO;
        for (NSUInteger index = 0; index < shapeCount; ++index) {
            if (!ane::h14::supportsElementwise(candidate.unaryOperation,
                                               shapes[index])) continue;
            candidate.shape = shapes[index];
            candidate.elements = (NSUInteger)shapes[index].channels *
                shapes[index].height * shapes[index].width;
            candidate.unary = YES;
            *plan = candidate;
            return YES;
        }
        return NO;
    }
    if (!binaryEncoding(name, &candidate.binaryOperation) ||
        operation.arguments.count != 2 || !y || constantValue(x)) return NO;
    BOOL runtime = !constantValue(y);
    if (runtime) {
        if (!tensor(y, operation.result.type.shape)) return NO;
    } else if (y.type.kind != ANEValueTypeKindScalar ||
               y.type.elementType != ANEElementTypeFP16 ||
               y.producer.arguments.count ||
               !fp16Scalar(y.producer.attributes[@"val"], &candidate.scalarBits) ||
               candidate.scalarBits != 0x3800) {
        return NO;
    }
    for (NSUInteger index = 0; index < shapeCount; ++index) {
        if (!ane::h14::supportsElementwise(candidate.binaryOperation,
                                           shapes[index], !runtime)) continue;
        candidate.shape = shapes[index];
        candidate.elements = (NSUInteger)shapes[index].channels *
            shapes[index].height * shapes[index].width;
        candidate.scalarConstant = !runtime;
        *plan = candidate;
        return YES;
    }
    return NO;
}

@implementation ANEH14Compiler
+ (BOOL)compileMILData:(NSData *)milData
             modelRoot:(NSURL *)modelRoot
                format:(NSString *)format
       outputDirectory:(NSURL *)directory
           diagnostics:(ANEDiagnosticEngine *)diagnostics
                 error:(NSError **)error {
    (void)modelRoot;
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
            @"H14 parity encodes exactly one elementwise or unary operation",
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

    H14ParityPlan plan{};
    if (!parityPlan(operation, &plan))
        return reject(diagnostics,
            @"H14 supports only the decoded fp16 elementwise, scalar-constant, and unary parity envelope",
            operation, @"h14.outside-parity-envelope");

    ane::h14::Program program;
    NSData *payload = nil;
    NSUInteger inputCount = plan.unary || plan.scalarConstant ? 1 : 2;
    try {
        program = plan.unary
            ? ane::h14::encodeElementwise(plan.unaryOperation, plan.shape)
            : ane::h14::encodeElementwise(plan.binaryOperation, plan.shape,
                                          plan.scalarConstant, plan.scalarBits);
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

    NSArray<NSNumber *> *shape = operation.result.type.shape;
    NSMutableArray<NSDictionary *> *inputRecords = [NSMutableArray array];
    NSMutableDictionary<NSString *, NSDictionary *> *tensors =
        [NSMutableDictionary dictionary];
    NSMutableArray<ANEGraphValue *> *inputValues = [NSMutableArray array];
    [inputValues addObject:operation.operands[@"x"].value];
    if (inputCount == 2) [inputValues addObject:operation.operands[@"y"].value];
    for (NSUInteger index = 0; index < inputCount; ++index) {
        ANEGraphValue *value = inputValues[index];
        [inputRecords addObject:binding(value, shape, program.inputs[index])];
        tensors[value.name] = @{@"shape": shape,
            @"logicalBytes": @(plan.elements * 2), @"role": @"input"};
    }
    tensors[operation.result.name] = @{@"shape": shape,
        @"logicalBytes": @(plan.elements * 2), @"role": @"output"};
    NSDictionary *record = @{
        @"file": [@"program-0." stringByAppendingString:format],
        @"bytes": @(payload.length),
        @"taskDescriptors": @(program.taskCount),
        @"encoder": @"h14-oracle-parity",
        @"operation": operation.operationName,
        @"inputs": inputRecords,
        @"constantInputs": @{},
        @"outputs": @[binding(operation.result, shape, program.output)],
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
