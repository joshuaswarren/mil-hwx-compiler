#import "ANEH13Compiler.h"

#import "ANEBlobResolver.h"
#import "ANEGraphVerifier.h"
#import "MILLexer.h"
#import "MILGraphImporter.h"
#import "MILParser.h"
#include "H13Program.h"

#include <exception>

static BOOL reject(ANEDiagnosticEngine *diagnostics, NSString *message,
                   ANEGraphOperation *operation = nil) {
    ANESourceLocation start = ANESourceLocationMake(0, 1, 1);
    [diagnostics emitSeverity:ANEDiagnosticSeverityError
        code:@"h13.unsupported-program" message:message
        range:operation ? operation.range : ANESourceRangeMake(start, start)];
    return NO;
}

static BOOL tensor(ANEGraphValue *value, NSArray<NSNumber *> *shape) {
    return value.type.kind == ANEValueTypeKindTensor &&
        value.type.elementType == ANEElementTypeFP16 &&
        [value.type.shape isEqualToArray:shape];
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

static NSDictionary *binding(ANEGraphValue *value,
                             const ane::h13::TensorLayout &layout) {
    NSMutableArray *physical = [NSMutableArray arrayWithCapacity:6];
    for (std::uint64_t dimension : layout.nchw) [physical addObject:@(dimension)];
    NSUInteger elements = 1;
    for (NSNumber *dimension in value.type.shape)
        elements *= dimension.unsignedIntegerValue;
    return @{@"name": value.name, @"dtype": @"float16",
        @"shape": value.type.shape, @"logicalBytes": @(elements * 2),
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
    NSArray<NSString *> *binaryNames = @[@"add", @"mul", @"maximum", @"minimum"];
    NSUInteger binaryIndex = [binaryNames indexOfObject:name];
    ane::h13::Program program;
    NSArray<ANEGraphValue *> *inputs;
    try {
        if (binaryIndex != NSNotFound) {
            if (operation.arguments.count != 2 || function.inputs.count != 2 ||
                x == y || ![function.inputs containsObject:x] ||
                ![function.inputs containsObject:y] ||
                !tensor(x, @[@1, @64, @1, @1]) ||
                !tensor(y, x.type.shape) || !tensor(operation.result, x.type.shape))
                return reject(diagnostics,
                    @"H13 binary operations require two distinct fp16 [1,64,1,1] inputs and the same output shape", operation);
            const ane::h13::BinaryOperation operations[] = {
                ane::h13::BinaryOperation::Add, ane::h13::BinaryOperation::Multiply,
                ane::h13::BinaryOperation::Maximum, ane::h13::BinaryOperation::Minimum};
            program = ane::h13::encodeBinary(operations[binaryIndex]);
            inputs = @[x, y];
        } else if ([name isEqualToString:@"matmul"]) {
            NSUInteger reduction = x.type.shape.count == 2
                ? x.type.shape[1].unsignedIntegerValue : 0;
            BOOL transposeY = boolean(operation.arguments[@"transpose_y"], YES);
            if (operation.arguments.count != 4 || function.inputs.count != 1 ||
                function.inputs[0] != x || (reduction != 256 && reduction != 512) ||
                !tensor(x, @[@1, @(reduction)]) ||
                !tensor(y, transposeY ? @[@512, @(reduction)] : @[@(reduction), @512]) ||
                !tensor(operation.result, @[@1, @512]) ||
                ![y.producer.operationName isEqualToString:@"const"] ||
                !boolean(operation.arguments[@"transpose_x"], NO) ||
                (!transposeY && !boolean(operation.arguments[@"transpose_y"], NO)))
                return reject(diagnostics,
                    @"H13 matmul requires fp16 x[1,K], K=256 or 512, constant W, transpose_x=false, and a matching transpose_y weight shape", operation);
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
        for (NSUInteger index = 0; index < inputs.count; ++index)
            [inputRecords addObject:binding(inputs[index], program.inputs.at(index))];
        NSDictionary *manifest = @{
            @"schema": @"mil-hwxc.h13-anec-package.v1",
            @"target": @"H13", @"artifactFormat": @"anec",
            @"file": @"program-0.anec", @"bytes": @(anec.size()),
            @"taskDescriptors": @1, @"operation": name, @"inputs": inputRecords,
            @"outputs": @[binding(operation.result, program.output)],
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
