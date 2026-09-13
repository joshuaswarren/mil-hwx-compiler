#import "ANEGraphVerifier.h"

@implementation ANEGraphVerifier

+ (void)emit:(ANEDiagnosticEngine *)diagnostics
        code:(NSString *)code
     message:(NSString *)message
   operation:(ANEGraphOperation *)operation {
    [diagnostics emitSeverity:ANEDiagnosticSeverityError code:code
                      message:message range:operation.range];
}

+ (NSNumber *)integerConstantForValue:(ANEGraphValue *)value {
    if (value.type.kind != ANEValueTypeKindTensor ||
        value.type.elementType != ANEElementTypeInt32 ||
        value.type.shape.count != 0)
        return nil;
    ANEGraphOperation *producer = value.producer;
    if (![producer.operationName isEqualToString:@"const"] ||
        producer.results.count != 1)
        return nil;
    ANEGraphArgument *argument = producer.attributes[@"val"];
    if (argument.kind != ANEGraphArgumentKindCall ||
        argument.callArguments.count != 1 ||
        ![argument.calleeValueType isEqualToValueType:value.type])
        return nil;
    argument = argument.callArguments[0].value;
    if (argument.kind != ANEGraphArgumentKindInteger) return nil;
    return @(argument.text.longLongValue);
}

+ (BOOL)verifyUnarySameType:(ANEGraphOperation *)operation
                 diagnostics:(ANEDiagnosticEngine *)diagnostics
                        code:(NSString *)code {
    ANEGraphArgument *input = operation.operands[@"x"];
    ANEGraphValue *result = operation.results[0];
    if (!input || ![result.type isEqualToValueType:input.value.type]) {
        [self emit:diagnostics code:code
           message:[NSString stringWithFormat:
              @"%@ input and result types must match", operation.operationName]
         operation:operation];
        return NO;
    }
    return YES;
}

+ (BOOL)verifySplit:(ANEGraphOperation *)operation
        diagnostics:(ANEDiagnosticEngine *)diagnostics {
    ANEGraphValue *input = operation.operands[@"x"].value;
    NSNumber *axisValue = [self integerConstantForValue:
        operation.operands[@"axis"].value];
    NSNumber *countValue = [self integerConstantForValue:
        operation.operands[@"num_splits"].value];
    if (!input || !axisValue || !countValue) {
        [self emit:diagnostics code:@"ane.verify.split-constant-parameters"
            message:@"split requires x plus constant axis and num_splits values"
          operation:operation];
        return NO;
    }
    NSInteger count = countValue.integerValue;
    if (count <= 0 || operation.results.count != (NSUInteger)count) {
        [self emit:diagnostics code:@"ane.verify.split-result-arity"
            message:[NSString stringWithFormat:
                @"split declares %lu results for num_splits %ld",
                (unsigned long)operation.results.count, (long)count]
          operation:operation];
        return NO;
    }
    if (input.type.kind != ANEValueTypeKindTensor || input.type.shape.count == 0) {
        [self emit:diagnostics code:@"ane.verify.split-type"
            message:@"split requires a ranked tensor input" operation:operation];
        return NO;
    }
    NSInteger rank = input.type.shape.count;
    NSInteger axis = axisValue.integerValue;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) {
        [self emit:diagnostics code:@"ane.verify.split-axis"
            message:@"split axis is outside the input rank" operation:operation];
        return NO;
    }
    NSInteger dimension = input.type.shape[(NSUInteger)axis].integerValue;
    if (dimension <= 0 || dimension % count != 0) {
        [self emit:diagnostics code:@"ane.verify.split-shape"
            message:@"split dimension must be positive and divisible by num_splits"
          operation:operation];
        return NO;
    }
    NSMutableArray<NSNumber *> *shape = [input.type.shape mutableCopy];
    shape[(NSUInteger)axis] = @(dimension / count);
    ANEValueType *expected = [[ANEValueType alloc]
        initWithKind:ANEValueTypeKindTensor
         elementType:input.type.elementType shape:shape];
    for (ANEGraphValue *result in operation.results) {
        if (![result.type isEqualToValueType:expected]) {
            [self emit:diagnostics code:@"ane.verify.split-shape"
                message:@"split result types must match the ordered partition shape"
              operation:operation];
            return NO;
        }
    }
    return YES;
}

+ (BOOL)verifyOperation:(ANEGraphOperation *)operation
             diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (operation.results.count == 0) {
        [self emit:diagnostics code:@"ane.verify.result-arity"
            message:@"operation must define at least one result"
          operation:operation];
        return NO;
    }
    for (ANEGraphValue *result in operation.results) {
        if (result.producer != operation) {
            [self emit:diagnostics code:@"ane.verify.bad-producer"
                message:@"result does not point to its defining operation"
              operation:operation];
            return NO;
        }
    }
    NSString *name = operation.operationName;
    if ([name isEqualToString:@"split"])
        return [self verifySplit:operation diagnostics:diagnostics];
    if (operation.results.count != 1) {
        [self emit:diagnostics code:@"ane.verify.result-arity"
            message:[NSString stringWithFormat:
                @"%@ requires exactly one result", name]
          operation:operation];
        return NO;
    }
    ANEGraphValue *result = operation.results[0];
    if ([name isEqualToString:@"relu"])
        return [self verifyUnarySameType:operation diagnostics:diagnostics
                                    code:@"ane.verify.relu-type"];
    if ([name isEqualToString:@"softmax"] ||
        [name isEqualToString:@"dequantize"] ||
        [name isEqualToString:@"quantize"]) {
        ANEGraphArgument *input = operation.operands[@"x"] ?:
                                  operation.operands[@"input"];
        if (!input) {
            [self emit:diagnostics code:@"ane.verify.missing-input"
                message:[NSString stringWithFormat:@"%@ requires an input", name]
              operation:operation];
            return NO;
        }
        if ([name isEqualToString:@"softmax"] &&
            ![result.type isEqualToValueType:input.value.type]) {
            [self emit:diagnostics code:@"ane.verify.softmax-type"
                message:@"softmax input and result types must match"
              operation:operation];
            return NO;
        }
        if ([name isEqualToString:@"quantize"] &&
            (input.value.type.elementType != ANEElementTypeFP16 ||
             result.type.elementType != ANEElementTypeInt8)) {
            [self emit:diagnostics code:@"ane.verify.quantize-type"
                message:@"initial quantize path requires fp16 to int8"
              operation:operation];
            return NO;
        }
        if ([name isEqualToString:@"dequantize"] &&
            (input.value.type.elementType != ANEElementTypeInt8 ||
             result.type.elementType != ANEElementTypeFP16)) {
            [self emit:diagnostics code:@"ane.verify.dequantize-type"
                message:@"initial dequantize path requires int8 to fp16"
              operation:operation];
            return NO;
        }
    }
    if ([name isEqualToString:@"conv"]) {
        ANEGraphValue *input = operation.operands[@"x"].value;
        ANEGraphValue *weight = operation.operands[@"weight"].value;
        if (!input || !weight || input.type.kind != ANEValueTypeKindTensor ||
            weight.type.kind != ANEValueTypeKindTensor ||
            input.type.elementType != ANEElementTypeFP16 ||
            weight.type.elementType != ANEElementTypeFP16 ||
            result.type.elementType != ANEElementTypeFP16) {
            [self emit:diagnostics code:@"ane.verify.conv-type"
                message:@"initial convolution path requires fp16 tensors"
              operation:operation];
            return NO;
        }
    }
    return YES;
}

+ (BOOL)verifyModule:(ANEGraphModule *)module
          diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSSet<NSString *> *versions = [NSSet setWithArray:@[@"1", @"1.3"]];
    if (![versions containsObject:module.version]) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
            code:@"ane.verify.unsupported-program-version"
            message:[NSString stringWithFormat:
                @"unsupported MIL program version '%@'", module.version]
            range:ANESourceRangeMake(ANESourceLocationMake(0, 1, 1),
                                     ANESourceLocationMake(0, 1, 1))];
        return NO;
    }
    if (module.functions.count == 0) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
            code:@"ane.verify.empty-module" message:@"module has no functions"
            range:ANESourceRangeMake(ANESourceLocationMake(0, 1, 1),
                                     ANESourceLocationMake(0, 1, 1))];
        return NO;
    }
    NSString *expectedOpset = [module.version isEqualToString:@"1"]
        ? @"CoreML8" : @"ios18";
    for (ANEGraphFunction *function in module.functions) {
        if (![function.opset isEqualToString:expectedOpset]) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                code:@"ane.verify.unsupported-opset"
                message:[NSString stringWithFormat:
                    @"unsupported opset '%@' for program version %@",
                    function.opset, module.version]
                range:ANESourceRangeMake(ANESourceLocationMake(0, 1, 1),
                                         ANESourceLocationMake(0, 1, 1))];
            return NO;
        }
        for (ANEGraphOperation *operation in function.operations)
            if (![self verifyOperation:operation diagnostics:diagnostics])
                return NO;
    }
    return YES;
}

@end
