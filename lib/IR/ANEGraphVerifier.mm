#import "ANEGraphVerifier.h"

@implementation ANEGraphVerifier

+ (void)emit:(ANEDiagnosticEngine *)diagnostics
        code:(NSString *)code
     message:(NSString *)message
   operation:(ANEGraphOperation *)operation {
    [diagnostics emitSeverity:ANEDiagnosticSeverityError code:code
                      message:message range:operation.range];
}

+ (BOOL)verifyUnarySameType:(ANEGraphOperation *)operation
                 diagnostics:(ANEDiagnosticEngine *)diagnostics
                        code:(NSString *)code {
    ANEGraphArgument *input = operation.operands[@"x"];
    if (!input || ![operation.result.type isEqualToValueType:input.value.type]) {
        [self emit:diagnostics code:code
           message:[NSString stringWithFormat:
              @"%@ input and result types must match", operation.operationName]
         operation:operation];
        return NO;
    }
    return YES;
}

+ (BOOL)verifyOperation:(ANEGraphOperation *)operation
             diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (operation.result.producer != operation) {
        [self emit:diagnostics code:@"ane.verify.bad-producer"
            message:@"result does not point to its defining operation"
          operation:operation];
        return NO;
    }
    NSString *name = operation.operationName;
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
            ![operation.result.type isEqualToValueType:input.value.type]) {
            [self emit:diagnostics code:@"ane.verify.softmax-type"
                message:@"softmax input and result types must match"
              operation:operation];
            return NO;
        }
        if ([name isEqualToString:@"quantize"] &&
            (input.value.type.elementType != ANEElementTypeFP16 ||
             operation.result.type.elementType != ANEElementTypeInt8)) {
            [self emit:diagnostics code:@"ane.verify.quantize-type"
                message:@"initial quantize path requires fp16 to int8"
              operation:operation];
            return NO;
        }
        if ([name isEqualToString:@"dequantize"] &&
            (input.value.type.elementType != ANEElementTypeInt8 ||
             operation.result.type.elementType != ANEElementTypeFP16)) {
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
            operation.result.type.elementType != ANEElementTypeFP16) {
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
    if (module.functions.count == 0) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
                            code:@"ane.verify.empty-module"
                         message:@"module has no functions"
                           range:ANESourceRangeMake(
                               ANESourceLocationMake(0, 1, 1),
                               ANESourceLocationMake(0, 1, 1))];
        return NO;
    }
    for (ANEGraphFunction *function in module.functions)
        for (ANEGraphOperation *operation in function.operations)
            if (![self verifyOperation:operation diagnostics:diagnostics])
                return NO;
    return YES;
}

@end

