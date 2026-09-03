#import "MILGraphImporter.h"

@implementation MILGraphImporter

+ (ANEElementType)elementTypeForName:(NSString *)name {
    if ([name isEqualToString:@"fp16"]) return ANEElementTypeFP16;
    if ([name isEqualToString:@"fp32"]) return ANEElementTypeFP32;
    if ([name isEqualToString:@"int8"]) return ANEElementTypeInt8;
    if ([name isEqualToString:@"int32"]) return ANEElementTypeInt32;
    if ([name isEqualToString:@"uint64"]) return ANEElementTypeUInt64;
    if ([name isEqualToString:@"bool"]) return ANEElementTypeBool;
    if ([name isEqualToString:@"string"]) return ANEElementTypeString;
    return ANEElementTypeInvalid;
}

+ (ANEValueType *)importType:(MILTypeSyntax *)type
                 diagnostics:(ANEDiagnosticEngine *)diagnostics
                       range:(ANESourceRange)range {
    BOOL tensor = [type.name isEqualToString:@"tensor"];
    NSString *elementName = tensor && type.typeArguments.count
        ? type.typeArguments[0].name : type.name;
    ANEElementType elementType = [self elementTypeForName:elementName];
    if (elementType == ANEElementTypeInvalid) {
        [diagnostics emitSeverity:ANEDiagnosticSeverityError
                            code:@"mil.import.unsupported-type"
                         message:[NSString stringWithFormat:
                            @"unsupported MIL type '%@'", type.name]
                           range:range];
        return nil;
    }
    return [[ANEValueType alloc]
        initWithKind:tensor ? ANEValueTypeKindTensor : ANEValueTypeKindScalar
        elementType:elementType shape:tensor ? type.dimensions : @[]];
}

+ (ANEGraphArgument *)importExpression:(MILExpressionSyntax *)expression
                                symbols:(NSDictionary<NSString *, ANEGraphValue *> *)symbols
                            diagnostics:(ANEDiagnosticEngine *)diagnostics {
    if (expression.kind == MILExpressionKindIdentifier) {
        if ([expression.atom isEqualToString:@"true"] ||
            [expression.atom isEqualToString:@"false"]) {
            return [[ANEGraphArgument alloc]
                initWithKind:ANEGraphArgumentKindBoolean text:expression.atom
                value:nil calleeName:nil calleeValueType:nil
                callArguments:@[] elements:@[]
                range:expression.range];
        }
        ANEGraphValue *value = symbols[expression.atom];
        if (!value) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"mil.import.unknown-value"
                             message:[NSString stringWithFormat:
                                @"unknown SSA value '%@'", expression.atom]
                               range:expression.range];
            return nil;
        }
        return [[ANEGraphArgument alloc] initWithKind:ANEGraphArgumentKindValue
            text:nil value:value calleeName:nil calleeValueType:nil
            callArguments:@[] elements:@[]
            range:expression.range];
    }
    ANEGraphArgumentKind kind;
    switch (expression.kind) {
        case MILExpressionKindInteger: kind = ANEGraphArgumentKindInteger; break;
        case MILExpressionKindFloatingPoint:
            kind = ANEGraphArgumentKindFloatingPoint; break;
        case MILExpressionKindString: kind = ANEGraphArgumentKindString; break;
        case MILExpressionKindList: kind = ANEGraphArgumentKindList; break;
        case MILExpressionKindTuple: kind = ANEGraphArgumentKindTuple; break;
        case MILExpressionKindDictionary:
            kind = ANEGraphArgumentKindDictionary; break;
        case MILExpressionKindCall: kind = ANEGraphArgumentKindCall; break;
        case MILExpressionKindIdentifier:
            __builtin_unreachable();
    }
    NSMutableArray<ANEGraphArgument *> *elements = [NSMutableArray array];
    for (MILExpressionSyntax *element in expression.elements) {
        ANEGraphArgument *imported = [self importExpression:element
            symbols:symbols diagnostics:diagnostics];
        if (!imported) return nil;
        [elements addObject:imported];
    }
    NSMutableArray<ANEGraphNamedArgument *> *arguments = [NSMutableArray array];
    for (MILArgumentSyntax *argument in expression.arguments) {
        ANEGraphArgument *imported = [self importExpression:argument.value
            symbols:symbols diagnostics:diagnostics];
        if (!imported) return nil;
        [arguments addObject:[[ANEGraphNamedArgument alloc]
            initWithName:argument.name value:imported]];
    }
    NSString *callee = expression.calleeName ?: expression.calleeType.name;
    ANEValueType *calleeValueType = expression.calleeType
        ? [self importType:expression.calleeType diagnostics:diagnostics
                    range:expression.range] : nil;
    if (expression.calleeType && !calleeValueType) return nil;
    return [[ANEGraphArgument alloc] initWithKind:kind text:expression.atom
        value:nil calleeName:callee calleeValueType:calleeValueType
        callArguments:arguments elements:elements
        range:expression.range];
}

+ (NSDictionary<NSString *, ANEGraphArgument *> *)
    importArguments:(NSArray<MILArgumentSyntax *> *)arguments
             symbols:(NSDictionary<NSString *, ANEGraphValue *> *)symbols
         diagnostics:(ANEDiagnosticEngine *)diagnostics
               range:(ANESourceRange)range {
    NSMutableDictionary *result = [NSMutableDictionary dictionary];
    for (MILArgumentSyntax *argument in arguments) {
        if (!argument.name) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"mil.import.positional-operation-argument"
                             message:@"operation arguments must be named"
                               range:range];
            return nil;
        }
        if (result[argument.name]) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"mil.import.duplicate-argument"
                             message:[NSString stringWithFormat:
                                @"duplicate argument '%@'", argument.name]
                               range:range];
            return nil;
        }
        ANEGraphArgument *value = [self importExpression:argument.value
            symbols:symbols diagnostics:diagnostics];
        if (!value) return nil;
        result[argument.name] = value;
    }
    return result;
}

+ (ANEGraphFunction *)importFunction:(MILFunctionSyntax *)function
                          diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSMutableDictionary<NSString *, ANEGraphValue *> *symbols =
        [NSMutableDictionary dictionary];
    NSMutableArray<ANEGraphValue *> *inputs = [NSMutableArray array];
    for (MILParameterSyntax *parameter in function.parameters) {
        if (symbols[parameter.name]) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"mil.import.duplicate-value"
                             message:[NSString stringWithFormat:
                                @"duplicate SSA value '%@'", parameter.name]
                               range:function.range];
            return nil;
        }
        ANEValueType *type = [self importType:parameter.type
            diagnostics:diagnostics range:function.range];
        if (!type) return nil;
        ANEGraphValue *value = [[ANEGraphValue alloc]
            initWithName:parameter.name type:type];
        symbols[value.name] = value;
        [inputs addObject:value];
    }

    NSMutableArray<ANEGraphOperation *> *operations = [NSMutableArray array];
    for (MILOperationSyntax *operation in function.operations) {
        if (symbols[operation.resultName]) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"mil.import.duplicate-value"
                             message:[NSString stringWithFormat:
                                @"duplicate SSA value '%@'", operation.resultName]
                               range:operation.range];
            return nil;
        }
        NSDictionary *arguments = [self importArguments:operation.arguments
            symbols:symbols diagnostics:diagnostics range:operation.range];
        NSDictionary *attributes = [self importArguments:operation.attributes
            symbols:symbols diagnostics:diagnostics range:operation.range];
        ANEValueType *type = [self importType:operation.resultType
            diagnostics:diagnostics range:operation.range];
        if (!arguments || !attributes || !type) return nil;
        ANEGraphValue *result = [[ANEGraphValue alloc]
            initWithName:operation.resultName type:type];
        ANEGraphOperation *imported = [[ANEGraphOperation alloc]
            initWithOperationName:operation.operationName result:result
            arguments:arguments attributes:attributes range:operation.range];
        symbols[result.name] = result;
        [operations addObject:imported];
    }

    NSMutableArray<ANEGraphValue *> *returns = [NSMutableArray array];
    for (NSString *name in function.returnNames) {
        ANEGraphValue *value = symbols[name];
        if (!value) {
            [diagnostics emitSeverity:ANEDiagnosticSeverityError
                                code:@"mil.import.unknown-return"
                             message:[NSString stringWithFormat:
                                @"unknown return value '%@'", name]
                               range:function.range];
            return nil;
        }
        [returns addObject:value];
    }
    return [[ANEGraphFunction alloc] initWithName:function.name inputs:inputs
        operations:operations returnValues:returns];
}

+ (ANEGraphModule *)importProgram:(MILProgramSyntax *)program
                       diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSMutableArray<ANEGraphFunction *> *functions = [NSMutableArray array];
    for (MILFunctionSyntax *function in program.functions) {
        ANEGraphFunction *imported = [self importFunction:function
                                                  diagnostics:diagnostics];
        if (!imported) return nil;
        [functions addObject:imported];
    }
    return [[ANEGraphModule alloc] initWithFunctions:functions];
}

@end
