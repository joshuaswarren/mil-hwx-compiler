#import "MILPrinter.h"

@implementation MILPrinter

+ (NSString *)typeString:(MILTypeSyntax *)type {
    if ([type.name isEqualToString:@"tensor"]) {
        NSMutableArray<NSString *> *dimensions = [NSMutableArray array];
        for (NSNumber *dimension in type.dimensions)
            [dimensions addObject:dimension.stringValue];
        return [NSString stringWithFormat:@"tensor<%@, [%@]>",
            [self typeString:type.typeArguments[0]],
            [dimensions componentsJoinedByString:@", "]];
    }
    if (type.typeArguments.count == 0) return type.name;
    NSMutableArray<NSString *> *arguments = [NSMutableArray array];
    for (MILTypeSyntax *argument in type.typeArguments)
        [arguments addObject:[self typeString:argument]];
    return [NSString stringWithFormat:@"%@<%@>", type.name,
            [arguments componentsJoinedByString:@", "]];
}

+ (NSString *)expressionString:(MILExpressionSyntax *)expression {
    switch (expression.kind) {
        case MILExpressionKindIdentifier:
        case MILExpressionKindInteger:
        case MILExpressionKindFloatingPoint:
            return expression.atom;
        case MILExpressionKindString: {
            NSString *escaped = [expression.atom
                stringByReplacingOccurrencesOfString:@"\\" withString:@"\\\\"];
            escaped = [escaped stringByReplacingOccurrencesOfString:@"\""
                                                           withString:@"\\\""];
            return [NSString stringWithFormat:@"\"%@\"", escaped];
        }
        case MILExpressionKindList:
        case MILExpressionKindTuple:
        case MILExpressionKindDictionary: {
            NSMutableArray<NSString *> *elements = [NSMutableArray array];
            for (MILExpressionSyntax *element in expression.elements)
                [elements addObject:[self expressionString:element]];
            NSString *open = expression.kind == MILExpressionKindList
                ? @"[" : (expression.kind == MILExpressionKindTuple ? @"(" : @"{");
            NSString *close = expression.kind == MILExpressionKindList
                ? @"]" : (expression.kind == MILExpressionKindTuple ? @")" : @"}");
            return [NSString stringWithFormat:@"%@%@%@", open,
                [elements componentsJoinedByString:@", "], close];
        }
        case MILExpressionKindCall: {
            NSString *callee = expression.calleeType
                ? [self typeString:expression.calleeType]
                : expression.calleeName;
            return [NSString stringWithFormat:@"%@(%@)", callee,
                [self argumentString:expression.arguments]];
        }
    }
}

+ (NSString *)argumentString:(NSArray<MILArgumentSyntax *> *)arguments {
    NSMutableArray<NSString *> *parts = [NSMutableArray array];
    for (MILArgumentSyntax *argument in arguments) {
        NSString *value = [self expressionString:argument.value];
        [parts addObject:argument.name
            ? [NSString stringWithFormat:@"%@ = %@", argument.name, value]
            : value];
    }
    return [parts componentsJoinedByString:@", "];
}

+ (NSString *)stringForProgram:(MILProgramSyntax *)program {
    NSMutableString *result = [NSMutableString stringWithFormat:
        @"program(%@)\n", program.version];
    if (program.attributes.count)
        [result appendFormat:@"[%@]\n", [self argumentString:program.attributes]];
    [result appendString:@"{\n"];
    for (MILFunctionSyntax *function in program.functions) {
        NSMutableArray<NSString *> *parameters = [NSMutableArray array];
        for (MILParameterSyntax *parameter in function.parameters)
            [parameters addObject:[NSString stringWithFormat:@"%@ %@",
                [self typeString:parameter.type], parameter.name]];
        [result appendFormat:@"  func %@<%@>(%@) {\n", function.name,
            function.opset, [parameters componentsJoinedByString:@", "]];
        for (MILOperationSyntax *operation in function.operations) {
            [result appendFormat:@"    %@ %@ = %@(%@)",
                [self typeString:operation.resultType], operation.resultName,
                operation.operationName,
                [self argumentString:operation.arguments]];
            if (operation.attributes.count)
                [result appendFormat:@"[%@]",
                    [self argumentString:operation.attributes]];
            [result appendString:@";\n"];
        }
        [result appendFormat:@"  } -> (%@);\n",
            [function.returnNames componentsJoinedByString:@", "]];
    }
    [result appendString:@"}\n"];
    return result;
}

@end

