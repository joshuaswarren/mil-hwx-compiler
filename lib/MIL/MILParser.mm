#import "MILParser.h"

@interface MILParser () {
    NSArray<MILToken *> *_tokens;
    NSUInteger _index;
    ANEDiagnosticEngine *_diagnostics;
    BOOL _failed;
}
@end

@implementation MILParser

- (instancetype)initWithTokens:(NSArray<MILToken *> *)tokens
                    diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSParameterAssert(tokens.count > 0);
    NSParameterAssert(diagnostics != nil);
    self = [super init];
    if (self) {
        _tokens = [tokens copy];
        _diagnostics = diagnostics;
    }
    return self;
}

- (MILToken *)current { return _tokens[MIN(_index, _tokens.count - 1)]; }
- (MILToken *)lookahead:(NSUInteger)distance {
    return _tokens[MIN(_index + distance, _tokens.count - 1)];
}
- (BOOL)at:(MILTokenKind)kind { return self.current.kind == kind; }
- (MILToken *)take { MILToken *token = self.current; if (_index < _tokens.count) _index++; return token; }

- (void)error:(NSString *)code message:(NSString *)message token:(MILToken *)token {
    if (_failed) return;
    _failed = YES;
    [_diagnostics emitSeverity:ANEDiagnosticSeverityError
                          code:code message:message range:token.range];
}

- (MILToken *)expect:(MILTokenKind)kind label:(NSString *)label {
    if ([self at:kind]) return [self take];
    [self error:@"mil.parse.expected-token"
        message:[NSString stringWithFormat:@"expected %@, found '%@'",
                                           label, self.current.spelling]
          token:self.current];
    return nil;
}

- (MILToken *)expectIdentifier:(NSString *)spelling {
    MILToken *token = [self expect:MILTokenKindIdentifier label:spelling];
    if (token && spelling && ![token.spelling isEqualToString:spelling]) {
        [self error:@"mil.parse.expected-keyword"
            message:[NSString stringWithFormat:@"expected '%@', found '%@'",
                                               spelling, token.spelling]
              token:token];
        return nil;
    }
    return token;
}

- (MILTypeSyntax *)parseType {
    MILToken *name = [self expect:MILTokenKindIdentifier label:@"type"];
    if (!name) return nil;
    NSMutableArray<MILTypeSyntax *> *arguments = [NSMutableArray array];
    NSMutableArray<NSNumber *> *dimensions = [NSMutableArray array];
    if ([self at:MILTokenKindLess]) {
        [self take];
        MILTypeSyntax *first = [self parseType];
        if (!first) return nil;
        [arguments addObject:first];
        if ([name.spelling isEqualToString:@"tensor"]) {
            if (![self expect:MILTokenKindComma label:@"','"] ||
                ![self expect:MILTokenKindLBracket label:@"'['"])
                return nil;
            while (![self at:MILTokenKindRBracket] && !_failed) {
                MILToken *dimension = [self expect:MILTokenKindInteger
                                             label:@"tensor dimension"];
                if (!dimension) return nil;
                NSInteger value = dimension.spelling.integerValue;
                if (value < 0) {
                    [self error:@"mil.parse.negative-dimension"
                        message:@"tensor dimensions must be non-negative"
                          token:dimension];
                    return nil;
                }
                [dimensions addObject:@(value)];
                if (![self at:MILTokenKindComma]) break;
                [self take];
            }
            if (![self expect:MILTokenKindRBracket label:@"']'"]) return nil;
        } else {
            while ([self at:MILTokenKindComma]) {
                [self take];
                MILTypeSyntax *argument = [self parseType];
                if (!argument) return nil;
                [arguments addObject:argument];
            }
        }
        if (![self expect:MILTokenKindGreater label:@"'>'"]) return nil;
    }
    return [[MILTypeSyntax alloc] initWithName:name.spelling
                                  typeArguments:arguments
                                     dimensions:dimensions];
}

- (NSArray<MILExpressionSyntax *> *)parseElementsUntil:(MILTokenKind)closing {
    NSMutableArray<MILExpressionSyntax *> *elements = [NSMutableArray array];
    while (![self at:closing] && !_failed) {
        MILExpressionSyntax *element = [self parseExpression];
        if (!element) return nil;
        [elements addObject:element];
        if (![self at:MILTokenKindComma]) break;
        [self take];
    }
    return elements;
}

- (NSArray<MILArgumentSyntax *> *)parseArgumentsUntil:(MILTokenKind)closing {
    NSMutableArray<MILArgumentSyntax *> *arguments = [NSMutableArray array];
    while (![self at:closing] && !_failed) {
        NSString *name = nil;
        if ([self at:MILTokenKindIdentifier] &&
            [self lookahead:1].kind == MILTokenKindEqual) {
            name = [self take].spelling;
            [self take];
        }
        MILExpressionSyntax *value = [self parseExpression];
        if (!value) return nil;
        [arguments addObject:[[MILArgumentSyntax alloc] initWithName:name
                                                                value:value]];
        if (![self at:MILTokenKindComma]) break;
        [self take];
    }
    return arguments;
}

- (MILExpressionSyntax *)parseExpression {
    MILToken *startToken = self.current;
    if ([self at:MILTokenKindInteger] ||
        [self at:MILTokenKindFloatingPoint] ||
        [self at:MILTokenKindString]) {
        MILToken *token = [self take];
        MILExpressionKind kind = token.kind == MILTokenKindInteger
            ? MILExpressionKindInteger
            : (token.kind == MILTokenKindFloatingPoint
                ? MILExpressionKindFloatingPoint : MILExpressionKindString);
        return [[MILExpressionSyntax alloc] initWithKind:kind atom:token.spelling
            calleeType:nil calleeName:nil arguments:@[] elements:@[]
            range:token.range];
    }
    MILTokenKind opening = self.current.kind;
    if (opening == MILTokenKindLBracket || opening == MILTokenKindLParen ||
        opening == MILTokenKindLBrace) {
        [self take];
        MILTokenKind closing = opening == MILTokenKindLBracket
            ? MILTokenKindRBracket : (opening == MILTokenKindLParen
                ? MILTokenKindRParen : MILTokenKindRBrace);
        NSArray<MILExpressionSyntax *> *elements =
            [self parseElementsUntil:closing];
        MILToken *end = [self expect:closing label:@"closing delimiter"];
        if (!elements || !end) return nil;
        MILExpressionKind kind = opening == MILTokenKindLBracket
            ? MILExpressionKindList : (opening == MILTokenKindLParen
                ? MILExpressionKindTuple : MILExpressionKindDictionary);
        return [[MILExpressionSyntax alloc] initWithKind:kind atom:nil
            calleeType:nil calleeName:nil arguments:@[] elements:elements
            range:ANESourceRangeMake(startToken.range.start, end.range.end)];
    }
    if (![self at:MILTokenKindIdentifier]) {
        [self error:@"mil.parse.expected-expression"
            message:[NSString stringWithFormat:@"expected expression, found '%@'",
                                               self.current.spelling]
              token:self.current];
        return nil;
    }

    NSUInteger savedIndex = _index;
    MILTypeSyntax *possibleType = [self parseType];
    if (!possibleType) return nil;
    if (![self at:MILTokenKindLParen]) {
        if (possibleType.typeArguments.count || possibleType.dimensions.count) {
            [self error:@"mil.parse.expected-call"
                message:@"parameterized expression must be called"
                  token:self.current];
            return nil;
        }
        return [[MILExpressionSyntax alloc]
            initWithKind:MILExpressionKindIdentifier atom:possibleType.name
            calleeType:nil calleeName:nil arguments:@[] elements:@[]
            range:ANESourceRangeMake(startToken.range.start,
                                     _tokens[_index - 1].range.end)];
    }
    [self take];
    NSArray<MILArgumentSyntax *> *arguments =
        [self parseArgumentsUntil:MILTokenKindRParen];
    MILToken *end = [self expect:MILTokenKindRParen label:@"')'"];
    if (!arguments || !end) return nil;
    BOOL typedCallee = possibleType.typeArguments.count > 0 ||
                       possibleType.dimensions.count > 0;
    if (!typedCallee && savedIndex + 1 == _index) typedCallee = NO;
    return [[MILExpressionSyntax alloc] initWithKind:MILExpressionKindCall
        atom:nil calleeType:typedCallee ? possibleType : nil
        calleeName:typedCallee ? nil : possibleType.name
        arguments:arguments elements:@[]
        range:ANESourceRangeMake(startToken.range.start, end.range.end)];
}

- (NSArray<MILArgumentSyntax *> *)parseBracketedArguments {
    if (![self expect:MILTokenKindLBracket label:@"'['"]) return nil;
    NSArray<MILArgumentSyntax *> *arguments =
        [self parseArgumentsUntil:MILTokenKindRBracket];
    if (![self expect:MILTokenKindRBracket label:@"']'"]) return nil;
    return arguments;
}

- (MILOperationSyntax *)parseOperation {
    MILToken *start = self.current;
    MILTypeSyntax *type = [self parseType];
    MILToken *result = [self expect:MILTokenKindIdentifier label:@"result name"];
    if (!type || !result || ![self expect:MILTokenKindEqual label:@"'='"])
        return nil;
    MILToken *operation = [self expect:MILTokenKindIdentifier
                                 label:@"operation name"];
    if (!operation || ![self expect:MILTokenKindLParen label:@"'('"])
        return nil;
    NSArray<MILArgumentSyntax *> *arguments =
        [self parseArgumentsUntil:MILTokenKindRParen];
    if (![self expect:MILTokenKindRParen label:@"')'"]) return nil;
    NSArray<MILArgumentSyntax *> *attributes = @[];
    if ([self at:MILTokenKindLBracket]) {
        attributes = [self parseBracketedArguments];
        if (!attributes) return nil;
    }
    MILToken *semicolon = [self expect:MILTokenKindSemicolon label:@"';'"];
    if (!semicolon) return nil;
    return [[MILOperationSyntax alloc] initWithResultType:type
        resultName:result.spelling operationName:operation.spelling
        arguments:arguments attributes:attributes
        range:ANESourceRangeMake(start.range.start, semicolon.range.end)];
}

- (MILFunctionSyntax *)parseFunction {
    MILToken *start = [self expectIdentifier:@"func"];
    MILToken *name = [self expect:MILTokenKindIdentifier label:@"function name"];
    if (!start || !name || ![self expect:MILTokenKindLess label:@"'<'"])
        return nil;
    MILToken *opset = [self expect:MILTokenKindIdentifier label:@"opset"];
    if (!opset || ![self expect:MILTokenKindGreater label:@"'>'"] ||
        ![self expect:MILTokenKindLParen label:@"'('"])
        return nil;
    NSMutableArray<MILParameterSyntax *> *parameters = [NSMutableArray array];
    while (![self at:MILTokenKindRParen] && !_failed) {
        MILTypeSyntax *type = [self parseType];
        MILToken *parameterName = [self expect:MILTokenKindIdentifier
                                         label:@"parameter name"];
        if (!type || !parameterName) return nil;
        [parameters addObject:[[MILParameterSyntax alloc]
            initWithType:type name:parameterName.spelling]];
        if (![self at:MILTokenKindComma]) break;
        [self take];
    }
    if (![self expect:MILTokenKindRParen label:@"')'"] ||
        ![self expect:MILTokenKindLBrace label:@"'{'"])
        return nil;
    NSMutableArray<MILOperationSyntax *> *operations = [NSMutableArray array];
    while (![self at:MILTokenKindRBrace] && !_failed) {
        MILOperationSyntax *operation = [self parseOperation];
        if (!operation) return nil;
        [operations addObject:operation];
    }
    if (![self expect:MILTokenKindRBrace label:@"'}'"] ||
        ![self expect:MILTokenKindArrow label:@"'->'"] ||
        ![self expect:MILTokenKindLParen label:@"'('"])
        return nil;
    NSMutableArray<NSString *> *returns = [NSMutableArray array];
    while (![self at:MILTokenKindRParen] && !_failed) {
        MILToken *value = [self expect:MILTokenKindIdentifier
                                 label:@"return value"];
        if (!value) return nil;
        [returns addObject:value.spelling];
        if (![self at:MILTokenKindComma]) break;
        [self take];
    }
    if (![self expect:MILTokenKindRParen label:@"')'"]) return nil;
    MILToken *semicolon = [self expect:MILTokenKindSemicolon label:@"';'"];
    if (!semicolon) return nil;
    return [[MILFunctionSyntax alloc] initWithName:name.spelling
        opset:opset.spelling parameters:parameters operations:operations
        returnNames:returns
        range:ANESourceRangeMake(start.range.start, semicolon.range.end)];
}

- (MILProgramSyntax *)parseProgram {
    MILToken *start = [self expectIdentifier:@"program"];
    if (!start || ![self expect:MILTokenKindLParen label:@"'('"])
        return nil;
    MILToken *version = self.current;
    if (version.kind != MILTokenKindFloatingPoint &&
        version.kind != MILTokenKindInteger) {
        [self error:@"mil.parse.expected-version"
            message:@"expected program version" token:version];
        return nil;
    }
    [self take];
    if (![self expect:MILTokenKindRParen label:@"')'"]) return nil;
    NSArray<MILArgumentSyntax *> *attributes = @[];
    if ([self at:MILTokenKindLBracket]) {
        attributes = [self parseBracketedArguments];
        if (!attributes) return nil;
    }
    if (![self expect:MILTokenKindLBrace label:@"'{'"]) return nil;
    NSMutableArray<MILFunctionSyntax *> *functions = [NSMutableArray array];
    while (![self at:MILTokenKindRBrace] && !_failed) {
        MILFunctionSyntax *function = [self parseFunction];
        if (!function) return nil;
        [functions addObject:function];
    }
    MILToken *end = [self expect:MILTokenKindRBrace label:@"'}'"];
    if (!end || ![self at:MILTokenKindEndOfFile]) {
        if (!_failed)
            [self error:@"mil.parse.trailing-input"
                message:@"unexpected input after program" token:self.current];
        return nil;
    }
    return [[MILProgramSyntax alloc] initWithVersion:version.spelling
        attributes:attributes functions:functions
        range:ANESourceRangeMake(start.range.start, end.range.end)];
}

@end

