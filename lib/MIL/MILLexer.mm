#import "MILLexer.h"

@implementation MILToken

- (instancetype)initWithKind:(MILTokenKind)kind
                     spelling:(NSString *)spelling
                        range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _kind = kind;
        _spelling = [spelling copy];
        _range = range;
    }
    return self;
}

@end


@interface MILLexer () {
    NSData *_data;
    const uint8_t *_bytes;
    NSUInteger _length;
    NSUInteger _offset;
    NSUInteger _line;
    NSUInteger _column;
    ANEDiagnosticEngine *_diagnostics;
}
@end


@implementation MILLexer

- (instancetype)initWithData:(NSData *)data
                  diagnostics:(ANEDiagnosticEngine *)diagnostics {
    NSParameterAssert(data != nil);
    NSParameterAssert(diagnostics != nil);
    self = [super init];
    if (self) {
        _data = [data copy];
        _bytes = (const uint8_t *)_data.bytes;
        _length = _data.length;
        _line = 1;
        _column = 1;
        _diagnostics = diagnostics;
    }
    return self;
}

- (BOOL)isAtEnd {
    return _offset >= _length;
}

- (uint8_t)peek:(NSUInteger)distance {
    NSUInteger index = _offset + distance;
    return index < _length ? _bytes[index] : 0;
}

- (ANESourceLocation)location {
    return ANESourceLocationMake(_offset, _line, _column);
}

- (uint8_t)advance {
    if ([self isAtEnd]) return 0;
    uint8_t byte = _bytes[_offset++];
    if (byte == '\n') {
        _line++;
        _column = 1;
    } else {
        _column++;
    }
    return byte;
}

- (NSString *)sourceFrom:(NSUInteger)start to:(NSUInteger)end {
    NSData *slice = [_data subdataWithRange:NSMakeRange(start, end - start)];
    NSString *string = [[NSString alloc] initWithData:slice
                                             encoding:NSUTF8StringEncoding];
    return string ?: @"";
}

- (MILToken *)token:(MILTokenKind)kind
            spelling:(NSString *)spelling
               start:(ANESourceLocation)start {
    return [[MILToken alloc] initWithKind:kind spelling:spelling
                                    range:ANESourceRangeMake(start,
                                                            [self location])];
}

- (void)skipTrivia {
    while (![self isAtEnd]) {
        uint8_t c = [self peek:0];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            [self advance];
            continue;
        }
        if (c == '/' && [self peek:1] == '/') {
            [self advance];
            [self advance];
            while (![self isAtEnd] && [self peek:0] != '\n') [self advance];
            continue;
        }
        if (c == '/' && [self peek:1] == '*') {
            ANESourceLocation start = [self location];
            [self advance];
            [self advance];
            BOOL terminated = NO;
            while (![self isAtEnd]) {
                if ([self peek:0] == '*' && [self peek:1] == '/') {
                    [self advance];
                    [self advance];
                    terminated = YES;
                    break;
                }
                [self advance];
            }
            if (!terminated) {
                [_diagnostics emitSeverity:ANEDiagnosticSeverityError
                                      code:@"mil.lex.unterminated-comment"
                                   message:@"unterminated block comment"
                                     range:ANESourceRangeMake(start,
                                                              [self location])];
            }
            continue;
        }
        break;
    }
}

static BOOL isIdentifierStart(uint8_t c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

static BOOL isIdentifierContinue(uint8_t c) {
    return isIdentifierStart(c) || (c >= '0' && c <= '9');
}

static BOOL isDecimalDigit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static BOOL isHexDigit(uint8_t c) {
    return isDecimalDigit(c) || (c >= 'a' && c <= 'f') ||
           (c >= 'A' && c <= 'F');
}

- (MILToken *)lexIdentifier {
    ANESourceLocation start = [self location];
    NSUInteger begin = _offset;
    [self advance];
    while (isIdentifierContinue([self peek:0])) [self advance];
    return [self token:MILTokenKindIdentifier
              spelling:[self sourceFrom:begin to:_offset]
                 start:start];
}

- (MILToken *)lexNumber {
    ANESourceLocation start = [self location];
    NSUInteger begin = _offset;
    if ([self peek:0] == '-') [self advance];
    BOOL floating = NO;
    if ([self peek:0] == '0' &&
        ([self peek:1] == 'x' || [self peek:1] == 'X')) {
        [self advance];
        [self advance];
        while (isHexDigit([self peek:0])) [self advance];
        if ([self peek:0] == '.') {
            floating = YES;
            [self advance];
            while (isHexDigit([self peek:0])) [self advance];
        }
        if ([self peek:0] == 'p' || [self peek:0] == 'P') {
            floating = YES;
            [self advance];
            if ([self peek:0] == '+' || [self peek:0] == '-') [self advance];
            while (isDecimalDigit([self peek:0])) [self advance];
        }
    } else {
        while (isDecimalDigit([self peek:0])) [self advance];
        if ([self peek:0] == '.') {
            floating = YES;
            [self advance];
            while (isDecimalDigit([self peek:0])) [self advance];
        }
        if ([self peek:0] == 'e' || [self peek:0] == 'E') {
            floating = YES;
            [self advance];
            if ([self peek:0] == '+' || [self peek:0] == '-') [self advance];
            while (isDecimalDigit([self peek:0])) [self advance];
        }
    }
    return [self token:floating ? MILTokenKindFloatingPoint
                                 : MILTokenKindInteger
              spelling:[self sourceFrom:begin to:_offset]
                 start:start];
}

- (MILToken *)lexString {
    ANESourceLocation start = [self location];
    [self advance];
    NSMutableString *decoded = [NSMutableString string];
    BOOL terminated = NO;
    while (![self isAtEnd]) {
        uint8_t c = [self advance];
        if (c == '"') {
            terminated = YES;
            break;
        }
        if (c == '\n') break;
        if (c == '\\' && ![self isAtEnd]) {
            uint8_t escaped = [self advance];
            switch (escaped) {
                case 'n': [decoded appendString:@"\n"]; break;
                case 'r': [decoded appendString:@"\r"]; break;
                case 't': [decoded appendString:@"\t"]; break;
                case '\\': [decoded appendString:@"\\"]; break;
                case '"': [decoded appendString:@"\""]; break;
                default: [decoded appendFormat:@"%c", escaped]; break;
            }
        } else {
            [decoded appendFormat:@"%c", c];
        }
    }
    if (!terminated) {
        [_diagnostics emitSeverity:ANEDiagnosticSeverityError
                              code:@"mil.lex.unterminated-string"
                           message:@"unterminated string literal"
                             range:ANESourceRangeMake(start, [self location])];
    }
    return [self token:MILTokenKindString spelling:decoded start:start];
}

- (NSArray<MILToken *> *)lexAllTokens {
    NSMutableArray<MILToken *> *tokens = [NSMutableArray array];
    while (YES) {
        [self skipTrivia];
        ANESourceLocation start = [self location];
        if ([self isAtEnd]) {
            [tokens addObject:[self token:MILTokenKindEndOfFile
                                      spelling:@"" start:start]];
            break;
        }
        uint8_t c = [self peek:0];
        if (isIdentifierStart(c)) {
            [tokens addObject:[self lexIdentifier]];
            continue;
        }
        if (isDecimalDigit(c) ||
            (c == '-' && isDecimalDigit([self peek:1]))) {
            [tokens addObject:[self lexNumber]];
            continue;
        }
        if (c == '"') {
            [tokens addObject:[self lexString]];
            continue;
        }
        MILTokenKind kind;
        NSString *spelling = nil;
        if (c == '-' && [self peek:1] == '>') {
            [self advance];
            [self advance];
            kind = MILTokenKindArrow;
            spelling = @"->";
        } else {
            [self advance];
            switch (c) {
                case '(': kind = MILTokenKindLParen; spelling = @"("; break;
                case ')': kind = MILTokenKindRParen; spelling = @")"; break;
                case '{': kind = MILTokenKindLBrace; spelling = @"{"; break;
                case '}': kind = MILTokenKindRBrace; spelling = @"}"; break;
                case '[': kind = MILTokenKindLBracket; spelling = @"["; break;
                case ']': kind = MILTokenKindRBracket; spelling = @"]"; break;
                case '<': kind = MILTokenKindLess; spelling = @"<"; break;
                case '>': kind = MILTokenKindGreater; spelling = @">"; break;
                case ',': kind = MILTokenKindComma; spelling = @","; break;
                case ':': kind = MILTokenKindColon; spelling = @":"; break;
                case '=': kind = MILTokenKindEqual; spelling = @"="; break;
                case ';': kind = MILTokenKindSemicolon; spelling = @";"; break;
                default: {
                    NSString *message = [NSString stringWithFormat:
                        @"unexpected byte 0x%02x", c];
                    [_diagnostics emitSeverity:ANEDiagnosticSeverityError
                                          code:@"mil.lex.unexpected-character"
                                       message:message
                                         range:ANESourceRangeMake(
                                             start, [self location])];
                    continue;
                }
            }
        }
        [tokens addObject:[self token:kind spelling:spelling start:start]];
    }
    return tokens;
}

@end

