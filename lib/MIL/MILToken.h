#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"

typedef NS_ENUM(NSUInteger, MILTokenKind) {
    MILTokenKindEndOfFile,
    MILTokenKindIdentifier,
    MILTokenKindInteger,
    MILTokenKindFloatingPoint,
    MILTokenKindString,
    MILTokenKindLParen,
    MILTokenKindRParen,
    MILTokenKindLBrace,
    MILTokenKindRBrace,
    MILTokenKindLBracket,
    MILTokenKindRBracket,
    MILTokenKindLess,
    MILTokenKindGreater,
    MILTokenKindComma,
    MILTokenKindColon,
    MILTokenKindEqual,
    MILTokenKindSemicolon,
    MILTokenKindArrow,
};

@interface MILToken : NSObject
@property(nonatomic, readonly) MILTokenKind kind;
@property(nonatomic, readonly, copy) NSString *spelling;
@property(nonatomic, readonly) ANESourceRange range;

- (instancetype)initWithKind:(MILTokenKind)kind
                     spelling:(NSString *)spelling
                        range:(ANESourceRange)range;
@end

