#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

@class MILExpressionSyntax;

@interface MILTypeSyntax : NSObject
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly, copy) NSArray<MILTypeSyntax *> *typeArguments;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *dimensions;
- (instancetype)initWithName:(NSString *)name
                typeArguments:(NSArray<MILTypeSyntax *> *)typeArguments
                   dimensions:(NSArray<NSNumber *> *)dimensions;
@end

typedef NS_ENUM(NSUInteger, MILExpressionKind) {
    MILExpressionKindIdentifier,
    MILExpressionKindInteger,
    MILExpressionKindFloatingPoint,
    MILExpressionKindString,
    MILExpressionKindList,
    MILExpressionKindTuple,
    MILExpressionKindDictionary,
    MILExpressionKindCall,
};

@interface MILArgumentSyntax : NSObject
@property(nonatomic, readonly, copy, nullable) NSString *name;
@property(nonatomic, readonly) MILExpressionSyntax *value;
- (instancetype)initWithName:(nullable NSString *)name
                         value:(MILExpressionSyntax *)value;
@end

@interface MILExpressionSyntax : NSObject
@property(nonatomic, readonly) MILExpressionKind kind;
@property(nonatomic, readonly, copy, nullable) NSString *atom;
@property(nonatomic, readonly, nullable) MILTypeSyntax *calleeType;
@property(nonatomic, readonly, copy, nullable) NSString *calleeName;
@property(nonatomic, readonly, copy) NSArray<MILArgumentSyntax *> *arguments;
@property(nonatomic, readonly, copy) NSArray<MILExpressionSyntax *> *elements;
@property(nonatomic, readonly) ANESourceRange range;
- (instancetype)initWithKind:(MILExpressionKind)kind
                          atom:(nullable NSString *)atom
                    calleeType:(nullable MILTypeSyntax *)calleeType
                    calleeName:(nullable NSString *)calleeName
                     arguments:(NSArray<MILArgumentSyntax *> *)arguments
                      elements:(NSArray<MILExpressionSyntax *> *)elements
                         range:(ANESourceRange)range;
@end

@interface MILParameterSyntax : NSObject
@property(nonatomic, readonly) MILTypeSyntax *type;
@property(nonatomic, readonly, copy) NSString *name;
- (instancetype)initWithType:(MILTypeSyntax *)type name:(NSString *)name;
@end

@interface MILOperationSyntax : NSObject
@property(nonatomic, readonly) MILTypeSyntax *resultType;
@property(nonatomic, readonly, copy) NSString *resultName;
@property(nonatomic, readonly, copy) NSString *operationName;
@property(nonatomic, readonly, copy) NSArray<MILArgumentSyntax *> *arguments;
@property(nonatomic, readonly, copy) NSArray<MILArgumentSyntax *> *attributes;
@property(nonatomic, readonly) ANESourceRange range;
- (instancetype)initWithResultType:(MILTypeSyntax *)resultType
                         resultName:(NSString *)resultName
                      operationName:(NSString *)operationName
                          arguments:(NSArray<MILArgumentSyntax *> *)arguments
                         attributes:(NSArray<MILArgumentSyntax *> *)attributes
                              range:(ANESourceRange)range;
@end

@interface MILFunctionSyntax : NSObject
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly, copy) NSString *opset;
@property(nonatomic, readonly, copy) NSArray<MILParameterSyntax *> *parameters;
@property(nonatomic, readonly, copy) NSArray<MILOperationSyntax *> *operations;
@property(nonatomic, readonly, copy) NSArray<NSString *> *returnNames;
@property(nonatomic, readonly) ANESourceRange range;
- (instancetype)initWithName:(NSString *)name
                         opset:(NSString *)opset
                    parameters:(NSArray<MILParameterSyntax *> *)parameters
                     operations:(NSArray<MILOperationSyntax *> *)operations
                    returnNames:(NSArray<NSString *> *)returnNames
                          range:(ANESourceRange)range;
@end

@interface MILProgramSyntax : NSObject
@property(nonatomic, readonly, copy) NSString *version;
@property(nonatomic, readonly, copy) NSArray<MILArgumentSyntax *> *attributes;
@property(nonatomic, readonly, copy) NSArray<MILFunctionSyntax *> *functions;
@property(nonatomic, readonly) ANESourceRange range;
- (instancetype)initWithVersion:(NSString *)version
                       attributes:(NSArray<MILArgumentSyntax *> *)attributes
                        functions:(NSArray<MILFunctionSyntax *> *)functions
                            range:(ANESourceRange)range;
@end

NS_ASSUME_NONNULL_END
