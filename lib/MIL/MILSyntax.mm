#import "MILSyntax.h"

@implementation MILTypeSyntax
- (instancetype)initWithName:(NSString *)name
                typeArguments:(NSArray<MILTypeSyntax *> *)typeArguments
                   dimensions:(NSArray<NSNumber *> *)dimensions {
    self = [super init];
    if (self) {
        _name = [name copy];
        _typeArguments = [typeArguments copy];
        _dimensions = [dimensions copy];
    }
    return self;
}
@end

@implementation MILArgumentSyntax
- (instancetype)initWithName:(NSString *)name
                         value:(MILExpressionSyntax *)value {
    self = [super init];
    if (self) {
        _name = [name copy];
        _value = value;
    }
    return self;
}
@end

@implementation MILExpressionSyntax
- (instancetype)initWithKind:(MILExpressionKind)kind
                          atom:(NSString *)atom
                    calleeType:(MILTypeSyntax *)calleeType
                    calleeName:(NSString *)calleeName
                     arguments:(NSArray<MILArgumentSyntax *> *)arguments
                      elements:(NSArray<MILExpressionSyntax *> *)elements
                         range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _kind = kind;
        _atom = [atom copy];
        _calleeType = calleeType;
        _calleeName = [calleeName copy];
        _arguments = [arguments copy];
        _elements = [elements copy];
        _range = range;
    }
    return self;
}
@end

@implementation MILParameterSyntax
- (instancetype)initWithType:(MILTypeSyntax *)type name:(NSString *)name {
    self = [super init];
    if (self) {
        _type = type;
        _name = [name copy];
    }
    return self;
}
@end

@implementation MILOperationSyntax
- (instancetype)initWithResultType:(MILTypeSyntax *)resultType
                         resultName:(NSString *)resultName
                      operationName:(NSString *)operationName
                          arguments:(NSArray<MILArgumentSyntax *> *)arguments
                         attributes:(NSArray<MILArgumentSyntax *> *)attributes
                              range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _resultType = resultType;
        _resultName = [resultName copy];
        _operationName = [operationName copy];
        _arguments = [arguments copy];
        _attributes = [attributes copy];
        _range = range;
    }
    return self;
}
@end

@implementation MILFunctionSyntax
- (instancetype)initWithName:(NSString *)name
                         opset:(NSString *)opset
                    parameters:(NSArray<MILParameterSyntax *> *)parameters
                     operations:(NSArray<MILOperationSyntax *> *)operations
                    returnNames:(NSArray<NSString *> *)returnNames
                          range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _name = [name copy];
        _opset = [opset copy];
        _parameters = [parameters copy];
        _operations = [operations copy];
        _returnNames = [returnNames copy];
        _range = range;
    }
    return self;
}
@end

@implementation MILProgramSyntax
- (instancetype)initWithVersion:(NSString *)version
                       attributes:(NSArray<MILArgumentSyntax *> *)attributes
                        functions:(NSArray<MILFunctionSyntax *> *)functions
                            range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _version = [version copy];
        _attributes = [attributes copy];
        _functions = [functions copy];
        _range = range;
    }
    return self;
}
@end

