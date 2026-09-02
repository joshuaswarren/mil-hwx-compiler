#import "ANEGraphIR.h"

@implementation ANEValueType
- (instancetype)initWithKind:(ANEValueTypeKind)kind
                   elementType:(ANEElementType)elementType
                         shape:(NSArray<NSNumber *> *)shape {
    self = [super init];
    if (self) {
        _kind = kind;
        _elementType = elementType;
        _shape = [shape copy];
    }
    return self;
}
- (BOOL)isEqualToValueType:(ANEValueType *)other {
    return other && _kind == other.kind && _elementType == other.elementType &&
           [_shape isEqualToArray:other.shape];
}
@end

@interface ANEGraphValue ()
@property(nonatomic, readwrite, weak, nullable) ANEGraphOperation *producer;
@end

@implementation ANEGraphValue
- (instancetype)initWithName:(NSString *)name type:(ANEValueType *)type {
    self = [super init];
    if (self) {
        _name = [name copy];
        _type = type;
    }
    return self;
}
- (void)setDefiningOperation:(ANEGraphOperation *)producer {
    NSParameterAssert(_producer == nil);
    _producer = producer;
}
@end

@implementation ANEGraphNamedArgument
- (instancetype)initWithName:(NSString *)name value:(ANEGraphArgument *)value {
    self = [super init];
    if (self) {
        _name = [name copy];
        _value = value;
    }
    return self;
}
@end

@implementation ANEGraphArgument
- (instancetype)initWithKind:(ANEGraphArgumentKind)kind
                          text:(NSString *)text
                         value:(ANEGraphValue *)value
                    calleeName:(NSString *)calleeName
               calleeValueType:(ANEValueType *)calleeValueType
                 callArguments:(NSArray<ANEGraphNamedArgument *> *)callArguments
                      elements:(NSArray<ANEGraphArgument *> *)elements
                         range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _kind = kind;
        _text = [text copy];
        _value = value;
        _calleeName = [calleeName copy];
        _calleeValueType = calleeValueType;
        _callArguments = [callArguments copy];
        NSMutableDictionary *named = [NSMutableDictionary dictionary];
        for (ANEGraphNamedArgument *argument in callArguments)
            if (argument.name) named[argument.name] = argument;
        _namedArguments = [named copy];
        _elements = [elements copy];
        _range = range;
    }
    return self;
}
@end

@implementation ANEGraphOperation
- (instancetype)initWithOperationName:(NSString *)operationName
                                result:(ANEGraphValue *)result
                             arguments:(NSDictionary<NSString *,ANEGraphArgument *> *)arguments
                            attributes:(NSDictionary<NSString *,ANEGraphArgument *> *)attributes
                                 range:(ANESourceRange)range {
    self = [super init];
    if (self) {
        _operationName = [operationName copy];
        _result = result;
        _arguments = [arguments copy];
        _attributes = [attributes copy];
        NSMutableDictionary *operands = [NSMutableDictionary dictionary];
        [arguments enumerateKeysAndObjectsUsingBlock:
            ^(NSString *key, ANEGraphArgument *argument, BOOL *stop) {
                (void)stop;
                if (argument.kind == ANEGraphArgumentKindValue)
                    operands[key] = argument;
            }];
        _operands = [operands copy];
        _range = range;
        [result setDefiningOperation:self];
    }
    return self;
}
@end

@implementation ANEGraphFunction
- (instancetype)initWithName:(NSString *)name
                        inputs:(NSArray<ANEGraphValue *> *)inputs
                    operations:(NSArray<ANEGraphOperation *> *)operations
                   returnValues:(NSArray<ANEGraphValue *> *)returnValues {
    self = [super init];
    if (self) {
        _name = [name copy];
        _inputs = [inputs copy];
        _operations = [operations copy];
        _returnValues = [returnValues copy];
    }
    return self;
}
@end

@implementation ANEGraphModule
- (instancetype)initWithFunctions:(NSArray<ANEGraphFunction *> *)functions {
    self = [super init];
    if (self) _functions = [functions copy];
    return self;
}
@end
