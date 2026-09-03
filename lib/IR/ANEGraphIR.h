#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, ANEElementType) {
    ANEElementTypeInvalid,
    ANEElementTypeFP16,
    ANEElementTypeFP32,
    ANEElementTypeInt8,
    ANEElementTypeInt32,
    ANEElementTypeUInt64,
    ANEElementTypeBool,
    ANEElementTypeString,
};

typedef NS_ENUM(NSUInteger, ANEValueTypeKind) {
    ANEValueTypeKindScalar,
    ANEValueTypeKindTensor,
};

@interface ANEValueType : NSObject
@property(nonatomic, readonly) ANEValueTypeKind kind;
@property(nonatomic, readonly) ANEElementType elementType;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *shape;
- (instancetype)initWithKind:(ANEValueTypeKind)kind
                   elementType:(ANEElementType)elementType
                         shape:(NSArray<NSNumber *> *)shape;
- (BOOL)isEqualToValueType:(ANEValueType *)other;
@end

@class ANEGraphOperation;

@interface ANEGraphValue : NSObject
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly) ANEValueType *type;
@property(nonatomic, readonly, weak, nullable) ANEGraphOperation *producer;
- (instancetype)initWithName:(NSString *)name type:(ANEValueType *)type;
- (void)setDefiningOperation:(ANEGraphOperation *)producer;
@end

typedef NS_ENUM(NSUInteger, ANEGraphArgumentKind) {
    ANEGraphArgumentKindValue,
    ANEGraphArgumentKindInteger,
    ANEGraphArgumentKindFloatingPoint,
    ANEGraphArgumentKindString,
    ANEGraphArgumentKindBoolean,
    ANEGraphArgumentKindList,
    ANEGraphArgumentKindTuple,
    ANEGraphArgumentKindDictionary,
    ANEGraphArgumentKindCall,
};

@class ANEGraphArgument;

@interface ANEGraphNamedArgument : NSObject
@property(nonatomic, readonly, copy, nullable) NSString *name;
@property(nonatomic, readonly) ANEGraphArgument *value;
- (instancetype)initWithName:(nullable NSString *)name
                         value:(ANEGraphArgument *)value;
@end

@interface ANEGraphArgument : NSObject
@property(nonatomic, readonly) ANEGraphArgumentKind kind;
@property(nonatomic, readonly, copy, nullable) NSString *text;
@property(nonatomic, readonly, nullable) ANEGraphValue *value;
@property(nonatomic, readonly, copy, nullable) NSString *calleeName;
@property(nonatomic, readonly, nullable) ANEValueType *calleeValueType;
@property(nonatomic, readonly, copy) NSArray<ANEGraphNamedArgument *> *callArguments;
@property(nonatomic, readonly, copy) NSDictionary<NSString *, ANEGraphNamedArgument *> *namedArguments;
@property(nonatomic, readonly, copy) NSArray<ANEGraphArgument *> *elements;
@property(nonatomic, readonly) ANESourceRange range;
- (instancetype)initWithKind:(ANEGraphArgumentKind)kind
                          text:(nullable NSString *)text
                         value:(nullable ANEGraphValue *)value
                    calleeName:(nullable NSString *)calleeName
               calleeValueType:(nullable ANEValueType *)calleeValueType
                 callArguments:(NSArray<ANEGraphNamedArgument *> *)callArguments
                      elements:(NSArray<ANEGraphArgument *> *)elements
                         range:(ANESourceRange)range;
@end

@interface ANEGraphOperation : NSObject
@property(nonatomic, readonly, copy) NSString *operationName;
@property(nonatomic, readonly) ANEGraphValue *result;
@property(nonatomic, readonly, copy) NSDictionary<NSString *, ANEGraphArgument *> *arguments;
@property(nonatomic, readonly, copy) NSDictionary<NSString *, ANEGraphArgument *> *attributes;
@property(nonatomic, readonly, copy) NSDictionary<NSString *, ANEGraphArgument *> *operands;
@property(nonatomic, readonly) ANESourceRange range;
- (instancetype)initWithOperationName:(NSString *)operationName
                                result:(ANEGraphValue *)result
                             arguments:(NSDictionary<NSString *, ANEGraphArgument *> *)arguments
                            attributes:(NSDictionary<NSString *, ANEGraphArgument *> *)attributes
                                 range:(ANESourceRange)range;
@end

@interface ANEGraphFunction : NSObject
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly, copy) NSArray<ANEGraphValue *> *inputs;
@property(nonatomic, readonly, copy) NSArray<ANEGraphOperation *> *operations;
@property(nonatomic, readonly, copy) NSArray<ANEGraphValue *> *returnValues;
- (instancetype)initWithName:(NSString *)name
                        inputs:(NSArray<ANEGraphValue *> *)inputs
                    operations:(NSArray<ANEGraphOperation *> *)operations
                   returnValues:(NSArray<ANEGraphValue *> *)returnValues;
@end

@interface ANEGraphModule : NSObject
@property(nonatomic, readonly, copy) NSArray<ANEGraphFunction *> *functions;
- (instancetype)initWithFunctions:(NSArray<ANEGraphFunction *> *)functions;
@end

NS_ASSUME_NONNULL_END
