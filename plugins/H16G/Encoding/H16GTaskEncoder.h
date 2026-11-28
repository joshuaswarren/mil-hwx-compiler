#import <Foundation/Foundation.h>

#import "ANEScheduledGraph.h"
#import "H16GEncodedTask.h"
#import "H16GTarget.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GTaskEncodingRequest : NSObject
@property(nonatomic, readonly, copy) NSArray<NSString *> *stageOperations;
@property(nonatomic, readonly, copy) NSArray<NSString *> *inputIdentifiers;
@property(nonatomic, readonly, copy) NSArray<NSArray<NSNumber *> *> *inputShapes;
@property(nonatomic, readonly, copy) NSString *outputIdentifier;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readonly) ANELegalNumericMode numericMode;
@property(nonatomic, readonly) ANEScheduledBridgeStorage outputStorage;
/// Constant scalar operand for packet families that carry one. Required for
/// the scalar-scale family; ignored by every other family.
@property(nonatomic, readonly, nullable, strong) NSNumber *scalarOperand;
/// Element types of the input/output surfaces, when the caller knows them.
/// Required for the boundary-conversion family, which selects the decoded
/// direction from the type pair; ignored by every other family.
@property(nonatomic, readonly, copy, nullable)
    NSArray<NSNumber *> *inputElementTypes;
@property(nonatomic, readonly) ANEElementType outputElementType;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                        inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                        outputIdentifier:(NSString *)outputIdentifier
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                        inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                        outputIdentifier:(NSString *)outputIdentifier
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                           scalarOperand:(nullable NSNumber *)scalarOperand;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                        inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                        outputIdentifier:(NSString *)outputIdentifier
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                           scalarOperand:(nullable NSNumber *)scalarOperand
                      inputElementTypes:(nullable NSArray<NSNumber *> *)inputElementTypes
                      outputElementType:(ANEElementType)outputElementType;
@end

@interface H16GTaskEncoder : NSObject
+ (nullable H16GEncodedTask *)encodeRequest:(H16GTaskEncodingRequest *)request
                                      target:(H16GTarget *)target
                                       error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
