#import <Foundation/Foundation.h>

#import "ANEScheduledGraph.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GOptionalUInt : NSObject
@property(nonatomic, readonly) BOOL available;
@property(nonatomic, readonly) NSUInteger value;
@property(nonatomic, readonly, copy) NSString *provenance;
- (instancetype)initUnavailableWithProvenance:(NSString *)provenance;
- (instancetype)initWithValue:(NSUInteger)value provenance:(NSString *)provenance;
@end

typedef NS_ENUM(NSUInteger, H16GTaskPacketFamily) {
    H16GTaskPacketFamilySquareMatmul,
    H16GTaskPacketFamilyUnaryLUT,
    H16GTaskPacketFamilyScalarScale,
    H16GTaskPacketFamilyMatrixRowALU,
    H16GTaskPacketFamilyRowStateALU,
    H16GTaskPacketFamilyReduction,
    H16GTaskPacketFamilyMatrixRowDivision,
    H16GTaskPacketFamilySquareALU,
    H16GTaskPacketFamilyPrimitiveFallback,
};

typedef NS_ENUM(NSUInteger, H16GTaskSemanticConstraint) {
    H16GTaskSemanticConstraintNone,
    H16GTaskSemanticConstraintNonTransposedMatmul,
    /// One tensor operand and one finite constant scalar operand.
    H16GTaskSemanticConstraintScalarOperand,
    H16GTaskSemanticConstraintLastAxis,
};

typedef NS_ENUM(NSUInteger, H16GProgramCompositionAction) {
    H16GProgramCompositionActionTiledPostOperation,
    H16GProgramCompositionActionPrimitiveFallback,
    /// The consumer's constant scalar multiplies into the producer's
    /// decoded output-scale field; the consumer task disappears.
    H16GProgramCompositionActionMatmulOutputScale,
    /// The producer's constant scalar multiplies into the consumer's
    /// decoded table input-scale field; the producer task disappears.
    H16GProgramCompositionActionLUTInputScale,
    /// Emit two decoded task forms in one program with the intermediate
    /// retained in on-chip SRAM.
    H16GProgramCompositionActionSRAMTaskChain,
};

@interface H16GTaskCapability : NSObject
@property(nonatomic, readonly, copy) NSArray<NSString *> *stageOperations;
@property(nonatomic, readonly, copy) NSArray<NSArray<NSNumber *> *> *inputShapes;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readonly) ANELegalNumericMode numericMode;
@property(nonatomic, readonly) ANEScheduledBridgeStorage outputStorage;
@property(nonatomic, readonly) H16GTaskPacketFamily packetFamily;
@property(nonatomic, readonly) NSUInteger geometry;
@property(nonatomic, readonly) NSUInteger programTaskCount;
@property(nonatomic, readonly) NSUInteger standaloneTaskCount;
@property(nonatomic, readonly) H16GTaskSemanticConstraint semanticConstraint;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry
                        programTaskCount:(NSUInteger)programTaskCount
                      semanticConstraint:(H16GTaskSemanticConstraint)semanticConstraint;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry
                        programTaskCount:(NSUInteger)programTaskCount
                     standaloneTaskCount:(NSUInteger)standaloneTaskCount
                      semanticConstraint:(H16GTaskSemanticConstraint)semanticConstraint;
@end

@interface H16GProgramCompositionCapability : NSObject
@property(nonatomic, readonly) H16GTaskPacketFamily producerPacketFamily;
@property(nonatomic, readonly) H16GTaskPacketFamily consumerPacketFamily;
@property(nonatomic, readonly, copy) NSString *producerOperationName;
@property(nonatomic, readonly, copy) NSString *consumerOperationName;
@property(nonatomic, readonly) NSUInteger producerGeometry;
@property(nonatomic, readonly) NSUInteger consumerGeometry;
@property(nonatomic, readonly) ANEScheduledBridgeStorage bridgeStorage;
@property(nonatomic, readonly) H16GProgramCompositionAction action;
@property(nonatomic, readonly) NSUInteger consumerTaskCountContribution;
@property(nonatomic, readonly, copy) NSString *provenance;
- (instancetype)initWithProducerPacketFamily:
    (H16GTaskPacketFamily)producerPacketFamily
    consumerPacketFamily:(H16GTaskPacketFamily)consumerPacketFamily
    producerOperationName:(NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    producerGeometry:(NSUInteger)producerGeometry
    consumerGeometry:(NSUInteger)consumerGeometry
    bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage
    action:(H16GProgramCompositionAction)action
    provenance:(NSString *)provenance;
- (instancetype)initWithProducerPacketFamily:
    (H16GTaskPacketFamily)producerPacketFamily
    consumerPacketFamily:(H16GTaskPacketFamily)consumerPacketFamily
    producerOperationName:(NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    producerGeometry:(NSUInteger)producerGeometry
    consumerGeometry:(NSUInteger)consumerGeometry
    bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage
    action:(H16GProgramCompositionAction)action
    consumerTaskCountContribution:(NSUInteger)consumerTaskCountContribution
    provenance:(NSString *)provenance;
@end

@interface H16GTarget : NSObject
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly) NSUInteger sramBankCount;
@property(nonatomic, readonly) NSUInteger sramGranuleBytes;
@property(nonatomic, readonly) NSUInteger computeSetCount;
@property(nonatomic, readonly) NSUInteger ioSurfaceAlignment;
@property(nonatomic, readonly) NSUInteger outputSlabBytes;
@property(nonatomic, readonly) NSUInteger workingSetBytes;
@property(nonatomic, readonly) NSUInteger maximumTensorDimension;
@property(nonatomic, readonly) NSUInteger minimumSurfaceRowBytes;
@property(nonatomic, readonly) NSUInteger layoutDMAProgramMaxInputBytes;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *layoutBlockSizes;
@property(nonatomic, readonly) H16GOptionalUInt *maximumProgramInputCount;
@property(nonatomic, readonly) H16GOptionalUInt *maximumProgramTaskCount;
@property(nonatomic, readonly) NSUInteger maximumTaskDescriptorByteLength;
+ (instancetype)currentTarget;
- (BOOL)supportsOperationKind:(ANEOperationKind)kind;
- (BOOL)supportsFusedEpilogue:(NSString *)operationName
                 producerKind:(ANEOperationKind)producerKind;
- (nullable H16GTaskCapability *)taskCapabilityForStageOperations:
    (NSArray<NSString *> *)stageOperations
    inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
    outputShape:(NSArray<NSNumber *> *)outputShape
    numericMode:(ANELegalNumericMode)numericMode
    outputStorage:(ANEScheduledBridgeStorage)outputStorage;
- (nullable H16GProgramCompositionCapability *)
    programCompositionCapabilityFrom:(H16GTaskPacketFamily)producerPacketFamily
    to:(H16GTaskPacketFamily)consumerPacketFamily
    producerOperationName:(NSString *)producerOperationName
    consumerOperationName:(NSString *)consumerOperationName
    producerGeometry:(NSUInteger)producerGeometry
    consumerGeometry:(NSUInteger)consumerGeometry
    bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage;
- (BOOL)validateSpaceDepthInputShape:(NSArray<NSNumber *> *)inputShape
                         outputShape:(NSArray<NSNumber *> *)outputShape
                           blockSize:(NSUInteger)blockSize
                        depthToSpace:(BOOL)depthToSpace
                              reason:(NSString * _Nullable * _Nullable)reason;
@end

NS_ASSUME_NONNULL_END
