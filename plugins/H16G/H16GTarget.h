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
};

@interface H16GTaskCapability : NSObject
@property(nonatomic, readonly, copy) NSArray<NSString *> *stageOperations;
@property(nonatomic, readonly, copy) NSArray<NSArray<NSNumber *> *> *inputShapes;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readonly) ANELegalNumericMode numericMode;
@property(nonatomic, readonly) ANEScheduledBridgeStorage outputStorage;
@property(nonatomic, readonly) H16GTaskPacketFamily packetFamily;
@property(nonatomic, readonly) NSUInteger geometry;
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage
                            packetFamily:(H16GTaskPacketFamily)packetFamily
                                geometry:(NSUInteger)geometry;
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
@property(nonatomic, readonly) H16GOptionalUInt *maximumProgramTaskCount;
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
- (BOOL)validateSpaceDepthInputShape:(NSArray<NSNumber *> *)inputShape
                         outputShape:(NSArray<NSNumber *> *)outputShape
                           blockSize:(NSUInteger)blockSize
                        depthToSpace:(BOOL)depthToSpace
                              reason:(NSString * _Nullable * _Nullable)reason;
@end

NS_ASSUME_NONNULL_END
