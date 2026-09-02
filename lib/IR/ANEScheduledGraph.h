#import <Foundation/Foundation.h>

#import "ANEOperationGraph.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, ANEScheduledSurfaceRole) {
    ANEScheduledSurfaceRoleExternalInput,
    ANEScheduledSurfaceRoleConstant,
    ANEScheduledSurfaceRoleIntermediate,
    ANEScheduledSurfaceRoleCarry,
    ANEScheduledSurfaceRoleOutput,
};

typedef NS_ENUM(NSUInteger, ANEScheduledBridgeStorage) {
    ANEScheduledBridgeStorageExternal,
    ANEScheduledBridgeStorageConstant,
    ANEScheduledBridgeStorageSRAM,
    ANEScheduledBridgeStorageCarry,
};

typedef NS_ENUM(NSUInteger, ANEScheduledTopology) {
    ANEScheduledTopologyDirect,
    ANEScheduledTopologyOnlineReduction,
    ANEScheduledTopologyAssociativeScan,
};

typedef NS_ENUM(NSUInteger, ANEScheduledCommandKind) {
    ANEScheduledCommandKindDMALoad,
    ANEScheduledCommandKindDMAInter,
    ANEScheduledCommandKindDMAStore,
    ANEScheduledCommandKindCompute,
    ANEScheduledCommandKindWait,
};

typedef NS_ENUM(NSUInteger, ANETileStrategy) {
    ANETileStrategyDirect,
    ANETileStrategyMatrixRows,
    ANETileStrategyLayoutConv,
    ANETileStrategyLayoutDMA3,
    ANETileStrategyLayoutStoreStream,
    ANETileStrategyLayoutLoadStream,
};

@interface ANETilePlan : NSObject
@property(nonatomic, readonly) NSUInteger rows;
@property(nonatomic, readonly) NSUInteger count;
@property(nonatomic, readonly) ANETileStrategy strategy;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *inputShape;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readonly) NSUInteger descriptorCount;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *iterationShape;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *tileShape;
- (instancetype)initWithRows:(NSUInteger)rows count:(NSUInteger)count;
- (instancetype)initWithRows:(NSUInteger)rows
                        count:(NSUInteger)count
                     strategy:(ANETileStrategy)strategy
                   inputShape:(NSArray<NSNumber *> *)inputShape
                  outputShape:(NSArray<NSNumber *> *)outputShape
              descriptorCount:(NSUInteger)descriptorCount;
@end

@interface ANEScheduledStage : NSObject
@property(nonatomic, readonly, copy) NSString *sourceNodeIdentifier;
@property(nonatomic, readonly, copy) NSString *operationName;
@property(nonatomic, readonly) ANEOperationKind operationKind;
@property(nonatomic, readonly, copy) NSArray<NSString *> *inputIdentifiers;
@property(nonatomic, readonly, copy) NSString *outputIdentifier;
@property(nonatomic, readonly) ANEScheduledBridgeStorage bridgeStorage;
- (instancetype)initWithSourceNodeIdentifier:(NSString *)sourceNodeIdentifier
                                operationName:(NSString *)operationName
                                operationKind:(ANEOperationKind)operationKind
                             inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                             outputIdentifier:(NSString *)outputIdentifier
                                bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage;
@end

@interface ANEScheduledSurface : NSObject
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly) ANEScheduledSurfaceRole role;
@property(nonatomic, readonly) ANEElementType elementType;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *shape;
@property(nonatomic, readonly) NSUInteger byteLength;
@property(nonatomic, readonly) NSUInteger firstTask;
@property(nonatomic, readonly) NSUInteger lastTask;
@property(nonatomic, readonly) BOOL sramAllocated;
@property(nonatomic, readonly) NSUInteger sramOffset;
@property(nonatomic, readonly) NSUInteger bank;
- (instancetype)initWithIdentifier:(NSString *)identifier
                               role:(ANEScheduledSurfaceRole)role
                        elementType:(ANEElementType)elementType
                              shape:(NSArray<NSNumber *> *)shape
                          firstTask:(NSUInteger)firstTask;
- (void)extendLifetimeThroughTask:(NSUInteger)task;
- (void)assignSRAMOffset:(NSUInteger)offset bankCount:(NSUInteger)bankCount
                  granule:(NSUInteger)granule;
@end

@interface ANEScheduledCommand : NSObject
@property(nonatomic, readonly) ANEScheduledCommandKind kind;
@property(nonatomic, readonly, copy) NSArray<NSString *> *inputs;
@property(nonatomic, readonly, copy) NSArray<NSString *> *outputs;
- (instancetype)initWithKind:(ANEScheduledCommandKind)kind
                       inputs:(NSArray<NSString *> *)inputs
                      outputs:(NSArray<NSString *> *)outputs;
@end

@interface ANEScheduledTask : NSObject
@property(nonatomic, readonly) NSUInteger index;
@property(nonatomic, readonly, copy) NSString *regionIdentifier;
@property(nonatomic, readonly, copy) NSString *nodeIdentifier;
@property(nonatomic, readonly, copy) NSArray<NSString *> *sourceNodeIdentifiers;
@property(nonatomic, readonly) ANEOperationKind operationKind;
@property(nonatomic, readonly) ANELegalNumericMode numericMode;
@property(nonatomic, readonly) ANETilePlan *tilePlan;
@property(nonatomic, readonly, copy) NSArray<ANEScheduledCommand *> *commands;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *dependencies;
@property(nonatomic, readonly, copy) NSArray<ANEScheduledStage *> *stages;
@property(nonatomic, readonly) NSUInteger waveIndex;
@property(nonatomic, readonly) ANEScheduledTopology topology;
- (instancetype)initWithIndex:(NSUInteger)index
              regionIdentifier:(NSString *)regionIdentifier
                 nodeIdentifier:(NSString *)nodeIdentifier
           sourceNodeIdentifiers:(NSArray<NSString *> *)sourceNodeIdentifiers
                  operationKind:(ANEOperationKind)operationKind
                    numericMode:(ANELegalNumericMode)numericMode
                       tilePlan:(ANETilePlan *)tilePlan
                       commands:(NSArray<ANEScheduledCommand *> *)commands
                    dependencies:(NSArray<NSNumber *> *)dependencies
                         stages:(NSArray<ANEScheduledStage *> *)stages
                      waveIndex:(NSUInteger)waveIndex
                       topology:(ANEScheduledTopology)topology;
@end

@interface ANEScheduledRegionPlan : NSObject
@property(nonatomic, readonly, copy) NSString *regionIdentifier;
@property(nonatomic, readonly) ANEScheduledTopology topology;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *taskIndexes;
@property(nonatomic, readonly, copy) NSArray<NSString *> *stageIdentifiers;
@property(nonatomic, readonly) NSUInteger queryTileCount;
@property(nonatomic, readonly) NSUInteger keyValueTileCount;
@property(nonatomic, readonly) NSUInteger tileRows;
@property(nonatomic, readonly, copy) NSArray<NSString *> *carriedSurfaceIdentifiers;
@property(nonatomic, readonly, copy) NSArray<NSString *> *elidedSurfaceIdentifiers;
@property(nonatomic, readonly, copy) NSArray<NSString *> *boundaryInputIdentifiers;
@property(nonatomic, readonly, copy) NSString *outputIdentifier;
- (instancetype)initWithRegionIdentifier:(NSString *)regionIdentifier
                                 topology:(ANEScheduledTopology)topology
                              taskIndexes:(NSArray<NSNumber *> *)taskIndexes
                         stageIdentifiers:(NSArray<NSString *> *)stageIdentifiers;
- (instancetype)initWithRegionIdentifier:(NSString *)regionIdentifier
                                 topology:(ANEScheduledTopology)topology
                              taskIndexes:(NSArray<NSNumber *> *)taskIndexes
                         stageIdentifiers:(NSArray<NSString *> *)stageIdentifiers
                           queryTileCount:(NSUInteger)queryTileCount
                        keyValueTileCount:(NSUInteger)keyValueTileCount
                                 tileRows:(NSUInteger)tileRows
                carriedSurfaceIdentifiers:(NSArray<NSString *> *)carriedSurfaceIdentifiers
                 elidedSurfaceIdentifiers:(NSArray<NSString *> *)elidedSurfaceIdentifiers;
- (instancetype)initWithRegionIdentifier:(NSString *)regionIdentifier
                                 topology:(ANEScheduledTopology)topology
                              taskIndexes:(NSArray<NSNumber *> *)taskIndexes
                         stageIdentifiers:(NSArray<NSString *> *)stageIdentifiers
                           queryTileCount:(NSUInteger)queryTileCount
                        keyValueTileCount:(NSUInteger)keyValueTileCount
                                 tileRows:(NSUInteger)tileRows
                carriedSurfaceIdentifiers:(NSArray<NSString *> *)carriedSurfaceIdentifiers
                 elidedSurfaceIdentifiers:(NSArray<NSString *> *)elidedSurfaceIdentifiers
                 boundaryInputIdentifiers:(NSArray<NSString *> *)boundaryInputIdentifiers
                         outputIdentifier:(NSString *)outputIdentifier;
@end

@interface ANEScheduledGraph : NSObject
@property(nonatomic, readonly, copy) NSArray<ANEScheduledSurface *> *surfaces;
@property(nonatomic, readonly, copy) NSArray<ANEScheduledTask *> *tasks;
@property(nonatomic, readonly, copy) NSArray<ANEScheduledRegionPlan *> *composedPlans;
@property(nonatomic, readonly) NSUInteger peakSRAMBytes;
- (instancetype)initWithSurfaces:(NSArray<ANEScheduledSurface *> *)surfaces
                            tasks:(NSArray<ANEScheduledTask *> *)tasks
                    composedPlans:(NSArray<ANEScheduledRegionPlan *> *)composedPlans
                    peakSRAMBytes:(NSUInteger)peakSRAMBytes;
- (NSString *)textualDescription;
@end

NS_ASSUME_NONNULL_END
