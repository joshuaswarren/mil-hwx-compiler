#import <Foundation/Foundation.h>

#import "H16GConvChainEncoder.h"
#import "H16GTarget.h"

NS_ASSUME_NONNULL_BEGIN

@interface H16GEncodedTask : NSObject
@property(nonatomic, readonly, strong) H16GEncodedTDProgram *tdProgram;
@property(nonatomic, readonly, copy) NSData *constantRegion;
@property(nonatomic, readonly, copy) NSArray<NSString *> *stageOperations;
@property(nonatomic, readonly, copy) NSArray<NSString *> *inputIdentifiers;
@property(nonatomic, readonly, copy) NSArray<NSArray<NSNumber *> *> *inputShapes;
@property(nonatomic, readonly, copy) NSString *outputIdentifier;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *outputShape;
@property(nonatomic, readonly) NSUInteger taskCount;
@property(nonatomic, readonly) BOOL scratchBacked;
@property(nonatomic, readonly) NSUInteger scratchAllocationByteLength;
@property(nonatomic, readonly) NSUInteger outputResourceIndex;
@property(nonatomic, readonly) H16GTaskPacketFamily packetFamily;
@property(nonatomic, readonly) NSUInteger geometry;
@property(nonatomic, readonly) ANELegalNumericMode numericMode;
@property(nonatomic, readonly) ANEScheduledBridgeStorage outputStorage;
@property(nonatomic, readonly, copy) NSString *compositionOperationName;
/// The constant scalar operand of a task whose packet family carries one
/// (scalar scale). Nil for every other family.
@property(nonatomic, readonly, nullable, strong) NSNumber *scalarOperand;
- (instancetype)initWithTDProgram:(H16GEncodedTDProgram *)tdProgram
                   constantRegion:(NSData *)constantRegion
                  stageOperations:(NSArray<NSString *> *)stageOperations
                 inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                      inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                 outputIdentifier:(NSString *)outputIdentifier
                      outputShape:(NSArray<NSNumber *> *)outputShape
                        taskCount:(NSUInteger)taskCount
                    scratchBacked:(BOOL)scratchBacked
      scratchAllocationByteLength:(NSUInteger)scratchAllocationByteLength
               outputResourceIndex:(NSUInteger)outputResourceIndex
                     packetFamily:(H16GTaskPacketFamily)packetFamily
                         geometry:(NSUInteger)geometry
                      numericMode:(ANELegalNumericMode)numericMode
                    outputStorage:(ANEScheduledBridgeStorage)outputStorage
          compositionOperationName:(NSString *)compositionOperationName;
- (instancetype)initWithTDProgram:(H16GEncodedTDProgram *)tdProgram
                   constantRegion:(NSData *)constantRegion
                  stageOperations:(NSArray<NSString *> *)stageOperations
                 inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                      inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                 outputIdentifier:(NSString *)outputIdentifier
                      outputShape:(NSArray<NSNumber *> *)outputShape
                        taskCount:(NSUInteger)taskCount
                    scratchBacked:(BOOL)scratchBacked
      scratchAllocationByteLength:(NSUInteger)scratchAllocationByteLength
               outputResourceIndex:(NSUInteger)outputResourceIndex
                     packetFamily:(H16GTaskPacketFamily)packetFamily
                         geometry:(NSUInteger)geometry
                      numericMode:(ANELegalNumericMode)numericMode
                    outputStorage:(ANEScheduledBridgeStorage)outputStorage
          compositionOperationName:(NSString *)compositionOperationName
                    scalarOperand:(nullable NSNumber *)scalarOperand;
@end

NS_ASSUME_NONNULL_END
