#import "H16GEncodedTask.h"

@implementation H16GEncodedTask
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
          compositionOperationName:(NSString *)compositionOperationName {
    return [self initWithTDProgram:tdProgram constantRegion:constantRegion
        stageOperations:stageOperations inputIdentifiers:inputIdentifiers
        inputShapes:inputShapes outputIdentifier:outputIdentifier
        outputShape:outputShape taskCount:taskCount
        scratchBacked:scratchBacked
        scratchAllocationByteLength:scratchAllocationByteLength
        outputResourceIndex:outputResourceIndex packetFamily:packetFamily
        geometry:geometry numericMode:numericMode outputStorage:outputStorage
        compositionOperationName:compositionOperationName scalarOperand:nil];
}

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
                    scalarOperand:(NSNumber *)scalarOperand {
    self = [super init];
    if (self) {
        _tdProgram = tdProgram;
        _constantRegion = [constantRegion copy];
        _stageOperations = [stageOperations copy];
        _inputIdentifiers = [inputIdentifiers copy];
        _inputShapes = [inputShapes copy];
        _outputIdentifier = [outputIdentifier copy];
        _outputShape = [outputShape copy];
        _taskCount = taskCount;
        _scratchBacked = scratchBacked;
        _scratchAllocationByteLength = scratchAllocationByteLength;
        _outputResourceIndex = outputResourceIndex;
        _packetFamily = packetFamily;
        _geometry = geometry;
        _numericMode = numericMode;
        _outputStorage = outputStorage;
        _compositionOperationName = [compositionOperationName copy];
        _scalarOperand = scalarOperand;
    }
    return self;
}
@end
