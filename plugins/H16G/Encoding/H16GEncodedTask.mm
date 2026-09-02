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
               outputResourceIndex:(NSUInteger)outputResourceIndex {
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
    }
    return self;
}
@end
