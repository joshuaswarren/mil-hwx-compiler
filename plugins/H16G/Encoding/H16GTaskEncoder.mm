#import "H16GTaskEncoder.h"

#import "H16GLUTEncoder.h"
#import "H16GMatmulEncoder.h"
#import "H16GALUEncoder.h"
#import "H16GBroadcastALUEncoder.h"
#import "H16GMatrixRowDivisionEncoder.h"
#import "H16GReduceEncoder.h"

static NSString *const H16GTaskEncoderErrorDomain = @"ANE.H16G.TaskEncoder";

static void setError(NSError **error, NSString *message) {
    if (error) *error = [NSError errorWithDomain:H16GTaskEncoderErrorDomain
        code:1 userInfo:@{NSLocalizedDescriptionKey:message}];
}

@implementation H16GTaskEncodingRequest
- (instancetype)initWithStageOperations:(NSArray<NSString *> *)stageOperations
                        inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                             inputShapes:(NSArray<NSArray<NSNumber *> *> *)inputShapes
                        outputIdentifier:(NSString *)outputIdentifier
                             outputShape:(NSArray<NSNumber *> *)outputShape
                             numericMode:(ANELegalNumericMode)numericMode
                           outputStorage:(ANEScheduledBridgeStorage)outputStorage {
    self = [super init];
    if (self) {
        _stageOperations = [stageOperations copy];
        _inputIdentifiers = [inputIdentifiers copy];
        _inputShapes = [inputShapes copy];
        _outputIdentifier = [outputIdentifier copy];
        _outputShape = [outputShape copy];
        _numericMode = numericMode;
        _outputStorage = outputStorage;
    }
    return self;
}
@end

@implementation H16GTaskEncoder
+ (H16GEncodedTask *)encodeRequest:(H16GTaskEncodingRequest *)request
                             target:(H16GTarget *)target
                              error:(NSError **)error {
    H16GTaskCapability *capability = [target
        taskCapabilityForStageOperations:request.stageOperations
        inputShapes:request.inputShapes outputShape:request.outputShape
        numericMode:request.numericMode outputStorage:request.outputStorage];
    if (!capability) {
        setError(error, @"scheduled task has no decoded H16G capability row");
        return nil;
    }
    H16GEncodedTDProgram *program = nil;
    NSData *constants = [NSData data];
    NSUInteger taskCount = 0;
    BOOL scratchBacked = NO;
    NSUInteger scratchAllocationBytes = 0;
    NSUInteger outputResourceIndex = request.inputIdentifiers.count;
    NSArray<NSString *> *encodedInputIdentifiers = request.inputIdentifiers;
    NSArray<NSArray<NSNumber *> *> *encodedInputShapes = request.inputShapes;
    switch (capability.packetFamily) {
        case H16GTaskPacketFamilySquareMatmul:
            program = [H16GMatmulEncoder encodeSquareSize:capability.geometry
                                                    error:error];
            taskCount = [H16GMatmulEncoder
                tileCountForSquareSize:capability.geometry] + 1;
            scratchBacked = YES;
            scratchAllocationBytes = capability.geometry * capability.geometry * 2;
            break;
        case H16GTaskPacketFamilyUnaryLUT:
            program = [H16GLUTEncoder
                encodeOperationName:request.stageOperations.lastObject
                inputShape:request.inputShapes.firstObject error:error];
            constants = [H16GLUTEncoder
                constantRegionForOperationName:request.stageOperations.lastObject
                error:error];
            taskCount = 1;
            break;
        case H16GTaskPacketFamilyScalarScale:
            program = [H16GBroadcastALUEncoder
                encodeScalarScaleForMatrixRows:capability.geometry
                columns:capability.geometry
                measuredScaleKind:H16GMeasuredScaleInverseSqrt128
                error:error];
            taskCount = 1;
            break;
        case H16GTaskPacketFamilyMatrixRowALU:
            program = [H16GBroadcastALUEncoder
                encodeMatrixRowOperation:request.stageOperations.lastObject
                rows:capability.geometry columns:capability.geometry
                error:error];
            taskCount = 1;
            encodedInputIdentifiers = [[request.inputIdentifiers
                reverseObjectEnumerator] allObjects];
            encodedInputShapes = [[request.inputShapes
                reverseObjectEnumerator] allObjects];
            break;
        case H16GTaskPacketFamilyRowStateALU:
            program = [H16GBroadcastALUEncoder
                encodeRowOperation:request.stageOperations.lastObject
                rows:capability.geometry error:error];
            taskCount = 1;
            encodedInputIdentifiers = [[request.inputIdentifiers
                reverseObjectEnumerator] allObjects];
            encodedInputShapes = [[request.inputShapes
                reverseObjectEnumerator] allObjects];
            break;
        case H16GTaskPacketFamilyReduction: {
            H16GReduceEncoding *reduction = [H16GReduceEncoder
                encodeOperationName:request.stageOperations.lastObject
                inputShape:request.inputShapes.firstObject axis:3 error:error];
            program = reduction.tdProgram;
            taskCount = reduction.taskCount;
            break;
        }
        case H16GTaskPacketFamilyMatrixRowDivision: {
            H16GMatrixRowDivisionEncoding *division =
                [H16GMatrixRowDivisionEncoder encodeRows:capability.geometry
                    columns:capability.geometry error:error];
            program = division.tdProgram;
            constants = division.constantRegion;
            taskCount = division.taskCount;
            encodedInputIdentifiers = [[request.inputIdentifiers
                reverseObjectEnumerator] allObjects];
            encodedInputShapes = [[request.inputShapes
                reverseObjectEnumerator] allObjects];
            break;
        }
        case H16GTaskPacketFamilySquareALU:
            program = [H16GALUEncoder
                encodeOperationName:request.stageOperations.lastObject
                squareSize:capability.geometry error:error];
            taskCount = 1;
            outputResourceIndex = 1;
            break;
    }
    if (!program || !constants) return nil;
    return [[H16GEncodedTask alloc] initWithTDProgram:program
        constantRegion:constants stageOperations:request.stageOperations
        inputIdentifiers:encodedInputIdentifiers inputShapes:encodedInputShapes
        outputIdentifier:request.outputIdentifier outputShape:request.outputShape
        taskCount:taskCount scratchBacked:scratchBacked
        scratchAllocationByteLength:scratchAllocationBytes
        outputResourceIndex:outputResourceIndex];
}
@end
