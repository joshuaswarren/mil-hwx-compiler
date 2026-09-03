#import "ANEScheduledGraph.h"

static NSUInteger byteWidth(ANEElementType type) {
    switch (type) {
        case ANEElementTypeFP16: return 2;
        case ANEElementTypeFP32: return 4;
        case ANEElementTypeInt8: return 1;
        case ANEElementTypeInt32: return 4;
        case ANEElementTypeUInt64: return 8;
        case ANEElementTypeBool: return 1;
        case ANEElementTypeString:
        case ANEElementTypeInvalid: return 0;
    }
}

@implementation ANETilePlan
- (instancetype)initWithRows:(NSUInteger)rows count:(NSUInteger)count {
    return [self initWithRows:rows count:count strategy:ANETileStrategyDirect
        inputShape:@[] outputShape:@[] descriptorCount:count];
}
- (instancetype)initWithRows:(NSUInteger)rows
                        count:(NSUInteger)count
                     strategy:(ANETileStrategy)strategy
                   inputShape:(NSArray<NSNumber *> *)inputShape
                  outputShape:(NSArray<NSNumber *> *)outputShape
              descriptorCount:(NSUInteger)descriptorCount {
    self = [super init];
    if (self) {
        _rows = rows;
        _count = count;
        _strategy = strategy;
        _inputShape = [inputShape copy];
        _outputShape = [outputShape copy];
        _descriptorCount = descriptorCount;
        _iterationShape = outputShape.count ? [outputShape copy] : @[@(count)];
        if (outputShape.count >= 2) {
            NSMutableArray<NSNumber *> *tileShape = [outputShape mutableCopy];
            tileShape[tileShape.count - 2] = @(rows);
            _tileShape = [tileShape copy];
        } else {
            _tileShape = @[@(rows)];
        }
    }
    return self;
}
@end

@implementation ANEScheduledStage
- (instancetype)initWithSourceNodeIdentifier:(NSString *)sourceNodeIdentifier
                                operationName:(NSString *)operationName
                                operationKind:(ANEOperationKind)operationKind
                             inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
                             outputIdentifier:(NSString *)outputIdentifier
                                bridgeStorage:(ANEScheduledBridgeStorage)bridgeStorage {
    self = [super init];
    if (self) {
        _sourceNodeIdentifier = [sourceNodeIdentifier copy];
        _operationName = [operationName copy];
        _operationKind = operationKind;
        _inputIdentifiers = [inputIdentifiers copy];
        _outputIdentifier = [outputIdentifier copy];
        _bridgeStorage = bridgeStorage;
    }
    return self;
}
@end

@interface ANEScheduledSurface ()
@property(nonatomic) NSUInteger lastTask;
@property(nonatomic) BOOL sramAllocated;
@property(nonatomic) NSUInteger sramOffset;
@property(nonatomic) NSUInteger bank;
@end

@implementation ANEScheduledSurface
- (instancetype)initWithIdentifier:(NSString *)identifier
                               role:(ANEScheduledSurfaceRole)role
                        elementType:(ANEElementType)elementType
                              shape:(NSArray<NSNumber *> *)shape
                          firstTask:(NSUInteger)firstTask {
    self = [super init];
    if (self) {
        _identifier = [identifier copy]; _role = role; _elementType = elementType;
        _shape = [shape copy]; _firstTask = firstTask; _lastTask = firstTask;
        NSUInteger elements = 1;
        for (NSNumber *dimension in shape) elements *= dimension.unsignedIntegerValue;
        _byteLength = elements * byteWidth(elementType);
    }
    return self;
}
- (void)extendLifetimeThroughTask:(NSUInteger)task { _lastTask = MAX(_lastTask, task); }
- (void)assignSRAMOffset:(NSUInteger)offset bankCount:(NSUInteger)bankCount
                  granule:(NSUInteger)granule {
    _sramAllocated = YES; _sramOffset = offset;
    _bank = (offset / granule) % bankCount;
}
@end

@implementation ANEScheduledCommand
- (instancetype)initWithKind:(ANEScheduledCommandKind)kind
                       inputs:(NSArray<NSString *> *)inputs
                      outputs:(NSArray<NSString *> *)outputs {
    self = [super init];
    if (self) { _kind = kind; _inputs = [inputs copy]; _outputs = [outputs copy]; }
    return self;
}
@end

@implementation ANEScheduledTask
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
                       topology:(ANEScheduledTopology)topology {
    self = [super init];
    if (self) {
        _index = index; _regionIdentifier = [regionIdentifier copy];
        _nodeIdentifier = [nodeIdentifier copy];
        _sourceNodeIdentifiers = [sourceNodeIdentifiers copy];
        _operationKind = operationKind;
        _numericMode = numericMode; _tilePlan = tilePlan;
        _commands = [commands copy]; _dependencies = [dependencies copy];
        _stages = [stages copy]; _waveIndex = waveIndex; _topology = topology;
    }
    return self;
}
@end

@implementation ANEScheduledRegionPlan
- (instancetype)initWithRegionIdentifier:(NSString *)regionIdentifier
                                 topology:(ANEScheduledTopology)topology
                              taskIndexes:(NSArray<NSNumber *> *)taskIndexes
                         stageIdentifiers:(NSArray<NSString *> *)stageIdentifiers {
    return [self initWithRegionIdentifier:regionIdentifier topology:topology
        taskIndexes:taskIndexes stageIdentifiers:stageIdentifiers
        queryTileCount:0 keyValueTileCount:0 tileRows:0
        carriedSurfaceIdentifiers:@[] elidedSurfaceIdentifiers:@[]
        boundaryInputIdentifiers:@[] outputIdentifier:@""];
}
- (instancetype)initWithRegionIdentifier:(NSString *)regionIdentifier
                                 topology:(ANEScheduledTopology)topology
                              taskIndexes:(NSArray<NSNumber *> *)taskIndexes
                         stageIdentifiers:(NSArray<NSString *> *)stageIdentifiers
                           queryTileCount:(NSUInteger)queryTileCount
                        keyValueTileCount:(NSUInteger)keyValueTileCount
                                 tileRows:(NSUInteger)tileRows
                carriedSurfaceIdentifiers:(NSArray<NSString *> *)carriedSurfaceIdentifiers
                 elidedSurfaceIdentifiers:(NSArray<NSString *> *)elidedSurfaceIdentifiers {
    return [self initWithRegionIdentifier:regionIdentifier topology:topology
        taskIndexes:taskIndexes stageIdentifiers:stageIdentifiers
        queryTileCount:queryTileCount keyValueTileCount:keyValueTileCount
        tileRows:tileRows carriedSurfaceIdentifiers:carriedSurfaceIdentifiers
        elidedSurfaceIdentifiers:elidedSurfaceIdentifiers
        boundaryInputIdentifiers:@[] outputIdentifier:@""];
}
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
                         outputIdentifier:(NSString *)outputIdentifier {
    self = [super init];
    if (self) {
        _regionIdentifier = [regionIdentifier copy];
        _topology = topology;
        _taskIndexes = [taskIndexes copy];
        _stageIdentifiers = [stageIdentifiers copy];
        _queryTileCount = queryTileCount;
        _keyValueTileCount = keyValueTileCount;
        _tileRows = tileRows;
        _carriedSurfaceIdentifiers = [carriedSurfaceIdentifiers copy];
        _elidedSurfaceIdentifiers = [elidedSurfaceIdentifiers copy];
        _boundaryInputIdentifiers = [boundaryInputIdentifiers copy];
        _outputIdentifier = [outputIdentifier copy];
    }
    return self;
}
@end

@implementation ANEScheduledGraph
- (instancetype)initWithSurfaces:(NSArray<ANEScheduledSurface *> *)surfaces
                            tasks:(NSArray<ANEScheduledTask *> *)tasks
                    composedPlans:(NSArray<ANEScheduledRegionPlan *> *)composedPlans
                    peakSRAMBytes:(NSUInteger)peakSRAMBytes {
    self = [super init];
    if (self) { _surfaces = [surfaces copy]; _tasks = [tasks copy];
        _composedPlans = [composedPlans copy]; _peakSRAMBytes = peakSRAMBytes; }
    return self;
}
- (NSString *)textualDescription {
    NSMutableString *text = [NSMutableString string];
    for (ANEScheduledTask *task in _tasks)
        [text appendFormat:@"task %lu %@ kind=%lu mode=%lu tile=%lux%lu deps=%@\n",
            (unsigned long)task.index, task.nodeIdentifier,
            (unsigned long)task.operationKind, (unsigned long)task.numericMode,
            (unsigned long)task.tilePlan.rows, (unsigned long)task.tilePlan.count,
            task.dependencies];
    return text;
}
@end
