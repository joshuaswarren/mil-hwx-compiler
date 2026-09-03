#import "ANEProgramPartition.h"

@implementation ANEProgramTaskDescriptor
- (instancetype)initWithScheduledTaskIndex:(NSUInteger)scheduledTaskIndex
    inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
    outputIdentifier:(NSString *)outputIdentifier
    encodedTaskCount:(NSUInteger)encodedTaskCount
    outputByteLength:(NSUInteger)outputByteLength {
    return [self initWithScheduledTaskIndex:scheduledTaskIndex
        inputIdentifiers:inputIdentifiers outputIdentifier:outputIdentifier
        encodedTaskCount:encodedTaskCount
        composedTaskCountContribution:encodedTaskCount
        outputByteLength:outputByteLength];
}
- (instancetype)initWithScheduledTaskIndex:(NSUInteger)scheduledTaskIndex
    inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
    outputIdentifier:(NSString *)outputIdentifier
    encodedTaskCount:(NSUInteger)encodedTaskCount
    composedTaskCountContribution:(NSUInteger)composedTaskCountContribution
    outputByteLength:(NSUInteger)outputByteLength {
    self = [super init];
    if (self) {
        _scheduledTaskIndex = scheduledTaskIndex;
        _inputIdentifiers = [inputIdentifiers copy];
        _outputIdentifier = [outputIdentifier copy];
        _encodedTaskCount = encodedTaskCount;
        _composedTaskCountContribution = composedTaskCountContribution;
        _outputByteLength = outputByteLength;
    }
    return self;
}
@end

@interface ANEProgramPartition ()
- (instancetype)initWithScheduledTaskIndexes:(NSArray<NSNumber *> *)taskIndexes
    boundaryInputIdentifiers:(NSArray<NSString *> *)boundaryInputs
    boundaryOutputIdentifiers:(NSArray<NSString *> *)boundaryOutputs
    encodedTaskCount:(NSUInteger)encodedTaskCount
    internalStorageByteLength:(NSUInteger)internalStorageByteLength;
@end

@implementation ANEProgramPartition
- (instancetype)initWithScheduledTaskIndexes:(NSArray<NSNumber *> *)taskIndexes
    boundaryInputIdentifiers:(NSArray<NSString *> *)boundaryInputs
    boundaryOutputIdentifiers:(NSArray<NSString *> *)boundaryOutputs
    encodedTaskCount:(NSUInteger)encodedTaskCount
    internalStorageByteLength:(NSUInteger)internalStorageByteLength {
    self = [super init];
    if (self) {
        _scheduledTaskIndexes = [taskIndexes copy];
        _boundaryInputIdentifiers = [boundaryInputs copy];
        _boundaryOutputIdentifiers = [boundaryOutputs copy];
        _encodedTaskCount = encodedTaskCount;
        _internalStorageByteLength = internalStorageByteLength;
    }
    return self;
}
- (NSString *)textualDescription {
    return [NSString stringWithFormat:
        @"tasks=%@ inputs=%@ outputs=%@ encoded=%lu internal=%lu",
        [_scheduledTaskIndexes componentsJoinedByString:@","],
        [_boundaryInputIdentifiers componentsJoinedByString:@","],
        [_boundaryOutputIdentifiers componentsJoinedByString:@","],
        (unsigned long)_encodedTaskCount,
        (unsigned long)_internalStorageByteLength];
}
@end

@interface ANEProgramTransitionRecord ()
- (instancetype)initWithProducerTaskIndex:(NSUInteger)producer
                        consumerTaskIndex:(NSUInteger)consumer
                                 accepted:(BOOL)accepted
                                   reason:(NSString *)reason;
@end

@implementation ANEProgramTransitionRecord
- (instancetype)initWithProducerTaskIndex:(NSUInteger)producer
                        consumerTaskIndex:(NSUInteger)consumer
                                 accepted:(BOOL)accepted
                                   reason:(NSString *)reason {
    self = [super init];
    if (self) {
        _producerTaskIndex = producer;
        _consumerTaskIndex = consumer;
        _accepted = accepted;
        _reason = [reason copy];
    }
    return self;
}
- (NSString *)textualDescription {
    return [NSString stringWithFormat:
        @"transition producer=%lu consumer=%lu result=%@ reason=%@",
        (unsigned long)_producerTaskIndex, (unsigned long)_consumerTaskIndex,
        _accepted ? @"composed" : @"declined", _reason];
}
@end

static NSDictionary<NSString *, NSNumber *> *consumerCounts(
    NSArray<ANEProgramTaskDescriptor *> *tasks) {
    NSMutableDictionary<NSString *, NSNumber *> *counts =
        [NSMutableDictionary dictionary];
    for (ANEProgramTaskDescriptor *task in tasks)
        for (NSString *identifier in task.inputIdentifiers)
            counts[identifier] = @(counts[identifier].unsignedIntegerValue + 1);
    return [counts copy];
}

static BOOL addWithoutOverflow(NSUInteger left, NSUInteger right,
                               NSUInteger *result) {
    if (right > NSUIntegerMax - left) return NO;
    *result = left + right;
    return YES;
}

static ANEProgramPartition *makePartition(
    NSArray<ANEProgramTaskDescriptor *> *allTasks,
    NSRange range,
    NSSet<NSString *> *finalOutputs,
    NSDictionary<NSString *, NSNumber *> *allConsumerCounts) {
    NSMutableArray<NSNumber *> *indexes = [NSMutableArray array];
    NSMutableArray<NSString *> *boundaryInputs = [NSMutableArray array];
    NSMutableSet<NSString *> *boundaryInputSet = [NSMutableSet set];
    NSMutableSet<NSString *> *produced = [NSMutableSet set];
    NSUInteger encodedTaskCount = 0;
    BOOL encodedTaskCountValid = YES;

    for (NSUInteger index = range.location; index < NSMaxRange(range); index++) {
        ANEProgramTaskDescriptor *task = allTasks[index];
        [indexes addObject:@(task.scheduledTaskIndex)];
        for (NSString *input in task.inputIdentifiers)
            if (![produced containsObject:input] &&
                ![boundaryInputSet containsObject:input]) {
                [boundaryInputSet addObject:input];
                [boundaryInputs addObject:input];
            }
        [produced addObject:task.outputIdentifier];
        NSUInteger nextCount = 0;
        NSUInteger contribution = index == range.location
            ? task.encodedTaskCount : task.composedTaskCountContribution;
        encodedTaskCountValid &= addWithoutOverflow(
            encodedTaskCount, contribution, &nextCount);
        encodedTaskCount = nextCount;
    }

    NSMutableDictionary<NSString *, NSNumber *> *insideConsumerCounts =
        [NSMutableDictionary dictionary];
    for (NSUInteger index = range.location; index < NSMaxRange(range); index++)
        for (NSString *input in allTasks[index].inputIdentifiers)
            insideConsumerCounts[input] =
                @(insideConsumerCounts[input].unsignedIntegerValue + 1);

    NSMutableArray<NSString *> *boundaryOutputs = [NSMutableArray array];
    NSUInteger internalStorageBytes = 0;
    BOOL internalStorageValid = YES;
    for (NSUInteger index = range.location; index < NSMaxRange(range); index++) {
        ANEProgramTaskDescriptor *task = allTasks[index];
        NSUInteger allConsumers =
            allConsumerCounts[task.outputIdentifier].unsignedIntegerValue;
        NSUInteger insideConsumers =
            insideConsumerCounts[task.outputIdentifier].unsignedIntegerValue;
        BOOL isBoundary = [finalOutputs containsObject:task.outputIdentifier] ||
                          allConsumers > insideConsumers;
        if (isBoundary) [boundaryOutputs addObject:task.outputIdentifier];
        if (insideConsumers > 0 && !isBoundary) {
            NSUInteger nextBytes = 0;
            internalStorageValid &= addWithoutOverflow(
                internalStorageBytes, task.outputByteLength, &nextBytes);
            internalStorageBytes = nextBytes;
        }
    }
    if (!encodedTaskCountValid) encodedTaskCount = NSUIntegerMax;
    if (!internalStorageValid) internalStorageBytes = NSUIntegerMax;
    return [[ANEProgramPartition alloc]
        initWithScheduledTaskIndexes:indexes
        boundaryInputIdentifiers:boundaryInputs
        boundaryOutputIdentifiers:boundaryOutputs
        encodedTaskCount:encodedTaskCount
        internalStorageByteLength:internalStorageBytes];
}

static NSString *partitionLimitFailure(ANEProgramPartition *partition,
                                       NSUInteger maximumInputCount,
                                       NSNumber *maximumTaskCount,
                                       NSUInteger workingSetBytes) {
    if (partition.boundaryInputIdentifiers.count > maximumInputCount)
        return [NSString stringWithFormat:
            @"external input count %lu exceeds the program input limit %lu",
            (unsigned long)partition.boundaryInputIdentifiers.count,
            (unsigned long)maximumInputCount];
    if (maximumTaskCount &&
        partition.encodedTaskCount > maximumTaskCount.unsignedIntegerValue)
        return [NSString stringWithFormat:
            @"encoded task count %lu exceeds the program task limit %lu",
            (unsigned long)partition.encodedTaskCount,
            (unsigned long)maximumTaskCount.unsignedIntegerValue];
    if (partition.internalStorageByteLength > workingSetBytes)
        return [NSString stringWithFormat:
            @"internal storage %lu bytes exceeds the working set %lu bytes",
            (unsigned long)partition.internalStorageByteLength,
            (unsigned long)workingSetBytes];
    return nil;
}

static NSString *boundaryOutputFailure(ANEProgramPartition *partition) {
    return [NSString stringWithFormat:
        @"program ending here would expose %lu boundary outputs (%@); "
         "one output per program is supported",
        (unsigned long)partition.boundaryOutputIdentifiers.count,
        [partition.boundaryOutputIdentifiers componentsJoinedByString:@","]];
}

@implementation ANEProgramPartitionPlanner
+ (NSArray<ANEProgramPartition *> *)partitionsForTasks:
    (NSArray<ANEProgramTaskDescriptor *> *)tasks
    finalOutputIdentifiers:(NSArray<NSString *> *)finalOutputIdentifiers
    maximumInputCount:(NSUInteger)maximumInputCount
    maximumTaskCount:(NSNumber *)maximumTaskCount
    workingSetBytes:(NSUInteger)workingSetBytes
    canCompose:(ANEProgramCompositionPredicate)canCompose {
    return [self partitionsForTasks:tasks
        finalOutputIdentifiers:finalOutputIdentifiers
        maximumInputCount:maximumInputCount maximumTaskCount:maximumTaskCount
        workingSetBytes:workingSetBytes
        reasonedCanCompose:^BOOL(ANEProgramTaskDescriptor *producer,
                                 ANEProgramTaskDescriptor *consumer,
                                 NSString **reason) {
            if (canCompose(producer, consumer)) return YES;
            *reason = @"target declined the transition";
            return NO;
        }
        transitions:nil];
}

+ (NSArray<ANEProgramPartition *> *)partitionsForTasks:
    (NSArray<ANEProgramTaskDescriptor *> *)tasks
    finalOutputIdentifiers:(NSArray<NSString *> *)finalOutputIdentifiers
    maximumInputCount:(NSUInteger)maximumInputCount
    maximumTaskCount:(NSNumber *)maximumTaskCount
    workingSetBytes:(NSUInteger)workingSetBytes
    reasonedCanCompose:(ANEProgramCompositionReasonedPredicate)canCompose
    transitions:(NSMutableArray<ANEProgramTransitionRecord *> *)transitions {
    if (tasks.count == 0) return @[];
    NSDictionary<NSString *, NSNumber *> *counts = consumerCounts(tasks);
    NSSet<NSString *> *finalOutputs =
        [NSSet setWithArray:finalOutputIdentifiers];
    NSMutableArray<ANEProgramPartition *> *partitions =
        [NSMutableArray array];
    // Keyed by the consumer position so a pair re-evaluated from a later
    // partition start overwrites the earlier, provisional decision.
    NSMutableDictionary<NSNumber *, ANEProgramTransitionRecord *> *records =
        [NSMutableDictionary dictionary];
    void (^record)(NSUInteger, BOOL, NSString *) =
        ^(NSUInteger consumerPosition, BOOL accepted, NSString *reason) {
            records[@(consumerPosition)] = [[ANEProgramTransitionRecord alloc]
                initWithProducerTaskIndex:
                    tasks[consumerPosition - 1].scheduledTaskIndex
                consumerTaskIndex:tasks[consumerPosition].scheduledTaskIndex
                accepted:accepted reason:reason];
        };
    NSUInteger partitionStart = 0;
    while (partitionStart < tasks.count) {
        NSUInteger bestEnd = partitionStart;
        NSUInteger lastExtended = partitionStart;
        NSString *pendingBoundaryReason = nil;
        for (NSUInteger candidateEnd = partitionStart + 1;
             candidateEnd < tasks.count; ++candidateEnd) {
            if (!maximumTaskCount) {
                record(candidateEnd, NO,
                       @"target has no measured program task limit");
                break;
            }
            ANEProgramTaskDescriptor *producer = tasks[candidateEnd - 1];
            ANEProgramTaskDescriptor *consumer = tasks[candidateEnd];
            BOOL directEdge = [consumer.inputIdentifiers
                containsObject:producer.outputIdentifier];
            if (!directEdge) {
                record(candidateEnd, NO, [NSString stringWithFormat:
                    @"consumer does not read the producer output '%@'",
                    producer.outputIdentifier]);
                break;
            }
            NSString *reason = nil;
            if (!canCompose(producer, consumer, &reason)) {
                record(candidateEnd, NO,
                       reason ?: @"target declined the transition");
                break;
            }

            NSRange trialRange = NSMakeRange(
                partitionStart, candidateEnd - partitionStart + 1);
            ANEProgramPartition *trial = makePartition(
                tasks, trialRange, finalOutputs, counts);
            NSString *limitFailure = partitionLimitFailure(
                trial, maximumInputCount, maximumTaskCount, workingSetBytes);
            if (limitFailure) {
                record(candidateEnd, NO, limitFailure);
                break;
            }
            lastExtended = candidateEnd;
            if (trial.boundaryOutputIdentifiers.count == 1) {
                bestEnd = candidateEnd;
                pendingBoundaryReason = nil;
            } else {
                pendingBoundaryReason = boundaryOutputFailure(trial);
            }
        }
        for (NSUInteger position = partitionStart + 1;
             position <= bestEnd; ++position)
            record(position, YES, @"composed within one program");
        for (NSUInteger position = bestEnd + 1;
             position <= lastExtended; ++position)
            record(position, NO, pendingBoundaryReason ?:
                   @"program would expose more than one boundary output");
        NSRange finishedRange = NSMakeRange(
            partitionStart, bestEnd - partitionStart + 1);
        [partitions addObject:makePartition(
            tasks, finishedRange, finalOutputs, counts)];
        partitionStart = bestEnd + 1;
    }
    if (transitions) {
        NSArray<NSNumber *> *positions = [records.allKeys
            sortedArrayUsingSelector:@selector(compare:)];
        for (NSNumber *position in positions)
            [transitions addObject:records[position]];
    }
    return [partitions copy];
}
@end
