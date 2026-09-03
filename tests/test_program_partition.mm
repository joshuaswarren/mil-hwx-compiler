#import <Foundation/Foundation.h>

#import "ANEProgramPartition.h"

static int failures = 0;

static void expect(BOOL condition, NSString *message) {
    if (!condition) {
        fprintf(stderr, "FAIL: %s\n", message.UTF8String);
        failures++;
    }
}

static ANEProgramTaskDescriptor *task(NSUInteger index,
                                      NSArray<NSString *> *inputs,
                                      NSString *output,
                                      NSUInteger taskCount,
                                      NSUInteger outputBytes) {
    return [[ANEProgramTaskDescriptor alloc] initWithScheduledTaskIndex:index
        inputIdentifiers:inputs outputIdentifier:output
        encodedTaskCount:taskCount outputByteLength:outputBytes];
}

static ANEProgramTaskDescriptor *composedTask(
    NSUInteger index, NSArray<NSString *> *inputs, NSString *output,
    NSUInteger standaloneCount, NSUInteger composedContribution,
    NSUInteger outputBytes) {
    return [[ANEProgramTaskDescriptor alloc] initWithScheduledTaskIndex:index
        inputIdentifiers:inputs outputIdentifier:output
        encodedTaskCount:standaloneCount
        composedTaskCountContribution:composedContribution
        outputByteLength:outputBytes];
}

static NSArray<ANEProgramPartition *> *plan(
    NSArray<ANEProgramTaskDescriptor *> *tasks,
    NSArray<NSString *> *outputs,
    NSUInteger maximumInputs,
    NSNumber *maximumTasks,
    NSUInteger workingSetBytes,
    ANEProgramCompositionPredicate predicate) {
    return [ANEProgramPartitionPlanner partitionsForTasks:tasks
        finalOutputIdentifiers:outputs maximumInputCount:maximumInputs
        maximumTaskCount:maximumTasks workingSetBytes:workingSetBytes
        canCompose:predicate];
}

static BOOL indexesEqual(ANEProgramPartition *partition,
                         NSArray<NSNumber *> *indexes) {
    return [partition.scheduledTaskIndexes isEqualToArray:indexes];
}

static void testLinearChainFormsOnePartition(void) {
    NSArray *tasks = @[
        task(4, @[@"a", @"b"], @"middle", 2, 512),
        task(7, @[@"middle", @"bias"], @"result", 1, 512),
    ];
    NSArray *partitions = plan(tasks, @[@"result"], 8, @8, 4096,
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            return producer.scheduledTaskIndex == 4 &&
                   consumer.scheduledTaskIndex == 7;
        });
    expect(partitions.count == 1 &&
           indexesEqual(partitions.firstObject, @[@4, @7]),
           @"a supported linear transition forms one partition");
    ANEProgramPartition *partition = partitions.firstObject;
    expect([partition.boundaryInputIdentifiers
               isEqualToArray:@[@"a", @"b", @"bias"]],
           @"partition inputs exclude its internal value");
    expect([partition.boundaryOutputIdentifiers
               isEqualToArray:@[@"result"]],
           @"partition output contains the requested graph result");
    expect(partition.encodedTaskCount == 3 &&
           partition.internalStorageByteLength == 512,
           @"partition accounts for encoded tasks and internal storage");
}

static void testUnknownTransitionPreservesSeparateTasks(void) {
    NSArray *tasks = @[
        task(0, @[@"x"], @"middle", 1, 256),
        task(1, @[@"middle"], @"y", 1, 256),
    ];
    NSArray *partitions = plan(tasks, @[@"y"], 8, @8, 4096,
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            (void)producer;
            (void)consumer;
            return NO;
        });
    expect(partitions.count == 2 &&
           indexesEqual(partitions[0], @[@0]) &&
           indexesEqual(partitions[1], @[@1]),
           @"an unsupported transition keeps one task per partition");
}

static void testFanoutEndsTheProducerPartition(void) {
    NSArray *tasks = @[
        task(0, @[@"x"], @"shared", 1, 256),
        task(1, @[@"shared"], @"left", 1, 256),
        task(2, @[@"shared"], @"right", 1, 256),
    ];
    NSArray *partitions = plan(tasks, @[@"left", @"right"], 8, @8, 4096,
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            (void)producer;
            (void)consumer;
            return YES;
        });
    expect(partitions.count == 3,
           @"a value with two consumers remains an external partition boundary");
}

static void testFanoutContainedByPartitionStaysInternal(void) {
    NSArray *tasks = @[
        task(0, @[@"x"], @"scores", 1, 256),
        task(1, @[@"scores"], @"scaled", 1, 256),
        task(2, @[@"scaled"], @"stats", 1, 64),
        task(3, @[@"scaled", @"stats"], @"y", 1, 256),
    ];
    ANEProgramCompositionPredicate allow =
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            (void)producer;
            (void)consumer;
            return YES;
        };
    NSArray *partitions = plan(tasks, @[@"y"], 8, @8, 4096, allow);
    expect(partitions.count == 1 &&
           indexesEqual(partitions.firstObject, @[@0, @1, @2, @3]),
           @"fanout stays internal when every consumer is in the program");
    ANEProgramPartition *partition = partitions.firstObject;
    expect([partition.boundaryOutputIdentifiers
               isEqualToArray:@[@"y"]],
           @"contained fanout does not create another program output");
}

static void testLimitsIndependentlyEndPartitions(void) {
    NSArray *inputLimitedTasks = @[
        task(0, @[@"a", @"b"], @"middle", 1, 256),
        task(1, @[@"middle", @"c"], @"y", 1, 256),
    ];
    ANEProgramCompositionPredicate allow =
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            (void)producer;
            (void)consumer;
            return YES;
        };
    expect(plan(inputLimitedTasks, @[@"y"], 2, @8, 4096, allow).count == 2,
           @"the external-input limit ends a partition");

    NSArray *taskLimitedTasks = @[
        task(0, @[@"x"], @"middle", 2, 256),
        task(1, @[@"middle"], @"y", 2, 256),
    ];
    expect(plan(taskLimitedTasks, @[@"y"], 8, @3, 4096, allow).count == 2,
           @"the encoded-task limit ends a partition");

    NSArray *storageLimitedTasks = @[
        task(0, @[@"x"], @"middle", 1, 1024),
        task(1, @[@"middle"], @"y", 1, 256),
    ];
    expect(plan(storageLimitedTasks, @[@"y"], 8, @8, 512, allow).count == 2,
           @"the internal-storage limit ends a partition");
}

static void testComposedTaskCountUsesTransitionContribution(void) {
    NSArray *tasks = @[
        composedTask(0, @[@"x"], @"middle", 2, 2, 256),
        composedTask(1, @[@"middle"], @"y", 1, 0, 256),
    ];
    ANEProgramCompositionPredicate allow =
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            (void)producer;
            (void)consumer;
            return YES;
        };
    NSArray *combined = plan(tasks, @[@"y"], 8, @2, 4096, allow);
    ANEProgramPartition *combinedPartition = combined.firstObject;
    expect(combined.count == 1 && combinedPartition.encodedTaskCount == 2,
           @"a consumed successor contributes no additional hardware task");
    NSArray *separate = plan(tasks, @[@"y"], 8, @1, 4096, allow);
    ANEProgramPartition *separateConsumer = separate.count > 1
        ? separate[1] : nil;
    expect(separate.count == 2 && separateConsumer.encodedTaskCount == 1,
           @"the same successor uses its standalone count at a partition boundary");
}

static void testUnavailableTaskLimitDeclinesCompositionAndIsDeterministic(void) {
    NSArray *tasks = @[
        task(2, @[@"x"], @"a", 3, 128),
        task(5, @[@"a"], @"b", 4, 128),
        task(9, @[@"b"], @"y", 5, 128),
    ];
    ANEProgramCompositionPredicate allow =
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer) {
            (void)producer;
            (void)consumer;
            return YES;
        };
    NSArray *first = plan(tasks, @[@"y"], 8, nil, 4096, allow);
    NSArray *second = plan(tasks, @[@"y"], 8, nil, 4096, allow);
    expect(first.count == 3 &&
           indexesEqual(first[0], @[@2]) &&
           indexesEqual(first[1], @[@5]) &&
           indexesEqual(first[2], @[@9]),
           @"an unavailable task limit keeps tasks in separate programs");
    expect([[first.firstObject textualDescription]
               isEqualToString:[second.firstObject textualDescription]],
           @"the same task order produces the same partition description");
}

static NSArray<ANEProgramTransitionRecord *> *transitionsFor(
    NSArray<ANEProgramTaskDescriptor *> *tasks,
    NSArray<NSString *> *outputs, NSUInteger maximumInputs,
    NSNumber *maximumTasks, NSUInteger workingSetBytes,
    ANEProgramCompositionReasonedPredicate predicate,
    NSArray<ANEProgramPartition *> **partitionsOut) {
    NSMutableArray<ANEProgramTransitionRecord *> *records =
        [NSMutableArray array];
    NSArray *partitions = [ANEProgramPartitionPlanner partitionsForTasks:tasks
        finalOutputIdentifiers:outputs maximumInputCount:maximumInputs
        maximumTaskCount:maximumTasks workingSetBytes:workingSetBytes
        reasonedCanCompose:predicate transitions:records];
    if (partitionsOut) *partitionsOut = partitions;
    return records;
}

static ANEProgramCompositionReasonedPredicate allowAll(void) {
    return ^BOOL(ANEProgramTaskDescriptor *producer,
                 ANEProgramTaskDescriptor *consumer, NSString **reason) {
        (void)producer;
        (void)consumer;
        (void)reason;
        return YES;
    };
}

static void testTransitionRecordsNameEveryDeclineReason(void) {
    NSArray *chain = @[
        task(0, @[@"x"], @"a", 1, 256),
        task(1, @[@"a"], @"b", 1, 256),
        task(2, @[@"b"], @"c", 1, 256),
        task(3, @[@"c"], @"y", 1, 256),
    ];
    NSArray<ANEProgramPartition *> *partitions = nil;
    NSArray<ANEProgramTransitionRecord *> *records = transitionsFor(
        chain, @[@"y"], 8, @8, 4096,
        ^BOOL(ANEProgramTaskDescriptor *producer,
              ANEProgramTaskDescriptor *consumer, NSString **reason) {
            (void)producer;
            if (consumer.scheduledTaskIndex == 2) {
                *reason = @"no capability row for this pair";
                return NO;
            }
            return YES;
        }, &partitions);
    expect(records.count == 3,
           @"every adjacent scheduled pair produces exactly one record");
    expect(records.count == 3 && records[0].accepted &&
           records[0].producerTaskIndex == 0 &&
           records[0].consumerTaskIndex == 1,
           @"a composed transition is recorded as accepted");
    expect(records.count == 3 && !records[1].accepted &&
           [records[1].reason isEqualToString:
               @"no capability row for this pair"],
           @"a target decline keeps the target's exact reason");
    expect(records.count == 3 && records[2].accepted,
           @"planning restarts after a decline and composes the tail");
    expect(partitions.count == 2 &&
           indexesEqual(partitions[0], @[@0, @1]) &&
           indexesEqual(partitions[1], @[@2, @3]),
           @"records agree with the emitted partitions");
    expect([records[1].textualDescription containsString:@"result=declined"] &&
           [records[1].textualDescription containsString:@"producer=1"] &&
           [records[1].textualDescription containsString:@"consumer=2"],
           @"the textual record names both tasks and the outcome");

    NSArray *noEdge = @[
        task(0, @[@"x"], @"a", 1, 256),
        task(1, @[@"x"], @"y", 1, 256),
    ];
    records = transitionsFor(noEdge, @[@"a", @"y"], 8, @8, 4096, allowAll(),
                             nil);
    expect(records.count == 1 && !records[0].accepted &&
           [records[0].reason containsString:@"does not read the producer output 'a'"],
           @"a missing direct edge is reported with the producer output");

    NSArray *inputLimited = @[
        task(0, @[@"a", @"b"], @"middle", 1, 256),
        task(1, @[@"middle", @"c"], @"y", 1, 256),
    ];
    records = transitionsFor(inputLimited, @[@"y"], 2, @8, 4096, allowAll(),
                             nil);
    expect(records.count == 1 && !records[0].accepted &&
           [records[0].reason containsString:@"external input count 3 exceeds"],
           @"the input limit decline states the counted inputs");

    NSArray *taskLimited = @[
        task(0, @[@"x"], @"middle", 2, 256),
        task(1, @[@"middle"], @"y", 2, 256),
    ];
    records = transitionsFor(taskLimited, @[@"y"], 8, @3, 4096, allowAll(),
                             nil);
    expect(records.count == 1 && !records[0].accepted &&
           [records[0].reason containsString:@"encoded task count 4 exceeds"],
           @"the task limit decline states the counted tasks");

    NSArray *storageLimited = @[
        task(0, @[@"x"], @"middle", 1, 1024),
        task(1, @[@"middle"], @"y", 1, 256),
    ];
    records = transitionsFor(storageLimited, @[@"y"], 8, @8, 512, allowAll(),
                             nil);
    expect(records.count == 1 && !records[0].accepted &&
           [records[0].reason containsString:@"internal storage 1024 bytes exceeds"],
           @"the storage limit decline states the counted bytes");

    records = transitionsFor(taskLimited, @[@"y"], 8, nil, 4096, allowAll(),
                             nil);
    expect(records.count == 1 && !records[0].accepted &&
           [records[0].reason containsString:@"no measured program task limit"],
           @"an unavailable task limit is reported as such");

    NSArray *fanout = @[
        task(0, @[@"x"], @"shared", 1, 256),
        task(1, @[@"shared"], @"left", 1, 256),
        task(2, @[@"shared"], @"right", 1, 256),
    ];
    records = transitionsFor(fanout, @[@"left", @"right"], 8, @8, 4096,
                             allowAll(), nil);
    expect(records.count == 2 && !records[0].accepted &&
           [records[0].reason containsString:@"boundary outputs"],
           @"a fanout decline names the extra boundary outputs");
}

int main(void) {
    @autoreleasepool {
        testTransitionRecordsNameEveryDeclineReason();
        testLinearChainFormsOnePartition();
        testUnknownTransitionPreservesSeparateTasks();
        testFanoutEndsTheProducerPartition();
        testFanoutContainedByPartitionStaysInternal();
        testLimitsIndependentlyEndPartitions();
        testComposedTaskCountUsesTransitionContribution();
        testUnavailableTaskLimitDeclinesCompositionAndIsDeterministic();
        printf("program partition planning: %s\n",
               failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
