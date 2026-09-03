#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ANEProgramTaskDescriptor : NSObject
@property(nonatomic, readonly) NSUInteger scheduledTaskIndex;
@property(nonatomic, readonly, copy) NSArray<NSString *> *inputIdentifiers;
@property(nonatomic, readonly, copy) NSString *outputIdentifier;
@property(nonatomic, readonly) NSUInteger encodedTaskCount;
@property(nonatomic, readonly) NSUInteger composedTaskCountContribution;
@property(nonatomic, readonly) NSUInteger outputByteLength;
- (instancetype)initWithScheduledTaskIndex:(NSUInteger)scheduledTaskIndex
    inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
    outputIdentifier:(NSString *)outputIdentifier
    encodedTaskCount:(NSUInteger)encodedTaskCount
    outputByteLength:(NSUInteger)outputByteLength;
- (instancetype)initWithScheduledTaskIndex:(NSUInteger)scheduledTaskIndex
    inputIdentifiers:(NSArray<NSString *> *)inputIdentifiers
    outputIdentifier:(NSString *)outputIdentifier
    encodedTaskCount:(NSUInteger)encodedTaskCount
    composedTaskCountContribution:(NSUInteger)composedTaskCountContribution
    outputByteLength:(NSUInteger)outputByteLength;
@end

typedef BOOL (^ANEProgramCompositionPredicate)(
    ANEProgramTaskDescriptor *producer,
    ANEProgramTaskDescriptor *consumer);

/// A composition predicate that also reports why a transition declined.
/// `reason` is filled only when the predicate returns NO.
typedef BOOL (^ANEProgramCompositionReasonedPredicate)(
    ANEProgramTaskDescriptor *producer,
    ANEProgramTaskDescriptor *consumer,
    NSString * _Nullable * _Nonnull reason);

@interface ANEProgramPartition : NSObject
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *scheduledTaskIndexes;
@property(nonatomic, readonly, copy) NSArray<NSString *> *boundaryInputIdentifiers;
@property(nonatomic, readonly, copy) NSArray<NSString *> *boundaryOutputIdentifiers;
@property(nonatomic, readonly) NSUInteger encodedTaskCount;
@property(nonatomic, readonly) NSUInteger internalStorageByteLength;
- (NSString *)textualDescription;
@end

/// One planner decision about a producer-to-consumer transition. Every
/// adjacent scheduled pair produces exactly one record so a profile can list
/// every rejected transition with its exact reason.
@interface ANEProgramTransitionRecord : NSObject
@property(nonatomic, readonly) NSUInteger producerTaskIndex;
@property(nonatomic, readonly) NSUInteger consumerTaskIndex;
@property(nonatomic, readonly) BOOL accepted;
@property(nonatomic, readonly, copy) NSString *reason;
- (NSString *)textualDescription;
@end

@interface ANEProgramPartitionPlanner : NSObject
+ (NSArray<ANEProgramPartition *> *)partitionsForTasks:
    (NSArray<ANEProgramTaskDescriptor *> *)tasks
    finalOutputIdentifiers:(NSArray<NSString *> *)finalOutputIdentifiers
    maximumInputCount:(NSUInteger)maximumInputCount
    maximumTaskCount:(nullable NSNumber *)maximumTaskCount
    workingSetBytes:(NSUInteger)workingSetBytes
    canCompose:(ANEProgramCompositionPredicate)canCompose;
+ (NSArray<ANEProgramPartition *> *)partitionsForTasks:
    (NSArray<ANEProgramTaskDescriptor *> *)tasks
    finalOutputIdentifiers:(NSArray<NSString *> *)finalOutputIdentifiers
    maximumInputCount:(NSUInteger)maximumInputCount
    maximumTaskCount:(nullable NSNumber *)maximumTaskCount
    workingSetBytes:(NSUInteger)workingSetBytes
    reasonedCanCompose:(ANEProgramCompositionReasonedPredicate)canCompose
    transitions:(nullable NSMutableArray<ANEProgramTransitionRecord *> *)transitions;
@end

NS_ASSUME_NONNULL_END
