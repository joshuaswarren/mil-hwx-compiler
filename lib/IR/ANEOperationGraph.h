#import <Foundation/Foundation.h>

#import "ANEDiagnostic.h"
#import "ANEGraphIR.h"

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, ANEOperationKind) {
    ANEOperationKindConstant,
    ANEOperationKindConv,
    ANEOperationKindMatmul,
    ANEOperationKindALU,
    ANEOperationKindLUT,
    ANEOperationKindReduce,
    ANEOperationKindLayout,
    ANEOperationKindQuantize,
    ANEOperationKindDequantize,
    ANEOperationKindHighLevel,
    ANEOperationKindUnsupported,
};

typedef NS_ENUM(NSUInteger, ANEPhysicalLayout) {
    ANEPhysicalLayoutUnknown,
    ANEPhysicalLayoutLinear,
    ANEPhysicalLayoutMatrix,
    ANEPhysicalLayoutTensor4D,
    ANEPhysicalLayoutNCHW,
    ANEPhysicalLayoutBHSD,
};

typedef NS_ENUM(NSUInteger, ANELegalNumericMode) {
    ANELegalNumericModeFP16,
    ANELegalNumericModeW8A8InputBoundary,
    ANELegalNumericModeW8A8Packed,
    ANELegalNumericModeW8A8OutputBoundary,
};

@class ANEOperationNode;

@interface ANEOperationNode : NSObject
@property(nonatomic, readonly) NSUInteger ordinal;
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly, copy) NSString *operationName;
@property(nonatomic, readonly) ANEOperationKind kind;
@property(nonatomic, readonly) ANEValueType *outputType;
@property(nonatomic, readonly, nullable) ANEGraphOperation *sourceOperation;
@property(nonatomic, readonly, copy) NSArray<ANEOperationNode *> *inputs;
@property(nonatomic, readonly, copy) NSArray<NSString *> *externalValueNames;
@property(nonatomic, readonly, copy) NSArray<ANEOperationNode *> *users;
@property(nonatomic, readonly) ANEPhysicalLayout physicalLayout;
@property(nonatomic, readonly) ANELegalNumericMode numericMode;
@property(nonatomic, readonly) BOOL foldedIntoNumericBoundary;
- (instancetype)initWithIdentifier:(NSString *)identifier
                      operationName:(NSString *)operationName
                               kind:(ANEOperationKind)kind
                         outputType:(ANEValueType *)outputType
                             inputs:(NSArray<ANEOperationNode *> *)inputs
                 externalValueNames:(NSArray<NSString *> *)externalValueNames
                    sourceOperation:(nullable ANEGraphOperation *)sourceOperation;
- (void)applyPhysicalLayout:(ANEPhysicalLayout)layout
                numericMode:(ANELegalNumericMode)numericMode;
- (void)markFoldedIntoNumericBoundary;
@end

@interface ANEGraphRegion : NSObject
@property(nonatomic, readonly, copy) NSString *identifier;
@property(nonatomic, readonly, copy) NSArray<ANEOperationNode *> *nodes;
@property(nonatomic, readonly, copy) NSArray<NSString *> *materializedValues;
- (instancetype)initWithIdentifier:(NSString *)identifier
                              nodes:(NSArray<ANEOperationNode *> *)nodes
                 materializedValues:(NSArray<NSString *> *)materializedValues;
@end

@interface ANEOperationGraph : NSObject
@property(nonatomic, readonly) ANEGraphFunction *sourceFunction;
@property(nonatomic, readonly, copy) NSArray<ANEOperationNode *> *nodes;
@property(nonatomic, readonly, copy) NSArray<ANEGraphRegion *> *regions;
@property(nonatomic, readonly, copy) NSArray<NSString *> *outputValueNames;
- (nullable instancetype)initWithFunction:(ANEGraphFunction *)function
                              diagnostics:(ANEDiagnosticEngine *)diagnostics;
- (nullable ANEOperationNode *)nodeForValueName:(NSString *)valueName;
- (BOOL)replaceNode:(ANEOperationNode *)node
          withNodes:(NSArray<ANEOperationNode *> *)nodes
    replacementNode:(ANEOperationNode *)replacement
         diagnostics:(ANEDiagnosticEngine *)diagnostics;
- (void)removeNodes:(NSArray<ANEOperationNode *> *)nodes;
- (void)setPlannedRegions:(NSArray<ANEGraphRegion *> *)regions;
- (NSString *)textualDescription;
@end

NS_ASSUME_NONNULL_END
