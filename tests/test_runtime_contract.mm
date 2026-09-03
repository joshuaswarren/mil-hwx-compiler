#import <Foundation/Foundation.h>

#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"
#import "ANEProvisionedRuntime.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    ++failures;
}

static ANEHWXArtifact *artifact(void) {
    NSArray<ANEHWXBinding *> *bindings = @[
        [[ANEHWXBinding alloc] initWithIdentifier:@"input"
            role:ANESurfaceRoleInput logicalByteLength:96
            allocationByteLength:128 ioSurfaceIndex:0
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:96],
        [[ANEHWXBinding alloc] initWithIdentifier:@"embedded-weight"
            role:ANESurfaceRoleWeight logicalByteLength:32
            allocationByteLength:32 ioSurfaceIndex:-1],
        [[ANEHWXBinding alloc] initWithIdentifier:@"output"
            role:ANESurfaceRoleOutput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:1
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
    ];
    return [[ANEHWXArtifact alloc]
        initWithImage:[NSMutableData dataWithLength:16]
                                        bindings:bindings];
}

static ANEHWXArtifact *artifactWithBindings(NSArray<ANEHWXBinding *> *bindings) {
    return [[ANEHWXArtifact alloc]
        initWithImage:[NSMutableData dataWithLength:16] bindings:bindings];
}

static ANEExecutableBundle *bundleWithArtifacts(
    NSArray<ANEHWXArtifact *> *artifacts) {
    return [[ANEExecutableBundle alloc] initWithTarget:@"H16G"
        artifacts:artifacts dispatchPlan:@[@0] passTrace:@[]];
}

static ANEHWXBinding *ioBinding(NSString *identifier, ANESurfaceRole role,
                               NSInteger index, NSUInteger bytes) {
    return [[ANEHWXBinding alloc] initWithIdentifier:identifier role:role
        logicalByteLength:bytes allocationByteLength:bytes
        ioSurfaceIndex:index rowStrideBytes:16 planeStrideBytes:32
        batchStrideBytes:bytes];
}

static ANEHWXArtifact *dispatchArtifact(NSArray<NSString *> *inputs,
                                        NSString *output,
                                        NSUInteger bytes) {
    NSMutableArray<ANEHWXBinding *> *bindings = [NSMutableArray array];
    NSInteger index = 0;
    for (NSString *input in inputs)
        [bindings addObject:ioBinding(input, ANESurfaceRoleInput,
                                      index++, bytes)];
    [bindings addObject:ioBinding(output, ANESurfaceRoleOutput, index, bytes)];
    return artifactWithBindings(bindings);
}

static ANEExecutableBundle *twoArtifactBundle(void) {
    return [[ANEExecutableBundle alloc] initWithTarget:@"H16G"
        artifacts:@[
            dispatchArtifact(@[@"a", @"b"], @"matrix", 64),
            dispatchArtifact(@[@"matrix"], @"y", 64),
        ] dispatchPlan:@[@0, @1] passTrace:@[]];
}

static void testAllocatesManifestSizedSurfaces(void) {
    NSError *error = nil;
    ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifact()])
        modelHash:@"test-cache-key" qos:21 error:&error];
    expect(runtime != nil && error == nil, @"single-artifact bundle is accepted");
    NSArray<ANEIOSurfaceBuffer *> *inputs =
        [runtime createInputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *outputs =
        [runtime createOutputBuffersWithError:&error];
    expect(inputs.count == 1 && inputs[0].allocationByteLength == 128 &&
           inputs[0].logicalByteLength == 96,
           @"input IOSurface uses manifest allocation and logical lengths");
    expect(outputs.count == 1 && outputs[0].allocationByteLength == 64,
           @"output IOSurface uses manifest allocation length");
}

static void testFailsClosedOutsideSupportedRuntimeContract(void) {
    NSError *error = nil;
    ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifact()])
        modelHash:@"test-cache-key" qos:21 error:&error];
    expect(![runtime evaluateInputs:@[] outputs:@[] error:&error] &&
           error.code == ANERuntimeErrorNotLoaded,
           @"evaluation before load fails closed");

    ANEEvaluationProfile *profile = [[ANEEvaluationProfile alloc] init];
    expect(profile.entries.count == 0 && profile.totalMicroseconds == 0.0 &&
           profile.performanceStatisticsMask == 0,
           @"a fresh profile is empty and requests no hardware statistics");
    error = nil;
    expect(![runtime evaluateInputs:@[] outputs:@[] profile:profile
                              error:&error] &&
           error.code == ANERuntimeErrorNotLoaded &&
           profile.entries.count == 0,
           @"profiled evaluation before load fails closed without entries");
}

static void testBundlePreservesOperationsAndTrace(void) {
    ANEHWXArtifact *described = [[ANEHWXArtifact alloc]
        initWithImage:[NSMutableData dataWithLength:16]
        bindings:artifact().bindings operations:@[@"matmul", @"gelu"]];
    ANEExecutableBundle *bundle = [[ANEExecutableBundle alloc]
        initWithTarget:@"H16G" artifacts:@[described] dispatchPlan:@[@0]
        passTrace:@[] compositionTrace:@[
            @"transition producer=0 consumer=1 result=declined reason=test"]];
    expect([bundle.artifacts[0].operations
               isEqualToArray:@[@"matmul", @"gelu"]] &&
           bundle.compositionTrace.count == 1,
           @"bundles carry program operations and the composition trace");
    expect(artifact().operations.count == 0 &&
           bundleWithArtifacts(@[artifact()]).compositionTrace.count == 0,
           @"undescribed artifacts and bundles default to empty records");
}

static void testSharesIntermediateSurfaceByIdentifier(void) {
    NSError *error = nil;
    ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:twoArtifactBundle()
        modelHashes:@[@"matmul-cache-key", @"gelu-cache-key"]
        qos:21 error:&error];
    expect(runtime != nil && error == nil,
           @"valid two-artifact dispatch is accepted");
    NSArray<ANEIOSurfaceBuffer *> *inputs =
        [runtime createInputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *outputs =
        [runtime createOutputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *matmul =
        [runtime surfaceBuffersForArtifactAtIndex:0 error:&error];
    NSArray<ANEIOSurfaceBuffer *> *gelu =
        [runtime surfaceBuffersForArtifactAtIndex:1 error:&error];
    expect(inputs.count == 2 && outputs.count == 1,
           @"runtime exposes only graph boundary surfaces");
    expect(matmul.count == 3 && gelu.count == 2 &&
           matmul[2] == gelu[0] &&
           [matmul[2].identifier isEqualToString:@"matrix"],
           @"producer and consumer receive the same intermediate IOSurface");
}

static void testRejectsMalformedDispatchPlans(void) {
    NSError *error = nil;
    ANEExecutableBundle *valid = twoArtifactBundle();
    ANEExecutableBundle *missingProducer = [[ANEExecutableBundle alloc]
        initWithTarget:@"H16G" artifacts:valid.artifacts
        dispatchPlan:@[@1] passTrace:@[]];
    ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:missingProducer modelHashes:@[@"a", @"b"]
        qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"dispatch plan cannot omit an intermediate producer");

    error = nil;
    ANEExecutableBundle *repeatedWrite = [[ANEExecutableBundle alloc]
        initWithTarget:@"H16G" artifacts:@[
            dispatchArtifact(@[@"a"], @"matrix", 64),
            dispatchArtifact(@[@"b"], @"matrix", 64),
        ] dispatchPlan:@[@0, @1] passTrace:@[]];
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:repeatedWrite modelHashes:@[@"a", @"b"]
        qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"two artifacts cannot write the same surface");

    error = nil;
    ANEExecutableBundle *cycle = [[ANEExecutableBundle alloc]
        initWithTarget:@"H16G" artifacts:@[
            dispatchArtifact(@[@"right"], @"left", 64),
            dispatchArtifact(@[@"left"], @"right", 64),
        ] dispatchPlan:@[@0, @1] passTrace:@[]];
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:cycle modelHashes:@[@"a", @"b"]
        qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"cyclic surface dependencies are rejected");

    error = nil;
    ANEExecutableBundle *badLayout = [[ANEExecutableBundle alloc]
        initWithTarget:@"H16G" artifacts:@[
            dispatchArtifact(@[@"a"], @"matrix", 64),
            dispatchArtifact(@[@"matrix"], @"y", 128),
        ] dispatchPlan:@[@0, @1] passTrace:@[]];
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:badLayout modelHashes:@[@"a", @"b"]
        qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"shared surface layouts must agree across artifacts");
}

static void testRejectsMalformedBindingManifest(void) {
    NSArray<ANEHWXBinding *> *duplicateIndices = @[
        [[ANEHWXBinding alloc] initWithIdentifier:@"input"
            role:ANESurfaceRoleInput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:0
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
        [[ANEHWXBinding alloc] initWithIdentifier:@"output"
            role:ANESurfaceRoleOutput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:0
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
    ];
    NSError *error = nil;
    ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifactWithBindings(duplicateIndices)])
        modelHash:@"test-cache-key" qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"duplicate IOSurface indices are rejected");

    NSArray<ANEHWXBinding *> *badStrides = @[
        [[ANEHWXBinding alloc] initWithIdentifier:@"input"
            role:ANESurfaceRoleInput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:0
            rowStrideBytes:32 planeStrideBytes:16 batchStrideBytes:64],
        [[ANEHWXBinding alloc] initWithIdentifier:@"output"
            role:ANESurfaceRoleOutput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:1
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
    ];
    error = nil;
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifactWithBindings(badStrides)])
        modelHash:@"test-cache-key" qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"non-monotonic surface strides are rejected");

    NSArray<ANEHWXBinding *> *badWeightIndex = @[
        [[ANEHWXBinding alloc] initWithIdentifier:@"input"
            role:ANESurfaceRoleInput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:0
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
        [[ANEHWXBinding alloc] initWithIdentifier:@"weight"
            role:ANESurfaceRoleWeight logicalByteLength:32
            allocationByteLength:32 ioSurfaceIndex:1],
        [[ANEHWXBinding alloc] initWithIdentifier:@"output"
            role:ANESurfaceRoleOutput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:1
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
    ];
    error = nil;
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifactWithBindings(badWeightIndex)])
        modelHash:@"test-cache-key" qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"weight bindings cannot claim IOSurface slots");

    NSArray<ANEHWXBinding *> *duplicateIdentifiers = @[
        [[ANEHWXBinding alloc] initWithIdentifier:@"surface"
            role:ANESurfaceRoleInput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:0
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
        [[ANEHWXBinding alloc] initWithIdentifier:@"surface"
            role:ANESurfaceRoleOutput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:1
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
    ];
    error = nil;
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifactWithBindings(
            duplicateIdentifiers)]) modelHash:@"test-cache-key" qos:21
        error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"duplicate binding identifiers are rejected");

    NSArray<ANEHWXBinding *> *gappedIndices = @[
        [[ANEHWXBinding alloc] initWithIdentifier:@"input"
            role:ANESurfaceRoleInput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:0
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
        [[ANEHWXBinding alloc] initWithIdentifier:@"output"
            role:ANESurfaceRoleOutput logicalByteLength:64
            allocationByteLength:64 ioSurfaceIndex:2
            rowStrideBytes:16 planeStrideBytes:32 batchStrideBytes:64],
    ];
    error = nil;
    runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:bundleWithArtifacts(@[artifactWithBindings(gappedIndices)])
        modelHash:@"test-cache-key" qos:21 error:&error];
    expect(runtime == nil && error.code == ANERuntimeErrorUnsupportedBundle,
           @"gapped IOSurface indices are rejected");
}

int main(void) {
    @autoreleasepool {
        testAllocatesManifestSizedSurfaces();
        testFailsClosedOutsideSupportedRuntimeContract();
        testBundlePreservesOperationsAndTrace();
        testSharesIntermediateSurfaceByIdentifier();
        testRejectsMalformedDispatchPlans();
        testRejectsMalformedBindingManifest();
        printf("runtime contract: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
