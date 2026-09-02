#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>

#import "ANECompiler.h"
#import "ANEDiagnostic.h"
#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"
#import "HWXImage.h"

#include <stdio.h>

static int failures = 0;
static void expect(BOOL condition, NSString *message) {
    if (condition) return;
    fprintf(stderr, "FAIL: %s\n", message.UTF8String);
    ++failures;
}

static NSData *fixture(NSString *name) {
    return [NSData dataWithContentsOfFile:
        [@"tests/fixtures" stringByAppendingPathComponent:name]];
}

static NSString *sha256(NSData *data) {
    uint8_t digest[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes, (CC_LONG)data.length, digest);
    NSMutableString *result = [NSMutableString stringWithCapacity:64];
    for (NSUInteger index = 0; index < sizeof(digest); ++index)
        [result appendFormat:@"%02x", digest[index]];
    return result;
}

static void expectSemanticSectionHashes(NSData *actualData,
                                        NSString *textHash,
                                        NSString *kernelHash,
                                        NSString *label) {
    NSError *error = nil;
    HWXImage *actual = [HWXImage imageWithData:actualData error:&error];
    HWXSection *actualText = [actual firstSectionNamed:@"__text"
                                             inSegment:@"__TEXT"];
    HWXSection *actualKern = [actual firstSectionNamed:@"__kern_0"
                                             inSegment:@"__KERN_0"];
    expect(actual != nil && error == nil,
           [label stringByAppendingString:@" reparses"]);
    expect([[sha256(actualText.data) lowercaseString] isEqualToString:textHash],
           [label stringByAppendingString:@" descriptor hash matches"]);
    expect([[sha256(actualKern.data) lowercaseString] isEqualToString:kernelHash],
           [label stringByAppendingString:@" constants hash matches"]);
}

static ANEExecutableBundle *compileCase(ANECompiler *compiler,
                                        NSString *fixtureName,
                                        NSString *modelName,
                                        ANEDiagnosticEngine *diagnostics) {
    NSURL *modelRoot = [NSURL fileURLWithPath:
        [@"tests/models" stringByAppendingPathComponent:modelName]
        isDirectory:YES];
    return [compiler compileMILData:fixture(fixtureName)
                          modelRoot:modelRoot target:@"H16G"
                         diagnostics:diagnostics];
}

static void testThreeEndToEndCompilations(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    NSArray<NSDictionary<NSString *, NSString *> *> *cases = @[
        @{@"fixture": @"conv_relu.mil", @"model": @"conv_relu",
          @"text": @"62a90244a4edc71a70d6f0630a2d72371b7dcb39d5194d22369781cb54094bff",
          @"kernel": @"44f8459ee020ef8a01d767df4e238eef58953984af4be906101e9f7bc1a4a1bb"},
        @{@"fixture": @"attention.mil", @"model": @"attention",
          @"text": @"a2d9c65e6440b7fcd1b01c40934c2fdedb5a76575d9c0a5bb86f2becd2e9a95c",
          @"kernel": @"b7b6085a1edc7def0f0bb2fc1fe345f1ba9d9a1a47e55b4516273149323b54d2"},
        @{@"fixture": @"w8a8_conv_chain.mil", @"model": @"w8a8",
          @"text": @"185de64f2a80436a319e7b7ef561a9ee05b9fba45142ddfc11e409e055ae69f8",
          @"kernel": @"54e18123ce030cef0fa8ac906f358e6bcba6ce19ee083173875ee21717407155"},
    ];
    for (NSDictionary<NSString *, NSString *> *testCase in cases) {
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = compileCase(
            compiler, testCase[@"fixture"], testCase[@"model"], diagnostics);
        expect(bundle != nil && diagnostics.errorCount == 0,
               [testCase[@"fixture"] stringByAppendingString:@" compiles"]);
        expect(bundle.artifacts.count == 1 &&
               bundle.dispatchPlan.count == 1,
               @"exact patterns each produce one dispatchable HWX");
        expect([bundle.passTrace isEqualToArray:@[
            @"mil.import-operation-graph", @"ane.normalize", @"ane.decompose",
            @"ane.fuse", @"h16g.legalize", @"h16g.plan", @"h16g.encode",
            @"h16g.write-object",
        ]], @"driver records the staged lowering pass trace");
        expectSemanticSectionHashes(bundle.artifacts[0].image,
                                    testCase[@"text"],
                                    testCase[@"kernel"],
                                    testCase[@"fixture"]);
    }
}

static void testBundleSerialization(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = compileCase(
        compiler, @"conv_relu.mil", @"conv_relu", diagnostics);
    NSURL *directory = [NSURL fileURLWithPath:@"build/bundle-conv"
                                  isDirectory:YES];
    NSError *error = nil;
    expect([bundle writeToDirectory:directory error:&error] && error == nil,
           @"bundle serializes atomically");
    NSData *manifestData = [NSData dataWithContentsOfURL:
        [directory URLByAppendingPathComponent:@"manifest.json"]];
    NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:
        manifestData options:0 error:&error];
    expect([manifest[@"target"] isEqualToString:@"H16G"] &&
           [manifest[@"artifacts"] count] == 1,
           @"serialized manifest records target and artifact");
    expect([[NSFileManager defaultManager] fileExistsAtPath:
        [[directory URLByAppendingPathComponent:@"program-0.hwx"] path]],
        @"serialized bundle contains emitted HWX");
    ANEExecutableBundle *roundTrip =
        [ANEExecutableBundle bundleWithContentsOfDirectory:directory
                                                      error:&error];
    expect(roundTrip != nil && error == nil,
           @"serialized bundle deserializes");
    expect([roundTrip.target isEqualToString:bundle.target] &&
           [roundTrip.passTrace isEqualToArray:bundle.passTrace] &&
           [roundTrip.dispatchPlan isEqualToArray:bundle.dispatchPlan] &&
           [roundTrip.artifacts[0].image isEqualToData:
               bundle.artifacts[0].image],
           @"bundle round trip preserves target, plan, trace and HWX bytes");
    ANEHWXBinding *roundTripInput = roundTrip.artifacts[0].bindings[0];
    expect(roundTripInput.logicalByteLength == 524288 &&
           roundTripInput.allocationByteLength == 524288 &&
           roundTripInput.rowStrideBytes == 128,
           @"bundle round trip preserves binding layout");
}

static void testUnsupportedTargetFailsClosed(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [compiler
        compileMILData:fixture(@"conv_relu.mil")
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                            isDirectory:YES]
        target:@"H15" diagnostics:diagnostics];
    expect(bundle == nil && diagnostics.errorCount == 1,
           @"unsupported target fails with one diagnostic");
}

static void testBundlePathEscapeFailsClosed(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = compileCase(
        compiler, @"conv_relu.mil", @"conv_relu", diagnostics);
    NSURL *directory = [NSURL fileURLWithPath:@"build/bundle-invalid"
                                  isDirectory:YES];
    NSError *error = nil;
    expect([bundle writeToDirectory:directory error:&error],
           @"invalid-bundle test starts from a valid serialized bundle");
    NSURL *manifestURL =
        [directory URLByAppendingPathComponent:@"manifest.json"];
    NSMutableDictionary *manifest = [[NSJSONSerialization
        JSONObjectWithData:[NSData dataWithContentsOfURL:manifestURL]
        options:NSJSONReadingMutableContainers error:&error] mutableCopy];
    manifest[@"artifacts"][0][@"file"] = @"../program-0.hwx";
    NSData *mutated = [NSJSONSerialization dataWithJSONObject:manifest
                                                       options:0 error:&error];
    expect([mutated writeToURL:manifestURL options:NSDataWritingAtomic
                         error:&error], @"invalid bundle manifest is written");
    ANEExecutableBundle *loaded =
        [ANEExecutableBundle bundleWithContentsOfDirectory:directory
                                                      error:&error];
    expect(loaded == nil && error != nil,
           @"bundle artifact path escape fails closed");
}

static void testMalformedExternalConstantFailsClosed(void) {
    NSFileManager *fileManager = NSFileManager.defaultManager;
    NSString *modelRoot = @"build/models/w8a8-invalid";
    NSString *weightsDirectory =
        [modelRoot stringByAppendingPathComponent:@"weights"];
    NSError *error = nil;
    expect([fileManager createDirectoryAtPath:weightsDirectory
                  withIntermediateDirectories:YES attributes:nil error:&error],
           @"malformed-blob test creates its private model directory");
    NSMutableData *blob = [[NSData dataWithContentsOfFile:
        @"tests/models/w8a8/weights/weight.bin"] mutableCopy];
    expect(blob.length == 16704, @"malformed-blob source fixture is complete");
    uint64_t zero = 0;
    [blob replaceBytesInRange:NSMakeRange(64 + 8, sizeof(zero))
                    withBytes:&zero];
    NSString *blobPath = [weightsDirectory
        stringByAppendingPathComponent:@"weight.bin"];
    expect([blob writeToFile:blobPath options:NSDataWritingAtomic error:&error],
           @"malformed-blob fixture is written");

    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [compiler
        compileMILData:fixture(@"w8a8_conv_chain.mil")
        modelRoot:[NSURL fileURLWithPath:modelRoot isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle == nil && diagnostics.errorCount == 1,
           @"malformed external constant fails closed");
    expect([diagnostics.diagnostics.firstObject.code
               isEqualToString:@"ane.model.invalid-blob-header"],
           @"malformed external constant reports its exact contract failure");
}

static void testModelPathEscapeFailsClosed(void) {
    NSString *source = [[NSString alloc] initWithData:fixture(@"conv_relu.mil")
                                             encoding:NSUTF8StringEncoding];
    source = [source stringByReplacingOccurrencesOfString:
        @"@model_path/weights/weight.bin"
        withString:@"@model_path/../outside.bin"];
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = [compiler
        compileMILData:[source dataUsingEncoding:NSUTF8StringEncoding]
        modelRoot:[NSURL fileURLWithPath:@"tests/models/conv_relu"
                            isDirectory:YES]
        target:@"H16G" diagnostics:diagnostics];
    expect(bundle == nil && diagnostics.errorCount == 1,
           @"model path escape fails closed");
    expect([diagnostics.diagnostics.firstObject.code
               isEqualToString:@"ane.model.path-escape"],
           @"model path escape reports its exact contract failure");
}

static void testCleanEncoderNeedsNoResourceConfiguration(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = compileCase(
        compiler, @"conv_relu.mil", @"conv_relu", diagnostics);
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"clean emission does not load legacy descriptor rows or skeletons");
}

static NSArray<NSString *> *bindingIdentifiers(ANEHWXArtifact *artifact) {
    NSMutableArray<NSString *> *identifiers = [NSMutableArray array];
    for (ANEHWXBinding *binding in artifact.bindings)
        [identifiers addObject:binding.identifier];
    return [identifiers copy];
}

static void testMatmulGELUUsesOrderedPrimitiveArtifacts(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    for (NSString *fixtureName in @[@"matmul_gelu_128.mil",
                                    @"matmul_gelu_256.mil"]) {
        ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *bundle = compileCase(
            compiler, fixtureName, @"conv_relu", diagnostics);
        expect(bundle != nil && diagnostics.errorCount == 0,
               [fixtureName stringByAppendingString:@" compiles"]);
        expect(bundle.artifacts.count == 2 &&
               [bundle.dispatchPlan isEqualToArray:@[@0, @1]],
               @"matmul and GELU produce two ordered primitive artifacts");
        expect([bundle.sharedSurfaceIdentifiers isEqualToArray:@[@"matrix"]],
               @"the dispatch plan records its shared matrix surface");
        if (bundle.artifacts.count == 2) {
            expect([bindingIdentifiers(bundle.artifacts[0])
                       isEqualToArray:@[@"a", @"b", @"matrix"]],
                   @"matmul artifact exposes graph inputs and matrix output");
            expect([bindingIdentifiers(bundle.artifacts[1])
                       isEqualToArray:@[@"matrix", @"y"]],
                   @"GELU artifact consumes the same matrix identifier");
        }

        ANEDiagnosticEngine *repeatDiagnostics =
            [[ANEDiagnosticEngine alloc] init];
        ANEExecutableBundle *repeat = compileCase(
            compiler, fixtureName, @"conv_relu", repeatDiagnostics);
        BOOL deterministic = repeat.artifacts.count == bundle.artifacts.count;
        for (NSUInteger index = 0;
             deterministic && index < bundle.artifacts.count; ++index)
            deterministic = [repeat.artifacts[index].image
                isEqualToData:bundle.artifacts[index].image];
        expect(deterministic && repeatDiagnostics.errorCount == 0,
               @"composed primitive artifact bytes are deterministic");

        NSURL *directory = [NSURL fileURLWithPath:
            [@"build" stringByAppendingPathComponent:
                [fixtureName stringByDeletingPathExtension]]
            isDirectory:YES];
        NSError *error = nil;
        expect([bundle writeToDirectory:directory error:&error],
               @"composed bundle serializes");
        ANEExecutableBundle *roundTrip =
            [ANEExecutableBundle bundleWithContentsOfDirectory:directory
                                                          error:&error];
        expect(roundTrip != nil &&
               [roundTrip.dispatchPlan isEqualToArray:@[@0, @1]] &&
               [roundTrip.sharedSurfaceIdentifiers
                   isEqualToArray:@[@"matrix"]],
               @"composed dispatch and shared surface survive serialization");
    }
}

static void testSingleTileOnlineReductionUsesPrimitiveArtifacts(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = compileCase(
        compiler, @"fa2_fp16_s128_d128.mil", @"conv_relu", diagnostics);
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"single-tile FP16 online reduction compiles");
    expect(bundle.artifacts.count == 8 &&
           [bundle.dispatchPlan isEqualToArray:
               @[@0,@1,@2,@3,@4,@5,@6,@7]],
           @"online reduction emits eight ordered primitive artifacts");
    expect(bundle.sharedSurfaceIdentifiers.count == 7,
           @"each internal online-reduction value is a shared surface");
    if (bundle.artifacts.count == 8) {
        expect([bindingIdentifiers(bundle.artifacts.firstObject)
                   isEqualToArray:@[@"q", @"k", @"scores"]],
               @"the first contraction consumes the graph Q and K inputs");
        expect([bindingIdentifiers(bundle.artifacts.lastObject)
                   isEqualToArray:@[@"probabilities", @"v", @"y"]],
               @"the final contraction writes the graph output");
        expect([bindingIdentifiers(bundle.artifacts[6])
                   isEqualToArray:@[@"probabilities.sum",
                                    @"probabilities.exp",
                                    @"probabilities"]],
               @"normalization uses the measured denominator-first order");
    }
}

static void testAffineScanUsesOrderedPrimitiveArtifacts(void) {
    ANECompiler *compiler = [[ANECompiler alloc] init];
    ANEDiagnosticEngine *diagnostics = [[ANEDiagnosticEngine alloc] init];
    ANEExecutableBundle *bundle = compileCase(
        compiler, @"affine_scan_fp16_4.mil", @"conv_relu", diagnostics);
    expect(bundle != nil && diagnostics.errorCount == 0,
           @"four-transition FP16 affine scan compiles");
    expect(bundle.artifacts.count == 8 &&
           [bundle.dispatchPlan isEqualToArray:
               @[@0,@1,@2,@3,@4,@5,@6,@7]],
           @"affine scan emits one ordered primitive per multiply and add");
    expect(bundle.sharedSurfaceIdentifiers.count == 7,
           @"affine scan keeps every internal transition on a shared surface");
    if (bundle.artifacts.count == 8) {
        expect([bindingIdentifiers(bundle.artifacts.firstObject)
                   isEqualToArray:@[@"state", @"a0", @"p0"]],
               @"the scan starts from the external state and first factor");
        expect([bindingIdentifiers(bundle.artifacts.lastObject)
                   isEqualToArray:@[@"p3", @"b3", @"y"]],
               @"the final update writes the graph output");
    }
}

int main(void) {
    @autoreleasepool {
        testThreeEndToEndCompilations();
        testBundleSerialization();
        testUnsupportedTargetFailsClosed();
        testBundlePathEscapeFailsClosed();
        testMalformedExternalConstantFailsClosed();
        testModelPathEscapeFailsClosed();
        testCleanEncoderNeedsNoResourceConfiguration();
        testMatmulGELUUsesOrderedPrimitiveArtifacts();
        testSingleTileOnlineReductionUsesPrimitiveArtifacts();
        testAffineScanUsesOrderedPrimitiveArtifacts();
        printf("compiler e2e: %s\n", failures == 0 ? "PASS" : "FAIL");
    }
    return failures == 0 ? 0 : 1;
}
