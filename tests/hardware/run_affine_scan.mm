#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static _Float16 initialValue(NSUInteger run, NSUInteger index) {
    NSInteger value = (NSInteger)(index % 13) - 6;
    return (_Float16)((float)value * (run == 1 ? 0.03125f : 0.0625f));
}

static _Float16 factorValue(NSUInteger run, NSUInteger chunk,
                            NSUInteger index) {
    float base = 0.5f + (float)chunk * 0.0625f;
    float variation = (float)((index + run + chunk) % 3) * 0.03125f;
    return (_Float16)(base + variation);
}

static _Float16 updateValue(NSUInteger run, NSUInteger chunk,
                            NSUInteger index) {
    NSInteger value = (NSInteger)((index * 3 + chunk + run) % 9) - 4;
    return (_Float16)((float)value * 0.015625f);
}

static ANEIOSurfaceBuffer *bufferNamed(
    NSArray<ANEIOSurfaceBuffer *> *buffers, NSString *identifier) {
    for (ANEIOSurfaceBuffer *buffer in buffers)
        if ([buffer.identifier isEqualToString:identifier]) return buffer;
    return nil;
}

static void fill(ANEIOSurfaceBuffer *buffer,
                 _Float16 (^value)(NSUInteger)) {
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    _Float16 *data = (_Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    NSUInteger count = buffer.logicalByteLength / sizeof(_Float16);
    for (NSUInteger index = 0; index < count; ++index) data[index] = value(index);
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
}

static NSUInteger fp16ULPDistance(_Float16 left, _Float16 right) {
    uint16_t leftBits = 0;
    uint16_t rightBits = 0;
    memcpy(&leftBits, &left, sizeof(leftBits));
    memcpy(&rightBits, &right, sizeof(rightBits));
    NSUInteger leftOrdered = (leftBits & 0x8000)
        ? 0x8000 - (leftBits & 0x7fff) : 0x8000 + leftBits;
    NSUInteger rightOrdered = (rightBits & 0x8000)
        ? 0x8000 - (rightBits & 0x7fff) : 0x8000 + rightBits;
    return leftOrdered > rightOrdered
        ? leftOrdered - rightOrdered : rightOrdered - leftOrdered;
}

static BOOL validate(ANEIOSurfaceBuffer *output, NSUInteger run) {
    IOSurfaceLock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(output.ioSurface);
    NSUInteger count = output.logicalByteLength / sizeof(_Float16);
    NSUInteger exactMismatches = 0;
    NSUInteger overFourULP = 0;
    NSUInteger maximumULP = 0;
    float maximumError = 0.0f;
    for (NSUInteger index = 0; index < count; ++index) {
        _Float16 expected = initialValue(run, index);
        for (NSUInteger chunk = 0; chunk < 4; ++chunk) {
            expected = (_Float16)(expected * factorValue(run, chunk, index));
            expected = (_Float16)(expected + updateValue(run, chunk, index));
        }
        float error = fabsf((float)actual[index] - (float)expected);
        NSUInteger ulp = fp16ULPDistance(actual[index], expected);
        maximumError = fmaxf(maximumError, error);
        maximumULP = MAX(maximumULP, ulp);
        if (error != 0.0f) ++exactMismatches;
        if (!isfinite((float)actual[index]) || ulp > 4) {
            if (overFourULP < 8)
                printf("MISMATCH run=%lu index=%lu expected=%g actual=%g "
                       "ulp=%lu\n",
                    (unsigned long)run, (unsigned long)index,
                    (float)expected, (float)actual[index],
                    (unsigned long)ulp);
            ++overFourULP;
        }
    }
    IOSurfaceUnlock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    printf("HARDWARE affine-scan run=%lu elements=%lu exact_mismatches=%lu "
           "over_four_ulp=%lu max_ulp=%lu max_abs_error=%g\n",
           (unsigned long)run, (unsigned long)count,
           (unsigned long)exactMismatches, (unsigned long)overFourULP,
           (unsigned long)maximumULP, maximumError);
    return overFourULP == 0;
}

static void reportArtifactOutputs(ANEProvisionedRuntime *runtime) {
    for (NSUInteger artifactIndex = 0;
         artifactIndex < runtime.bundle.artifacts.count; ++artifactIndex) {
        NSError *error = nil;
        NSArray<ANEIOSurfaceBuffer *> *buffers =
            [runtime surfaceBuffersForArtifactAtIndex:artifactIndex
                                                 error:&error];
        ANEIOSurfaceBuffer *output = buffers.lastObject;
        if (!output) continue;
        IOSurfaceLock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *data =
            (const _Float16 *)IOSurfaceGetBaseAddress(output.ioSurface);
        printf("TRACE affine-scan artifact=%lu output=%s first=%g\n",
               (unsigned long)artifactIndex, output.identifier.UTF8String,
               (float)data[0]);
        IOSurfaceUnlock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    }
}

static ANEIOSurfaceBuffer *artifactOutput(ANEProvisionedRuntime *runtime,
                                          NSUInteger artifactIndex) {
    return [[runtime surfaceBuffersForArtifactAtIndex:artifactIndex error:nil]
        lastObject];
}

static BOOL validateStageRounding(ANEProvisionedRuntime *runtime,
                                  NSArray<ANEIOSurfaceBuffer *> *inputs) {
    ANEIOSurfaceBuffer *previous = bufferNamed(inputs, @"state");
    NSUInteger violations = 0;
    NSUInteger maximumULP = 0;
    for (NSUInteger chunk = 0; chunk < 4; ++chunk) {
        ANEIOSurfaceBuffer *factor = bufferNamed(inputs,
            [NSString stringWithFormat:@"a%lu", (unsigned long)chunk]);
        ANEIOSurfaceBuffer *update = bufferNamed(inputs,
            [NSString stringWithFormat:@"b%lu", (unsigned long)chunk]);
        ANEIOSurfaceBuffer *product = artifactOutput(runtime, chunk * 2);
        ANEIOSurfaceBuffer *sum = artifactOutput(runtime, chunk * 2 + 1);
        NSArray<ANEIOSurfaceBuffer *> *buffers =
            @[previous, factor, update, product, sum];
        for (ANEIOSurfaceBuffer *buffer in buffers)
            IOSurfaceLock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
        const _Float16 *previousData =
            (const _Float16 *)IOSurfaceGetBaseAddress(previous.ioSurface);
        const _Float16 *factorData =
            (const _Float16 *)IOSurfaceGetBaseAddress(factor.ioSurface);
        const _Float16 *updateData =
            (const _Float16 *)IOSurfaceGetBaseAddress(update.ioSurface);
        const _Float16 *productData =
            (const _Float16 *)IOSurfaceGetBaseAddress(product.ioSurface);
        const _Float16 *sumData =
            (const _Float16 *)IOSurfaceGetBaseAddress(sum.ioSurface);
        NSUInteger count = product.logicalByteLength / sizeof(_Float16);
        for (NSUInteger index = 0; index < count; ++index) {
            _Float16 expectedProduct =
                (_Float16)(previousData[index] * factorData[index]);
            _Float16 expectedSum =
                (_Float16)(productData[index] + updateData[index]);
            NSUInteger productULP =
                fp16ULPDistance(productData[index], expectedProduct);
            NSUInteger sumULP = fp16ULPDistance(sumData[index], expectedSum);
            maximumULP = MAX(maximumULP, MAX(productULP, sumULP));
            if (!isfinite((float)productData[index]) ||
                !isfinite((float)sumData[index]) || productULP > 1 ||
                sumULP > 1)
                ++violations;
        }
        for (ANEIOSurfaceBuffer *buffer in [buffers reverseObjectEnumerator])
            IOSurfaceUnlock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
        previous = sum;
    }
    printf("HARDWARE affine-scan stage_rounding violations=%lu max_ulp=%lu\n",
           (unsigned long)violations, (unsigned long)maximumULP);
    return violations == 0;
}

static BOOL validateFirstArtifact(ANEExecutableBundle *bundle,
                                  NSString *modelHash) {
    ANEExecutableBundle *single = [[ANEExecutableBundle alloc]
        initWithTarget:bundle.target artifacts:@[bundle.artifacts.firstObject]
        dispatchPlan:@[@0] passTrace:@[@"first-stage-isolation"]];
    NSError *error = nil;
    ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
        initWithBundle:single modelHash:modelHash qos:21 error:&error];
    NSArray<ANEIOSurfaceBuffer *> *inputs =
        [runtime createInputBuffersWithError:&error];
    NSArray<ANEIOSurfaceBuffer *> *outputs =
        [runtime createOutputBuffersWithError:&error];
    ANEIOSurfaceBuffer *state = bufferNamed(inputs, @"state");
    ANEIOSurfaceBuffer *factor = bufferNamed(inputs, @"a0");
    if (!runtime || !state || !factor || outputs.count != 1) return NO;
    fill(state, ^_Float16(NSUInteger index) {
        return initialValue(1, index);
    });
    fill(factor, ^_Float16(NSUInteger index) {
        return factorValue(1, 0, index);
    });
    BOOL valid = [runtime loadWithError:&error] &&
        [runtime evaluateInputs:inputs outputs:outputs error:&error];
    ANEIOSurfaceBuffer *output = outputs.firstObject;
    IOSurfaceLock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(output.ioSurface);
    NSUInteger count = output.logicalByteLength / sizeof(_Float16);
    for (NSUInteger index = 0; valid && index < count; ++index) {
        _Float16 expected = (_Float16)(initialValue(1, index) *
                                      factorValue(1, 0, index));
        valid = actual[index] == expected;
    }
    IOSurfaceUnlock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    if (runtime.loaded) [runtime unloadWithError:nil];
    printf("HARDWARE affine-scan first-stage-isolation valid=%d\n", valid);
    return valid;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 10) return 64;
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        for (int index = 2; index < argc; ++index)
            [hashes addObject:[NSString stringWithUTF8String:argv[index]]];
        if (!validateFirstArtifact(bundle, hashes.firstObject)) return 3;
        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        ANEIOSurfaceBuffer *state = bufferNamed(inputs, @"state");
        ANEIOSurfaceBuffer *output = bufferNamed(outputs, @"y");
        if (!runtime || !state || !output || inputs.count != 9) return 2;
        BOOL valid = [runtime loadWithError:&error];
        if (!valid) fprintf(stderr, "load: %s\n", error.description.UTF8String);
        @try {
            for (NSUInteger run = 1; valid && run <= 2; ++run) {
                fill(state, ^_Float16(NSUInteger index) {
                    return initialValue(run, index);
                });
                for (NSUInteger chunk = 0; chunk < 4; ++chunk) {
                    ANEIOSurfaceBuffer *factor = bufferNamed(inputs,
                        [NSString stringWithFormat:@"a%lu", (unsigned long)chunk]);
                    ANEIOSurfaceBuffer *update = bufferNamed(inputs,
                        [NSString stringWithFormat:@"b%lu", (unsigned long)chunk]);
                    fill(factor, ^_Float16(NSUInteger index) {
                        return factorValue(run, chunk, index);
                    });
                    fill(update, ^_Float16(NSUInteger index) {
                        return updateValue(run, chunk, index);
                    });
                }
                BOOL evaluated = [runtime evaluateInputs:inputs outputs:outputs
                                                    error:&error];
                if (!evaluated)
                    fprintf(stderr, "evaluate: %s\n",
                            error.description.UTF8String);
                valid = evaluated && validateStageRounding(runtime, inputs) &&
                    validate(output, run);
                if (!valid) reportArtifactOutputs(runtime);
            }
        } @finally {
            if (runtime.loaded) [runtime unloadWithError:nil];
        }
        printf("SUMMARY affine-scan valid=%d runs=2\n", valid);
        return valid ? 0 : 1;
    }
}
