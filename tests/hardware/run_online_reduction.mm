#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static const NSUInteger kSize = 128;

static float qValue(NSUInteger run, NSUInteger row, NSUInteger column) {
    return run == 1 ? 0.0f : (row == column ? 1.0f : 0.0f);
}

static float kValue(NSUInteger run, NSUInteger row, NSUInteger column) {
    return run == 1 ? 0.0f : (row == column ? 1.0f : 0.0f);
}

static float vValue(NSUInteger row, NSUInteger column) {
    NSInteger value = (NSInteger)((row * 3 + column * 5) % 17) - 8;
    return (float)value * 0.03125f;
}

static ANEIOSurfaceBuffer *bufferNamed(
    NSArray<ANEIOSurfaceBuffer *> *buffers, NSString *identifier) {
    for (ANEIOSurfaceBuffer *buffer in buffers)
        if ([buffer.identifier isEqualToString:identifier]) return buffer;
    return nil;
}

static void fillMatrix(ANEIOSurfaceBuffer *buffer,
                       float (^value)(NSUInteger, NSUInteger)) {
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    _Float16 *data = (_Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    for (NSUInteger row = 0; row < kSize; ++row)
        for (NSUInteger column = 0; column < kSize; ++column)
            data[row * kSize + column] = (_Float16)value(row, column);
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
}

static void buildReference(NSUInteger run, float *output) {
    float *probabilities = (float *)calloc(kSize * kSize, sizeof(float));
    for (NSUInteger query = 0; query < kSize; ++query) {
        float maximum = -INFINITY;
        for (NSUInteger key = 0; key < kSize; ++key) {
            float score = 0.0f;
            for (NSUInteger dimension = 0; dimension < kSize; ++dimension)
                score += qValue(run, query, dimension) *
                         kValue(run, dimension, key);
            score *= 0.08838834764831845f;
            probabilities[query * kSize + key] = score;
            maximum = fmaxf(maximum, score);
        }
        float sum = 0.0f;
        for (NSUInteger key = 0; key < kSize; ++key) {
            float value = expf(probabilities[query * kSize + key] - maximum);
            probabilities[query * kSize + key] = value;
            sum += value;
        }
        for (NSUInteger key = 0; key < kSize; ++key)
            probabilities[query * kSize + key] /= sum;
    }
    for (NSUInteger query = 0; query < kSize; ++query)
        for (NSUInteger column = 0; column < kSize; ++column) {
            float value = 0.0f;
            for (NSUInteger key = 0; key < kSize; ++key)
                value += probabilities[query * kSize + key] *
                         vValue(key, column);
            output[query * kSize + column] = value;
        }
    free(probabilities);
}

static BOOL validate(ANEIOSurfaceBuffer *buffer, NSUInteger run) {
    float *reference = (float *)malloc(kSize * kSize * sizeof(float));
    buildReference(run, reference);
    IOSurfaceLock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    for (NSUInteger index = 0; index < kSize * kSize; ++index) {
        float error = fabsf((float)actual[index] - reference[index]);
        maximumError = fmaxf(maximumError, error);
        if (!isfinite((float)actual[index]) || error > 0.01f) {
            if (mismatches < 8)
                printf("MISMATCH run=%lu index=%lu expected=%g actual=%g\n",
                    (unsigned long)run, (unsigned long)index,
                    reference[index], (float)actual[index]);
            ++mismatches;
        }
    }
    IOSurfaceUnlock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    free(reference);
    printf("HARDWARE online-reduction run=%lu elements=%lu mismatches=%lu "
           "max_abs_error=%g\n", (unsigned long)run,
           (unsigned long)(kSize * kSize), (unsigned long)mismatches,
           maximumError);
    return mismatches == 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 10) {
            fprintf(stderr, "usage: %s BUNDLE_DIR MODEL_HASH...\n", argv[0]);
            return 64;
        }
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        for (int index = 2; index < argc; ++index)
            [hashes addObject:[NSString stringWithUTF8String:argv[index]]];
        BOOL individualLoadsValid = YES;
        for (NSInteger rawIndex = (NSInteger)bundle.artifacts.count - 1;
             rawIndex >= 0; --rawIndex) {
            NSUInteger index = (NSUInteger)rawIndex;
            ANEExecutableBundle *single = [[ANEExecutableBundle alloc]
                initWithTarget:bundle.target artifacts:@[bundle.artifacts[index]]
                dispatchPlan:@[@0] passTrace:@[@"load-isolation"]];
            NSError *loadError = nil;
            ANEProvisionedRuntime *singleRuntime = [[ANEProvisionedRuntime alloc]
                initWithBundle:single modelHash:hashes[index] qos:21
                error:&loadError];
            BOOL loaded = [singleRuntime loadWithError:&loadError];
            printf("LOAD artifact=%lu result=%d error=%s\n",
                   (unsigned long)index, loaded,
                   loadError ? loadError.description.UTF8String : "(none)");
            individualLoadsValid = individualLoadsValid && loaded;
            if (loaded) [singleRuntime unloadWithError:nil];
        }
        if (!individualLoadsValid) return 3;
        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        ANEIOSurfaceBuffer *q = bufferNamed(inputs, @"q");
        ANEIOSurfaceBuffer *k = bufferNamed(inputs, @"k");
        ANEIOSurfaceBuffer *v = bufferNamed(inputs, @"v");
        ANEIOSurfaceBuffer *y = bufferNamed(outputs, @"y");
        if (!runtime || !q || !k || !v || !y) {
            fprintf(stderr, "runtime setup: %s\n", error.description.UTF8String);
            return 2;
        }
        BOOL valid = [runtime loadWithError:&error];
        if (!valid) fprintf(stderr, "load: %s\n", error.description.UTF8String);
        @try {
            for (NSUInteger run = 1; valid && run <= 2; ++run) {
                fillMatrix(q, ^float(NSUInteger row, NSUInteger column) {
                    return qValue(run, row, column);
                });
                fillMatrix(k, ^float(NSUInteger row, NSUInteger column) {
                    return kValue(run, row, column);
                });
                fillMatrix(v, ^float(NSUInteger row, NSUInteger column) {
                    return vValue(row, column);
                });
                IOSurfaceLock(y.ioSurface, 0, NULL);
                memset(IOSurfaceGetBaseAddress(y.ioSurface), 0xff,
                       y.allocationByteLength);
                IOSurfaceUnlock(y.ioSurface, 0, NULL);
                error = nil;
                BOOL evaluated = [runtime evaluateInputs:inputs
                    outputs:outputs error:&error];
                if (!evaluated)
                    fprintf(stderr, "evaluate: %s\n",
                            error.description.UTF8String);
                valid = evaluated && validate(y, run);
            }
        } @finally {
            if (runtime.loaded) [runtime unloadWithError:nil];
        }
        printf("SUMMARY online-reduction valid=%d runs=2\n", valid);
        return valid ? 0 : 1;
    }
}
