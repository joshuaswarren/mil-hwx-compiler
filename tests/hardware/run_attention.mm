#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash =
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static const NSUInteger kHeads = 4;
static const NSUInteger kSequence = 64;
static const NSUInteger kWidth = 64;
static const NSUInteger kInputElements =
    kSequence * kHeads * (3 * kWidth);
static const NSUInteger kOutputElements = 16384;

static float qValue(NSUInteger run, NSUInteger sequence, NSUInteger head,
                    NSUInteger dimension) {
    if (run == 1) return 0.0f;
    NSInteger numerator = (NSInteger)((sequence * 3 + dimension + head) % 7) - 3;
    return (float)numerator * 0.03125f;
}

static float kValue(NSUInteger run, NSUInteger sequence, NSUInteger head,
                    NSUInteger dimension) {
    if (run == 1) return 0.0f;
    NSInteger numerator = (NSInteger)((sequence + dimension * 2 + head) % 5) - 2;
    return (float)numerator * 0.03125f;
}

static float vValue(NSUInteger run, NSUInteger sequence, NSUInteger head,
                    NSUInteger dimension) {
    if (run == 1) {
        return -0.25f + (float)head * 0.125f +
               (float)sequence * 0.00390625f;
    }
    NSInteger numerator =
        (NSInteger)((sequence * 5 + dimension * 3 + head) % 11) - 5;
    return (float)numerator * 0.125f;
}

static void fillInput(IOSurfaceRef input, NSUInteger run) {
    IOSurfaceLock(input, 0, NULL);
    _Float16 *values = (_Float16 *)IOSurfaceGetBaseAddress(input);
    memset(values, 0, kInputElements * sizeof(_Float16));
    for (NSUInteger sequence = 0; sequence < kSequence; ++sequence) {
        for (NSUInteger head = 0; head < kHeads; ++head) {
            NSUInteger base = (sequence * kHeads + head) * 192;
            for (NSUInteger dimension = 0; dimension < kWidth; ++dimension) {
                values[base + dimension] =
                    (_Float16)qValue(run, sequence, head, dimension);
                values[base + 64 + dimension] =
                    (_Float16)kValue(run, sequence, head, dimension);
                values[base + 128 + dimension] =
                    (_Float16)vValue(run, sequence, head, dimension);
            }
        }
    }
    IOSurfaceUnlock(input, 0, NULL);
}

static void buildReference(NSUInteger run, float *output) {
    const NSUInteger matrixElements = kHeads * kWidth * kWidth;
    float *probabilities = (float *)calloc(matrixElements, sizeof(float));
    for (NSUInteger head = 0; head < kHeads; ++head) {
        for (NSUInteger queryDimension = 0; queryDimension < kWidth;
             ++queryDimension) {
            float maximum = -INFINITY;
            for (NSUInteger keyDimension = 0; keyDimension < kWidth;
                 ++keyDimension) {
                float score = 0.0f;
                for (NSUInteger sequence = 0; sequence < kSequence;
                     ++sequence) {
                    score += qValue(run, sequence, head, queryDimension) *
                             kValue(run, sequence, head, keyDimension);
                }
                score *= 0.125f;
                NSUInteger index = (head * kWidth + queryDimension) *
                                   kWidth + keyDimension;
                probabilities[index] = score;
                maximum = fmaxf(maximum, score);
            }
            float denominator = 0.0f;
            for (NSUInteger keyDimension = 0; keyDimension < kWidth;
                 ++keyDimension) {
                NSUInteger index = (head * kWidth + queryDimension) *
                                   kWidth + keyDimension;
                probabilities[index] = expf(probabilities[index] - maximum);
                denominator += probabilities[index];
            }
            for (NSUInteger keyDimension = 0; keyDimension < kWidth;
                 ++keyDimension) {
                NSUInteger index = (head * kWidth + queryDimension) *
                                   kWidth + keyDimension;
                probabilities[index] /= denominator;
            }
        }
    }

    for (NSUInteger sequence = 0; sequence < kSequence; ++sequence) {
        for (NSUInteger head = 0; head < kHeads; ++head) {
            for (NSUInteger queryDimension = 0; queryDimension < kWidth;
                 ++queryDimension) {
                float context = 0.0f;
                for (NSUInteger keyDimension = 0; keyDimension < kWidth;
                     ++keyDimension) {
                    NSUInteger probabilityIndex =
                        (head * kWidth + queryDimension) * kWidth +
                        keyDimension;
                    context += probabilities[probabilityIndex] *
                               vValue(run, sequence, head, keyDimension);
                }
                NSUInteger outputIndex =
                    (sequence * kHeads + head) * kWidth + queryDimension;
                output[outputIndex] = context;
            }
        }
    }
    free(probabilities);
}

static BOOL validateOutput(IOSurfaceRef output, NSUInteger run) {
    float *reference = (float *)malloc(kOutputElements * sizeof(float));
    buildReference(run, reference);
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(output);
    printf("OUTPUT run=%lu first-by-head=", (unsigned long)run);
    for (NSUInteger head = 0; head < kHeads; ++head) {
        NSUInteger index = head * kWidth;
        printf("%sh%lu:%g", head ? "," : "", (unsigned long)head,
               (float)actual[index]);
    }
    printf("\n");
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    for (NSUInteger index = 0; index < kOutputElements; ++index) {
        float value = (float)actual[index];
        float error = fabsf(value - reference[index]);
        maximumError = fmaxf(maximumError, error);
        if (!isfinite(value) || error > 0.006f) {
            if (mismatches < 8)
                printf("MISMATCH run=%lu index=%lu expected=%g actual=%g "
                       "error=%g\n", (unsigned long)run,
                       (unsigned long)index, reference[index], value, error);
            ++mismatches;
        }
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    free(reference);
    printf("HARDWARE attention run=%lu elements=%lu mismatches=%lu "
           "max_abs_error=%g\n", (unsigned long)run,
           (unsigned long)kOutputElements, (unsigned long)mismatches,
           maximumError);
    return mismatches == 0;
}

static void printError(NSString *operation, NSError *error) {
    printf("%s error=%s\n", operation.UTF8String,
           error ? error.description.UTF8String : "(none)");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr, "usage: %s BUNDLE_DIR CACHE_HWX\n", argv[0]);
            return 64;
        }
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        if (!bundle) {
            printError(@"bundle load", error);
            return 2;
        }
        NSString *cachePath = [NSString stringWithUTF8String:argv[2]];
        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHash:kModelHash qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        if (!runtime || inputs.count != 1 || outputs.count != 1) {
            printError(@"runtime setup", error);
            return 3;
        }
        IOSurfaceRef input = inputs[0].ioSurface;
        IOSurfaceRef output = outputs[0].ioSurface;
        printf("CACHE path=%s\n", cachePath.fileSystemRepresentation);
        BOOL loaded = [runtime loadWithError:&error];
        printf("LOAD result=%d\n", loaded);
        printError(@"load", error);
        BOOL allValid = loaded;
        @try {
            if (allValid) {
                for (NSUInteger run = 1; run <= 2; ++run) {
                    fillInput(input, run);
                    IOSurfaceLock(output, 0, NULL);
                    uint16_t *bits = (uint16_t *)IOSurfaceGetBaseAddress(output);
                    for (NSUInteger index = 0; index < kOutputElements; ++index)
                        bits[index] = 0x7e00;
                    IOSurfaceUnlock(output, 0, NULL);
                    error = nil;
                    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
                    BOOL evaluated = [runtime evaluateInputs:inputs
                                                     outputs:outputs
                                                       error:&error];
                    double elapsedMicroseconds =
                        (CFAbsoluteTimeGetCurrent() - started) * 1.0e6;
                    printf("EVAL run=%lu result=%d time_us=%.1f\n",
                           (unsigned long)run, evaluated,
                           elapsedMicroseconds);
                    printError(@"evaluate", error);
                    BOOL runValid = evaluated && validateOutput(output, run);
                    allValid = allValid && runValid;
                }
            }
        } @finally {
            if (loaded) {
                NSError *unloadError = nil;
                [runtime unloadWithError:&unloadError];
                printError(@"unload", unloadError);
            }
        }
        printf("SUMMARY attention valid=%d runs=2\n", allValid);
        return allValid ? 0 : 1;
    }
}
