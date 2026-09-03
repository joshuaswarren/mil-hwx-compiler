#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash =
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static const NSUInteger kChannels = 64;
static const NSUInteger kPlaneElements = 4096;
static const NSUInteger kElements = 262144;
static const float kScale = 0.125f;

static float inputValue(NSUInteger channel, NSUInteger spatialIndex,
                        NSUInteger run) {
    NSUInteger modulus = run == 1 ? 7 : 11;
    NSInteger center = run == 1 ? 3 : 5;
    NSInteger numerator =
        (NSInteger)((channel * 3 + spatialIndex * (run + 1)) % modulus) -
        center;
    return (float)numerator * 0.125f;
}

static int8_t weightValue(NSUInteger layer, NSUInteger outputChannel,
                          NSUInteger inputChannel) {
    NSUInteger sourceChannel = (outputChannel + layer + 1) % kChannels;
    return inputChannel == sourceChannel ? 8 : 0;
}

static float quantizeDequantize(float value) {
    float rounded = nearbyintf((float)(_Float16)value / kScale);
    rounded = fmaxf(-128.0f, fminf(127.0f, rounded));
    return (float)(_Float16)(rounded * kScale);
}

static void buildReference(NSUInteger run, _Float16 *reference) {
    float activation[64];
    float next[64];
    for (NSUInteger spatialIndex = 0; spatialIndex < kPlaneElements;
         ++spatialIndex) {
        for (NSUInteger channel = 0; channel < kChannels; ++channel)
            activation[channel] = inputValue(channel, spatialIndex, run);
        for (NSUInteger layer = 0; layer < 4; ++layer) {
            for (NSUInteger outputChannel = 0; outputChannel < kChannels;
                 ++outputChannel) {
                float accumulator = 0.0f;
                for (NSUInteger inputChannel = 0; inputChannel < kChannels;
                     ++inputChannel) {
                    float weight = (float)weightValue(
                        layer, outputChannel, inputChannel) * kScale;
                    accumulator += activation[inputChannel] * weight;
                }
                float fp16Result = (float)(_Float16)accumulator;
                next[outputChannel] = layer < 3
                    ? quantizeDequantize(fp16Result) : fp16Result;
            }
            memcpy(activation, next, sizeof(activation));
        }
        for (NSUInteger channel = 0; channel < kChannels; ++channel)
            reference[channel * kPlaneElements + spatialIndex] =
                (_Float16)activation[channel];
    }
}

static void fillInput(IOSurfaceRef input, NSUInteger run) {
    IOSurfaceLock(input, 0, NULL);
    _Float16 *values = (_Float16 *)IOSurfaceGetBaseAddress(input);
    for (NSUInteger channel = 0; channel < kChannels; ++channel)
        for (NSUInteger spatialIndex = 0; spatialIndex < kPlaneElements;
             ++spatialIndex)
            values[channel * kPlaneElements + spatialIndex] =
                (_Float16)inputValue(channel, spatialIndex, run);
    IOSurfaceUnlock(input, 0, NULL);
}

static BOOL validateOutput(IOSurfaceRef output, NSUInteger run) {
    NSMutableData *referenceData =
        [NSMutableData dataWithLength:kElements * sizeof(_Float16)];
    buildReference(run, (_Float16 *)referenceData.mutableBytes);
    const _Float16 *reference = (const _Float16 *)referenceData.bytes;
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    for (NSUInteger index = 0; index < kElements; ++index) {
        float expected = (float)reference[index];
        float value = (float)actual[index];
        float error = fabsf(value - expected);
        maximumError = fmaxf(maximumError, error);
        float tolerance = fmaxf(0.02f, 0.005f * fabsf(expected));
        if (!isfinite(value) || error > tolerance) {
            if (mismatches < 8)
                printf("MISMATCH run=%lu index=%lu expected=%g actual=%g "
                       "error=%g\n", (unsigned long)run,
                       (unsigned long)index, expected, value, error);
            ++mismatches;
        }
    }
    printf("HARDWARE W8A8 run=%lu elements=%lu mismatches=%lu "
           "max_abs_error=%g first=%g\n", (unsigned long)run,
           (unsigned long)kElements, (unsigned long)mismatches, maximumError,
           (float)actual[0]);
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
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
                    for (NSUInteger index = 0; index < kElements; ++index)
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
        printf("SUMMARY W8A8 valid=%d runs=2\n", allValid);
        return allValid ? 0 : 1;
    }
}
