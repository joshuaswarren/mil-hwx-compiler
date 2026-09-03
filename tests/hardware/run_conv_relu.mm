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
static const NSUInteger kSpatial = 64;
static const NSUInteger kElements = kChannels * kSpatial * kSpatial;

static float inputValue(NSUInteger channel, NSUInteger row,
                        NSUInteger column) {
    NSUInteger spatialIndex = row * kSpatial + column;
    return (float)((NSInteger)((channel * 3 + spatialIndex) % 7) - 3) * 0.25f;
}

static float weightValue(NSUInteger outputChannel, NSUInteger inputChannel) {
    NSUInteger flatIndex = outputChannel * kChannels + inputChannel;
    return (float)((NSInteger)(flatIndex % 7) - 3) * 0.125f;
}

static float referenceValue(NSUInteger outputChannel, NSUInteger row,
                            NSUInteger column) {
    float accumulator = 0.0f;
    for (NSUInteger inputChannel = 0; inputChannel < kChannels; ++inputChannel)
        accumulator += inputValue(inputChannel, row, column) *
                       weightValue(outputChannel, inputChannel);
    return fmaxf(accumulator, 0.0f);
}

static BOOL validateOutput(IOSurfaceRef output, NSUInteger runIndex) {
    if (IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
        return NO;
    const _Float16 *actualValues =
        (const _Float16 *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    for (NSUInteger outputChannel = 0; outputChannel < kChannels;
         ++outputChannel) {
        for (NSUInteger row = 0; row < kSpatial; ++row) {
            for (NSUInteger column = 0; column < kSpatial; ++column) {
                NSUInteger index = (outputChannel * kSpatial + row) *
                                   kSpatial + column;
                float expected = referenceValue(outputChannel, row, column);
                float actual = (float)actualValues[index];
                float error = fabsf(actual - expected);
                maximumError = fmaxf(maximumError, error);
                float tolerance = fmaxf(0.002f, 0.002f * fabsf(expected));
                if (!isfinite(actual) || error > tolerance) {
                    if (mismatches < 8)
                        printf("MISMATCH run=%lu index=%lu expected=%g actual=%g "
                               "error=%g\n", (unsigned long)runIndex,
                               (unsigned long)index, expected, actual, error);
                    ++mismatches;
                }
            }
        }
    }
    printf("HARDWARE run=%lu elements=%lu mismatches=%lu max_abs_error=%g\n",
           (unsigned long)runIndex, (unsigned long)kElements,
           (unsigned long)mismatches, maximumError);
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

        IOSurfaceLock(input, 0, NULL);
        _Float16 *inputValues = (_Float16 *)IOSurfaceGetBaseAddress(input);
        for (NSUInteger channel = 0; channel < kChannels; ++channel)
            for (NSUInteger row = 0; row < kSpatial; ++row)
                for (NSUInteger column = 0; column < kSpatial; ++column) {
                    NSUInteger index = (channel * kSpatial + row) *
                                       kSpatial + column;
                    inputValues[index] =
                        (_Float16)inputValue(channel, row, column);
                }
        IOSurfaceUnlock(input, 0, NULL);

        BOOL loaded = [runtime loadWithError:&error];
        printf("LOAD result=%d\n", loaded);
        printError(@"load", error);
        BOOL allValid = loaded;

        @try {
            if (allValid) {
                for (NSUInteger run = 1; run <= 2; ++run) {
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
                    allValid = allValid && evaluated &&
                               validateOutput(output, run);
                }
            }
        } @finally {
            if (loaded) {
                NSError *unloadError = nil;
                [runtime unloadWithError:&unloadError];
                printError(@"unload", unloadError);
            }
        }
        printf("SUMMARY conv_relu valid=%d runs=2\n", allValid);
        return allValid ? 0 : 1;
    }
}
