#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash =
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static _Float16 inputValue(NSUInteger channel, NSUInteger row,
                           NSUInteger column) {
    NSInteger value = (NSInteger)((channel * 7 + row * 3 + column) % 23) - 11;
    return (_Float16)((float)value * 0.125f);
}

static BOOL validateLayout(IOSurfaceRef output, ANEHWXBinding *binding,
                           BOOL depthToSpace,
                           NSUInteger inputChannels,
                           NSUInteger inputSpatial, NSUInteger block,
                           NSUInteger runIndex) {
    NSUInteger outputChannels = depthToSpace
        ? inputChannels / (block * block) : inputChannels * block * block;
    NSUInteger outputSpatial = depthToSpace
        ? inputSpatial * block : inputSpatial / block;
    NSUInteger outputElements =
        outputChannels * outputSpatial * outputSpatial;
    if (IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess)
        return NO;
    const uint8_t *actual =
        (const uint8_t *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches = 0;
    for (NSUInteger outputChannel = 0; outputChannel < outputChannels;
         ++outputChannel) {
        for (NSUInteger outputRow = 0; outputRow < outputSpatial; ++outputRow) {
            for (NSUInteger outputColumn = 0; outputColumn < outputSpatial;
                 ++outputColumn) {
                NSUInteger inputChannel;
                NSUInteger inputRow;
                NSUInteger inputColumn;
                if (depthToSpace) {
                    NSUInteger blockRow = outputRow % block;
                    NSUInteger blockColumn = outputColumn % block;
                    inputChannel =
                        (blockRow * block + blockColumn) * outputChannels +
                        outputChannel;
                    inputRow = outputRow / block;
                    inputColumn = outputColumn / block;
                } else {
                    NSUInteger blockIndex = outputChannel / inputChannels;
                    inputChannel = outputChannel % inputChannels;
                    inputRow = outputRow * block + blockIndex / block;
                    inputColumn = outputColumn * block + blockIndex % block;
                }
                _Float16 expectedValue = inputValue(
                    inputChannel, inputRow, inputColumn);
                uint16_t expected = 0;
                memcpy(&expected, &expectedValue, sizeof(expected));
                NSUInteger index = (outputChannel * outputSpatial + outputRow) *
                    outputSpatial + outputColumn;
                NSUInteger offset=outputChannel*binding.planeStrideBytes+
                    outputRow*binding.rowStrideBytes+outputColumn*2;
                uint16_t actualValue=*(const uint16_t *)(actual+offset);
                if (actualValue == expected) continue;
                if (mismatches < 8)
                    printf("MISMATCH run=%lu index=%lu expected=0x%04x actual=0x%04x\n",
                           (unsigned long)runIndex, (unsigned long)index,
                           expected, actualValue);
                ++mismatches;
            }
        }
    }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    printf("HARDWARE %s run=%lu elements=%lu mismatches=%lu\n",
           depthToSpace ? "d2s" : "s2d", (unsigned long)runIndex,
           (unsigned long)outputElements,
           (unsigned long)mismatches);
    return mismatches == 0;
}

static void printError(NSString *operation, NSError *error) {
    printf("%s error=%s\n", operation.UTF8String,
           error ? error.description.UTF8String : "(none)");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 7) {
            fprintf(stderr,
                "usage: %s BUNDLE_DIR CACHE_HWX s2d|d2s C S B\n", argv[0]);
            return 64;
        }
        BOOL depthToSpace = strcmp(argv[3], "d2s") == 0;
        BOOL spaceToDepth = strcmp(argv[3], "s2d") == 0;
        NSUInteger inputChannels = (NSUInteger)strtoull(argv[4], NULL, 10);
        NSUInteger inputSpatial = (NSUInteger)strtoull(argv[5], NULL, 10);
        NSUInteger block = (NSUInteger)strtoull(argv[6], NULL, 10);
        if ((!depthToSpace && !spaceToDepth) || inputChannels == 0 ||
            inputSpatial == 0 || block == 0 ||
            (!depthToSpace && inputSpatial % block != 0) ||
            (depthToSpace && inputChannels % (block * block) != 0)) {
            fprintf(stderr, "invalid layout case\n");
            return 65;
        }
        NSUInteger outputChannels = depthToSpace
            ? inputChannels / (block * block) : inputChannels * block * block;
        NSUInteger outputSpatial = depthToSpace
            ? inputSpatial * block : inputSpatial / block;
        NSUInteger inputElements =
            inputChannels * inputSpatial * inputSpatial;
        NSUInteger outputElements =
            outputChannels * outputSpatial * outputSpatial;
        NSError *error = nil;
        NSURL *bundleURL = [NSURL fileURLWithPath:
            [NSString stringWithUTF8String:argv[1]] isDirectory:YES];
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:bundleURL error:&error];
        if (!bundle) {
            printError(@"bundle load", error);
            return 2;
        }
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
        if (inputs[0].logicalByteLength != inputElements * 2 ||
            outputs[0].logicalByteLength != outputElements * 2) {
            fprintf(stderr, "bundle shape does not match requested layout case\n");
            return 4;
        }
        printf("CACHE path=%s\n", argv[2]);
        ANEHWXBinding *inputBinding=bundle.artifacts[0].bindings[0];
        ANEHWXBinding *outputBinding=bundle.artifacts[0].bindings[1];
        IOSurfaceLock(inputs[0].ioSurface, 0, NULL);
        uint8_t *values =
            (uint8_t *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
        memset(values,0,inputs[0].allocationByteLength);
        for (NSUInteger channel = 0; channel < inputChannels; ++channel)
            for (NSUInteger row = 0; row < inputSpatial; ++row)
                for (NSUInteger column = 0; column < inputSpatial; ++column) {
                    NSUInteger offset=channel*inputBinding.planeStrideBytes+
                        row*inputBinding.rowStrideBytes+column*2;
                    *(_Float16 *)(values+offset)=inputValue(channel,row,column);
                }
        IOSurfaceUnlock(inputs[0].ioSurface, 0, NULL);

        BOOL loaded = [runtime loadWithError:&error];
        printf("LOAD result=%d\n", loaded);
        printError(@"load", error);
        BOOL valid = loaded;
        @try {
            for (NSUInteger run = 1; valid && run <= 2; ++run) {
                IOSurfaceLock(outputs[0].ioSurface, 0, NULL);
                uint8_t *poison =
                    (uint8_t *)IOSurfaceGetBaseAddress(outputs[0].ioSurface);
                memset(poison,0x7e,outputs[0].allocationByteLength);
                IOSurfaceUnlock(outputs[0].ioSurface, 0, NULL);
                error = nil;
                CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
                BOOL evaluated = [runtime evaluateInputs:inputs outputs:outputs
                                                    error:&error];
                double elapsedMicroseconds =
                    (CFAbsoluteTimeGetCurrent() - started) * 1.0e6;
                printf("EVAL %s run=%lu result=%d time_us=%.1f\n",
                       depthToSpace ? "d2s" : "s2d",
                       (unsigned long)run, evaluated, elapsedMicroseconds);
                printError(@"evaluate", error);
                valid = evaluated && validateLayout(outputs[0].ioSurface,
                    outputBinding,depthToSpace,inputChannels,inputSpatial,block,run);
            }
        } @finally {
            if (loaded) {
                NSError *unloadError = nil;
                [runtime unloadWithError:&unloadError];
                printError(@"unload", unloadError);
            }
        }
        printf("SUMMARY %s C=%lu S=%lu B=%lu valid=%d runs=2\n",
               depthToSpace ? "d2s" : "s2d",
               (unsigned long)inputChannels, (unsigned long)inputSpatial,
               (unsigned long)block, valid);
        return valid ? 0 : 1;
    }
}
