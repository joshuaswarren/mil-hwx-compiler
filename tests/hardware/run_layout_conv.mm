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
    NSInteger value = (NSInteger)((channel * 11 + row * 5 + column * 3) % 29) - 14;
    return (_Float16)((float)value * 0.0625f);
}

static BOOL validateIdentity(IOSurfaceRef output, NSUInteger channels,
                             NSUInteger run) {
    IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL);
    const uint16_t *actual = (const uint16_t *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches = 0;
    for (NSUInteger c = 0; c < channels; ++c)
        for (NSUInteger y = 0; y < 128; ++y)
            for (NSUInteger x = 0; x < 128; ++x) {
                _Float16 value = inputValue(c,y,x);
                uint16_t expected = 0;
                memcpy(&expected, &value, sizeof(expected));
                NSUInteger index = (c * 128 + y) * 128 + x;
                if (actual[index] == expected) continue;
                if (mismatches < 8)
                    printf("MISMATCH run=%lu index=%lu expected=0x%04x actual=0x%04x\n",
                           (unsigned long)run, (unsigned long)index,
                           expected, actual[index]);
                ++mismatches;
            }
    IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);
    printf("HARDWARE layout-conv run=%lu elements=%lu mismatches=%lu\n",
           (unsigned long)run, (unsigned long)(channels*128*128),
           (unsigned long)mismatches);
    return mismatches == 0;
}

static void printError(NSString *operation, NSError *error) {
    printf("%s error=%s\n", operation.UTF8String,
           error ? error.description.UTF8String : "(none)");
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr, "usage: %s BUNDLE_DIR CACHE_HWX NATURAL_CHANNELS\n",
                    argv[0]);
            return 64;
        }
        NSUInteger channels = (NSUInteger)strtoull(argv[3], NULL, 10);
        NSUInteger elements = channels * 128 * 128;
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        ANEProvisionedRuntime *runtime = bundle ? [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHash:kModelHash qos:21 error:&error] : nil;
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        if (!runtime || inputs.count != 1 || outputs.count != 1 ||
            inputs[0].logicalByteLength != elements*2 ||
            outputs[0].logicalByteLength != elements*2) {
            printError(@"runtime setup", error);
            return 2;
        }
        printf("CACHE path=%s\n", argv[2]);
        IOSurfaceLock(inputs[0].ioSurface, 0, NULL);
        _Float16 *input = (_Float16 *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
        for (NSUInteger c = 0; c < channels; ++c)
            for (NSUInteger y = 0; y < 128; ++y)
                for (NSUInteger x = 0; x < 128; ++x)
                    input[(c*128+y)*128+x] = inputValue(c,y,x);
        IOSurfaceUnlock(inputs[0].ioSurface, 0, NULL);

        BOOL valid = [runtime loadWithError:&error];
        printf("LOAD result=%d\n", valid);
        printError(@"load", error);
        @try {
            for (NSUInteger run = 1; valid && run <= 2; ++run) {
                IOSurfaceLock(outputs[0].ioSurface, 0, NULL);
                uint16_t *poison =
                    (uint16_t *)IOSurfaceGetBaseAddress(outputs[0].ioSurface);
                for (NSUInteger i = 0; i < elements; ++i) poison[i] = 0x7e00;
                IOSurfaceUnlock(outputs[0].ioSurface, 0, NULL);
                error = nil;
                CFAbsoluteTime start = CFAbsoluteTimeGetCurrent();
                BOOL evaluated = [runtime evaluateInputs:inputs outputs:outputs
                                                    error:&error];
                printf("EVAL layout-conv run=%lu result=%d time_us=%.1f\n",
                       (unsigned long)run, evaluated,
                       (CFAbsoluteTimeGetCurrent()-start)*1.0e6);
                printError(@"evaluate", error);
                valid = evaluated && validateIdentity(outputs[0].ioSurface,
                                                       channels,run);
            }
        } @finally {
            if (runtime.loaded) [runtime unloadWithError:nil];
        }
        printf("SUMMARY layout-conv C=%lu S=128 B=4 valid=%d runs=2\n",
               (unsigned long)channels, valid);
        return valid ? 0 : 1;
    }
}
