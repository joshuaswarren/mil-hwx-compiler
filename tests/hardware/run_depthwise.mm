#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSString *const kModelHash =
    @"56EEB7C58DD30EA059D75A62C53809DA7E7B2C1CF2856B0B3B67FC6ED03BCED2_"
     "DF3F619804A92FDB4057192DC43DD748EA778ADC52BC498CE80524C014B81119_"
     "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

static float inputValue(NSUInteger index) {
    return (float)((NSInteger)(index % 17) - 8) * 0.0625f;
}

static BOOL validateOutput(IOSurfaceRef output, NSUInteger elements,
                           NSUInteger run) {
    IOSurfaceLock(output,kIOSurfaceLockReadOnly,NULL);
    const _Float16 *values = (const _Float16 *)IOSurfaceGetBaseAddress(output);
    NSUInteger mismatches = 0;
    float maximumError = 0;
    for (NSUInteger index = 0; index < elements; ++index) {
        float expected = inputValue(index);
        float actual = (float)values[index];
        float error = fabsf(actual - expected);
        maximumError = fmaxf(maximumError,error);
        if (!isfinite(actual) || error != 0) {
            if (mismatches < 8)
                printf("MISMATCH run=%lu index=%lu expected=%g actual=%g\n",
                    (unsigned long)run,(unsigned long)index,expected,actual);
            ++mismatches;
        }
    }
    IOSurfaceUnlock(output,kIOSurfaceLockReadOnly,NULL);
    printf("HARDWARE depthwise run=%lu elements=%lu mismatches=%lu max_abs_error=%g\n",
        (unsigned long)run,(unsigned long)elements,
        (unsigned long)mismatches,maximumError);
    return mismatches == 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 3) {
            fprintf(stderr,"usage: %s BUNDLE_DIR CACHE_HWX\n",argv[0]);
            return 64;
        }
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        if (!bundle) {
            fprintf(stderr,"bundle load: %s\n",error.description.UTF8String);
            return 2;
        }
        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHash:kModelHash qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        if (!runtime || inputs.count != 1 || outputs.count != 1) {
            fprintf(stderr,"runtime setup: %s\n",error.description.UTF8String);
            return 3;
        }
        NSUInteger elements = inputs[0].logicalByteLength / sizeof(_Float16);
        IOSurfaceLock(inputs[0].ioSurface,0,NULL);
        _Float16 *input = (_Float16 *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
        for (NSUInteger index = 0; index < elements; ++index)
            input[index] = (_Float16)inputValue(index);
        IOSurfaceUnlock(inputs[0].ioSurface,0,NULL);
        BOOL valid = [runtime loadWithError:&error];
        printf("LOAD depthwise result=%d cache=%s error=%s\n",valid,argv[2],
            error ? error.description.UTF8String : "(none)");
        @try {
            for (NSUInteger run = 1; valid && run <= 2; ++run) {
                IOSurfaceLock(outputs[0].ioSurface,0,NULL);
                memset(IOSurfaceGetBaseAddress(outputs[0].ioSurface),0xff,
                    outputs[0].allocationByteLength);
                IOSurfaceUnlock(outputs[0].ioSurface,0,NULL);
                error = nil;
                CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
                BOOL evaluated = [runtime evaluateInputs:inputs outputs:outputs
                    error:&error];
                double microseconds =
                    (CFAbsoluteTimeGetCurrent()-started)*1.0e6;
                printf("EVAL depthwise run=%lu result=%d time_us=%.1f error=%s\n",
                    (unsigned long)run,evaluated,microseconds,
                    error ? error.description.UTF8String : "(none)");
                valid = evaluated && validateOutput(outputs[0].ioSurface,elements,run);
            }
        } @finally {
            if (runtime.loaded) [runtime unloadWithError:nil];
        }
        printf("SUMMARY depthwise valid=%d runs=2\n",valid);
        return valid ? 0 : 1;
    }
}
