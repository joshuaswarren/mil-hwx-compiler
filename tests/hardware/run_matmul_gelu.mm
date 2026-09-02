#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <math.h>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static _Float16 inputValue(NSUInteger run, NSUInteger row,
                           NSUInteger column) {
    NSInteger value = (NSInteger)((row * 5 + column * 3 + run) % 33) - 16;
    return (_Float16)((float)value * 0.125f);
}

static _Float16 scaleValue(NSUInteger run, NSUInteger column) {
    return (_Float16)(0.5f + 0.125f * (float)((column + run) % 5));
}

static NSUInteger sourceColumn(NSUInteger size, NSUInteger column) {
    return (column + 17) % size;
}

static _Float16 geluReference(_Float16 input) {
    float x = (float)input;
    return (_Float16)(0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2)));
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

static void fillInputs(NSArray<ANEIOSurfaceBuffer *> *inputs, NSUInteger run,
                       NSUInteger size) {
    IOSurfaceLock(inputs[0].ioSurface, 0, NULL);
    IOSurfaceLock(inputs[1].ioSurface, 0, NULL);
    _Float16 *left = (_Float16 *)IOSurfaceGetBaseAddress(inputs[0].ioSurface);
    _Float16 *right = (_Float16 *)IOSurfaceGetBaseAddress(inputs[1].ioSurface);
    memset(right, 0, inputs[1].allocationByteLength);
    for (NSUInteger row = 0; row < size; ++row)
        for (NSUInteger column = 0; column < size; ++column)
            left[row * size + column] = inputValue(run, row, column);
    for (NSUInteger column = 0; column < size; ++column)
        right[sourceColumn(size, column) * size + column] =
            scaleValue(run, column);
    IOSurfaceUnlock(inputs[1].ioSurface, 0, NULL);
    IOSurfaceUnlock(inputs[0].ioSurface, 0, NULL);
}

static BOOL validate(ANEIOSurfaceBuffer *matrix, ANEIOSurfaceBuffer *output,
                     NSUInteger run, NSUInteger size) {
    IOSurfaceLock(matrix.ioSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceLock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *product =
        (const _Float16 *)IOSurfaceGetBaseAddress(matrix.ioSurface);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(output.ioSurface);
    NSUInteger productViolations = 0;
    NSUInteger localGELUViolations = 0;
    NSUInteger endToEndViolations = 0;
    NSUInteger maximumProductULP = 0;
    float maximumLocalGELUError = 0.0f;
    float maximumEndToEndError = 0.0f;
    for (NSUInteger row = 0; row < size; ++row) {
        for (NSUInteger column = 0; column < size; ++column) {
            NSUInteger index = row * size + column;
            _Float16 expectedProduct = (_Float16)(
                inputValue(run, row, sourceColumn(size, column)) *
                scaleValue(run, column));
            NSUInteger productULP =
                fp16ULPDistance(product[index], expectedProduct);
            maximumProductULP = MAX(maximumProductULP, productULP);
            if (!isfinite((float)product[index]) || productULP > 1)
                ++productViolations;

            _Float16 localExpected = geluReference(product[index]);
            _Float16 endToEndExpected = geluReference(expectedProduct);
            float localError =
                fabsf((float)actual[index] - (float)localExpected);
            float endToEndError =
                fabsf((float)actual[index] - (float)endToEndExpected);
            maximumLocalGELUError = fmaxf(maximumLocalGELUError, localError);
            maximumEndToEndError =
                fmaxf(maximumEndToEndError, endToEndError);
            if (!isfinite((float)actual[index]) || localError > 0.01f)
                ++localGELUViolations;
            if (!isfinite((float)actual[index]) || endToEndError > 0.012f)
                ++endToEndViolations;
        }
    }
    IOSurfaceUnlock(output.ioSurface, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(matrix.ioSurface, kIOSurfaceLockReadOnly, NULL);
    printf("HARDWARE matmul-gelu N=%lu run=%lu product_over_one_ulp=%lu "
           "product_max_ulp=%lu local_gelu_violations=%lu "
           "local_gelu_max_error=%g end_to_end_violations=%lu "
           "end_to_end_max_error=%g\n",
           (unsigned long)size, (unsigned long)run,
           (unsigned long)productViolations,
           (unsigned long)maximumProductULP,
           (unsigned long)localGELUViolations, maximumLocalGELUError,
           (unsigned long)endToEndViolations, maximumEndToEndError);
    return productViolations == 0 && localGELUViolations == 0 &&
        endToEndViolations == 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) return 64;
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        NSArray<NSString *> *hashes = @[
            [NSString stringWithUTF8String:argv[2]],
            [NSString stringWithUTF8String:argv[3]],
        ];
        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *matmulBuffers =
            [runtime surfaceBuffersForArtifactAtIndex:0 error:&error];
        ANEIOSurfaceBuffer *matrix = matmulBuffers.lastObject;
        ANEIOSurfaceBuffer *output = outputs.firstObject;
        if (!runtime || bundle.artifacts.count != 2 || inputs.count != 2 ||
            outputs.count != 1 || !matrix || !output)
            return 2;
        NSUInteger elements = inputs[0].logicalByteLength / sizeof(_Float16);
        NSUInteger size = (NSUInteger)sqrt((double)elements);
        if (size * size != elements) return 3;
        BOOL valid = [runtime loadWithError:&error];
        if (!valid) fprintf(stderr, "load: %s\n", error.description.UTF8String);
        @try {
            for (NSUInteger run = 1; valid && run <= 2; ++run) {
                fillInputs(inputs, run, size);
                IOSurfaceLock(matrix.ioSurface, 0, NULL);
                memset(IOSurfaceGetBaseAddress(matrix.ioSurface), 0xff,
                       matrix.allocationByteLength);
                IOSurfaceUnlock(matrix.ioSurface, 0, NULL);
                IOSurfaceLock(output.ioSurface, 0, NULL);
                memset(IOSurfaceGetBaseAddress(output.ioSurface), 0xff,
                       output.allocationByteLength);
                IOSurfaceUnlock(output.ioSurface, 0, NULL);
                BOOL evaluated = [runtime evaluateInputs:inputs outputs:outputs
                                                    error:&error];
                if (!evaluated)
                    fprintf(stderr, "evaluate: %s\n",
                            error.description.UTF8String);
                valid = evaluated && validate(matrix, output, run, size);
            }
        } @finally {
            if (runtime.loaded) [runtime unloadWithError:nil];
        }
        printf("SUMMARY matmul-gelu N=%lu valid=%d runs=2\n",
               (unsigned long)size, valid);
        return valid ? 0 : 1;
    }
}
