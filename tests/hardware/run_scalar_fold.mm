// M4 numerical verification for the scalar-fold compositions on
// non-attention graphs.
//
//   run_scalar_fold WORKLOAD BUNDLE_DIR MODEL_HASH...
//
// WORKLOAD is `matmul-scale` (y = mul(matmul(a, b), s)) or `scale-exp`
// (y = exp(mul(x, s))). The tool loads the compiled bundle, evaluates it on
// the ANE, and compares the output against a CPU reference with FP16-aware
// tolerances. It prints the program count so the same tool verifies both the
// composed bundle and the forced standalone bundle.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>

#import <cmath>
#import <cstdio>
#import <cstring>
#import <vector>

#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static ANEIOSurfaceBuffer *bufferNamed(
    NSArray<ANEIOSurfaceBuffer *> *buffers, NSString *identifier) {
    for (ANEIOSurfaceBuffer *buffer in buffers)
        if ([buffer.identifier isEqualToString:identifier]) return buffer;
    return nil;
}

static void fillMatrix(ANEIOSurfaceBuffer *buffer, NSUInteger size,
                       float (^value)(NSUInteger, NSUInteger)) {
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    _Float16 *data = (_Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    for (NSUInteger row = 0; row < size; ++row)
        for (NSUInteger column = 0; column < size; ++column)
            data[row * size + column] = (_Float16)value(row, column);
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
}

static float half(float value) { return (float)(_Float16)value; }

static BOOL compare(ANEIOSurfaceBuffer *buffer, const std::vector<float> &reference,
                    NSUInteger run, const char *workload, float tolerance,
                    float relative) {
    IOSurfaceLock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    const _Float16 *actual =
        (const _Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    for (NSUInteger index = 0; index < reference.size(); ++index) {
        float expected = reference[index];
        float got = (float)actual[index];
        float error = std::fabs(got - expected);
        float allowed = tolerance + relative * std::fabs(expected);
        maximumError = std::fmax(maximumError, error);
        if (!std::isfinite(got) || error > allowed) {
            if (mismatches < 8)
                std::printf("MISMATCH workload=%s run=%lu index=%lu expected=%g "
                            "actual=%g\n", workload, (unsigned long)run,
                            (unsigned long)index, expected, got);
            ++mismatches;
        }
    }
    IOSurfaceUnlock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    std::printf("HARDWARE %s run=%lu elements=%lu mismatches=%lu "
                "max_abs_error=%g\n", workload, (unsigned long)run,
                (unsigned long)reference.size(), (unsigned long)mismatches,
                maximumError);
    return mismatches == 0;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) {
            std::fprintf(stderr, "usage: %s WORKLOAD BUNDLE_DIR MODEL_HASH...\n",
                         argv[0]);
            return 64;
        }
        NSString *workload = [NSString stringWithUTF8String:argv[1]];
        BOOL matmulScale = [workload isEqualToString:@"matmul-scale"];
        BOOL scaleExp = [workload isEqualToString:@"scale-exp"];
        if (!matmulScale && !scaleExp) return 64;
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[2]] isDirectory:YES]
            error:&error];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        for (int index = 3; index < argc; ++index)
            [hashes addObject:[NSString stringWithUTF8String:argv[index]]];
        if (!bundle || hashes.count != bundle.artifacts.count) {
            std::fprintf(stderr, "bundle/hash mismatch\n");
            return 2;
        }
        std::printf("PROGRAMS workload=%s count=%lu\n", argv[1],
                    (unsigned long)bundle.artifacts.count);
        for (NSUInteger index = 0; index < bundle.artifacts.count; ++index)
            std::printf("PROGRAM index=%lu operations=%s\n",
                        (unsigned long)index,
                        [bundle.artifacts[index].operations
                            componentsJoinedByString:@","].UTF8String);

        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        ANEIOSurfaceBuffer *y = bufferNamed(outputs, @"y");
        if (!runtime || !y || ![runtime loadWithError:&error]) {
            std::fprintf(stderr, "runtime: %s\n",
                         error.description.UTF8String ?: "setup");
            return 3;
        }
        NSUInteger elements = y.logicalByteLength / sizeof(_Float16);
        NSUInteger size = (NSUInteger)std::sqrt((double)elements);
        BOOL valid = size * size == elements;
        std::vector<float> reference(elements);
        for (NSUInteger run = 1; valid && run <= 2; ++run) {
            if (matmulScale) {
                ANEIOSurfaceBuffer *a = bufferNamed(inputs, @"a");
                ANEIOSurfaceBuffer *b = bufferNamed(inputs, @"b");
                if (!a || !b) { valid = NO; break; }
                const float scale = 0.08838834764831845f;
                fillMatrix(a, size, ^float(NSUInteger row, NSUInteger column) {
                    NSInteger raw = (NSInteger)((row * 7 + column * 3 + run) % 23) - 11;
                    return (float)raw * 0.0625f;
                });
                fillMatrix(b, size, ^float(NSUInteger row, NSUInteger column) {
                    // Sparse B keeps the FP16 accumulation exact enough to
                    // check the fold, not the matmul.
                    NSUInteger source = (column + 5 * run) % size;
                    if (row != source) return 0.0f;
                    return 0.5f + 0.125f * (float)((column + run) % 5);
                });
                std::vector<float> A(elements), B(elements);
                IOSurfaceLock(a.ioSurface, kIOSurfaceLockReadOnly, NULL);
                IOSurfaceLock(b.ioSurface, kIOSurfaceLockReadOnly, NULL);
                const _Float16 *ap = (const _Float16 *)IOSurfaceGetBaseAddress(a.ioSurface);
                const _Float16 *bp = (const _Float16 *)IOSurfaceGetBaseAddress(b.ioSurface);
                for (NSUInteger i = 0; i < elements; ++i) { A[i] = (float)ap[i]; B[i] = (float)bp[i]; }
                IOSurfaceUnlock(a.ioSurface, kIOSurfaceLockReadOnly, NULL);
                IOSurfaceUnlock(b.ioSurface, kIOSurfaceLockReadOnly, NULL);
                for (NSUInteger row = 0; row < size; ++row)
                    for (NSUInteger column = 0; column < size; ++column) {
                        float sum = 0.0f;
                        for (NSUInteger k = 0; k < size; ++k)
                            sum += A[row * size + k] * B[k * size + column];
                        reference[row * size + column] = half(half(sum) * half(scale));
                    }
            } else {
                ANEIOSurfaceBuffer *x = bufferNamed(inputs, @"x");
                if (!x) { valid = NO; break; }
                const float scale = 0.25f;
                fillMatrix(x, size, ^float(NSUInteger row, NSUInteger column) {
                    NSInteger raw = (NSInteger)((row * 5 + column * 11 + run) % 41) - 20;
                    return (float)raw * 0.25f;
                });
                IOSurfaceLock(x.ioSurface, kIOSurfaceLockReadOnly, NULL);
                const _Float16 *xp = (const _Float16 *)IOSurfaceGetBaseAddress(x.ioSurface);
                for (NSUInteger i = 0; i < elements; ++i)
                    reference[i] = std::exp(half(half((float)xp[i]) * scale));
                IOSurfaceUnlock(x.ioSurface, kIOSurfaceLockReadOnly, NULL);
            }
            IOSurfaceLock(y.ioSurface, 0, NULL);
            std::memset(IOSurfaceGetBaseAddress(y.ioSurface), 0xff,
                        y.allocationByteLength);
            IOSurfaceUnlock(y.ioSurface, 0, NULL);
            error = nil;
            if (![runtime evaluateInputs:inputs outputs:outputs error:&error]) {
                std::fprintf(stderr, "evaluate: %s\n",
                             error.description.UTF8String ?: "unknown");
                valid = NO;
                break;
            }
            // FP16 table exp carries a few ULP of relative error; the
            // matmul fold is compared at FP16 rounding of the product.
            valid = compare(y, reference, run, argv[1],
                            matmulScale ? 0.004f : 0.02f,
                            matmulScale ? 0.002f : 0.01f);
        }
        [runtime unloadWithError:nil];
        std::printf("SUMMARY %s valid=%d programs=%lu\n", argv[1], valid,
                    (unsigned long)bundle.artifacts.count);
        return valid ? 0 : 1;
    }
}
