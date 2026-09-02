#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

#import <cmath>
#import <cstdio>
#import <cstdlib>
#import <cstring>
#import <vector>

#import "ANEAppleBaselineRuntime.h"
#import "ANEBenchmarkStats.h"
#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static NSArray<NSString *> *inputNamesForWorkload(NSString *workload) {
    if ([workload isEqualToString:@"fa2"])
        return @[@"q", @"k", @"v"];
    if ([workload isEqualToString:@"affine-scan"])
        return @[@"state", @"a0", @"b0", @"a1", @"b1", @"a2",
                 @"b2", @"a3", @"b3"];
    if ([workload isEqualToString:@"matmul-gelu"])
        return @[@"a", @"b"];
    return nil;
}

static double toleranceForWorkload(NSString *workload) {
    if ([workload isEqualToString:@"fa2"]) return 0.012;
    if ([workload isEqualToString:@"affine-scan"]) return 0.008;
    return 0.012;
}

static ANEIOSurfaceBuffer *bufferNamed(
    NSArray<ANEIOSurfaceBuffer *> *buffers, NSString *identifier) {
    for (ANEIOSurfaceBuffer *buffer in buffers)
        if ([buffer.identifier isEqualToString:identifier]) return buffer;
    return nil;
}

static BOOL orderedInputs(NSArray<ANEIOSurfaceBuffer *> *buffers,
                          NSArray<NSString *> *names,
                          NSArray<ANEIOSurfaceBuffer *> **ordered) {
    NSMutableArray<ANEIOSurfaceBuffer *> *result = [NSMutableArray array];
    for (NSString *name in names) {
        ANEIOSurfaceBuffer *buffer = bufferNamed(buffers, name);
        if (!buffer) return NO;
        [result addObject:buffer];
    }
    *ordered = [result copy];
    return YES;
}

static void fillBuffer(ANEIOSurfaceBuffer *buffer,
                       _Float16 (^value)(NSUInteger)) {
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    _Float16 *data = (_Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    NSUInteger count = buffer.logicalByteLength / sizeof(_Float16);
    for (NSUInteger index = 0; index < count; ++index)
        data[index] = value(index);
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
}

static BOOL fillFA2(NSArray<ANEIOSurfaceBuffer *> *inputs) {
    if (inputs.count != 3) return NO;
    NSUInteger elements = inputs[0].logicalByteLength / sizeof(_Float16);
    NSUInteger size = (NSUInteger)std::sqrt((double)elements);
    if (size * size != elements) return NO;
    fillBuffer(inputs[0], ^_Float16(NSUInteger index) {
        return (_Float16)(index / size == index % size ? 1.0f : 0.0f);
    });
    fillBuffer(inputs[1], ^_Float16(NSUInteger index) {
        return (_Float16)(index / size == index % size ? 1.0f : 0.0f);
    });
    fillBuffer(inputs[2], ^_Float16(NSUInteger index) {
        NSUInteger row = index / size;
        NSUInteger column = index % size;
        NSInteger raw = (NSInteger)((row * 3 + column * 5) % 17) - 8;
        return (_Float16)((float)raw * 0.03125f);
    });
    return YES;
}

static BOOL fillAffineScan(NSArray<ANEIOSurfaceBuffer *> *inputs) {
    if (inputs.count != 9) return NO;
    fillBuffer(inputs[0], ^_Float16(NSUInteger index) {
        NSInteger raw = (NSInteger)(index % 13) - 6;
        return (_Float16)((float)raw * 0.03125f);
    });
    for (NSUInteger chunk = 0; chunk < 4; ++chunk) {
        fillBuffer(inputs[1 + chunk * 2], ^_Float16(NSUInteger index) {
            float base = 0.5f + (float)chunk * 0.0625f;
            float variation = (float)((index + 1 + chunk) % 3) * 0.03125f;
            return (_Float16)(base + variation);
        });
        fillBuffer(inputs[2 + chunk * 2], ^_Float16(NSUInteger index) {
            NSInteger raw = (NSInteger)((index * 3 + chunk + 1) % 9) - 4;
            return (_Float16)((float)raw * 0.015625f);
        });
    }
    return YES;
}

static BOOL fillMatmulGELU(NSArray<ANEIOSurfaceBuffer *> *inputs) {
    if (inputs.count != 2) return NO;
    NSUInteger elements = inputs[0].logicalByteLength / sizeof(_Float16);
    NSUInteger size = (NSUInteger)std::sqrt((double)elements);
    if (size * size != elements ||
        inputs[1].logicalByteLength / sizeof(_Float16) != elements)
        return NO;
    fillBuffer(inputs[0], ^_Float16(NSUInteger index) {
        NSUInteger row = index / size;
        NSUInteger column = index % size;
        NSInteger raw = (NSInteger)((row * 5 + column * 3 + 1) % 33) - 16;
        return (_Float16)((float)raw * 0.125f);
    });
    fillBuffer(inputs[1], ^_Float16(NSUInteger index) {
        NSUInteger row = index / size;
        NSUInteger column = index % size;
        NSUInteger source = (column + 17) % size;
        if (row != source) return (_Float16)0.0f;
        return (_Float16)(0.5f + 0.125f * (float)((column + 1) % 5));
    });
    return YES;
}

static BOOL fillInputs(NSString *workload,
                       NSArray<ANEIOSurfaceBuffer *> *inputs) {
    if ([workload isEqualToString:@"fa2"]) return fillFA2(inputs);
    if ([workload isEqualToString:@"affine-scan"])
        return fillAffineScan(inputs);
    if ([workload isEqualToString:@"matmul-gelu"])
        return fillMatmulGELU(inputs);
    return NO;
}

static void poison(ANEIOSurfaceBuffer *buffer) {
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    std::memset(IOSurfaceGetBaseAddress(buffer.ioSurface), 0xff,
                buffer.allocationByteLength);
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
}

static NSData *copyOutput(ANEIOSurfaceBuffer *buffer) {
    IOSurfaceLock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    NSData *data = [NSData dataWithBytes:IOSurfaceGetBaseAddress(buffer.ioSurface)
                                 length:buffer.logicalByteLength];
    IOSurfaceUnlock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    return data;
}

static BOOL compareOutputs(NSString *workload, NSData *research,
                           NSData *apple) {
    if (research.length != apple.length || research.length % 2 != 0) return NO;
    const _Float16 *researchValues = (const _Float16 *)research.bytes;
    const _Float16 *appleValues = (const _Float16 *)apple.bytes;
    NSUInteger count = research.length / sizeof(_Float16);
    NSUInteger mismatches = 0;
    float maximumError = 0.0f;
    double tolerance = toleranceForWorkload(workload);
    for (NSUInteger index = 0; index < count; ++index) {
        float left = (float)researchValues[index];
        float right = (float)appleValues[index];
        float error = std::fabs(left - right);
        maximumError = std::fmax(maximumError, error);
        if (!std::isfinite(left) || !std::isfinite(right) ||
            error > tolerance) {
            if (mismatches < 8)
                std::printf("MISMATCH workload=%s index=%lu research=%g "
                            "anec=%g error=%g\n", workload.UTF8String,
                            (unsigned long)index, left, right, error);
            ++mismatches;
        }
    }
    std::printf("NUMERICAL workload=%s elements=%lu mismatches=%lu "
                "max_abs_error=%g tolerance=%g\n", workload.UTF8String,
                (unsigned long)count, (unsigned long)mismatches,
                maximumError, tolerance);
    return mismatches == 0;
}

static double elapsedMicroseconds(uint64_t start, uint64_t stop) {
    static mach_timebase_info_data_t timebase = {};
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)(stop - start) * (double)timebase.numer /
           (double)timebase.denom / 1000.0;
}

typedef BOOL (^Evaluation)(NSError **);

static BOOL evaluateRepeated(Evaluation evaluation, NSUInteger count,
                             std::vector<double> *samples, NSError **error) {
    for (NSUInteger iteration = 0; iteration < count; ++iteration) {
        @autoreleasepool {
            NSError *iterationError = nil;
            uint64_t started = mach_continuous_time();
            BOOL valid = evaluation(&iterationError);
            uint64_t stopped = mach_continuous_time();
            if (!valid) {
                if (error) *error = iterationError;
                return NO;
            }
            if (samples)
                samples->push_back(elapsedMicroseconds(started, stopped));
        }
    }
    return YES;
}

static BOOL measureBatch(Evaluation evaluation, NSUInteger iterations,
                         std::vector<double> *allSamples,
                         std::vector<double> *batchMedians, NSError **error) {
    std::vector<double> batch;
    if (!evaluateRepeated(evaluation, iterations, &batch, error)) return NO;
    ANELatencySummary summary{};
    if (!ANESummarizeLatencies(batch, &summary)) return NO;
    allSamples->insert(allSamples->end(), batch.begin(), batch.end());
    batchMedians->push_back(summary.medianMicroseconds);
    return YES;
}

static void printSummary(NSString *workload, const char *implementation,
                         const std::vector<double> &samples,
                         const std::vector<double> &batchMedians) {
    ANELatencySummary summary{};
    ANELatencySummary batches{};
    ANESummarizeLatencies(samples, &summary);
    ANESummarizeLatencies(batchMedians, &batches);
    std::printf("RESULT workload=%s implementation=%s samples=%lu batches=%lu "
                "batch_median_us=%.3f median_us=%.3f p95_us=%.3f "
                "min_us=%.3f max_us=%.3f mean_us=%.3f\n",
                workload.UTF8String, implementation,
                (unsigned long)samples.size(),
                (unsigned long)batchMedians.size(),
                batches.medianMicroseconds, summary.medianMicroseconds,
                summary.p95Microseconds, summary.minimumMicroseconds,
                summary.maximumMicroseconds, summary.meanMicroseconds);
}

static BOOL parseCount(const char *text, NSUInteger *value) {
    char *end = nullptr;
    unsigned long long raw = std::strtoull(text, &end, 10);
    if (!text[0] || !end || *end != '\0' || raw == 0 || raw > NSUIntegerMax)
        return NO;
    *value = (NSUInteger)raw;
    return YES;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 8) {
            std::fprintf(stderr, "usage: %s WORKLOAD MIL BUNDLE WARMUP "
                         "ITERATIONS BATCHES MODEL_HASH...\n", argv[0]);
            return 64;
        }
        NSString *workload = [NSString stringWithUTF8String:argv[1]];
        NSArray<NSString *> *inputNames = inputNamesForWorkload(workload);
        NSUInteger warmup = 0;
        NSUInteger iterations = 0;
        NSUInteger batches = 0;
        if (!inputNames || !parseCount(argv[4], &warmup) ||
            !parseCount(argv[5], &iterations) ||
            !parseCount(argv[6], &batches))
            return 64;

        NSError *error = nil;
        NSData *milData = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:argv[2]] options:0 error:&error];
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[3]] isDirectory:YES]
            error:&error];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        for (int index = 7; index < argc; ++index)
            [hashes addObject:[NSString stringWithUTF8String:argv[index]]];
        if (!milData || !bundle || hashes.count != bundle.artifacts.count) {
            std::fprintf(stderr, "setup: %s\n",
                         error.description.UTF8String ?: "invalid bundle/hash count");
            return 2;
        }

        ANEProvisionedRuntime *research = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *researchInputs =
            [research createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *researchOutputs =
            [research createOutputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *appleInputs = nil;
        ANEIOSurfaceBuffer *output = bufferNamed(researchOutputs, @"y");
        if (!research || !output ||
            !orderedInputs(researchInputs, inputNames, &appleInputs) ||
            !fillInputs(workload, appleInputs)) {
            std::fprintf(stderr, "surface setup: %s\n",
                         error.description.UTF8String ?: "binding mismatch");
            return 3;
        }

        CFAbsoluteTime researchLoadStarted = CFAbsoluteTimeGetCurrent();
        BOOL researchLoaded = [research loadWithError:&error];
        double researchLoadMicroseconds =
            (CFAbsoluteTimeGetCurrent() - researchLoadStarted) * 1.0e6;
        ANEAppleBaselineRuntime *apple = [[ANEAppleBaselineRuntime alloc]
            initWithMILData:milData qos:21 error:&error];
        BOOL appleLoaded = apple && [apple loadWithError:&error];
        if (!researchLoaded || !appleLoaded) {
            std::fprintf(stderr, "load: %s\n",
                         error.description.UTF8String ?: "unknown failure");
            if (research.loaded) [research unloadWithError:nil];
            return 4;
        }
        std::printf("PREPARE workload=%s anec_compile_us=%.1f "
                    "anec_load_us=%.1f research_load_us=%.1f "
                    "research_artifacts=%lu\n", workload.UTF8String,
                    apple.compileMicroseconds, apple.loadMicroseconds,
                    researchLoadMicroseconds,
                    (unsigned long)bundle.artifacts.count);

        Evaluation researchEvaluation = ^BOOL(NSError **evaluationError) {
            return [research evaluateInputs:researchInputs
                                    outputs:researchOutputs
                                      error:evaluationError];
        };
        Evaluation appleEvaluation = ^BOOL(NSError **evaluationError) {
            return [apple evaluateInputs:appleInputs outputs:@[output]
                                   error:evaluationError];
        };

        BOOL valid = YES;
        poison(output);
        valid = researchEvaluation(&error);
        NSData *researchOutput = valid ? copyOutput(output) : nil;
        poison(output);
        valid = valid && appleEvaluation(&error);
        NSData *appleOutput = valid ? copyOutput(output) : nil;
        valid = valid && compareOutputs(workload, researchOutput, appleOutput);

        std::vector<double> researchSamples;
        std::vector<double> appleSamples;
        std::vector<double> researchBatchMedians;
        std::vector<double> appleBatchMedians;
        valid = valid && evaluateRepeated(researchEvaluation, warmup, nullptr,
                                          &error);
        valid = valid && evaluateRepeated(appleEvaluation, warmup, nullptr,
                                          &error);
        for (NSUInteger batch = 0; valid && batch < batches; ++batch) {
            if (batch % 2 == 0) {
                valid = measureBatch(researchEvaluation, iterations,
                    &researchSamples, &researchBatchMedians, &error) &&
                    measureBatch(appleEvaluation, iterations,
                    &appleSamples, &appleBatchMedians, &error);
            } else {
                valid = measureBatch(appleEvaluation, iterations,
                    &appleSamples, &appleBatchMedians, &error) &&
                    measureBatch(researchEvaluation, iterations,
                    &researchSamples, &researchBatchMedians, &error);
            }
        }
        if (valid) {
            printSummary(workload, "aneccompile", appleSamples,
                         appleBatchMedians);
            printSummary(workload, "research", researchSamples,
                         researchBatchMedians);
            ANELatencySummary appleBatches{};
            ANELatencySummary researchBatches{};
            ANESummarizeLatencies(appleBatchMedians, &appleBatches);
            ANESummarizeLatencies(researchBatchMedians, &researchBatches);
            std::printf("COMPARISON workload=%s research_over_anec=%.3f "
                        "anec_speedup=%.3f\n", workload.UTF8String,
                        researchBatches.medianMicroseconds /
                            appleBatches.medianMicroseconds,
                        researchBatches.medianMicroseconds /
                            appleBatches.medianMicroseconds);
        } else {
            std::fprintf(stderr, "benchmark: %s\n",
                         error.description.UTF8String ?: "numerical failure");
        }
        [apple unloadWithError:nil];
        [research unloadWithError:nil];
        std::printf("SUMMARY compiler-ab workload=%s valid=%d\n",
                    workload.UTF8String, valid);
        return valid ? 0 : 1;
    }
}
