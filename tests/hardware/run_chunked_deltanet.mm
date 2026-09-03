#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#import "ANEBenchmarkStats.h"
#import "ANEAppleBaselineRuntime.h"
#import "ANEExecutableBundle.h"
#import "ANEHWXArtifact.h"
#import "ANEProvisionedRuntime.h"

static constexpr NSUInteger kSize = 128;

struct CaseData {
    std::vector<_Float16> q;
    std::vector<_Float16> k;
    std::vector<_Float16> v;
    std::vector<_Float16> beta;
    std::vector<_Float16> logDecay;
    std::vector<_Float16> state;
};

static ANEIOSurfaceBuffer *bufferNamed(
    NSArray<ANEIOSurfaceBuffer *> *buffers, NSString *identifier) {
    for (ANEIOSurfaceBuffer *buffer in buffers)
        if ([buffer.identifier isEqualToString:identifier]) return buffer;
    return nil;
}

static NSArray<ANEIOSurfaceBuffer *> *orderedBuffers(
    NSArray<ANEIOSurfaceBuffer *> *buffers, NSArray<NSString *> *identifiers) {
    NSMutableArray<ANEIOSurfaceBuffer *> *ordered = [NSMutableArray array];
    for (NSString *identifier in identifiers) {
        ANEIOSurfaceBuffer *buffer = bufferNamed(buffers, identifier);
        if (!buffer) return nil;
        [ordered addObject:buffer];
    }
    return [ordered copy];
}

static ANEHWXBinding *bindingNamed(ANEExecutableBundle *bundle,
                                   NSString *identifier) {
    for (ANEHWXArtifact *artifact in bundle.artifacts)
        for (ANEHWXBinding *binding in artifact.bindings)
            if ([binding.identifier isEqualToString:identifier]) return binding;
    return nil;
}

static BOOL writeTensor(ANEExecutableBundle *bundle,
                        NSArray<ANEIOSurfaceBuffer *> *buffers,
                        NSString *identifier, NSUInteger rows,
                        NSUInteger columns,
                        _Float16 (^value)(NSUInteger, NSUInteger)) {
    ANEIOSurfaceBuffer *buffer = bufferNamed(buffers, identifier);
    ANEHWXBinding *binding = bindingNamed(bundle, identifier);
    if (!buffer || !binding || binding.rowStrideBytes < columns * 2 ||
        binding.allocationByteLength < rows * binding.rowStrideBytes)
        return NO;
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    uint8_t *base = (uint8_t *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    std::memset(base, 0, buffer.allocationByteLength);
    for (NSUInteger row = 0; row < rows; ++row) {
        _Float16 *destination =
            (_Float16 *)(base + row * binding.rowStrideBytes);
        for (NSUInteger column = 0; column < columns; ++column)
            destination[column] = value(row, column);
    }
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
    return YES;
}

static std::vector<_Float16> readMatrix(ANEExecutableBundle *bundle,
                                        ANEIOSurfaceBuffer *buffer) {
    std::vector<_Float16> values(kSize * kSize);
    ANEHWXBinding *binding = bindingNamed(bundle, buffer.identifier);
    if (!binding || binding.rowStrideBytes < kSize * 2) return {};
    IOSurfaceLock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    const uint8_t *base =
        (const uint8_t *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    for (NSUInteger row = 0; row < kSize; ++row) {
        const _Float16 *source =
            (const _Float16 *)(base + row * binding.rowStrideBytes);
        std::memcpy(values.data() + row * kSize, source,
                    kSize * sizeof(_Float16));
    }
    IOSurfaceUnlock(buffer.ioSurface, kIOSurfaceLockReadOnly, NULL);
    return values;
}

static CaseData makeCase(BOOL activeUpdate) {
    CaseData data;
    data.q.assign(kSize * kSize, (_Float16)0.0f);
    data.k.assign(kSize * kSize, (_Float16)0.0f);
    data.v.resize(kSize * kSize);
    data.beta.resize(kSize);
    data.logDecay.resize(kSize);
    data.state.assign(kSize * kSize, (_Float16)0.0f);
    for (NSUInteger row = 0; row < kSize; ++row) {
        data.k[row * kSize + row] = (_Float16)0.96875f;
        data.k[row * kSize + (row + 1) % kSize] = (_Float16)0.25f;
        data.q[row * kSize + row] = (_Float16)0.9375f;
        data.q[row * kSize + (row + 1) % kSize] = (_Float16)-0.25f;
        data.q[row * kSize + (row + 2) % kSize] = (_Float16)0.25f;
        data.beta[row] = activeUpdate
            ? (_Float16)(0.09375f + 0.015625f * (float)(row % 3))
            : (_Float16)0.0f;
        data.logDecay[row] =
            (_Float16)(-0.001953125f * (float)(1 + row % 3));
        data.state[row * kSize + row] = (_Float16)0.03125f;
        data.state[row * kSize + (row + 5) % kSize] = (_Float16)-0.015625f;
        for (NSUInteger column = 0; column < kSize; ++column) {
            NSInteger raw =
                (NSInteger)((row * 7 + column * 11 + 3) % 17) - 8;
            data.v[row * kSize + column] =
                (_Float16)((float)raw * 0.00390625f);
        }
    }
    return data;
}

static BOOL fillCase(ANEExecutableBundle *bundle,
                     NSArray<ANEIOSurfaceBuffer *> *inputs,
                     const CaseData &data) {
    BOOL valid = YES;
    valid = valid && writeTensor(bundle, inputs, @"q", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            return data.q[row * kSize + column];
        });
    valid = valid && writeTensor(bundle, inputs, @"k", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            return data.k[row * kSize + column];
        });
    valid = valid && writeTensor(bundle, inputs, @"kt", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            return data.k[column * kSize + row];
        });
    valid = valid && writeTensor(bundle, inputs, @"v", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            return data.v[row * kSize + column];
        });
    valid = valid && writeTensor(bundle, inputs, @"beta", kSize, 1,
        ^_Float16(NSUInteger row, NSUInteger column) {
            (void)column;
            return data.beta[row];
        });
    valid = valid && writeTensor(bundle, inputs, @"log_decay_diagonal",
        kSize, kSize, ^_Float16(NSUInteger row, NSUInteger column) {
            return row == column ? data.logDecay[row] : (_Float16)0.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"state", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            return data.state[row * kSize + column];
        });
    valid = valid && writeTensor(bundle, inputs, @"negative_strict_lower",
        kSize, kSize, ^_Float16(NSUInteger row, NSUInteger column) {
            return column < row ? (_Float16)-1.0f : (_Float16)0.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"lower_inclusive",
        kSize, kSize, ^_Float16(NSUInteger row, NSUInteger column) {
            return column <= row ? (_Float16)1.0f : (_Float16)0.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"upper_inclusive",
        kSize, kSize, ^_Float16(NSUInteger row, NSUInteger column) {
            return column >= row ? (_Float16)1.0f : (_Float16)0.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"identity", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            return row == column ? (_Float16)1.0f : (_Float16)0.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"ones", kSize, kSize,
        ^_Float16(NSUInteger row, NSUInteger column) {
            (void)row;
            (void)column;
            return (_Float16)1.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"last_row_selector",
        kSize, kSize, ^_Float16(NSUInteger row, NSUInteger column) {
            (void)row;
            return column + 1 == kSize ? (_Float16)1.0f : (_Float16)0.0f;
        });
    valid = valid && writeTensor(bundle, inputs, @"negative_ones",
        kSize, kSize, ^_Float16(NSUInteger row, NSUInteger column) {
            (void)row;
            (void)column;
            return (_Float16)-1.0f;
        });
    return valid;
}

static void sequentialReference(const CaseData &data,
                                std::vector<float> *output,
                                std::vector<float> *finalState) {
    std::vector<float> state(kSize * kSize);
    for (NSUInteger index = 0; index < state.size(); ++index)
        state[index] = (float)data.state[index];
    output->assign(kSize * kSize, 0.0f);
    std::vector<float> delta(kSize);
    for (NSUInteger token = 0; token < kSize; ++token) {
        float decay = std::exp((float)data.logDecay[token]);
        for (float &element : state) element *= decay;
        for (NSUInteger column = 0; column < kSize; ++column) {
            float prediction = 0.0f;
            for (NSUInteger row = 0; row < kSize; ++row)
                prediction += (float)data.k[token * kSize + row] *
                              state[row * kSize + column];
            delta[column] = (float)data.beta[token] *
                ((float)data.v[token * kSize + column] - prediction);
        }
        for (NSUInteger row = 0; row < kSize; ++row)
            for (NSUInteger column = 0; column < kSize; ++column)
                state[row * kSize + column] +=
                    (float)data.k[token * kSize + row] * delta[column];
        for (NSUInteger column = 0; column < kSize; ++column)
            for (NSUInteger row = 0; row < kSize; ++row)
                (*output)[token * kSize + column] +=
                    (float)data.q[token * kSize + row] *
                    state[row * kSize + column];
    }
    *finalState = std::move(state);
}

static BOOL compareMatrix(NSString *caseName, NSString *outputName,
                          const std::vector<_Float16> &actual,
                          const std::vector<float> &expected) {
    if (actual.size() != expected.size()) return NO;
    double squaredError = 0.0;
    double squaredReference = 0.0;
    float maximumError = 0.0f;
    NSUInteger nonfinite = 0;
    NSUInteger overOnePercent = 0;
    for (NSUInteger index = 0; index < actual.size(); ++index) {
        float observed = (float)actual[index];
        float error = std::fabs(observed - expected[index]);
        if (!std::isfinite(observed)) ++nonfinite;
        if (error > 0.01f) ++overOnePercent;
        maximumError = std::fmax(maximumError, error);
        squaredError += (double)error * error;
        squaredReference += (double)expected[index] * expected[index];
    }
    double relativeL2 = std::sqrt(squaredError /
        std::fmax(squaredReference, 1.0e-30));
    std::printf("NUMERICAL chunked-deltanet case=%s output=%s elements=%lu "
                "nonfinite=%lu over_0.01=%lu max_abs_error=%g rel_l2=%g\n",
                caseName.UTF8String, outputName.UTF8String,
                (unsigned long)actual.size(), (unsigned long)nonfinite,
                (unsigned long)overOnePercent, maximumError, relativeL2);
    return nonfinite == 0 && maximumError <= 0.01f && relativeL2 <= 0.05;
}

static BOOL runCase(ANEProvisionedRuntime *runtime,
                    NSArray<ANEIOSurfaceBuffer *> *inputs,
                    NSArray<ANEIOSurfaceBuffer *> *outputs,
                    NSString *caseName, BOOL activeUpdate, NSError **error) {
    CaseData data = makeCase(activeUpdate);
    if (!fillCase(runtime.bundle, inputs, data)) return NO;
    if (![runtime evaluateInputs:inputs outputs:outputs error:error]) return NO;
    std::vector<float> expectedOutput;
    std::vector<float> expectedState;
    sequentialReference(data, &expectedOutput, &expectedState);
    ANEIOSurfaceBuffer *output = bufferNamed(outputs, @"output");
    ANEIOSurfaceBuffer *finalState = bufferNamed(outputs, @"final_state");
    return output && finalState &&
        compareMatrix(caseName, @"output",
                      readMatrix(runtime.bundle, output), expectedOutput) &&
        compareMatrix(caseName, @"final_state",
                      readMatrix(runtime.bundle, finalState), expectedState);
}

static double elapsedMicroseconds(uint64_t start, uint64_t stop) {
    static mach_timebase_info_data_t timebase = {};
    if (timebase.denom == 0) mach_timebase_info(&timebase);
    return (double)(stop - start) * (double)timebase.numer /
           (double)timebase.denom / 1000.0;
}

static NSUInteger environmentCount(const char *name, NSUInteger fallback) {
    const char *text = std::getenv(name);
    if (!text || !text[0]) return fallback;
    char *end = nullptr;
    unsigned long raw = std::strtoul(text, &end, 10);
    return end && *end == '\0' && raw > 0 ? (NSUInteger)raw : fallback;
}

typedef BOOL (^Evaluation)(NSError **);

static BOOL evaluateRepeated(Evaluation evaluation, NSUInteger iterations,
                             std::vector<double> *samples, NSError **error) {
    for (NSUInteger index = 0; index < iterations; ++index) {
        uint64_t start = mach_continuous_time();
        NSError *iterationError = nil;
        BOOL valid = evaluation(&iterationError);
        uint64_t stop = mach_continuous_time();
        if (!valid) {
            if (error) *error = iterationError;
            return NO;
        }
        if (samples) samples->push_back(elapsedMicroseconds(start, stop));
    }
    return YES;
}

static BOOL measureBatch(Evaluation evaluation, NSUInteger iterations,
                         std::vector<double> *samples,
                         std::vector<double> *batchMedians, NSError **error) {
    std::vector<double> batch;
    if (!evaluateRepeated(evaluation, iterations, &batch, error)) return NO;
    ANELatencySummary summary{};
    if (!ANESummarizeLatencies(batch, &summary)) return NO;
    samples->insert(samples->end(), batch.begin(), batch.end());
    batchMedians->push_back(summary.medianMicroseconds);
    return YES;
}

static ANELatencySummary printPerformance(
    const char *implementation, const std::vector<double> &samples,
    const std::vector<double> &batchMedians, NSUInteger warmup) {
    ANELatencySummary summary{};
    ANELatencySummary batches{};
    ANESummarizeLatencies(samples, &summary);
    ANESummarizeLatencies(batchMedians, &batches);
    std::printf("PERF chunked-deltanet implementation=%s warmup=%lu "
                "samples=%lu batches=%lu batch_median_us=%.3f "
                "median_us=%.3f p95_us=%.3f "
                "min_us=%.3f max_us=%.3f mean_us=%.3f\n",
                implementation, (unsigned long)warmup,
                (unsigned long)samples.size(),
                (unsigned long)batchMedians.size(),
                batches.medianMicroseconds, summary.medianMicroseconds,
                summary.p95Microseconds, summary.minimumMicroseconds,
                summary.maximumMicroseconds, summary.meanMicroseconds);
    return batches;
}

static BOOL benchmark(ANEProvisionedRuntime *research,
                      ANEAppleBaselineRuntime *apple,
                      NSArray<ANEIOSurfaceBuffer *> *researchInputs,
                      NSArray<ANEIOSurfaceBuffer *> *appleInputs,
                      NSArray<ANEIOSurfaceBuffer *> *outputs,
                      NSError **error) {
    NSUInteger warmup = environmentCount("CD_BENCHMARK_WARMUP", 10);
    NSUInteger iterations = environmentCount("CD_BENCHMARK_ITERATIONS", 50);
    NSUInteger batches = environmentCount("CD_BENCHMARK_BATCHES", 5);
    Evaluation researchEvaluation = ^BOOL(NSError **evaluationError) {
        return [research evaluateInputs:researchInputs outputs:outputs
                                  error:evaluationError];
    };
    Evaluation appleEvaluation = ^BOOL(NSError **evaluationError) {
        return [apple evaluateInputs:appleInputs outputs:outputs
                               error:evaluationError];
    };
    if (!evaluateRepeated(researchEvaluation, warmup, nullptr, error) ||
        !evaluateRepeated(appleEvaluation, warmup, nullptr, error))
        return NO;
    std::vector<double> researchSamples;
    std::vector<double> appleSamples;
    std::vector<double> researchMedians;
    std::vector<double> appleMedians;
    for (NSUInteger batch = 0; batch < batches; ++batch) {
        BOOL researchFirst = batch % 2 == 0;
        Evaluation first = researchFirst ? researchEvaluation : appleEvaluation;
        Evaluation second = researchFirst ? appleEvaluation : researchEvaluation;
        std::vector<double> *firstSamples =
            researchFirst ? &researchSamples : &appleSamples;
        std::vector<double> *secondSamples =
            researchFirst ? &appleSamples : &researchSamples;
        std::vector<double> *firstMedians =
            researchFirst ? &researchMedians : &appleMedians;
        std::vector<double> *secondMedians =
            researchFirst ? &appleMedians : &researchMedians;
        if (!measureBatch(first, iterations, firstSamples, firstMedians,
                          error) ||
            !measureBatch(second, iterations, secondSamples, secondMedians,
                          error))
            return NO;
    }
    ANELatencySummary appleSummary = printPerformance(
        "aneccompile", appleSamples, appleMedians, warmup);
    ANELatencySummary researchSummary = printPerformance(
        "research", researchSamples, researchMedians, warmup);
    std::printf("COMPARISON chunked-deltanet research_over_anec=%.3f "
                "research_speedup=%.3f research_artifacts=%lu\n",
                researchSummary.medianMicroseconds /
                    appleSummary.medianMicroseconds,
                appleSummary.medianMicroseconds /
                    researchSummary.medianMicroseconds,
                (unsigned long)research.bundle.artifacts.count);
    return YES;
}

static BOOL benchmarkResearchOnly(
    ANEProvisionedRuntime *research,
    NSArray<ANEIOSurfaceBuffer *> *inputs,
    NSArray<ANEIOSurfaceBuffer *> *outputs, NSError **error) {
    NSUInteger warmup = environmentCount("CD_BENCHMARK_WARMUP", 10);
    NSUInteger iterations = environmentCount("CD_BENCHMARK_ITERATIONS", 50);
    NSUInteger batches = environmentCount("CD_BENCHMARK_BATCHES", 5);
    Evaluation evaluation = ^BOOL(NSError **evaluationError) {
        return [research evaluateInputs:inputs outputs:outputs
                                  error:evaluationError];
    };
    if (!evaluateRepeated(evaluation, warmup, nullptr, error)) return NO;
    std::vector<double> samples;
    std::vector<double> medians;
    for (NSUInteger batch = 0; batch < batches; ++batch)
        if (!measureBatch(evaluation, iterations, &samples, &medians, error))
            return NO;
    printPerformance("research", samples, medians, warmup);
    return YES;
}

static BOOL validateApple(ANEExecutableBundle *bundle,
                          ANEAppleBaselineRuntime *apple,
                          NSArray<ANEIOSurfaceBuffer *> *appleInputs,
                          NSArray<ANEIOSurfaceBuffer *> *outputs,
                          NSError **error) {
    CaseData data = makeCase(YES);
    if (!fillCase(bundle, appleInputs, data) ||
        ![apple evaluateInputs:appleInputs outputs:outputs error:error])
        return NO;
    std::vector<float> expectedOutput;
    std::vector<float> expectedState;
    sequentialReference(data, &expectedOutput, &expectedState);
    return compareMatrix(@"anec-active-update", @"output",
                         readMatrix(bundle, bufferNamed(outputs, @"output")),
                         expectedOutput) &&
        compareMatrix(@"anec-active-update", @"final_state",
                      readMatrix(bundle, bufferNamed(outputs, @"final_state")),
                      expectedState);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 4) return 64;
        NSError *error = nil;
        NSData *milData = [NSData dataWithContentsOfFile:
            [NSString stringWithUTF8String:argv[1]] options:0 error:&error];
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[2]] isDirectory:YES]
            error:&error];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        for (int index = 3; index < argc; ++index)
            [hashes addObject:[NSString stringWithUTF8String:argv[index]]];
        if (!milData || !bundle || bundle.artifacts.count != 58 ||
            bundle.dispatchPlan.count != 58 ||
            hashes.count != bundle.artifacts.count) {
            std::fprintf(stderr, "bundle: %s\n",
                         error.description.UTF8String ?: "unexpected graph");
            return 2;
        }
        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *appleInputs = orderedBuffers(inputs, @[
            @"q", @"k", @"kt", @"v", @"beta", @"log_decay_diagonal",
            @"state", @"negative_strict_lower", @"lower_inclusive",
            @"upper_inclusive", @"identity", @"ones",
            @"last_row_selector", @"negative_ones",
        ]);
        NSArray<ANEIOSurfaceBuffer *> *orderedOutputs = orderedBuffers(
            outputs, @[@"output", @"final_state"]);
        if (!runtime || !appleInputs || !orderedOutputs ||
            inputs.count != 14 || outputs.count != 2) {
            std::fprintf(stderr, "surfaces: %s\n",
                         error.description.UTF8String ?: "binding mismatch");
            return 3;
        }
        BOOL valid = [runtime loadWithError:&error];
        if (!valid)
            std::fprintf(stderr, "load: %s\n", error.description.UTF8String);
        @try {
            valid = valid && runCase(runtime, inputs, outputs,
                                     @"decay-only", NO, &error);
            valid = valid && runCase(runtime, inputs, outputs,
                                     @"active-update", YES, &error);
            ANEAppleBaselineRuntime *apple = [[ANEAppleBaselineRuntime alloc]
                initWithMILData:milData qos:21 error:&error];
            if (!apple) {
                std::string message = error.description.UTF8String ?: "";
                if (ANEIsInvalidMILProgramError(message)) {
                    std::printf("BASELINE_UNAVAILABLE chunked-deltanet "
                                "reason=InvalidMILProgram\n");
                    valid = valid && fillCase(bundle, inputs, makeCase(YES));
                    valid = valid && benchmarkResearchOnly(
                        runtime, inputs, orderedOutputs, &error);
                } else {
                    valid = NO;
                    std::fprintf(stderr, "ANECCompile: %s\n",
                                 error.description.UTF8String);
                }
            } else {
                valid = valid && [apple loadWithError:&error];
                if (valid)
                    std::printf("PREPARE chunked-deltanet "
                                "anec_compile_us=%.1f anec_load_us=%.1f "
                                "research_artifacts=58\n",
                                apple.compileMicroseconds,
                                apple.loadMicroseconds);
                valid = valid && validateApple(bundle, apple, appleInputs,
                                                orderedOutputs, &error);
                valid = valid && fillCase(bundle, inputs, makeCase(YES));
                valid = valid && benchmark(runtime, apple, inputs,
                                           appleInputs, orderedOutputs,
                                           &error);
                if (apple.loaded) [apple unloadWithError:nil];
            }
            if (!valid && error)
                std::fprintf(stderr, "evaluate: %s\n",
                             error.description.UTF8String);
        } @finally {
            if (runtime.loaded) [runtime unloadWithError:nil];
        }
        std::printf("SUMMARY chunked-deltanet C=128 D=128 generic_ops=58 "
                    "valid=%d\n", valid);
        return valid ? 0 : 1;
    }
}
