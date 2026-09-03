// Profiles every program submission of a compiled bundle on the M4 ANE.
//
// For each dispatched program the tool reports the operations it executes,
// its bound surfaces with logical and physical sizes, and stable per-program
// costs: IOSurface wrapping, request construction, the synchronous submit
// wall time, and the accelerator execution time the runtime reports. It also
// prints the compiler's composition trace so every declined transition is
// visible next to the cost it leaves on the table.
#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>

#import <algorithm>
#import <cstdio>
#import <cstdlib>
#import <cstring>
#import <vector>

#import "ANEBenchmarkStats.h"
#import "ANEExecutableBundle.h"
#import "ANEProvisionedRuntime.h"

static const char *roleName(ANESurfaceRole role) {
    switch (role) {
        case ANESurfaceRoleInput: return "input";
        case ANESurfaceRoleWeight: return "constant";
        case ANESurfaceRoleIntermediate: return "intermediate";
        case ANESurfaceRoleOutput: return "output";
    }
    return "unknown";
}

static void fillDeterministic(ANEIOSurfaceBuffer *buffer, NSUInteger seed) {
    IOSurfaceLock(buffer.ioSurface, 0, NULL);
    _Float16 *data = (_Float16 *)IOSurfaceGetBaseAddress(buffer.ioSurface);
    NSUInteger count = buffer.logicalByteLength / sizeof(_Float16);
    for (NSUInteger index = 0; index < count; ++index) {
        NSInteger raw = (NSInteger)((index * (3 + seed) + seed * 7) % 17) - 8;
        data[index] = (_Float16)((float)raw * 0.03125f);
    }
    IOSurfaceUnlock(buffer.ioSurface, 0, NULL);
}

static BOOL parseCount(const char *text, NSUInteger *value) {
    char *end = nullptr;
    unsigned long long raw = std::strtoull(text, &end, 10);
    if (!text[0] || !end || *end != '\0' || raw > NSUIntegerMax) return NO;
    *value = (NSUInteger)raw;
    return YES;
}

struct ProgramSamples {
    std::vector<double> wrap;
    std::vector<double> request;
    std::vector<double> submit;
    std::vector<double> hardware;
    std::vector<double> hostWait;
};

static void printSummaryLine(const char *label, NSUInteger program,
                             const std::vector<double> &samples) {
    ANELatencySummary summary{};
    if (!ANESummarizeLatencies(samples, &summary)) return;
    std::printf("PROFILE program=%lu metric=%s samples=%lu median_us=%.3f "
                "p95_us=%.3f min_us=%.3f max_us=%.3f mean_us=%.3f\n",
                (unsigned long)program, label, (unsigned long)samples.size(),
                summary.medianMicroseconds, summary.p95Microseconds,
                summary.minimumMicroseconds, summary.maximumMicroseconds,
                summary.meanMicroseconds);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc < 5) {
            std::fprintf(stderr, "usage: %s BUNDLE_DIR WARMUP REPEATS "
                         "MODEL_HASH...\n", argv[0]);
            return 64;
        }
        NSUInteger warmup = 0;
        NSUInteger repeats = 0;
        if (!parseCount(argv[2], &warmup) || !parseCount(argv[3], &repeats) ||
            repeats == 0)
            return 64;
        NSError *error = nil;
        ANEExecutableBundle *bundle = [ANEExecutableBundle
            bundleWithContentsOfDirectory:[NSURL fileURLWithPath:
                [NSString stringWithUTF8String:argv[1]] isDirectory:YES]
            error:&error];
        NSMutableArray<NSString *> *hashes = [NSMutableArray array];
        for (int index = 4; index < argc; ++index)
            [hashes addObject:[NSString stringWithUTF8String:argv[index]]];
        if (!bundle || hashes.count != bundle.artifacts.count) {
            std::fprintf(stderr, "bundle/hash mismatch: %s\n",
                         error.description.UTF8String ?: "count");
            return 2;
        }

        std::printf("BUNDLE programs=%lu dispatch=%s shared_surfaces=%lu\n",
                    (unsigned long)bundle.artifacts.count,
                    [bundle.dispatchPlan componentsJoinedByString:@","]
                        .UTF8String,
                    (unsigned long)bundle.sharedSurfaceIdentifiers.count);
        for (NSString *line in bundle.compositionTrace)
            std::printf("COMPOSITION %s\n", line.UTF8String);
        for (NSUInteger index = 0; index < bundle.artifacts.count; ++index) {
            ANEHWXArtifact *artifact = bundle.artifacts[index];
            std::printf("PROGRAM index=%lu operations=%s image_bytes=%lu\n",
                        (unsigned long)index,
                        artifact.operations.count
                            ? [artifact.operations componentsJoinedByString:@","]
                                  .UTF8String
                            : "(unrecorded)",
                        (unsigned long)artifact.image.length);
            for (ANEHWXBinding *binding in artifact.bindings)
                std::printf("  SURFACE program=%lu role=%s identifier=%s "
                            "logical_bytes=%lu allocation_bytes=%lu "
                            "row_stride=%lu plane_stride=%lu batch_stride=%lu "
                            "iosurface_index=%ld\n",
                            (unsigned long)index, roleName(binding.role),
                            binding.identifier.UTF8String,
                            (unsigned long)binding.logicalByteLength,
                            (unsigned long)binding.allocationByteLength,
                            (unsigned long)binding.rowStrideBytes,
                            (unsigned long)binding.planeStrideBytes,
                            (unsigned long)binding.batchStrideBytes,
                            (long)binding.ioSurfaceIndex);
        }

        ANEProvisionedRuntime *runtime = [[ANEProvisionedRuntime alloc]
            initWithBundle:bundle modelHashes:hashes qos:21 error:&error];
        NSArray<ANEIOSurfaceBuffer *> *inputs =
            [runtime createInputBuffersWithError:&error];
        NSArray<ANEIOSurfaceBuffer *> *outputs =
            [runtime createOutputBuffersWithError:&error];
        if (!runtime || !inputs || !outputs) {
            std::fprintf(stderr, "runtime setup: %s\n",
                         error.description.UTF8String ?: "unknown");
            return 3;
        }
        NSUInteger seed = 1;
        for (ANEIOSurfaceBuffer *buffer in inputs)
            fillDeterministic(buffer, seed++);
        if (![runtime loadWithError:&error]) {
            std::fprintf(stderr, "load: %s\n",
                         error.description.UTF8String ?: "unknown");
            return 4;
        }

        ANEEvaluationProfile *profile = [[ANEEvaluationProfile alloc] init];
        const char *maskText = std::getenv("ANE_PROFILE_PERF_STATS_MASK");
        profile.performanceStatisticsMask = maskText
            ? (unsigned int)std::strtoul(maskText, nullptr, 0) : 0xffffffffu;

        BOOL valid = YES;
        for (NSUInteger iteration = 0; valid && iteration < warmup; ++iteration)
            valid = [runtime evaluateInputs:inputs outputs:outputs
                                    profile:profile error:&error];
        if (!valid) {
            std::fprintf(stderr, "warmup evaluate: %s\n",
                         error.description.UTF8String ?: "unknown");
            [runtime unloadWithError:nil];
            return 5;
        }
        std::printf("PERFSTATS mask=0x%x first_hardware_ns=%s\n",
                    profile.performanceStatisticsMask,
                    [[profile.entries valueForKey:
                        @"hardwareExecutionNanoseconds"]
                        componentsJoinedByString:@","].UTF8String);

        NSUInteger programCount = bundle.dispatchPlan.count;
        std::vector<ProgramSamples> perProgram(programCount);
        std::vector<double> chainProfiled;
        std::vector<double> chainPlain;
        std::vector<double> chainSumOfSubmits;
        for (NSUInteger iteration = 0; valid && iteration < repeats;
             ++iteration) {
            @autoreleasepool {
                valid = [runtime evaluateInputs:inputs outputs:outputs
                                        profile:profile error:&error];
                if (!valid) break;
                chainProfiled.push_back(profile.totalMicroseconds);
                double submitSum = 0.0;
                for (NSUInteger position = 0;
                     position < profile.entries.count &&
                     position < programCount; ++position) {
                    ANEDispatchProfileEntry *entry = profile.entries[position];
                    ProgramSamples &samples = perProgram[position];
                    samples.wrap.push_back(entry.surfaceWrapMicroseconds);
                    samples.request.push_back(entry.requestBuildMicroseconds);
                    samples.submit.push_back(entry.submitMicroseconds);
                    double hardwareMicroseconds =
                        (double)entry.hardwareExecutionNanoseconds / 1000.0;
                    samples.hardware.push_back(hardwareMicroseconds);
                    samples.hostWait.push_back(
                        std::max(0.0, entry.submitMicroseconds -
                                      hardwareMicroseconds));
                    submitSum += entry.submitMicroseconds;
                }
                chainSumOfSubmits.push_back(submitSum);
            }
        }
        for (NSUInteger iteration = 0; valid && iteration < repeats;
             ++iteration) {
            @autoreleasepool {
                uint64_t started = mach_absolute_time();
                valid = [runtime evaluateInputs:inputs outputs:outputs
                                          error:&error];
                uint64_t stopped = mach_absolute_time();
                static mach_timebase_info_data_t timebase = {};
                if (timebase.denom == 0) mach_timebase_info(&timebase);
                chainPlain.push_back((double)(stopped - started) *
                                     (double)timebase.numer /
                                     (double)timebase.denom / 1000.0);
            }
        }
        if (!valid) {
            std::fprintf(stderr, "evaluate: %s\n",
                         error.description.UTF8String ?: "unknown");
            [runtime unloadWithError:nil];
            return 6;
        }

        for (NSUInteger position = 0; position < programCount; ++position) {
            NSUInteger artifactIndex =
                bundle.dispatchPlan[position].unsignedIntegerValue;
            ANEHWXArtifact *artifact = bundle.artifacts[artifactIndex];
            std::printf("PROFILE program=%lu operations=%s\n",
                        (unsigned long)artifactIndex,
                        artifact.operations.count
                            ? [artifact.operations componentsJoinedByString:@","]
                                  .UTF8String
                            : "(unrecorded)");
            ProgramSamples &samples = perProgram[position];
            printSummaryLine("surface_wrap", artifactIndex, samples.wrap);
            printSummaryLine("request_build", artifactIndex, samples.request);
            printSummaryLine("submit_wall", artifactIndex, samples.submit);
            printSummaryLine("hardware_execution", artifactIndex,
                             samples.hardware);
            printSummaryLine("host_wait", artifactIndex, samples.hostWait);
        }
        ANELatencySummary profiled{};
        ANELatencySummary plain{};
        ANELatencySummary submits{};
        ANESummarizeLatencies(chainProfiled, &profiled);
        ANESummarizeLatencies(chainPlain, &plain);
        ANESummarizeLatencies(chainSumOfSubmits, &submits);
        std::printf("CHAIN profiled_median_us=%.3f plain_median_us=%.3f "
                    "sum_of_submit_medians_us=%.3f repeats=%lu programs=%lu\n",
                    profiled.medianMicroseconds, plain.medianMicroseconds,
                    submits.medianMicroseconds, (unsigned long)repeats,
                    (unsigned long)programCount);
        [runtime unloadWithError:nil];
        std::printf("SUMMARY profile-program-chain valid=%d\n", valid);
        return valid ? 0 : 1;
    }
}
