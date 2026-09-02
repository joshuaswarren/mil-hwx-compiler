#include "ANEBenchmarkStats.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static int failures = 0;

static void expect(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

static bool closeEnough(double actual, double expected) {
    return std::fabs(actual - expected) < 1.0e-12;
}

int main() {
    ANELatencySummary summary{};
    expect(!ANESummarizeLatencies({}, &summary),
           "empty sample sets must be rejected");
    expect(!ANESummarizeLatencies({1.0}, nullptr),
           "a missing summary destination must be rejected");
    expect(!ANESummarizeLatencies({1.0, -1.0}, &summary),
           "negative latency samples must be rejected");
    expect(!ANESummarizeLatencies(
               {1.0, std::numeric_limits<double>::infinity()}, &summary),
           "non-finite latency samples must be rejected");

    const std::vector<double> samples = {
        5, 1, 4, 2, 3, 6, 100, 8, 7, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    };
    expect(ANESummarizeLatencies(samples, &summary),
           "valid samples must produce a summary");
    expect(closeEnough(summary.minimumMicroseconds, 1.0),
           "minimum must use the lowest sample");
    expect(closeEnough(summary.medianMicroseconds, 10.5),
           "median must average the two middle samples");
    expect(closeEnough(summary.p95Microseconds, 19.0),
           "p95 must use nearest-rank selection");
    expect(closeEnough(summary.maximumMicroseconds, 100.0),
           "maximum must use the highest sample");
    expect(closeEnough(summary.meanMicroseconds, 14.5),
           "mean must include every sample");

    expect(ANESummarizeLatencies({9.0, 2.0, 5.0}, &summary),
           "odd sample sets must produce a summary");
    expect(closeEnough(summary.medianMicroseconds, 5.0),
           "odd median must select the middle sample");
    expect(closeEnough(summary.p95Microseconds, 9.0),
           "small-set p95 must remain in range");

    expect(ANEIsInvalidMILProgramError(
               "ANECCompile failed: err=(InvalidMILProgram)"),
           "Apple InvalidMILProgram failures must be classified explicitly");
    expect(!ANEIsInvalidMILProgramError(
               "ANECCompile failed: framework unavailable"),
           "other Apple compiler failures must not be treated as unsupported MIL");

    if (failures != 0) return 1;
    std::printf("benchmark statistics: PASS\n");
    return 0;
}
