#include "ANEBenchmarkStats.h"

#include <algorithm>
#include <cmath>
#include <numeric>

bool ANESummarizeLatencies(const std::vector<double> &samples,
                           ANELatencySummary *summary) {
    if (!summary || samples.empty()) return false;
    for (double sample : samples)
        if (!std::isfinite(sample) || sample < 0.0) return false;

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());
    const size_t count = sorted.size();
    const size_t middle = count / 2;
    const double median = count % 2 == 0
        ? (sorted[middle - 1] + sorted[middle]) * 0.5
        : sorted[middle];
    const size_t p95Index =
        static_cast<size_t>(std::ceil(0.95 * static_cast<double>(count))) - 1;

    summary->minimumMicroseconds = sorted.front();
    summary->medianMicroseconds = median;
    summary->p95Microseconds = sorted[p95Index];
    summary->maximumMicroseconds = sorted.back();
    summary->meanMicroseconds =
        std::accumulate(sorted.begin(), sorted.end(), 0.0) /
        static_cast<double>(count);
    return true;
}
