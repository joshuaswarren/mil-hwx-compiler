#pragma once

#include <string>
#include <vector>

struct ANELatencySummary {
    double minimumMicroseconds;
    double medianMicroseconds;
    double p95Microseconds;
    double maximumMicroseconds;
    double meanMicroseconds;
};

bool ANESummarizeLatencies(const std::vector<double> &samples,
                           ANELatencySummary *summary);
bool ANEIsInvalidMILProgramError(const std::string &message);
