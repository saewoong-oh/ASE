#pragma once

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

namespace ase {

class RMSMatcher {
public:
    RMSMatcher(const std::vector<std::string>& stem_names,
               const std::string& anchor = "drums",
               double smoothing = 0.15,
               double max_gain_db = 18.0,
               double min_gain_db = -18.0,
               double peak_limit = 0.98);

    std::map<std::string, double> compute_gains(
        const std::map<std::string, double>& ref_rms,
        const std::map<std::string, double>& live_rms,
        double confidence = 1.0);

    // Public state for your dashboard/UI
    std::map<std::string, double> gains;
    double overall_gain_db = 0.0;
    std::map<std::string, double> live_rel_db;
    std::map<std::string, double> ref_rel_db;

private:
    std::vector<std::string> names_;
    std::string anchor_;
    double alpha_;
    double max_g_;
    double min_g_;
    double limit_;
};

} // namespace ase
