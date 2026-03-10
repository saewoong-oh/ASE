#include "rms_matcher.h"

namespace ase {

RMSMatcher::RMSMatcher(const std::vector<std::string>& stem_names,
                       const std::string& anchor,
                       double smoothing, double max_gain_db,
                       double min_gain_db, double peak_limit)
    : names_(stem_names), alpha_(smoothing), limit_(peak_limit) {
    
    // Verify anchor exists in names, otherwise default to first stem
    if (std::find(names_.begin(), names_.end(), anchor) != names_.end()) {
        anchor_ = anchor;
    } else if (!names_.empty()) {
        anchor_ = names_[0];
    }

    max_g_ = std::pow(10.0, max_gain_db / 20.0);
    min_g_ = std::pow(10.0, min_gain_db / 20.0);

    for (const auto& name : names_) {
        gains[name] = 1.0;
        live_rel_db[name] = 0.0;
        ref_rel_db[name] = 0.0;
    }
}

std::map<std::string, double> RMSMatcher::compute_gains(
    const std::map<std::string, double>& ref_rms,
    const std::map<std::string, double>& live_rms,
    double confidence) {

    double ref_a = 1e-10;
    if (ref_rms.count(anchor_)) ref_a = std::max(ref_rms.at(anchor_), 1e-10);

    double live_a = 1e-10;
    if (live_rms.count(anchor_)) live_a = std::max(live_rms.at(anchor_), 1e-10);

    // Overall level (raw anchor gain)
    double raw_anchor_gain = ref_a / live_a;
    overall_gain_db = 20.0 * std::log10(std::max(raw_anchor_gain, 1e-10));

    for (const auto& name : names_) {
        double r = 0.0;
        if (ref_rms.count(name)) r = ref_rms.at(name);
        
        double l = 0.0;
        if (live_rms.count(name)) l = live_rms.at(name);

        ref_rel_db[name] = 20.0 * std::log10(std::max(r, 1e-10) / ref_a);
        live_rel_db[name] = 20.0 * std::log10(std::max(l, 1e-10) / live_a);

        double target = 1.0;
        if (name != anchor_) {
            double ref_ratio = r / ref_a;
            double live_ratio = std::max(l / live_a, 1e-10);
            target = ref_ratio / live_ratio;
        }

        // Clamp
        target = std::max(min_g_, std::min(max_g_, target));

        // Blend with confidence
        target = confidence * target + (1.0 - confidence) * 1.0;

        // EMA smooth
        double prev = gains[name];
        gains[name] = alpha_ * prev + (1.0 - alpha_) * target;
    }

    // Force anchor to exactly 1.0
    gains[anchor_] = 1.0;

    return gains;
}

} // namespace ase
