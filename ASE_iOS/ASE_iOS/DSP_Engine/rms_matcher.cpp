#include "rms_matcher.h"

namespace ase {

RMSMatcher::RMSMatcher(const std::vector<std::string>& stem_names,
                       const std::string& anchor,
                       double smoothing, double max_gain_db,
                       double min_gain_db, double peak_limit)
    : names_(stem_names), alpha_(smoothing), limit_(peak_limit) {
    
    // Verify that the requested anchor stem exists; fall back to the first stem if not
    if (std::find(names_.begin(), names_.end(), anchor) != names_.end()) {
        anchor_ = anchor;
    } else if (!names_.empty()) {
        anchor_ = names_[0];
    }

    // Convert dB gain limits to linear scale for clamping
    max_g_ = std::pow(10.0, max_gain_db / 20.0);
    min_g_ = std::pow(10.0, min_gain_db / 20.0);

    // Initialize all gains to unity (no adjustment) and all dB readings to 0
    for (const auto& name : names_) {
        gains[name] = 1.0;
        live_rel_db[name] = 0.0;
        ref_rel_db[name] = 0.0;
    }
}

/**
 * Compute per-stem gain advisories.
 *
 * Algorithm:
 *   1. Determine anchor levels in both reference and live (floor at -200 dB)
 *   2. Compute overall level offset (anchor ref / anchor live)
 *   3. For each non-anchor stem:
 *      a. Compute its ratio relative to the anchor in both ref and live
 *      b. Target gain = ref_ratio / live_ratio (what would fix the balance)
 *      c. Clamp to [min_gain, max_gain] to prevent extreme corrections
 *      d. Blend with unity using confidence (low confidence → no adjustment)
 *      e. Apply EMA smoothing against the previous gain value
 *   4. Force the anchor stem's gain to exactly 1.0 (it's the reference)
 */
std::map<std::string, double> RMSMatcher::compute_gains(
    const std::map<std::string, double>& ref_rms,
    const std::map<std::string, double>& live_rms,
    double confidence) {

    // Get anchor RMS values with a floor to prevent division by zero
    double ref_a = 1e-10;
    if (ref_rms.count(anchor_)) ref_a = std::max(ref_rms.at(anchor_), 1e-10);

    double live_a = 1e-10;
    if (live_rms.count(anchor_)) live_a = std::max(live_rms.at(anchor_), 1e-10);

    // Overall level: raw ratio between reference and live anchor levels (in dB)
    double raw_anchor_gain = ref_a / live_a;
    overall_gain_db = 20.0 * std::log10(std::max(raw_anchor_gain, 1e-10));

    for (const auto& name : names_) {
        // Get this stem's RMS in both reference and live
        double r = 0.0;
        if (ref_rms.count(name)) r = ref_rms.at(name);
        
        double l = 0.0;
        if (live_rms.count(name)) l = live_rms.at(name);

        // Compute relative dB levels (each stem relative to its own anchor)
        ref_rel_db[name] = 20.0 * std::log10(std::max(r, 1e-10) / ref_a);
        live_rel_db[name] = 20.0 * std::log10(std::max(l, 1e-10) / live_a);

        // Compute target gain for non-anchor stems
        double target = 1.0;
        if (name != anchor_) {
            // Ratio of this stem to anchor in the reference mix
            double ref_ratio = r / ref_a;
            // Ratio of this stem to anchor in the live mix (floored)
            double live_ratio = std::max(l / live_a, 1e-10);
            // The gain needed to make the live ratio match the reference ratio
            target = ref_ratio / live_ratio;
        }

        // Clamp to the configured gain range to prevent extreme adjustments
        target = std::max(min_g_, std::min(max_g_, target));

        // Confidence blending: when confidence is low, bias toward unity (no change).
        // This prevents the mixer from making wild adjustments when the tracker
        // isn't sure where we are in the song.
        target = confidence * target + (1.0 - confidence) * 1.0;

        // EMA smoothing: prevents jarring frame-to-frame gain changes
        double prev = gains[name];
        gains[name] = alpha_ * prev + (1.0 - alpha_) * target;
    }

    // The anchor stem is always exactly 1.0 — it's the absolute reference
    gains[anchor_] = 1.0;

    return gains;
}

} // namespace ase