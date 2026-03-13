#pragma once
/**
 * rms_matcher.h
 * ─────────────
 * Per-stem gain advisory calculator for real-time mix matching.
 *
 * Given the reference per-stem RMS levels at the current song position and
 * the estimated live per-stem RMS levels (from frequency-band filtering),
 * computes gain multipliers that would bring the live mix closer to the
 * reference balance.
 *
 * Uses an anchor stem (typically drums) as the absolute level reference.
 * All other stems are adjusted relative to the anchor. Gains are smoothed
 * with an EMA filter and blended with confidence to avoid wild swings
 * during uncertain tracking periods.
 */

#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

namespace ase {

class RMSMatcher {
public:
    /**
     * @param stem_names    Ordered list of stem names to track
     * @param anchor        Name of the anchor stem (used as the level reference)
     * @param smoothing     EMA smoothing coefficient (0 = instant, 1 = frozen)
     * @param max_gain_db   Maximum gain advisory in dB (positive = boost)
     * @param min_gain_db   Minimum gain advisory in dB (negative = cut)
     * @param peak_limit    Peak limiter threshold (currently unused, reserved)
     */
    RMSMatcher(const std::vector<std::string>& stem_names,
               const std::string& anchor = "drums",
               double smoothing = 0.15,
               double max_gain_db = 18.0,
               double min_gain_db = -18.0,
               double peak_limit = 0.98);

    /**
     * Compute per-stem gain advisories from reference and live RMS values.
     *
     * @param ref_rms     Reference per-stem RMS at the current estimated position
     * @param live_rms    Live per-stem RMS estimated from the current audio frame
     * @param confidence  Tracker confidence [0, 1]; low confidence biases gains toward unity
     * @return            Map of stem name → gain multiplier (linear scale)
     */
    std::map<std::string, double> compute_gains(
        const std::map<std::string, double>& ref_rms,
        const std::map<std::string, double>& live_rms,
        double confidence = 1.0);

    // ── Public state (read by the UI layer) ─────────────────────────

    /// Per-stem gain multipliers (linear scale). 1.0 = no change.
    std::map<std::string, double> gains;

    /// Overall level difference in dB between reference and live anchor stems.
    double overall_gain_db = 0.0;

    /// Per-stem live level relative to the live anchor, in dB.
    std::map<std::string, double> live_rel_db;

    /// Per-stem reference level relative to the reference anchor, in dB.
    std::map<std::string, double> ref_rel_db;

private:
    std::vector<std::string> names_;   // Ordered stem names
    std::string anchor_;                // Anchor stem name
    double alpha_;                      // EMA smoothing coefficient
    double max_g_;                      // Maximum gain (linear scale)
    double min_g_;                      // Minimum gain (linear scale)
    double limit_;                      // Peak limiter threshold (reserved)
};

} // namespace ase