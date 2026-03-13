#pragma once
/**
 * position_tracker.h
 * ──────────────────
 * Real-time song position tracker using chroma-based sequence matching.
 *
 * Given a reference chromagram (from offline analysis), this class estimates
 * the current playback position within the reference by matching a sliding
 * window of live chroma frames against the reference using FFT-accelerated
 * cross-correlation.
 *
 * Key features:
 *   - Adaptive search window: wide during acquisition, narrow once locked
 *   - Gaussian continuity prior to penalize large position jumps
 *   - EMA-smoothed confidence score for robust lock/unlock decisions
 *   - Slow tempo ratio estimation for gradual drift compensation
 *   - Graceful degradation during silence (coasts at estimated tempo)
 */

#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include "dft_engine.h"

namespace ase {

class PositionTracker {
public:
    /**
     * Construct a position tracker with pre-analyzed reference chroma.
     *
     * @param reference_chroma     2D chroma array [N_frames][12], will be L2-normalized internally
     * @param history_frames       Maximum number of live frames in the sliding template window
     * @param search_radius        Search radius (frames) around expected position when locked
     * @param sigma                Width of the Gaussian continuity prior (frames)
     * @param confidence_threshold Minimum similarity to accept a position update
     * @param max_tempo            Upper bound for tempo ratio estimation
     * @param min_tempo            Lower bound for tempo ratio estimation
     */
    PositionTracker(const std::vector<std::vector<double>>& reference_chroma,
                    int    history_frames      = 128,
                    int    search_radius       = 200,
                    double sigma               = 4.0,
                    double confidence_threshold = 0.20,
                    double max_tempo           = 1.15,
                    double min_tempo           = 0.85);

    /// Reset all tracking state. Optionally set a starting position.
    void reset(int start_pos = 0);

    /**
     * Process one live chroma frame and return the estimated position.
     *
     * @param live_chroma  12-element chroma vector from the current audio frame
     * @return             Pair of (estimated_frame_index, confidence)
     */
    std::pair<int, double> process(const std::vector<double>& live_chroma);

    /// Return normalized progress through the reference [0.0, 1.0].
    double progress()     const;

    /// Return whether the tracker has acquired a stable lock on the song position.
    bool   locked()       const;

    /// Return the raw similarity score from the most recent match.
    double match_quality() const;

private:
    // ── Reference data ──────────────────────────────────────────────
    std::vector<std::vector<double>> ref_norm;   // L2-normalized reference chroma [N][12]
    int N;                                        // Total number of reference frames

    // ── Configuration parameters ────────────────────────────────────
    int    history_len;     // Maximum template window length (frames)
    int    radius;          // Search radius when locked (frames)
    double sigma_base;      // Gaussian continuity prior base width
    double conf_thresh;     // Minimum confidence to accept a match
    double max_tempo;       // Upper tempo ratio bound
    double min_tempo;       // Lower tempo ratio bound
    double max_step;        // Maximum forward position jump per frame
    double min_step;        // Maximum backward position correction per frame

    // ── Tracking state ──────────────────────────────────────────────
    double pos;             // Current estimated position (fractional frame index)
    int    j;               // Frame counter (total frames processed)
    double confidence;      // Published confidence value [0, 1]
    double tempo_ratio;     // Estimated tempo ratio (live speed / reference speed)
    double conf_ema;        // Exponentially smoothed confidence
    bool   locked_;         // Whether the tracker is confidently locked
    int    lock_count;      // Consecutive frames with good matches
    double last_sim;        // Raw similarity from the most recent match

    /// Sliding window of recent L2-normalized live chroma frames.
    std::deque<std::vector<double>> live_history;
};

} // namespace ase