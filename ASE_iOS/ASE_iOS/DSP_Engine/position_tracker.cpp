#include "position_tracker.h"
#include <numeric>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace ase {

// ────────────────────────────────────────────────────────────────────
// Construction
// ────────────────────────────────────────────────────────────────────

PositionTracker::PositionTracker(const std::vector<std::vector<double>>& reference_chroma,
                                 int    history_frames,
                                 int    search_radius,
                                 double sigma,
                                 double confidence_threshold,
                                 double max_tempo_,
                                 double min_tempo_)
    : N(static_cast<int>(reference_chroma.size())),
      history_len(history_frames),
      radius(search_radius),
      sigma_base(sigma),
      conf_thresh(confidence_threshold),
      max_tempo(max_tempo_),
      min_tempo(min_tempo_),
      // Max step is derived from tempo bounds to prevent unrealistic jumps
      max_step(std::max(2.0, std::ceil(max_tempo_ * 2.0))),
      min_step(-2.0),
      pos(0.0), j(0),
      confidence(0.0), tempo_ratio(1.0),
      conf_ema(0.0), locked_(false),
      lock_count(0), last_sim(0.0)
{
    // L2-normalize each reference chroma frame so that dot products yield cosine similarity.
    // This makes matching invariant to overall volume differences between reference frames.
    ref_norm.resize(N, std::vector<double>(12, 0.0));
    for (int i = 0; i < N; ++i) {
        double norm = 0.0;
        for (int c = 0; c < 12; ++c)
            norm += reference_chroma[i][c] * reference_chroma[i][c];
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1.0;  // Avoid division by zero for silent frames
        for (int c = 0; c < 12; ++c)
            ref_norm[i][c] = reference_chroma[i][c] / norm;
    }
}

// ────────────────────────────────────────────────────────────────────
// Reset
// ────────────────────────────────────────────────────────────────────

/// Reset all internal state to begin tracking from a given position.
void PositionTracker::reset(int start_pos) {
    pos        = static_cast<double>(start_pos);
    j          = 0;
    confidence = 0.0;
    tempo_ratio = 1.0;
    conf_ema   = 0.0;
    locked_    = false;
    lock_count = 0;
    last_sim   = 0.0;
    live_history.clear();
}

// ────────────────────────────────────────────────────────────────────
// Process one live chroma frame
// ────────────────────────────────────────────────────────────────────

/**
 * Core tracking algorithm. For each incoming chroma frame:
 *
 *   1. Normalize the frame and append to the sliding history window
 *   2. Predict the expected position based on current tempo estimate
 *   3. Choose search window size based on lock state:
 *      - Very early (< 15 frames): minimal search, building history
 *      - Unlocked: wide search to find the initial position
 *      - Locked: narrow search around the expected position
 *   4. Run FFT cross-correlation between the live history template
 *      and the reference chroma within the search window
 *   5. Score each candidate: similarity × Gaussian continuity prior
 *   6. Update position with gain proportional to match quality
 *   7. Slowly adapt the tempo ratio estimate when confidently locked
 *   8. Update lock state based on consecutive good matches
 */
std::pair<int, double> PositionTracker::process(const std::vector<double>& live_chroma) {

    // --- Normalize incoming chroma frame ---
    double ln = 0.0;
    for (double v : live_chroma) ln += v * v;
    ln = std::sqrt(ln);

    // Handle silence: coast forward at current tempo, decay confidence
    if (ln < 1e-3) {
        live_history.push_back(std::vector<double>(12, 0.0));
        if (static_cast<int>(live_history.size()) > history_len)
            live_history.pop_front();

        pos = std::max(0.0, std::min(static_cast<double>(N - 1), pos + tempo_ratio));
        confidence  *= 0.95;        // Gradual confidence decay during silence
        lock_count   = std::max(0, lock_count - 1);
        locked_      = lock_count >= 5;
        j++;
        return {static_cast<int>(std::round(pos)), confidence};
    }

    // L2-normalize the incoming live chroma frame
    std::vector<double> lc(12);
    for (int c = 0; c < 12; ++c) lc[c] = live_chroma[c] / ln;

    // Maintain the sliding window of recent live frames
    live_history.push_back(lc);
    if (static_cast<int>(live_history.size()) > history_len)
        live_history.pop_front();

    int T = static_cast<int>(live_history.size());   // Current template length

    // Predicted next position based on current tempo estimate
    double expected = std::max(0.0, std::min(static_cast<double>(N - 1), pos + tempo_ratio));

    // --- Adaptive search window configuration ---
    int r;
    double sig;
    
    if (T < 15) {
        // Very early: barely any history, use minimal search to avoid false locks
        r   = 2;
        sig = sigma_base * 0.5;
    } else if (!locked_) {
        // Unlocked: wide search to find the correct position anywhere in the song
        r   = std::min(1500, N / 2);
        sig = sigma_base * 20.0;      // Very wide Gaussian: don't penalize distant matches
    } else {
        // Locked: tight search around the expected position
        r   = radius;
        sig = sigma_base;
    }

    // Compute the reference index range to search within
    int search_center_start = std::max(0, static_cast<int>(expected) - T + 1);
    int lo = std::max(0, search_center_start - r);
    int hi = std::min(N, static_cast<int>(expected) + r + 1);
    int S  = hi - lo;   // Total source frames in the search window

    // If the search window is smaller than the template, skip matching
    if (S < T) {
        pos = expected;
        j++;
        return {static_cast<int>(std::round(pos)), confidence};
    }

    // --- Build template and source matrices for cross-correlation ---
    
    // Flatten the live history deque into a contiguous row-major array [T × 12]
    std::vector<double> tpl(T * 12);
    {
        int row = 0;
        for (const auto& frame : live_history) {
            for (int c = 0; c < 12; ++c)
                tpl[row * 12 + c] = frame[c];
            ++row;
        }
    }

    // Extract the search window from the reference chroma [S × 12]
    std::vector<double> src(S * 12);
    for (int i = 0; i < S; ++i)
        for (int c = 0; c < 12; ++c)
            src[i * 12 + c] = ref_norm[lo + i][c];

    // --- FFT cross-correlation ---
    // Returns normalized similarity at each valid alignment offset
    RVec sims = chroma_sequence_match(tpl.data(), T,
                                      src.data(), S,
                                      12);

    if (sims.empty()) {
        pos = expected;
        j++;
        return {static_cast<int>(std::round(pos)), confidence};
    }

    int result_len = static_cast<int>(sims.size());

    // --- Find the best match: similarity × Gaussian continuity prior ---
    double best_score = -1e30;
    int    best_d     = 0;
    double best_sim   = 0.0;
    double best_pos   = expected;

    for (int d = 0; d < result_len; ++d) {
        // The reference position this alignment corresponds to (end of the template window)
        double ref_pos = static_cast<double>(lo + d + T - 1);
        // Deviation from the expected position
        double offset  = ref_pos - expected;
        // Gaussian continuity prior: penalizes large deviations from expected position
        double cont    = std::exp(-0.5 * (offset / sig) * (offset / sig));
        // Combined score: similarity weighted by positional plausibility
        double score   = sims[d] * cont;
        if (score > best_score) {
            best_score = score;
            best_d     = d;
            best_sim   = sims[d];
            best_pos   = ref_pos;
        }
    }

    last_sim = best_sim;

    // --- Update confidence using exponential moving average ---
    conf_ema   = 0.75 * conf_ema + 0.25 * best_sim;
    confidence = std::max(0.0, std::min(1.0, conf_ema));

    // --- Position update, gated by confidence ---
    if (best_sim >= conf_thresh) {
        double innovation = best_pos - expected;
        double clamped_innovation;

        if (!locked_ && T >= 15) {
            // Unlocked with enough history: allow large jumps for initial acquisition
            clamped_innovation = innovation;
        } else {
            // Locked: clamp the innovation to prevent wild jumps from spurious matches
            clamped_innovation = std::max(min_step, std::min(max_step, innovation));
        }

        // Adaptive Kalman-like gain: higher similarity → stronger position correction
        double gain = std::min(0.85, best_sim);
        if (best_sim < 0.6) {
            // Further attenuate the correction when match quality is marginal
            gain *= (best_sim / 0.6);
        }

        // Apply the weighted correction to the expected position
        pos = std::max(0.0, std::min(static_cast<double>(N - 1),
                                     expected + gain * clamped_innovation));

        // Tempo estimation: slowly adapt when confidently locked
        if (j > 15 && locked_ && best_sim > 0.65) {
            // Measure the effective step size from the position update
            double measured_step = pos - (expected - tempo_ratio);
            double t = std::max(min_tempo, std::min(max_tempo, measured_step));
            // Very slow IIR filter (0.5% per frame) to avoid tempo oscillation
            tempo_ratio = 0.995 * tempo_ratio + 0.005 * t;
        } else {
            // Not confident enough: slowly drift tempo back toward 1.0 (no drift)
            tempo_ratio = 0.995 * tempo_ratio + 0.005 * 1.0;
        }

        // Increment lock counter (capped at 60 to limit state accumulation)
        lock_count = std::min(lock_count + 1, 60);
        locked_    = lock_count >= 5;

    } else {
        // Poor match: coast at expected position, decay lock state
        pos        = expected;
        lock_count = std::max(0, lock_count - 2);  // Decay faster than we grow
        locked_    = lock_count >= 3;
        
        // Drift tempo back toward unity during uncertain periods
        tempo_ratio = 0.995 * tempo_ratio + 0.005 * 1.0;
    }

    j++;
    return {static_cast<int>(std::round(pos)), confidence};
}

// ────────────────────────────────────────────────────────────────────
// Accessors
// ────────────────────────────────────────────────────────────────────

/// Return normalized progress [0.0, 1.0] through the reference song.
double PositionTracker::progress() const {
    return pos / std::max(1.0, static_cast<double>(N - 1));
}

/// Return whether the tracker has acquired a stable position lock.
bool PositionTracker::locked() const {
    return locked_;
}

/// Return the raw cosine similarity from the most recent match attempt.
double PositionTracker::match_quality() const {
    return last_sim;
}

} // namespace ase