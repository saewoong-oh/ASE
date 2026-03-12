#include "position_tracker.h"
#include <numeric>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace ase {

// ----------------------------------------------------------------
// Construction
// ----------------------------------------------------------------
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
      max_step(std::max(2.0, std::ceil(max_tempo_ * 2.0))),
      min_step(-2.0),
      pos(0.0), j(0),
      confidence(0.0), tempo_ratio(1.0),
      conf_ema(0.0), locked_(false),
      lock_count(0), last_sim(0.0)
{
    // Standard L2 Normalisation
    ref_norm.resize(N, std::vector<double>(12, 0.0));
    for (int i = 0; i < N; ++i) {
        double norm = 0.0;
        for (int c = 0; c < 12; ++c)
            norm += reference_chroma[i][c] * reference_chroma[i][c];
        norm = std::sqrt(norm);
        if (norm < 1e-12) norm = 1.0;
        for (int c = 0; c < 12; ++c)
            ref_norm[i][c] = reference_chroma[i][c] / norm;
    }
}

// ----------------------------------------------------------------
// Reset
// ----------------------------------------------------------------
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

// ----------------------------------------------------------------
// Process one live chroma frame
// ----------------------------------------------------------------
std::pair<int, double> PositionTracker::process(const std::vector<double>& live_chroma) {

    // --- Normalise incoming frame ---
    double ln = 0.0;
    for (double v : live_chroma) ln += v * v;
    ln = std::sqrt(ln);

    // Silence / noise floor
    if (ln < 1e-3) {
        live_history.push_back(std::vector<double>(12, 0.0));
        if (static_cast<int>(live_history.size()) > history_len)
            live_history.pop_front();

        pos = std::max(0.0, std::min(static_cast<double>(N - 1), pos + tempo_ratio));
        confidence  *= 0.95;
        lock_count   = std::max(0, lock_count - 1);
        locked_      = lock_count >= 5;
        j++;
        return {static_cast<int>(std::round(pos)), confidence};
    }

    // Normalised live frame
    std::vector<double> lc(12);
    for (int c = 0; c < 12; ++c) lc[c] = live_chroma[c] / ln;

    live_history.push_back(lc);
    if (static_cast<int>(live_history.size()) > history_len)
        live_history.pop_front();

    int T = static_cast<int>(live_history.size());   // actual history length

    double expected = std::max(0.0, std::min(static_cast<double>(N - 1), pos + tempo_ratio));

    // --- Search window ---
    int r;
    double sig;
    
    if (T < 15) {
        r   = 2;
        sig = sigma_base * 0.5;
    } else if (!locked_) {
        r   = std::min(1500, N / 2);
        sig = sigma_base * 20.0;
    } else {
        r   = radius;
        sig = sigma_base;
    }

    int search_center_start = std::max(0, static_cast<int>(expected) - T + 1);
    int lo = std::max(0, search_center_start - r);
    int hi = std::min(N, static_cast<int>(expected) + r + 1);
    int S  = hi - lo;   // source window length

    if (S < T) {
        pos = expected;
        j++;
        return {static_cast<int>(std::round(pos)), confidence};
    }

    std::vector<double> tpl(T * 12);
    {
        int row = 0;
        for (const auto& frame : live_history) {
            for (int c = 0; c < 12; ++c)
                tpl[row * 12 + c] = frame[c];
            ++row;
        }
    }

    std::vector<double> src(S * 12);
    for (int i = 0; i < S; ++i)
        for (int c = 0; c < 12; ++c)
            src[i * 12 + c] = ref_norm[lo + i][c];

    RVec sims = chroma_sequence_match(tpl.data(), T,
                                      src.data(), S,
                                      12);

    if (sims.empty()) {
        pos = expected;
        j++;
        return {static_cast<int>(std::round(pos)), confidence};
    }

    int result_len = static_cast<int>(sims.size());

    double best_score = -1e30;
    int    best_d     = 0;
    double best_sim   = 0.0;
    double best_pos   = expected;

    for (int d = 0; d < result_len; ++d) {
        double ref_pos = static_cast<double>(lo + d + T - 1);
        double offset  = ref_pos - expected;
        double cont    = std::exp(-0.5 * (offset / sig) * (offset / sig));
        double score   = sims[d] * cont;
        if (score > best_score) {
            best_score = score;
            best_d     = d;
            best_sim   = sims[d];
            best_pos   = ref_pos;
        }
    }

    last_sim = best_sim;

    // --- EMA confidence ---
    conf_ema   = 0.75 * conf_ema + 0.25 * best_sim;
    confidence = std::max(0.0, std::min(1.0, conf_ema));

    // --- Position update ---
    if (best_sim >= conf_thresh) {
        double innovation = best_pos - expected;
        double clamped_innovation;

        if (!locked_ && T >= 15) {
            clamped_innovation = innovation;
        } else {
            clamped_innovation = std::max(min_step, std::min(max_step, innovation));
        }

        double gain = std::min(0.85, best_sim);
        if (best_sim < 0.6) {
            gain *= (best_sim / 0.6);
        }

        pos = std::max(0.0, std::min(static_cast<double>(N - 1),
                                     expected + gain * clamped_innovation));

        if (j > 15 && locked_ && best_sim > 0.65) {
            double measured_step = pos - (expected - tempo_ratio);
            double t = std::max(min_tempo, std::min(max_tempo, measured_step));
            tempo_ratio = 0.995 * tempo_ratio + 0.005 * t;
        } else {
            tempo_ratio = 0.995 * tempo_ratio + 0.005 * 1.0;
        }

        lock_count = std::min(lock_count + 1, 60);
        locked_    = lock_count >= 5;

    } else {
        pos        = expected;
        lock_count = std::max(0, lock_count - 2);
        locked_    = lock_count >= 3;
        
        tempo_ratio = 0.995 * tempo_ratio + 0.005 * 1.0;
    }

    j++;
    return {static_cast<int>(std::round(pos)), confidence};
}

// ----------------------------------------------------------------
// Accessors
// ----------------------------------------------------------------
double PositionTracker::progress() const {
    return pos / std::max(1.0, static_cast<double>(N - 1));
}

bool PositionTracker::locked() const {
    return locked_;
}

double PositionTracker::match_quality() const {
    return last_sim;
}

} // namespace ase

