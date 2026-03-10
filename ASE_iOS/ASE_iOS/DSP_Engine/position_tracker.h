#pragma once
#include <vector>
#include <deque>
#include <cmath>
#include <algorithm>
#include "dft_engine.h"

namespace ase {

class PositionTracker {
public:
    PositionTracker(const std::vector<std::vector<double>>& reference_chroma,
                    int    history_frames      = 128,
                    int    search_radius       = 200,
                    double sigma               = 4.0,
                    double confidence_threshold = 0.20,
                    double max_tempo           = 1.15,
                    double min_tempo           = 0.85);

    void reset(int start_pos = 0);

    // Returns {position, confidence}
    std::pair<int, double> process(const std::vector<double>& live_chroma);

    double progress()     const;
    bool   locked()       const;
    double match_quality() const;

private:
    // Reference data
    std::vector<std::vector<double>> ref_norm;   // pre-normalised, shape [N][12]
    int N;

    // Parameters
    int    history_len;
    int    radius;
    double sigma_base;
    double conf_thresh;
    double max_tempo;
    double min_tempo;
    double max_step;
    double min_step;

    // State
    double pos;
    int    j;
    double confidence;
    double tempo_ratio;
    double conf_ema;
    bool   locked_;
    int    lock_count;
    double last_sim;

    std::deque<std::vector<double>> live_history;
};

} // namespace ase

