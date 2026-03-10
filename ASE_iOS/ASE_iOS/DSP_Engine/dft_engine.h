#pragma once

#include <vector>
#include <complex>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <cstring>
#include <deque>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace ase {

using Complex  = std::complex<double>;
using CVec     = std::vector<Complex>;
using RVec     = std::vector<double>;
using Chroma   = std::array<double, 12>;

enum class Window { HANN, HAMMING, BLACKMAN_HARRIS, RECTANGULAR };

struct Peak {
    double freq;
    double amp;
    double phase;
    int    bin;
};

struct Partial {
    int                 id;
    std::vector<double> times;
    std::vector<double> freqs;
    std::vector<double> amps;
    std::vector<double> phases;
    bool                active = true;
    int                 gap    = 0;
};

struct Note {
    double start;
    double end;
    double freq;
    double amp;
    int    midi;
};

RVec chroma_sequence_match(
    const double* tpl_data, int T,
    const double* src_data, int S,
    int n_chroma = 12);

RVec make_window(size_t n, Window type);
size_t next_pow2(size_t n);
void fft_inplace(CVec& x);
void ifft_inplace(CVec& x);
CVec fft_real(const double* data, size_t len, size_t fft_size);
double compute_rms(const double* data, size_t n);
int    freq_to_midi(double f);
double midi_to_freq(int m);
std::string midi_to_name(int m);

class STFT {
public:
    struct Frame {
        CVec   spectrum;
        RVec   magnitude;
        RVec   mag_db;
        RVec   phase;
        double time;
    };

    STFT(int fft_size, int hop_size, int sample_rate, Window win = Window::HANN);
    std::vector<Frame> analyze(const RVec& signal) const;
    Frame analyze_frame(const double* samples, int count, double time) const;
    std::vector<Peak> detect_peaks(const Frame& f, double threshold_db = -60.0) const;
    RVec synthesize(const std::vector<Frame>& frames) const;

    int    fft_size()    const { return n_; }
    int    hop_size()    const { return hop_; }
    int    sample_rate() const { return sr_; }
    double freq_res()    const { return double(sr_) / n_; }

private:
    int  n_, hop_, sr_;
    RVec win_;
    Peak refine_peak(const RVec& mag_db, const RVec& ph, int bin) const;
};

class PartialTracker {
public:
    PartialTracker(double tolerance_cents = 50.0, double min_partial_dur = 0.03, int max_gap_frames = 3);
    void feed(const std::vector<Peak>& peaks, double time);
    void finish();
    std::vector<Partial> completed() const { return completed_; }
    std::vector<Partial> all() const;
    static std::vector<Note> extract_notes(const std::vector<Partial>& partials, double min_note_dur = 0.05, double pitch_gate_cents = 80.0);
    std::vector<double> active_chroma() const;

private:
    double tol_, min_dur_;
    int    max_gap_, next_id_ = 0;
    std::vector<Partial> active_, completed_;

    static double cents_dist(double f1, double f2) {
        if (f1 <= 0 || f2 <= 0) return 1e9;
        return std::abs(1200.0 * std::log2(f1 / f2));
    }
};

class ChromaExtractor {
public:
    ChromaExtractor(int fft_size, int hop_size, int sample_rate, double tuning_ref = 440.0);
    std::vector<Chroma> analyze(const RVec& signal) const;
    Chroma analyze_frame(const RVec& magnitude) const;
private:
    int n_, hop_, sr_;
    double ref_;
    STFT stft_;
    std::vector<int> bin_chroma_;
    void build_mapping();
};

class HPSS {
public:
    HPSS(int fft_size, int hop_size, int sr, int time_kernel = 17, int freq_kernel = 17, double mask_power = 2.0, Window win = Window::HANN);
    bool feed(const double* samples, int count, double time);
    const RVec& harmonic_magnitude() const { return h_mag_; }
    const CVec& harmonic_spectrum()  const { return h_spec_; }
    const CVec& percussive_spectrum() const { return p_spec_; }
    int latency_frames() const { return t_half_; }
    double latency_seconds() const { return t_half_ * static_cast<double>(hop_) / sr_; }
    static std::pair<RVec, RVec> separate_signal(const RVec& signal, int fft_size, int hop_size, int sr, int time_kernel = 17, int freq_kernel = 17, double power = 2.0);

private:
    STFT stft_;
    int  hop_, sr_, t_kern_, f_kern_, t_half_, f_half_;
    double power_;
    std::deque<STFT::Frame> buf_;
    bool ready_ = false;
    RVec h_mag_;
    CVec h_spec_, p_spec_;
    void compute_masks();
    static double median_val(std::vector<double>& v);
};

} // namespace ase
