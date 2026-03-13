#pragma once
/**
 * dft_engine.h
 * ────────────
 * Core DSP primitives for the ASE (Audio Spectral Engine).
 *
 * Provides:
 *   - FFT / IFFT (radix-2 Cooley-Tukey, in-place)
 *   - Windowed STFT analysis and overlap-add synthesis
 *   - Spectral peak detection with parabolic interpolation
 *   - Sinusoidal partial tracking across frames
 *   - Note extraction from tracked partials
 *   - Chromagram extraction (12-bin pitch class energy)
 *   - HPSS (Harmonic-Percussive Source Separation) via median filtering
 *   - FFT-accelerated chroma sequence cross-correlation for position matching
 *
 * All processing uses double-precision floating point.
 */

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

// Convenience type aliases
using Complex  = std::complex<double>;
using CVec     = std::vector<Complex>;
using RVec     = std::vector<double>;
using Chroma   = std::array<double, 12>;  // 12 pitch classes: C, C#, D, ..., B

/// Supported window functions for STFT analysis.
enum class Window { HANN, HAMMING, BLACKMAN_HARRIS, RECTANGULAR };

/// A single spectral peak detected in one STFT frame.
struct Peak {
    double freq;    // Frequency in Hz (parabolic-interpolated)
    double amp;     // Linear amplitude
    double phase;   // Phase in radians
    int    bin;     // Nearest FFT bin index
};

/// A tracked sinusoidal partial — a series of peaks linked across consecutive frames.
struct Partial {
    int                 id;       // Unique identifier
    std::vector<double> times;    // Timestamps of each contributing frame
    std::vector<double> freqs;    // Instantaneous frequency at each frame
    std::vector<double> amps;     // Instantaneous amplitude at each frame
    std::vector<double> phases;   // Instantaneous phase at each frame
    bool                active = true;  // Whether this partial is still being tracked
    int                 gap    = 0;     // Number of consecutive frames without a matching peak
};

/// A musical note extracted from a group of stable partial segments.
struct Note {
    double start;   // Onset time in seconds
    double end;     // Offset time in seconds
    double freq;    // Average frequency in Hz
    double amp;     // Average linear amplitude
    int    midi;    // MIDI note number (69 = A4 = 440 Hz)
};

/**
 * FFT-accelerated sliding cross-correlation between a chroma template and source.
 *
 * Computes normalized cosine similarity at every valid alignment offset,
 * used by the PositionTracker for efficient song-position matching.
 *
 * @param tpl_data  Template chroma data, row-major [T × n_chroma]
 * @param T         Number of template frames
 * @param src_data  Source (reference) chroma data, row-major [S × n_chroma]
 * @param S         Number of source frames
 * @param n_chroma  Number of chroma bins (default 12)
 * @return          Vector of similarity scores, length (S - T + 1)
 */
RVec chroma_sequence_match(
    const double* tpl_data, int T,
    const double* src_data, int S,
    int n_chroma = 12);

// ── Utility functions ───────────────────────────────────────────────

/// Generate a window function of the specified type and length.
RVec make_window(size_t n, Window type);

/// Return the smallest power of 2 >= n.
size_t next_pow2(size_t n);

/// In-place radix-2 FFT (Cooley-Tukey decimation-in-time).
void fft_inplace(CVec& x);

/// In-place inverse FFT (conjugate trick + forward FFT).
void ifft_inplace(CVec& x);

/// Compute the FFT of real-valued data, zero-padded to fft_size.
CVec fft_real(const double* data, size_t len, size_t fft_size);

/// Compute RMS (root-mean-square) energy of a signal buffer.
double compute_rms(const double* data, size_t n);

/// Convert a frequency in Hz to the nearest MIDI note number.
int    freq_to_midi(double f);

/// Convert a MIDI note number to its frequency in Hz.
double midi_to_freq(int m);

/// Convert a MIDI note number to its name string (e.g., "C4", "A#3").
std::string midi_to_name(int m);


// ── STFT (Short-Time Fourier Transform) ─────────────────────────────

/**
 * Windowed STFT analyzer and synthesizer.
 *
 * Splits a signal into overlapping frames, applies a window function,
 * computes the FFT of each frame, and provides magnitude/phase/dB spectra.
 * Also supports overlap-add resynthesis and parabolic peak detection.
 */
class STFT {
public:
    /// One analyzed STFT frame containing spectral data and timestamp.
    struct Frame {
        CVec   spectrum;     // Full complex spectrum (length = fft_size)
        RVec   magnitude;    // Magnitude of positive frequencies (length = fft_size/2 + 1)
        RVec   mag_db;       // Magnitude in decibels
        RVec   phase;        // Phase of positive frequencies (radians)
        double time;         // Center time of this frame (seconds)
    };

    /// Construct an STFT analyzer with specified FFT size, hop size, sample rate, and window.
    STFT(int fft_size, int hop_size, int sample_rate, Window win = Window::HANN);

    /// Analyze an entire signal, returning a vector of Frames.
    std::vector<Frame> analyze(const RVec& signal) const;

    /// Analyze a single frame from a raw sample buffer at a given timestamp.
    Frame analyze_frame(const double* samples, int count, double time) const;

    /// Detect spectral peaks above a dB threshold, sorted by amplitude (descending).
    /// Uses parabolic interpolation for sub-bin frequency accuracy.
    std::vector<Peak> detect_peaks(const Frame& f, double threshold_db = -60.0) const;

    /// Resynthesize a time-domain signal from a sequence of STFT frames using overlap-add.
    RVec synthesize(const std::vector<Frame>& frames) const;

    // Accessors
    int    fft_size()    const { return n_; }
    int    hop_size()    const { return hop_; }
    int    sample_rate() const { return sr_; }
    double freq_res()    const { return double(sr_) / n_; }  // Frequency resolution (Hz per bin)

private:
    int  n_, hop_, sr_;
    RVec win_;  // Precomputed window coefficients

    /// Refine a peak's frequency and amplitude using parabolic interpolation
    /// on the dB magnitude spectrum around the given bin.
    Peak refine_peak(const RVec& mag_db, const RVec& ph, int bin) const;
};


// ── Partial Tracker ─────────────────────────────────────────────────

/**
 * Sinusoidal partial tracker — links spectral peaks across consecutive frames
 * into continuous frequency tracks (partials).
 *
 * Uses a greedy nearest-neighbor approach with a frequency tolerance in cents.
 * Partials that lose their peak for too many frames are finalized. Short
 * partials below the minimum duration are discarded as noise.
 */
class PartialTracker {
public:
    /// @param tolerance_cents   Maximum frequency deviation to continue a partial (in cents)
    /// @param min_partial_dur   Minimum duration for a partial to be kept (seconds)
    /// @param max_gap_frames    Maximum consecutive missed frames before a partial dies
    PartialTracker(double tolerance_cents = 50.0, double min_partial_dur = 0.03, int max_gap_frames = 3);

    /// Feed one frame's peaks into the tracker at the given timestamp.
    void feed(const std::vector<Peak>& peaks, double time);

    /// Finalize all remaining active partials (call after processing the last frame).
    void finish();

    /// Return all completed (finalized) partials.
    std::vector<Partial> completed() const { return completed_; }

    /// Return all partials (both active and completed).
    std::vector<Partial> all() const;

    /// Extract musical notes from a set of partials by grouping stable-pitch segments.
    /// @param min_note_dur      Minimum note duration to keep (seconds)
    /// @param pitch_gate_cents  Maximum pitch variation within a single note (cents)
    static std::vector<Note> extract_notes(const std::vector<Partial>& partials, double min_note_dur = 0.05, double pitch_gate_cents = 80.0);

    /// Return a 12-bin chroma vector from currently active partials.
    std::vector<double> active_chroma() const;

private:
    double tol_, min_dur_;
    int    max_gap_, next_id_ = 0;
    std::vector<Partial> active_, completed_;

    /// Compute the absolute distance in cents between two frequencies.
    static double cents_dist(double f1, double f2) {
        if (f1 <= 0 || f2 <= 0) return 1e9;
        return std::abs(1200.0 * std::log2(f1 / f2));
    }
};


// ── Chroma Extractor ────────────────────────────────────────────────

/**
 * Extracts 12-bin chromagram (pitch class energy distribution) from audio.
 *
 * Maps each FFT bin to one of the 12 pitch classes based on its frequency,
 * accumulates energy per pitch class, and L2-normalizes the result.
 * Only bins in the musically relevant range (27.5 Hz to 4186 Hz) are included.
 */
class ChromaExtractor {
public:
    /// @param fft_size     FFT window size
    /// @param hop_size     Hop size between frames
    /// @param sample_rate  Audio sample rate
    /// @param tuning_ref   Reference tuning frequency for A4 (default 440 Hz)
    ChromaExtractor(int fft_size, int hop_size, int sample_rate, double tuning_ref = 440.0);

    /// Analyze a complete signal and return one chroma vector per frame.
    std::vector<Chroma> analyze(const RVec& signal) const;

    /// Analyze a single pre-computed magnitude spectrum and return its chroma vector.
    Chroma analyze_frame(const RVec& magnitude) const;

private:
    int n_, hop_, sr_;
    double ref_;          // Tuning reference frequency (Hz)
    STFT stft_;           // Internal STFT for full-signal analysis
    std::vector<int> bin_chroma_;  // Maps each FFT bin index to a chroma index (0–11), or -1

    /// Build the FFT bin → chroma bin mapping table.
    void build_mapping();
};


// ── HPSS (Harmonic-Percussive Source Separation) ────────────────────

/**
 * Real-time Harmonic-Percussive Source Separation using median filtering.
 *
 * Separates a spectrogram into harmonic (sustained tones) and percussive
 * (transient) components by comparing time-direction and frequency-direction
 * medians of the magnitude spectrum. Operates in a streaming fashion with
 * a latency of (time_kernel / 2) frames.
 *
 * Based on: Fitzgerald, D. (2010). "Harmonic/Percussive Separation using
 * Median Filtering."
 */
class HPSS {
public:
    /// @param fft_size     FFT window size
    /// @param hop_size     Hop size between frames
    /// @param sr           Sample rate
    /// @param time_kernel  Median filter kernel size along time axis (must be odd)
    /// @param freq_kernel  Median filter kernel size along frequency axis (must be odd)
    /// @param mask_power   Exponent for soft masking (higher = harder masks)
    /// @param win          Window function type
    HPSS(int fft_size, int hop_size, int sr, int time_kernel = 17, int freq_kernel = 17, double mask_power = 2.0, Window win = Window::HANN);

    /// Feed one frame of audio. Returns true when output is available (after initial latency).
    bool feed(const double* samples, int count, double time);

    /// Get the harmonic magnitude spectrum of the most recently completed frame.
    const RVec& harmonic_magnitude() const { return h_mag_; }

    /// Get the harmonic complex spectrum (for resynthesis).
    const CVec& harmonic_spectrum()  const { return h_spec_; }

    /// Get the percussive complex spectrum (for resynthesis).
    const CVec& percussive_spectrum() const { return p_spec_; }

    /// Latency in frames introduced by the time-direction median filter.
    int latency_frames() const { return t_half_; }

    /// Latency in seconds.
    double latency_seconds() const { return t_half_ * static_cast<double>(hop_) / sr_; }

    /// Offline convenience: separate an entire signal into harmonic and percussive components.
    /// Returns a pair of (harmonic_signal, percussive_signal).
    static std::pair<RVec, RVec> separate_signal(const RVec& signal, int fft_size, int hop_size, int sr, int time_kernel = 17, int freq_kernel = 17, double power = 2.0);

private:
    STFT stft_;
    int  hop_, sr_, t_kern_, f_kern_, t_half_, f_half_;
    double power_;                      // Soft mask exponent
    std::deque<STFT::Frame> buf_;       // Sliding buffer of recent STFT frames
    bool ready_ = false;                // True once enough frames have accumulated
    RVec h_mag_;                        // Harmonic magnitude output
    CVec h_spec_, p_spec_;              // Harmonic and percussive complex spectra

    /// Compute harmonic/percussive masks from the buffered spectrogram.
    void compute_masks();

    /// Compute the median of a vector (modifies the vector in-place).
    static double median_val(std::vector<double>& v);
};

} // namespace ase