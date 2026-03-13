#include "dft_engine.h"
#include <cassert>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace ase {

// ════════════════════════════════════════════════════════════════════
//  Window Functions
// ════════════════════════════════════════════════════════════════════

RVec make_window(size_t n, Window type) {
    RVec w(n);
    if (n == 0) return w;
    double N1 = static_cast<double>(n - 1);
    for (size_t i = 0; i < n; ++i) {
        double x = (N1 > 0) ? (2.0 * M_PI * i / N1) : 0.0;
        switch (type) {
        case Window::HANN:
            w[i] = 0.5 * (1.0 - std::cos(x));
            break;
        case Window::HAMMING:
            w[i] = 0.54 - 0.46 * std::cos(x);
            break;
        case Window::BLACKMAN_HARRIS:
            // 4-term Blackman-Harris: excellent sidelobe suppression (-92 dB)
            w[i] = 0.35875 - 0.48829 * std::cos(x) + 0.14128 * std::cos(2.0 * x) - 0.01168 * std::cos(3.0 * x);
            break;
        case Window::RECTANGULAR:
        default:
            w[i] = 1.0;
            break;
        }
    }
    return w;
}

/// Return the smallest power of 2 that is >= n.
size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// ════════════════════════════════════════════════════════════════════
//  FFT Implementation (Radix-2 Cooley-Tukey)
// ════════════════════════════════════════════════════════════════════

/// Perform bit-reversal permutation on the complex vector.
/// Required as the first step of the iterative Cooley-Tukey FFT.
static void bit_reverse(CVec& x) {
    size_t N = x.size();
    size_t bits = 0;
    { size_t t = N; while (t > 1) { t >>= 1; ++bits; } }
    for (size_t i = 0; i < N; ++i) {
        size_t j = 0, m = i;
        for (size_t b = 0; b < bits; ++b) {
            j = (j << 1) | (m & 1);
            m >>= 1;
        }
        if (j > i) std::swap(x[i], x[j]);
    }
}

/// In-place forward FFT using the iterative Cooley-Tukey algorithm.
/// Input length must be a power of 2.
void fft_inplace(CVec& x) {
    size_t N = x.size();
    if (N <= 1) return;
    bit_reverse(x);
    // Butterfly stages: process sub-arrays of increasing length
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2.0 * M_PI / static_cast<double>(len);
        Complex wn(std::cos(angle), std::sin(angle));  // Principal Nth root of unity
        for (size_t i = 0; i < N; i += len) {
            Complex w(1.0, 0.0);  // Twiddle factor, starts at 1
            size_t half = len >> 1;
            for (size_t j = 0; j < half; ++j) {
                Complex u = x[i + j];
                Complex v = x[i + j + half] * w;
                x[i + j]          = u + v;   // Even butterfly output
                x[i + j + half]   = u - v;   // Odd butterfly output
                w *= wn;                       // Advance twiddle factor
            }
        }
    }
}

/// In-place inverse FFT using the conjugate trick:
/// IFFT(x) = conj(FFT(conj(x))) / N
void ifft_inplace(CVec& x) {
    for (auto& c : x) c = std::conj(c);
    fft_inplace(x);
    double inv = 1.0 / static_cast<double>(x.size());
    for (auto& c : x) c = std::conj(c) * inv;
}

/// Compute FFT of real-valued data, zero-padded to the next power of 2 >= fft_size.
CVec fft_real(const double* data, size_t len, size_t fft_size) {
    size_t N = next_pow2(std::max(len, fft_size));
    CVec x(N, Complex(0.0, 0.0));
    for (size_t i = 0; i < len; ++i) x[i] = Complex(data[i], 0.0);
    fft_inplace(x);
    return x;
}

// ════════════════════════════════════════════════════════════════════
//  Utility Functions
// ════════════════════════════════════════════════════════════════════

/// Compute root-mean-square energy of a buffer.
double compute_rms(const double* data, size_t n) {
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += data[i] * data[i];
    return std::sqrt(sum / static_cast<double>(n));
}

/// Convert frequency (Hz) to MIDI note number. Returns -1 for non-positive frequencies.
int freq_to_midi(double f) {
    if (f <= 0.0) return -1;
    return static_cast<int>(std::round(12.0 * std::log2(f / 440.0) + 69.0));
}

/// Convert MIDI note number to frequency (Hz). A4 (MIDI 69) = 440 Hz.
double midi_to_freq(int m) {
    return 440.0 * std::pow(2.0, (m - 69.0) / 12.0);
}

/// Convert MIDI note number to a human-readable name (e.g., "C4", "A#3").
std::string midi_to_name(int m) {
    static const char* names[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    if (m < 0 || m > 127) return "?";
    std::ostringstream os;
    os << names[m % 12] << (m / 12 - 1);
    return os.str();
}

// ════════════════════════════════════════════════════════════════════
//  STFT Implementation
// ════════════════════════════════════════════════════════════════════

STFT::STFT(int fft_size, int hop_size, int sample_rate, Window win)
    : n_(fft_size), hop_(hop_size), sr_(sample_rate) {
    win_ = make_window(n_, win);
}

/// Analyze an entire signal by sliding the STFT window with the configured hop size.
/// Frame timestamps are set to the center of each analysis window.
std::vector<STFT::Frame> STFT::analyze(const RVec& signal) const {
    std::vector<Frame> frames;
    for (int start = 0; start + n_ <= (int)signal.size(); start += hop_)
        frames.push_back(analyze_frame(signal.data() + start, n_, (start + n_ / 2.0) / sr_));
    return frames;
}

/// Analyze a single frame: apply window, compute FFT, extract magnitude/phase/dB.
STFT::Frame STFT::analyze_frame(const double* samples, int count, double time) const {
    Frame f; f.time = time;
    // Apply window and zero-pad if count < fft_size
    CVec x(n_, Complex(0, 0));
    int limit = std::min(count, n_);
    for (int i = 0; i < limit; ++i) x[i] = Complex(samples[i] * win_[i], 0.0);
    fft_inplace(x);
    f.spectrum = x;
    // Extract positive-frequency half of the spectrum
    size_t half = n_ / 2 + 1;
    f.magnitude.resize(half); f.mag_db.resize(half); f.phase.resize(half);
    for (size_t k = 0; k < half; ++k) {
        double mag = std::abs(x[k]);
        f.magnitude[k] = mag;
        f.mag_db[k] = 20.0 * std::log10(mag + 1e-12);  // +epsilon to avoid log(0)
        f.phase[k] = std::arg(x[k]);
    }
    return f;
}

/// Refine a spectral peak using parabolic interpolation on the dB magnitude spectrum.
/// This yields sub-bin frequency accuracy and a more precise amplitude estimate.
Peak STFT::refine_peak(const RVec& mag_db, const RVec& ph, int bin) const {
    Peak p; p.bin = bin;
    // Three-point parabolic interpolation using the dB values at [bin-1, bin, bin+1]
    double alpha = mag_db[bin - 1], beta = mag_db[bin], gamma = mag_db[bin + 1];
    double denom = alpha - 2.0 * beta + gamma;
    double offset = 0.0, peak_db = beta;
    if (std::abs(denom) > 1e-12) {
        offset = 0.5 * (alpha - gamma) / denom;      // Fractional bin offset [-0.5, +0.5]
        peak_db = beta - 0.25 * (alpha - gamma) * offset;  // Interpolated peak dB
    }
    p.freq = (bin + offset) * sr_ / n_;                // Convert bin to Hz
    p.amp = std::pow(10.0, peak_db / 20.0);            // Convert dB to linear amplitude
    // Linearly interpolate phase based on the fractional offset
    if (offset >= 0 && bin + 1 < (int)ph.size()) p.phase = ph[bin] + offset * (ph[bin+1]-ph[bin]);
    else if (bin - 1 >= 0) p.phase = ph[bin] + offset * (ph[bin]-ph[bin-1]);
    else p.phase = ph[bin];
    return p;
}

/// Detect local maxima in the dB magnitude spectrum above a threshold.
/// Returns peaks sorted by amplitude in descending order.
std::vector<Peak> STFT::detect_peaks(const Frame& f, double threshold_db) const {
    std::vector<Peak> peaks;
    // Find local maxima: bin must be higher than both neighbors and above threshold
    for (int k = 1; k < (int)f.mag_db.size() - 1; ++k) {
        if (f.mag_db[k] > threshold_db && f.mag_db[k] > f.mag_db[k-1] && f.mag_db[k] > f.mag_db[k+1])
            peaks.push_back(refine_peak(f.mag_db, f.phase, k));
    }
    // Sort by amplitude (descending) so the strongest peaks come first
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b){ return a.amp > b.amp; });
    return peaks;
}

/// Overlap-add synthesis: reconstruct a time-domain signal from STFT frames.
/// Uses the analysis window for both analysis and synthesis (squared-window normalization).
RVec STFT::synthesize(const std::vector<Frame>& frames) const {
    if (frames.empty()) return {};
    size_t out_len = (frames.size() - 1) * hop_ + n_;
    RVec output(out_len, 0.0), win_sum(out_len, 0.0);
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        CVec buf = frames[fi].spectrum; ifft_inplace(buf);
        for (size_t i = 0; i < (size_t)n_ && fi * hop_ + i < out_len; ++i) {
            output[fi * hop_ + i] += buf[i].real() * win_[i];
            win_sum[fi * hop_ + i] += win_[i] * win_[i];   // Squared-window normalization
        }
    }
    // Normalize by the accumulated window energy to compensate for overlap
    for (size_t i = 0; i < out_len; ++i) if (win_sum[i] > 1e-8) output[i] /= win_sum[i];
    return output;
}

// ════════════════════════════════════════════════════════════════════
//  Partial Tracker Implementation
// ════════════════════════════════════════════════════════════════════

PartialTracker::PartialTracker(double tolerance_cents, double min_partial_dur, int max_gap_frames)
    : tol_(tolerance_cents), min_dur_(min_partial_dur), max_gap_(max_gap_frames) {}

/// Feed one frame's spectral peaks into the tracker.
///
/// Algorithm:
///   1. For each active partial, find the nearest unmatched peak within tolerance
///   2. Extend matched partials; increment gap counter for unmatched ones
///   3. Finalize partials that have exceeded the maximum gap
///   4. Create new partials from any leftover unmatched peaks
void PartialTracker::feed(const std::vector<Peak>& peaks, double time) {
    std::vector<bool> used(peaks.size(), false);
    // Try to continue each active partial with the nearest matching peak
    for (auto& p : active_) {
        int best = -1; double best_d = tol_ + 1.0;
        for (int i = 0; i < (int)peaks.size(); ++i) {
            if (used[i]) continue;
            double d = cents_dist(p.freqs.back(), peaks[i].freq);
            if (d < best_d) { best_d = d; best = i; }
        }
        if (best >= 0 && best_d <= tol_) {
            // Extend the partial with the matched peak
            p.times.push_back(time); p.freqs.push_back(peaks[best].freq);
            p.amps.push_back(peaks[best].amp); p.phases.push_back(peaks[best].phase);
            p.gap = 0; used[best] = true;
        } else {
            // No match found — increment gap counter
            p.gap++;
        }
    }
    // Remove partials that have exceeded the maximum gap tolerance
    auto it = active_.begin();
    while (it != active_.end()) {
        if (it->gap > max_gap_) {
            // Only keep partials that lasted long enough to be musically meaningful
            if (it->times.back() - it->times.front() >= min_dur_) completed_.push_back(std::move(*it));
            it = active_.erase(it);
        } else ++it;
    }
    // Start new partials for any unmatched peaks
    for (int i = 0; i < (int)peaks.size(); ++i) {
        if (!used[i]) {
            Partial np; np.id = next_id_++; np.times.push_back(time);
            np.freqs.push_back(peaks[i].freq); np.amps.push_back(peaks[i].amp);
            np.phases.push_back(peaks[i].phase); active_.push_back(std::move(np));
        }
    }
}

/// Finalize all remaining active partials (call after the last frame).
/// Only partials meeting the minimum duration requirement are kept.
void PartialTracker::finish() {
    for (auto& p : active_) if (p.times.back() - p.times.front() >= min_dur_) completed_.push_back(std::move(p));
    active_.clear();
}

/// Return all partials (both completed and still active).
std::vector<Partial> PartialTracker::all() const {
    auto out = completed_; out.insert(out.end(), active_.begin(), active_.end());
    return out;
}

/// Extract musical notes from tracked partials.
///
/// Groups consecutive partial frames with stable pitch (within pitch_gate_cents)
/// into notes. If the pitch jumps by more than the gate, a new note begins.
/// Notes shorter than min_dur are discarded.
std::vector<Note> PartialTracker::extract_notes(const std::vector<Partial>& partials, double min_dur, double gate) {
    std::vector<Note> notes;
    for (auto& p : partials) {
        if (p.freqs.size() < 2) continue;
        // Walk through the partial, splitting at pitch discontinuities
        size_t start = 0; double f_sum = p.freqs[0], a_sum = p.amps[0]; int count = 1;
        for (size_t i = 1; i < p.freqs.size(); ++i) {
            if (cents_dist(p.freqs[i], f_sum/count) > gate) {
                // Pitch jump: finalize the current note segment
                if (p.times[i-1]-p.times[start] >= min_dur) notes.push_back({p.times[start], p.times[i-1], f_sum/count, a_sum/count, freq_to_midi(f_sum/count)});
                start = i; f_sum = p.freqs[i]; a_sum = p.amps[i]; count = 1;
            } else { f_sum += p.freqs[i]; a_sum += p.amps[i]; count++; }
        }
        // Finalize the last note segment
        if (p.times.back()-p.times[start] >= min_dur) notes.push_back({p.times[start], p.times.back(), f_sum/count, a_sum/count, freq_to_midi(f_sum/count)});
    }
    // Sort all notes by onset time for chronological output
    std::sort(notes.begin(), notes.end(), [](const Note& a, const Note& b){ return a.start < b.start; });
    return notes;
}

// ════════════════════════════════════════════════════════════════════
//  Chroma Extractor Implementation
// ════════════════════════════════════════════════════════════════════

ChromaExtractor::ChromaExtractor(int fft_size, int hop_size, int sr, double ref)
    : n_(fft_size), hop_(hop_size), sr_(sr), ref_(ref), stft_(fft_size, hop_size, sr) { build_mapping(); }

/// Build the lookup table mapping each FFT bin to a chroma index (0–11).
/// Only bins in the musically relevant range (A0=27.5 Hz to C8=4186 Hz) are mapped.
/// Bins outside this range get -1 (ignored during chroma accumulation).
void ChromaExtractor::build_mapping() {
    bin_chroma_.resize(n_/2+1, -1);
    for (size_t k = 1; k < bin_chroma_.size(); ++k) {
        double f = k * sr_ / n_;
        if (f >= 27.5 && f <= 4186.0) bin_chroma_[k] = (int(std::round(12.0 * std::log2(f / ref_) + 69.0)) % 12 + 12) % 12;
    }
}

/// Analyze a complete signal: run STFT then extract chroma from each frame's magnitude.
std::vector<Chroma> ChromaExtractor::analyze(const RVec& signal) const {
    auto frames = stft_.analyze(signal); std::vector<Chroma> out;
    for (auto& f : frames) out.push_back(analyze_frame(f.magnitude));
    return out;
}

/// Extract a single chroma vector from a magnitude spectrum.
/// Accumulates squared magnitude per pitch class and L2-normalizes the result.
Chroma ChromaExtractor::analyze_frame(const RVec& mag) const {
    Chroma c{}; double norm = 0;
    // Sum squared magnitudes into the appropriate chroma bins
    for (size_t k = 0; k < mag.size(); ++k) if (bin_chroma_[k] >= 0) c[bin_chroma_[k]] += mag[k] * mag[k];
    // L2 normalization so that overall volume doesn't affect the chroma profile
    for (double v : c) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12) for (double& v : c) v /= norm;
    return c;
}

// ════════════════════════════════════════════════════════════════════
//  HPSS (Harmonic-Percussive Source Separation)
// ════════════════════════════════════════════════════════════════════

HPSS::HPSS(int n, int h, int sr, int tk, int fk, double p, Window w)
    : stft_(n, h, sr, w), hop_(h), sr_(sr), t_kern_(tk|1), f_kern_(fk|1), power_(p) {
    // Ensure kernel sizes are odd, compute half-widths for median filtering
    t_half_ = t_kern_ / 2; f_half_ = f_kern_ / 2;
}

/// Compute the median of a vector using partial sorting (nth_element).
double HPSS::median_val(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}

/// Feed one frame of audio into the HPSS pipeline.
/// Returns false until enough frames have accumulated to fill the time median kernel.
bool HPSS::feed(const double* s, int c, double t) {
    buf_.push_back(stft_.analyze_frame(s, c, t));
    if ((int)buf_.size() < t_kern_) return false;  // Need t_kern_ frames before output is available
    while ((int)buf_.size() > t_kern_) buf_.pop_front();
    compute_masks();
    return true;
}

/// Compute harmonic and percussive soft masks from the buffered spectrogram.
///
/// For each frequency bin in the center frame:
///   - Time median (H): median of that bin across all buffered frames → harmonic estimate
///   - Frequency median (P): median of neighboring bins in the center frame → percussive estimate
///   - Soft mask: H^p / (H^p + P^p) gives the harmonic fraction
void HPSS::compute_masks() {
    const auto& mid = buf_[t_half_]; size_t bins = mid.magnitude.size(), N = mid.spectrum.size();
    h_mag_.resize(bins); h_spec_.assign(N, 0); p_spec_.assign(N, 0);
    for (size_t k = 0; k < bins; ++k) {
        // Time-direction median: captures sustained (harmonic) energy
        std::vector<double> tv(t_kern_); for (int t=0; t<t_kern_; ++t) tv[t] = buf_[t].magnitude[k];
        double H = median_val(tv);
        // Frequency-direction median: captures broadband (percussive) energy
        int lo = std::max(0, (int)k - f_half_), hi = std::min((int)bins-1, (int)k+f_half_);
        std::vector<double> fv(hi-lo+1); for (int j=lo; j<=hi; ++j) fv[j-lo] = mid.magnitude[j];
        double P = median_val(fv);
        // Soft mask: ratio of harmonic power to total power
        double hm = std::pow(H, power_) / (std::pow(H, power_) + std::pow(P, power_) + 1e-10);
        // Apply mask to magnitude and complex spectrum
        h_mag_[k] = mid.magnitude[k] * hm; h_spec_[k] = mid.spectrum[k] * hm; p_spec_[k] = mid.spectrum[k] * (1.0-hm);
        // Mirror conjugate symmetry for proper IFFT
        if (k > 0 && k < N/2) { h_spec_[N-k] = std::conj(h_spec_[k]); p_spec_[N-k] = std::conj(p_spec_[k]); }
    }
}

/// Offline HPSS: separate an entire signal into harmonic and percussive components.
/// Computes the full spectrogram, applies median filtering, and resynthesizes both streams.
std::pair<RVec, RVec> HPSS::separate_signal(const RVec& sig, int n, int h, int sr, int tk, int fk, double p) {
    STFT st(n, h, sr); auto frames = st.analyze(sig); int T = frames.size(), bins = n/2+1, th = (tk|1)/2, fh = (fk|1)/2;
    std::vector<STFT::Frame> hf(T), pf(T);
    for (int t = 0; t < T; ++t) {
        hf[t].spectrum.resize(n); pf[t].spectrum.resize(n); hf[t].time = pf[t].time = frames[t].time;
        for (int k = 0; k < bins; ++k) {
            // Gather time-direction and frequency-direction neighborhoods
            std::vector<double> tv; for (int i=std::max(0, t-th); i<=std::min(T-1, t+th); ++i) tv.push_back(frames[i].magnitude[k]);
            std::vector<double> fv; for (int i=std::max(0, k-fh); i<=std::min(bins-1, k+fh); ++i) fv.push_back(frames[t].magnitude[i]);
            // Compute soft harmonic mask
            double hm = std::pow(median_val(tv), p) / (std::pow(median_val(tv), p) + std::pow(median_val(fv), p) + 1e-10);
            hf[t].spectrum[k] = frames[t].spectrum[k] * hm; pf[t].spectrum[k] = frames[t].spectrum[k] * (1.0-hm);
            if (k > 0 && k < n/2) { hf[t].spectrum[n-k] = std::conj(hf[t].spectrum[k]); pf[t].spectrum[n-k] = std::conj(pf[t].spectrum[k]); }
        }
    }
    return {st.synthesize(hf), st.synthesize(pf)};
}

// ════════════════════════════════════════════════════════════════════
//  Active Chroma from Partial Tracker
// ════════════════════════════════════════════════════════════════════

/// Compute a 12-bin chroma vector from currently active (in-progress) partials.
/// Each partial's most recent frequency is mapped to a chroma bin, weighted
/// by its squared amplitude. The result is L2-normalized.
std::vector<double> PartialTracker::active_chroma() const {
    std::vector<double> c(12, 0.0);
    for (const auto& p : active_) {
        if (p.freqs.empty()) continue;
        
        double freq = p.freqs.back();
        double amp  = p.amps.back();
        int midi = freq_to_midi(freq);
        
        if (midi >= 0 && midi < 128) {
            c[midi % 12] += amp * amp;  // Energy weighting (squared amplitude)
        }
    }
    
    // L2 normalization
    double norm = 0.0;
    for (double v : c) norm += v * v;
    norm = std::sqrt(norm);
    
    if (norm > 1e-12) {
        for (double& v : c) v /= norm;
    }
    return c;
}

// ════════════════════════════════════════════════════════════════════
//  FFT-Accelerated Chroma Sequence Cross-Correlation
// ════════════════════════════════════════════════════════════════════

/**
 * Compute normalized sliding cross-correlation between a chroma template
 * and a chroma source sequence using FFT convolution.
 *
 * This is the core matching function used by the PositionTracker. For each
 * valid alignment offset d, it computes the cosine similarity between
 * the template (recent live chroma history) and the corresponding segment
 * of the reference chroma.
 *
 * The correlation is computed per-chroma-channel via FFT, then normalized
 * by the geometric mean of template and window energies to produce a
 * cosine-similarity-like score in [0, 1].
 *
 * Complexity: O(C · N log N) where C=12 chroma bins and N=next_pow2(S+T),
 * much faster than the naive O(C · T · result_len) sliding dot product
 * when the search window is large.
 *
 * @param tpl  Template chroma data, row-major layout [T × C]
 * @param T    Number of template frames (live history length)
 * @param src  Source chroma data, row-major layout [S × C]
 * @param S    Number of source frames (reference search window)
 * @param C    Number of chroma bins (default 12)
 * @return     Vector of normalized similarity scores, length (S - T + 1)
 */
RVec chroma_sequence_match(
        const double* tpl, int T,
        const double* src, int S,
        int C) {
    
    if (T <= 0 || S <= 0 || T > S) return {};
    
    int result_len = S - T + 1;
    // Pad to next power of 2 for efficient FFT
    int N = static_cast<int>(next_pow2(static_cast<size_t>(S + T)));
    RVec correlation(result_len, 0.0);

    // Cross-correlate each chroma channel independently via FFT.
    // conj(FFT(template)) * FFT(source) in frequency domain = cross-correlation in time domain.
    for (int c = 0; c < C; ++c) {
        CVec A(N, Complex(0.0, 0.0));
        CVec B(N, Complex(0.0, 0.0));

        // Extract column c from the row-major template and source matrices
        for (int i = 0; i < T; ++i) A[i] = Complex(tpl[i * C + c], 0.0);
        for (int i = 0; i < S; ++i) B[i] = Complex(src[i * C + c], 0.0);

        fft_inplace(A);
        fft_inplace(B);

        // Multiply conjugate of A with B → cross-correlation in frequency domain
        CVec R(N);
        for (int i = 0; i < N; ++i) R[i] = std::conj(A[i]) * B[i];

        ifft_inplace(R);

        // Accumulate the real part of the IFFT into the correlation scores
        for (int d = 0; d < result_len; ++d) correlation[d] += R[d].real();
    }

    // Normalize each offset by the geometric mean of template and window energies.
    // This converts raw dot products into cosine-similarity-like scores.

    // Precompute total template energy (constant across all offsets)
    double tpl_energy = 0.0;
    for (int i = 0; i < T * C; ++i) tpl_energy += tpl[i] * tpl[i];
    if (tpl_energy < 1e-12) return RVec(result_len, 0.0);

    // Precompute cumulative source energy for efficient sliding-window energy lookup
    std::vector<double> cum(S + 1, 0.0);
    for (int i = 0; i < S; ++i) {
        double e = 0.0;
        for (int cc = 0; cc < C; ++cc) {
            double v = src[i * C + cc];
            e += v * v;
        }
        cum[i + 1] = cum[i] + e;
    }

    // Normalize each alignment offset
    for (int d = 0; d < result_len; ++d) {
        // Energy of the source window at offset d (length T)
        double win_energy = cum[d + T] - cum[d];
        double denom = std::sqrt(tpl_energy * win_energy);
        if (denom > 1e-12) correlation[d] /= denom;
        else correlation[d] = 0.0;
    }
    return correlation;
}

} // namespace ase