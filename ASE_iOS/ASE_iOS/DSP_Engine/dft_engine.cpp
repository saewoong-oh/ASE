#include "dft_engine.h"
#include <cassert>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace ase {

RVec make_window(size_t n, Window type) {
    RVec w(n);
    if (n == 0) return w;
    double N1 = static_cast<double>(n - 1);
    for (size_t i = 0; i < n; ++i) {
        double x = (N1 > 0) ? (2.0 * M_PI * i / N1) : 0.0;
        switch (type) {
        case Window::HANN: w[i] = 0.5 * (1.0 - std::cos(x)); break;
        case Window::HAMMING: w[i] = 0.54 - 0.46 * std::cos(x); break;
        case Window::BLACKMAN_HARRIS:
            w[i] = 0.35875 - 0.48829 * std::cos(x) + 0.14128 * std::cos(2.0 * x) - 0.01168 * std::cos(3.0 * x);
            break;
        case Window::RECTANGULAR:
        default: w[i] = 1.0; break;
        }
    }
    return w;
}

size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

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

void fft_inplace(CVec& x) {
    size_t N = x.size();
    if (N <= 1) return;
    bit_reverse(x);
    for (size_t len = 2; len <= N; len <<= 1) {
        double angle = -2.0 * M_PI / static_cast<double>(len);
        Complex wn(std::cos(angle), std::sin(angle));
        for (size_t i = 0; i < N; i += len) {
            Complex w(1.0, 0.0);
            size_t half = len >> 1;
            for (size_t j = 0; j < half; ++j) {
                Complex u = x[i + j];
                Complex v = x[i + j + half] * w;
                x[i + j] = u + v;
                x[i + j + half] = u - v;
                w *= wn;
            }
        }
    }
}

void ifft_inplace(CVec& x) {
    for (auto& c : x) c = std::conj(c);
    fft_inplace(x);
    double inv = 1.0 / static_cast<double>(x.size());
    for (auto& c : x) c = std::conj(c) * inv;
}

CVec fft_real(const double* data, size_t len, size_t fft_size) {
    size_t N = next_pow2(std::max(len, fft_size));
    CVec x(N, Complex(0.0, 0.0));
    for (size_t i = 0; i < len; ++i) x[i] = Complex(data[i], 0.0);
    fft_inplace(x);
    return x;
}

double compute_rms(const double* data, size_t n) {
    if (n == 0) return 0.0;
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += data[i] * data[i];
    return std::sqrt(sum / static_cast<double>(n));
}

int freq_to_midi(double f) {
    if (f <= 0.0) return -1;
    return static_cast<int>(std::round(12.0 * std::log2(f / 440.0) + 69.0));
}

double midi_to_freq(int m) {
    return 440.0 * std::pow(2.0, (m - 69.0) / 12.0);
}

std::string midi_to_name(int m) {
    static const char* names[] = {"C","C#","D","D#","E","F","F#","G","G#","A","A#","B"};
    if (m < 0 || m > 127) return "?";
    std::ostringstream os;
    os << names[m % 12] << (m / 12 - 1);
    return os.str();
}

STFT::STFT(int fft_size, int hop_size, int sample_rate, Window win)
    : n_(fft_size), hop_(hop_size), sr_(sample_rate) {
    win_ = make_window(n_, win);
}

std::vector<STFT::Frame> STFT::analyze(const RVec& signal) const {
    std::vector<Frame> frames;
    for (int start = 0; start + n_ <= (int)signal.size(); start += hop_)
        frames.push_back(analyze_frame(signal.data() + start, n_, (start + n_ / 2.0) / sr_));
    return frames;
}

STFT::Frame STFT::analyze_frame(const double* samples, int count, double time) const {
    Frame f; f.time = time;
    CVec x(n_, Complex(0, 0));
    int limit = std::min(count, n_);
    for (int i = 0; i < limit; ++i) x[i] = Complex(samples[i] * win_[i], 0.0);
    fft_inplace(x);
    f.spectrum = x;
    size_t half = n_ / 2 + 1;
    f.magnitude.resize(half); f.mag_db.resize(half); f.phase.resize(half);
    for (size_t k = 0; k < half; ++k) {
        double mag = std::abs(x[k]);
        f.magnitude[k] = mag;
        f.mag_db[k] = 20.0 * std::log10(mag + 1e-12);
        f.phase[k] = std::arg(x[k]);
    }
    return f;
}

Peak STFT::refine_peak(const RVec& mag_db, const RVec& ph, int bin) const {
    Peak p; p.bin = bin;
    double alpha = mag_db[bin - 1], beta = mag_db[bin], gamma = mag_db[bin + 1];
    double denom = alpha - 2.0 * beta + gamma;
    double offset = 0.0, peak_db = beta;
    if (std::abs(denom) > 1e-12) {
        offset = 0.5 * (alpha - gamma) / denom;
        peak_db = beta - 0.25 * (alpha - gamma) * offset;
    }
    p.freq = (bin + offset) * sr_ / n_;
    p.amp = std::pow(10.0, peak_db / 20.0);
    if (offset >= 0 && bin + 1 < (int)ph.size()) p.phase = ph[bin] + offset * (ph[bin+1]-ph[bin]);
    else if (bin - 1 >= 0) p.phase = ph[bin] + offset * (ph[bin]-ph[bin-1]);
    else p.phase = ph[bin];
    return p;
}

std::vector<Peak> STFT::detect_peaks(const Frame& f, double threshold_db) const {
    std::vector<Peak> peaks;
    for (int k = 1; k < (int)f.mag_db.size() - 1; ++k) {
        if (f.mag_db[k] > threshold_db && f.mag_db[k] > f.mag_db[k-1] && f.mag_db[k] > f.mag_db[k+1])
            peaks.push_back(refine_peak(f.mag_db, f.phase, k));
    }
    std::sort(peaks.begin(), peaks.end(), [](const Peak& a, const Peak& b){ return a.amp > b.amp; });
    return peaks;
}

RVec STFT::synthesize(const std::vector<Frame>& frames) const {
    if (frames.empty()) return {};
    size_t out_len = (frames.size() - 1) * hop_ + n_;
    RVec output(out_len, 0.0), win_sum(out_len, 0.0);
    for (size_t fi = 0; fi < frames.size(); ++fi) {
        CVec buf = frames[fi].spectrum; ifft_inplace(buf);
        for (size_t i = 0; i < (size_t)n_ && fi * hop_ + i < out_len; ++i) {
            output[fi * hop_ + i] += buf[i].real() * win_[i];
            win_sum[fi * hop_ + i] += win_[i] * win_[i];
        }
    }
    for (size_t i = 0; i < out_len; ++i) if (win_sum[i] > 1e-8) output[i] /= win_sum[i];
    return output;
}

PartialTracker::PartialTracker(double tolerance_cents, double min_partial_dur, int max_gap_frames)
    : tol_(tolerance_cents), min_dur_(min_partial_dur), max_gap_(max_gap_frames) {}

void PartialTracker::feed(const std::vector<Peak>& peaks, double time) {
    std::vector<bool> used(peaks.size(), false);
    for (auto& p : active_) {
        int best = -1; double best_d = tol_ + 1.0;
        for (int i = 0; i < (int)peaks.size(); ++i) {
            if (used[i]) continue;
            double d = cents_dist(p.freqs.back(), peaks[i].freq);
            if (d < best_d) { best_d = d; best = i; }
        }
        if (best >= 0 && best_d <= tol_) {
            p.times.push_back(time); p.freqs.push_back(peaks[best].freq);
            p.amps.push_back(peaks[best].amp); p.phases.push_back(peaks[best].phase);
            p.gap = 0; used[best] = true;
        } else p.gap++;
    }
    auto it = active_.begin();
    while (it != active_.end()) {
        if (it->gap > max_gap_) {
            if (it->times.back() - it->times.front() >= min_dur_) completed_.push_back(std::move(*it));
            it = active_.erase(it);
        } else ++it;
    }
    for (int i = 0; i < (int)peaks.size(); ++i) {
        if (!used[i]) {
            Partial np; np.id = next_id_++; np.times.push_back(time);
            np.freqs.push_back(peaks[i].freq); np.amps.push_back(peaks[i].amp);
            np.phases.push_back(peaks[i].phase); active_.push_back(std::move(np));
        }
    }
}

void PartialTracker::finish() {
    for (auto& p : active_) if (p.times.back() - p.times.front() >= min_dur_) completed_.push_back(std::move(p));
    active_.clear();
}

std::vector<Partial> PartialTracker::all() const {
    auto out = completed_; out.insert(out.end(), active_.begin(), active_.end());
    return out;
}

std::vector<Note> PartialTracker::extract_notes(const std::vector<Partial>& partials, double min_dur, double gate) {
    std::vector<Note> notes;
    for (auto& p : partials) {
        if (p.freqs.size() < 2) continue;
        size_t start = 0; double f_sum = p.freqs[0], a_sum = p.amps[0]; int count = 1;
        for (size_t i = 1; i < p.freqs.size(); ++i) {
            if (cents_dist(p.freqs[i], f_sum/count) > gate) {
                if (p.times[i-1]-p.times[start] >= min_dur) notes.push_back({p.times[start], p.times[i-1], f_sum/count, a_sum/count, freq_to_midi(f_sum/count)});
                start = i; f_sum = p.freqs[i]; a_sum = p.amps[i]; count = 1;
            } else { f_sum += p.freqs[i]; a_sum += p.amps[i]; count++; }
        }
        if (p.times.back()-p.times[start] >= min_dur) notes.push_back({p.times[start], p.times.back(), f_sum/count, a_sum/count, freq_to_midi(f_sum/count)});
    }
    std::sort(notes.begin(), notes.end(), [](const Note& a, const Note& b){ return a.start < b.start; });
    return notes;
}

ChromaExtractor::ChromaExtractor(int fft_size, int hop_size, int sr, double ref)
    : n_(fft_size), hop_(hop_size), sr_(sr), ref_(ref), stft_(fft_size, hop_size, sr) { build_mapping(); }

void ChromaExtractor::build_mapping() {
    bin_chroma_.resize(n_/2+1, -1);
    for (size_t k = 1; k < bin_chroma_.size(); ++k) {
        double f = k * sr_ / n_;
        if (f >= 27.5 && f <= 4186.0) bin_chroma_[k] = (int(std::round(12.0 * std::log2(f / ref_) + 69.0)) % 12 + 12) % 12;
    }
}

std::vector<Chroma> ChromaExtractor::analyze(const RVec& signal) const {
    auto frames = stft_.analyze(signal); std::vector<Chroma> out;
    for (auto& f : frames) out.push_back(analyze_frame(f.magnitude));
    return out;
}

Chroma ChromaExtractor::analyze_frame(const RVec& mag) const {
    Chroma c{}; double norm = 0;
    for (size_t k = 0; k < mag.size(); ++k) if (bin_chroma_[k] >= 0) c[bin_chroma_[k]] += mag[k] * mag[k];
    for (double v : c) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-12) for (double& v : c) v /= norm;
    return c;
}

HPSS::HPSS(int n, int h, int sr, int tk, int fk, double p, Window w)
    : stft_(n, h, sr, w), hop_(h), sr_(sr), t_kern_(tk|1), f_kern_(fk|1), power_(p) {
    t_half_ = t_kern_ / 2; f_half_ = f_kern_ / 2;
}

double HPSS::median_val(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::nth_element(v.begin(), v.begin() + v.size()/2, v.end());
    return v[v.size()/2];
}

bool HPSS::feed(const double* s, int c, double t) {
    buf_.push_back(stft_.analyze_frame(s, c, t));
    if ((int)buf_.size() < t_kern_) return false;
    while ((int)buf_.size() > t_kern_) buf_.pop_front();
    compute_masks();
    return true;
}

void HPSS::compute_masks() {
    const auto& mid = buf_[t_half_]; size_t bins = mid.magnitude.size(), N = mid.spectrum.size();
    h_mag_.resize(bins); h_spec_.assign(N, 0); p_spec_.assign(N, 0);
    for (size_t k = 0; k < bins; ++k) {
        std::vector<double> tv(t_kern_); for (int t=0; t<t_kern_; ++t) tv[t] = buf_[t].magnitude[k];
        double H = median_val(tv);
        int lo = std::max(0, (int)k - f_half_), hi = std::min((int)bins-1, (int)k+f_half_);
        std::vector<double> fv(hi-lo+1); for (int j=lo; j<=hi; ++j) fv[j-lo] = mid.magnitude[j];
        double P = median_val(fv);
        double hm = std::pow(H, power_) / (std::pow(H, power_) + std::pow(P, power_) + 1e-10);
        h_mag_[k] = mid.magnitude[k] * hm; h_spec_[k] = mid.spectrum[k] * hm; p_spec_[k] = mid.spectrum[k] * (1.0-hm);
        if (k > 0 && k < N/2) { h_spec_[N-k] = std::conj(h_spec_[k]); p_spec_[N-k] = std::conj(p_spec_[k]); }
    }
}

std::pair<RVec, RVec> HPSS::separate_signal(const RVec& sig, int n, int h, int sr, int tk, int fk, double p) {
    STFT st(n, h, sr); auto frames = st.analyze(sig); int T = frames.size(), bins = n/2+1, th = (tk|1)/2, fh = (fk|1)/2;
    std::vector<STFT::Frame> hf(T), pf(T);
    for (int t = 0; t < T; ++t) {
        hf[t].spectrum.resize(n); pf[t].spectrum.resize(n); hf[t].time = pf[t].time = frames[t].time;
        for (int k = 0; k < bins; ++k) {
            std::vector<double> tv; for (int i=std::max(0, t-th); i<=std::min(T-1, t+th); ++i) tv.push_back(frames[i].magnitude[k]);
            std::vector<double> fv; for (int i=std::max(0, k-fh); i<=std::min(bins-1, k+fh); ++i) fv.push_back(frames[t].magnitude[i]);
            double hm = std::pow(median_val(tv), p) / (std::pow(median_val(tv), p) + std::pow(median_val(fv), p) + 1e-10);
            hf[t].spectrum[k] = frames[t].spectrum[k] * hm; pf[t].spectrum[k] = frames[t].spectrum[k] * (1.0-hm);
            if (k > 0 && k < n/2) { hf[t].spectrum[n-k] = std::conj(hf[t].spectrum[k]); pf[t].spectrum[n-k] = std::conj(pf[t].spectrum[k]); }
        }
    }
    return {st.synthesize(hf), st.synthesize(pf)};
}

// ================================================================
// NEW: Active Chroma Extraction
// ================================================================
std::vector<double> PartialTracker::active_chroma() const {
    std::vector<double> c(12, 0.0);
    for (const auto& p : active_) {
        if (p.freqs.empty()) continue;
        
        double freq = p.freqs.back();
        double amp  = p.amps.back();
        int midi = freq_to_midi(freq);
        
        if (midi >= 0 && midi < 128) {
            c[midi % 12] += amp * amp;
        }
    }
    
    double norm = 0.0;
    for (double v : c) norm += v * v;
    norm = std::sqrt(norm);
    
    if (norm > 1e-12) {
        for (double& v : c) v /= norm;
    }
    return c;
}

// ================================================================
// NEW: FFT-Based Chroma Sequence Cross-Correlation
// ================================================================
RVec chroma_sequence_match(
        const double* tpl, int T,
        const double* src, int S,
        int C) {
    
    if (T <= 0 || S <= 0 || T > S) return {};
    
    int result_len = S - T + 1;
    int N = static_cast<int>(next_pow2(static_cast<size_t>(S + T)));
    RVec correlation(result_len, 0.0);

    // Cross-correlate each chroma channel using your native FFT
    for (int c = 0; c < C; ++c) {
        CVec A(N, Complex(0.0, 0.0));
        CVec B(N, Complex(0.0, 0.0));

        for (int i = 0; i < T; ++i) A[i] = Complex(tpl[i * C + c], 0.0);
        for (int i = 0; i < S; ++i) B[i] = Complex(src[i * C + c], 0.0);

        fft_inplace(A);
        fft_inplace(B);

        CVec R(N);
        for (int i = 0; i < N; ++i) R[i] = std::conj(A[i]) * B[i];

        ifft_inplace(R);

        for (int d = 0; d < result_len; ++d) correlation[d] += R[d].real();
    }

    // Normalize: cosine similarity
    double tpl_energy = 0.0;
    for (int i = 0; i < T * C; ++i) tpl_energy += tpl[i] * tpl[i];
    if (tpl_energy < 1e-12) return RVec(result_len, 0.0);

    std::vector<double> cum(S + 1, 0.0);
    for (int i = 0; i < S; ++i) {
        double e = 0.0;
        for (int cc = 0; cc < C; ++cc) {
            double v = src[i * C + cc];
            e += v * v;
        }
        cum[i + 1] = cum[i] + e;
    }

    for (int d = 0; d < result_len; ++d) {
        double win_energy = cum[d + T] - cum[d];
        double denom = std::sqrt(tpl_energy * win_energy);
        if (denom > 1e-12) correlation[d] /= denom;
        else correlation[d] = 0.0;
    }
    return correlation;
}

} // namespace ase

