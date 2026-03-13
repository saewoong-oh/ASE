#import "ASEWrapper.h"
#include "position_tracker.h"
#include "rms_matcher.h"
#include "dft_engine.h"
#include <map>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

/// Defines a frequency band with low/high bounds and a weighting factor.
/// Used to estimate per-stem energy by measuring specific spectral regions.
struct StemBandDef {
    double low_hz;
    double high_hz;
    double weight;
};

/// Frequency band definitions for each stem type, mirroring the Python-side STEM_BANDS.
/// Each stem is associated with one or more bands that capture its characteristic
/// spectral energy (e.g., drums have both a low kick band and a high cymbal band).
static std::map<std::string, std::vector<StemBandDef>> STEM_BANDS = {
    {"drums",  {{60.0,   250.0,  1.0}, {5000.0, 16000.0, 0.5}}},
    {"bass",   {{40.0,   300.0,  1.0}}},
    {"vocals", {{250.0,  3500.0, 1.0}}},
    {"other",  {{500.0,  8000.0, 1.0}}},
    {"guitar", {{150.0,  5000.0, 1.0}}},
    {"piano",  {{120.0,  4000.0, 1.0}}},
    {"keys",   {{120.0,  4000.0, 1.0}}},
};

/// Default frequency band used for any stem type not found in STEM_BANDS.
static std::vector<StemBandDef> DEFAULT_BAND = {{100.0, 8000.0, 1.0}};

@implementation ASEWrapper {
    // C++ engine components
    ase::PositionTracker* tracker;       // Chroma-based song position tracker
    ase::RMSMatcher* matcher;            // Per-stem gain advisory calculator
    ase::STFT* stft;                     // Short-time Fourier transform engine
    ase::ChromaExtractor* chroma_ext;    // Spectrum-to-chroma mapping

    // Reference data stored from loadReferenceChroma
    std::map<std::string, std::vector<double>> reference_rms_map;  // Per-stem reference RMS profiles
    std::vector<std::string> stem_names_cpp;                       // Ordered stem names
    std::string anchor_stem;                                       // Anchor stem for relative gain calculation

    // Ring buffer for accumulating incoming audio samples between hops.
    // Sized large enough to hold multiple FFT windows worth of audio.
    std::vector<double> ring_buf;
    int ring_buf_size;
    int ring_write_pos;        // Current write position in the ring buffer
    int samples_accumulated;   // Number of unprocessed samples waiting in the buffer

    // Analysis parameters
    int n_frames_total;        // Total number of reference frames
    int hop_size;              // Samples between consecutive analysis frames
    int sample_rate;           // Hardware sample rate (may differ from 44100)
    int fft_size_val;          // FFT window size

    // EMA (exponential moving average) smoothing for per-stem RMS output
    std::map<std::string, double> stem_ema;
    bool  ema_seeded;          // Whether the EMA has received its first value
    double ema_alpha;          // EMA smoothing coefficient (0 = no smoothing, 1 = no memory)

    // Level calibration: compensates for the gain difference between the
    // live microphone input and the reference recording levels.
    double level_scalar;            // Multiplicative gain correction factor
    bool   level_calibrated;        // Whether calibration is complete
    int    level_calib_count;       // Number of calibration frames accumulated
    double level_calib_live_sum;    // Running sum of live anchor stem RMS
    double level_calib_ref_sum;     // Running sum of reference anchor stem RMS

    // Diagnostic counters
    int processblock_entry_count;   // Total calls to processBlock:
    int hop_count;                  // Total hops (STFT frames) processed
}

/// Number of confident frames required to complete level calibration.
static const int LEVEL_CALIB_FRAMES = 48;

/// Default initializer — uses 44100 Hz as a safe fallback sample rate.
- (instancetype)init {
    return [self initWithSampleRate:44100.0];
}

/// Designated initializer. Creates all C++ DSP components at the specified sample rate.
///
/// The sample rate should match the actual hardware output/input rate from AVAudioEngine,
/// not the reference file's rate, to ensure FFT bin frequencies are correct.
- (instancetype)initWithSampleRate:(double)incomingSampleRate {
    self = [super init];
    if (self) {
        hop_size     = 1024;
        sample_rate  = (int)incomingSampleRate;
        fft_size_val = 4096;

        // Create C++ STFT and chroma extractor at the hardware sample rate
        stft       = new ase::STFT(fft_size_val, hop_size, sample_rate, ase::Window::HANN);
        chroma_ext = new ase::ChromaExtractor(fft_size_val, hop_size, sample_rate);

        // Ring buffer: 32768 samples ≈ 0.74s at 44100 Hz, enough for multiple FFT windows
        ring_buf_size = 32768;
        ring_buf.assign(ring_buf_size, 0.0);
        
        ring_write_pos      = 0;
        samples_accumulated = 0;

        // EMA smoothing factor for per-stem RMS (lower = smoother, higher = more responsive)
        ema_alpha  = 0.35;
        ema_seeded = false;

        // Level calibration starts uncalibrated with unity gain
        level_scalar         = 1.0;
        level_calibrated     = false;
        level_calib_count    = 0;
        level_calib_live_sum = 0.0;
        level_calib_ref_sum  = 0.0;

        processblock_entry_count = 0;
        hop_count                = 0;

        // Tracker and matcher are created when reference data is loaded
        tracker = nullptr;
        matcher = nullptr;

        NSLog(@"[ASEWrapper init] fft=%d hop=%d sr=%d",
              fft_size_val, hop_size, sample_rate);
    }
    return self;
}

- (void)dealloc {
    delete tracker; delete matcher; delete stft; delete chroma_ext;
}

/// Reset all tracking state to the beginning of the song.
/// Clears the ring buffer, resets the position tracker, and resets level calibration.
- (void)resetTracker {
    NSLog(@"[ASEWrapper resetTracker]");
    if (tracker) tracker->reset(0);

    // Clear the ring buffer if it has been allocated
    if (ring_buf.size() > 0) {
        std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    }
    
    ring_write_pos      = 0;
    samples_accumulated = 0;
    ema_seeded          = false;

    // Reset level calibration so it re-adapts to new input conditions
    level_scalar         = 1.0;
    level_calibrated     = false;
    level_calib_count    = 0;
    level_calib_live_sum = 0.0;
    level_calib_ref_sum  = 0.0;

    processblock_entry_count = 0;
    hop_count                = 0;

    // Reset all per-stem EMA values
    for (auto& kv : stem_ema) kv.second = 0.0;
}

/// Load reference analysis data from the Swift-decoded JSON map into C++ structures.
///
/// Creates the PositionTracker (for chroma-based song position matching) and
/// RMSMatcher (for per-stem gain advisory). Also determines the anchor stem
/// (preferring "drums" if available, otherwise the first stem).
- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS {

    NSLog(@"[loadReferenceChroma] frames=%lu  stems=%@",
          (unsigned long)chroma.count, stemNames);

    // Convert Objective-C 2D NSNumber array → C++ vector<vector<double>>
    std::vector<std::vector<double>> ref_chroma;
    for (NSArray<NSNumber *> *row in chroma) {
        std::vector<double> c_row;
        c_row.reserve(row.count);
        for (NSNumber *val in row) c_row.push_back(val.doubleValue);
        ref_chroma.push_back(c_row);
    }
    n_frames_total = (int)ref_chroma.size();

    // Convert stem names from NSString to std::string
    stem_names_cpp.clear();
    for (NSString *name in stemNames)
        stem_names_cpp.push_back(std::string([name UTF8String]));

    // Choose anchor stem: prefer "drums" because it has the most consistent
    // broadband energy, making it the most reliable level reference.
    anchor_stem = stem_names_cpp.empty() ? "" : stem_names_cpp[0];
    for (const auto& n : stem_names_cpp)
        if (n == "drums") { anchor_stem = "drums"; break; }

    // Convert per-stem RMS from NSDictionary → std::map
    reference_rms_map.clear();
    for (NSString *key in stemRMS) {
        std::string k([key UTF8String]);
        std::vector<double> v;
        for (NSNumber *n in stemRMS[key]) v.push_back(n.doubleValue);
        reference_rms_map[k] = v;
    }

    // Initialize per-stem EMA state
    stem_ema.clear();
    for (const auto& name : stem_names_cpp) stem_ema[name] = 0.0;
    ema_seeded = false;

    // Reset calibration and counters for the new reference
    level_scalar         = 1.0;
    level_calibrated     = false;
    level_calib_count    = 0;
    level_calib_live_sum = 0.0;
    level_calib_ref_sum  = 0.0;
    processblock_entry_count = 0;
    hop_count                = 0;

    // Destroy and recreate tracker and matcher with new reference data
    if (tracker) { delete tracker; tracker = nullptr; }
    if (matcher) { delete matcher; matcher = nullptr; }

    // Create position tracker with tuned parameters for real-time tracking
    tracker = new ase::PositionTracker(
        ref_chroma,
        300,    // history_frames: sliding template window length
        100,    // search_radius: frames to search around expected position when locked
        10,     // sigma: Gaussian continuity prior width
        0.15,   // confidence_threshold: minimum similarity to accept a match
        1.10,   // max_tempo: upper tempo ratio bound
        0.90    // min_tempo: lower tempo ratio bound
    );
    matcher = new ase::RMSMatcher(stem_names_cpp, anchor_stem);

    // Clear the ring buffer for fresh audio input
    if (ring_buf.size() > 0) {
        std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    }
    ring_write_pos      = 0;
    samples_accumulated = 0;

    NSLog(@"[loadReferenceChroma] DONE  N=%d  anchor=%s",
          n_frames_total, anchor_stem.c_str());
}

/// Compute RMS energy within a frequency band from an FFT magnitude spectrum.
///
/// @param mag        Magnitude spectrum (half-spectrum, bins 0 to N/2)
/// @param lo         Low frequency bound (Hz)
/// @param hi         High frequency bound (Hz)
/// @param N          FFT size
/// @param sr         Sample rate
/// @return           RMS energy in the specified band
- (double)computeBandEnergy:(const std::vector<double>&)mag
                      lowHz:(double)lo
                     highHz:(double)hi
                    fftSize:(int)N
                 sampleRate:(int)sr {
    // Convert frequency bounds to FFT bin indices
    int lo_bin = std::max(1, (int)std::ceil(lo * N / (double)sr));
    int hi_bin = std::min((int)mag.size() - 1, (int)std::floor(hi * N / (double)sr));
    if (lo_bin > hi_bin) return 0.0;
    // Compute RMS of the magnitude values within the band
    double e = 0.0;
    for (int k = lo_bin; k <= hi_bin; ++k) e += mag[k] * mag[k];
    return std::sqrt(e / (double)(hi_bin - lo_bin + 1));
}

/// Estimate per-stem RMS energy from a single FFT magnitude frame.
///
/// Uses the STEM_BANDS frequency band definitions to approximate what each
/// stem's energy would be, based solely on the full mix spectrum. Applies
/// EMA smoothing to reduce frame-to-frame jitter in the output.
///
/// @param mag        Magnitude spectrum from the current STFT frame
/// @param N          FFT size
/// @return           Map of stem name → smoothed RMS energy
- (std::map<std::string, double>)stemRMSFromMag:(const std::vector<double>&)mag
                                        fftSize:(int)N {
    std::map<std::string, double> out;
    for (const auto& name : stem_names_cpp) {
        // Look up the frequency bands for this stem type
        const std::vector<StemBandDef>* bands;
        auto it = STEM_BANDS.find(name);
        bands = (it != STEM_BANDS.end()) ? &it->second : &DEFAULT_BAND;

        // Sum weighted band energies across all bands for this stem
        double energy = 0.0;
        for (const auto& b : *bands)
            energy += [self computeBandEnergy:mag
                                        lowHz:b.low_hz
                                       highHz:b.high_hz
                                      fftSize:N
                                   sampleRate:sample_rate] * b.weight;

        // Apply exponential moving average for temporal smoothing
        double& ema = stem_ema[name];
        ema = ema_seeded ? (ema_alpha * energy + (1.0 - ema_alpha) * ema) : energy;
        out[name] = ema;
    }
    ema_seeded = true;
    return out;
}

/// Process a block of mono audio samples through the complete analysis pipeline.
///
/// Audio samples are written into a ring buffer. Whenever enough samples accumulate
/// for one hop, the following pipeline runs:
///   1. Extract an FFT-sized window from the ring buffer
///   2. Run STFT to get the magnitude spectrum
///   3. Extract a 12-bin chroma vector from the magnitude
///   4. Feed chroma into the PositionTracker to estimate song position
///   5. Estimate per-stem RMS from frequency bands
///   6. Look up reference RMS at the estimated position
///   7. Run level calibration (first N confident frames)
///   8. Compute per-stem gain advisories via RMSMatcher
///   9. Package results into an NSDictionary for Swift
///
/// Multiple hops may be processed per call if the incoming block is large.
/// Returns the result from the most recent hop, or an empty dict if no hop was completed.
- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length {

    processblock_entry_count++;
    
    // Guard: don't process if the engine isn't fully initialized
    if (!tracker || !matcher || length == 0 || ring_buf_size == 0) return @{};

    int N = fft_size_val;
    int M = ring_buf_size;

    // Write all incoming samples into the circular ring buffer
    for (int i = 0; i < (int)length; ++i) {
        ring_buf[ring_write_pos] = samples[i];
        ring_write_pos = (ring_write_pos + 1) % M;
        samples_accumulated++;
    }

    NSDictionary* lastResult = nil;

    // Process as many complete hops as are available in the buffer
    while (samples_accumulated >= hop_size) {
        samples_accumulated -= hop_size;
        hop_count++;

        // Calculate where this hop's FFT window ends in the ring buffer.
        // We need to look backwards from the current write position by
        // the number of still-unprocessed samples.
        int hop_end_pos = ring_write_pos - samples_accumulated;
        while (hop_end_pos < 0) hop_end_pos += M;
        hop_end_pos %= M;

        // The STFT window starts N samples before the hop end
        int read_start_pos = hop_end_pos - N;
        while (read_start_pos < 0) read_start_pos += M;
        read_start_pos %= M;

        // Copy N samples from the ring buffer into a contiguous frame buffer,
        // handling wrap-around at the ring buffer boundary
        std::vector<double> frame_buf(N);
        for (int i = 0; i < N; ++i) {
            frame_buf[i] = ring_buf[(read_start_pos + i) % M];
        }

        // Step 1: STFT analysis — compute magnitude spectrum
        auto stft_frame = stft->analyze_frame(frame_buf.data(), N, 0.0);
        std::vector<double> mag(stft_frame.magnitude.begin(),
                                stft_frame.magnitude.end());

        // Step 2: Chroma extraction — map spectrum to 12 pitch classes
        auto chroma_arr = chroma_ext->analyze_frame(mag);
        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());

        // Compute total chroma energy for diagnostics
        double chroma_energy = 0.0;
        for (double v : live_chroma) chroma_energy += v * v;

        // Step 3: Position tracking — find where we are in the reference song
        auto [pos, conf] = tracker->process(live_chroma);

        // Periodic diagnostic logging (every ~1 second at 44100/1024 ≈ 43 hops/s)
        if (hop_count % 43 == 0) {
            double ref_secs    = pos * (hop_size / (double)sample_rate);
            double total_secs  = n_frames_total * (hop_size / (double)sample_rate);
            NSLog(@"[HOP %4d] pos=%4d/%d (%.1f/%.1fs)  conf=%.3f",
                  hop_count, pos, n_frames_total,
                  ref_secs, total_secs, conf);
        }

        // Step 4: Estimate per-stem RMS from the live spectrum's frequency bands
        std::map<std::string, double> live_rms = [self stemRMSFromMag:mag fftSize:N];
        
        // Step 5: Look up reference per-stem RMS at the estimated position
        std::map<std::string, double> ref_rms;
        for (const auto& name : stem_names_cpp) {
            const auto& v = reference_rms_map[name];
            ref_rms[name] = (pos >= 0 && pos < (int)v.size()) ? v[pos] : 0.0;
        }

        // Step 6: Level calibration — learn the gain offset between live input
        // and reference levels using the anchor stem during the first N confident frames
        if (!level_calibrated && conf > 0.35) {
            double live_a = live_rms.count(anchor_stem) ? live_rms.at(anchor_stem) : 0.0;
            double ref_a  = ref_rms.count(anchor_stem)  ? ref_rms.at(anchor_stem)  : 0.0;
            if (live_a > 1e-6 && ref_a > 1e-6) {
                level_calib_live_sum += live_a;
                level_calib_ref_sum  += ref_a;
                level_calib_count++;
                if (level_calib_count >= LEVEL_CALIB_FRAMES) {
                    // Calibration complete: compute the scalar that maps live levels to reference levels
                    level_scalar     = level_calib_ref_sum / level_calib_live_sum;
                    level_calibrated = true;
                }
            }
        }

        // Step 7: Apply level calibration to the live RMS values
        std::map<std::string, double> live_rms_scaled;
        for (const auto& name : stem_names_cpp)
            live_rms_scaled[name] = live_rms.at(name) * level_scalar;

        // Step 8: Compute per-stem gain advisories (how much to boost/cut each stem)
        matcher->compute_gains(ref_rms, live_rms_scaled, conf);

        // Step 9: Package results into Objective-C dictionaries for Swift
        NSMutableDictionary *gD = [NSMutableDictionary new];  // Per-stem gain multipliers
        NSMutableDictionary *lD = [NSMutableDictionary new];  // Per-stem live relative dB
        NSMutableDictionary *rD = [NSMutableDictionary new];  // Per-stem reference relative dB
        for (const auto& name : stem_names_cpp) {
            NSString *n = [NSString stringWithUTF8String:name.c_str()];
            gD[n] = @(matcher->gains[name]);
            lD[n] = @(matcher->live_rel_db[name]);
            rD[n] = @(matcher->ref_rel_db[name]);
        }

        lastResult = @{
            @"position":        @(pos),
            @"confidence":      @(conf),
            @"overall_gain_db": @(matcher->overall_gain_db),
            @"ref_time":        @(pos * (hop_size / (double)sample_rate)),
            @"progress":        @((double)pos / (double)std::max(1, n_frames_total - 1)),
            @"stemGains":       gD,
            @"stemLive":        lD,
            @"stemRef":         rD,
            @"chroma_energy":   @(chroma_energy),
        };
    }

    // Return the most recent hop's result, or empty dict if no hop was processed
    return lastResult ?: @{};
}

@end