#import "ASEWrapper.h"
#include "position_tracker.h"
#include "rms_matcher.h"
#include "dft_engine.h"
#include <map>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

struct StemBandDef {
    double low_hz;
    double high_hz;
    double weight;
};

static std::map<std::string, std::vector<StemBandDef>> STEM_BANDS = {
    {"drums",  {{60.0,   250.0,  1.0}, {5000.0, 16000.0, 0.5}}},
    {"bass",   {{40.0,   300.0,  1.0}}},
    {"vocals", {{250.0,  3500.0, 1.0}}},
    {"other",  {{500.0,  8000.0, 1.0}}},
    {"guitar", {{150.0,  5000.0, 1.0}}},
    {"piano",  {{120.0,  4000.0, 1.0}}},
    {"keys",   {{120.0,  4000.0, 1.0}}},
};

static std::vector<StemBandDef> DEFAULT_BAND = {{100.0, 8000.0, 1.0}};

@implementation ASEWrapper {
    ase::PositionTracker* tracker;
    ase::RMSMatcher* matcher;
    ase::STFT* stft;
    ase::ChromaExtractor* chroma_ext;

    std::map<std::string, std::vector<double>> reference_rms_map;
    std::vector<std::string> stem_names_cpp;
    std::string anchor_stem;

    std::vector<double> ring_buf;
    int ring_write_pos;
    int samples_accumulated;

    int n_frames_total;
    int hop_size;
    int sample_rate;
    int fft_size_val;

    std::map<std::string, double> stem_ema;
    bool  ema_seeded;
    double ema_alpha;

    double level_scalar;
    bool   level_calibrated;
    int    level_calib_count;
    double level_calib_live_sum;
    double level_calib_ref_sum;

    int processblock_entry_count;
    int hop_count;          // total hops processed (for periodic logging)
}

static const int LEVEL_CALIB_FRAMES = 48;

- (instancetype)init {
    self = [super init];
    if (self) {
        hop_size     = 1024;
        sample_rate  = 44100;
        fft_size_val = 4096;

        stft       = new ase::STFT(fft_size_val, hop_size, sample_rate, ase::Window::HANN);
        chroma_ext = new ase::ChromaExtractor(fft_size_val, hop_size, sample_rate);

        ring_buf.assign(fft_size_val, 0.0);
        ring_write_pos      = 0;
        samples_accumulated = 0;

        ema_alpha  = 0.35;
        ema_seeded = false;

        level_scalar         = 1.0;
        level_calibrated     = false;
        level_calib_count    = 0;
        level_calib_live_sum = 0.0;
        level_calib_ref_sum  = 0.0;

        processblock_entry_count = 0;
        hop_count                = 0;

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

- (void)resetTracker {
    NSLog(@"[ASEWrapper resetTracker]");
    if (tracker) tracker->reset(0);

    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    ring_write_pos      = 0;
    samples_accumulated = 0;
    ema_seeded          = false;

    level_scalar         = 1.0;
    level_calibrated     = false;
    level_calib_count    = 0;
    level_calib_live_sum = 0.0;
    level_calib_ref_sum  = 0.0;

    processblock_entry_count = 0;
    hop_count                = 0;

    for (auto& kv : stem_ema) kv.second = 0.0;
}

- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS {

    NSLog(@"[loadReferenceChroma] frames=%lu  stems=%@",
          (unsigned long)chroma.count, stemNames);

    std::vector<std::vector<double>> ref_chroma;
    for (NSArray<NSNumber *> *row in chroma) {
        std::vector<double> c_row;
        c_row.reserve(row.count);
        for (NSNumber *val in row) c_row.push_back(val.doubleValue);
        ref_chroma.push_back(c_row);
    }
    n_frames_total = (int)ref_chroma.size();

    stem_names_cpp.clear();
    for (NSString *name in stemNames)
        stem_names_cpp.push_back(std::string([name UTF8String]));

    anchor_stem = stem_names_cpp.empty() ? "" : stem_names_cpp[0];
    for (const auto& n : stem_names_cpp)
        if (n == "drums") { anchor_stem = "drums"; break; }

    reference_rms_map.clear();
    for (NSString *key in stemRMS) {
        std::string k([key UTF8String]);
        std::vector<double> v;
        for (NSNumber *n in stemRMS[key]) v.push_back(n.doubleValue);
        reference_rms_map[k] = v;
    }

    for (const auto& name : stem_names_cpp) {
        const auto& rms = reference_rms_map[name];
        double peak = 0.0, sum = 0.0;
        for (double v : rms) { sum += v; peak = std::max(peak, v); }
        NSLog(@"[REF STATS] %s  frames=%d  mean=%.5f  peak=%.5f",
              name.c_str(), (int)rms.size(),
              rms.empty() ? 0.0 : sum / rms.size(), peak);
    }

    stem_ema.clear();
    for (const auto& name : stem_names_cpp) stem_ema[name] = 0.0;
    ema_seeded = false;

    level_scalar         = 1.0;
    level_calibrated     = false;
    level_calib_count    = 0;
    level_calib_live_sum = 0.0;
    level_calib_ref_sum  = 0.0;
    processblock_entry_count = 0;
    hop_count                = 0;

    if (tracker) { delete tracker; tracker = nullptr; }
    if (matcher) { delete matcher; matcher = nullptr; }

    tracker = new ase::PositionTracker(
        ref_chroma,
        200,    // history_frames <--- INCREASED THIS TO 150
        150,     // search_radius (local, when locked)
        3.0,    // sigma
        0.12,   // confidence_threshold
        1.10,   // max_tempo
        0.90    // min_tempo
    );
    matcher = new ase::RMSMatcher(stem_names_cpp, anchor_stem);

    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    ring_write_pos      = 0;
    samples_accumulated = 0;

    NSLog(@"[loadReferenceChroma] DONE  N=%d  anchor=%s",
          n_frames_total, anchor_stem.c_str());
}

- (double)computeBandEnergy:(const std::vector<double>&)mag
                      lowHz:(double)lo
                     highHz:(double)hi
                    fftSize:(int)N
                 sampleRate:(int)sr {
    int lo_bin = std::max(1, (int)std::ceil(lo * N / (double)sr));
    int hi_bin = std::min((int)mag.size() - 1, (int)std::floor(hi * N / (double)sr));
    if (lo_bin > hi_bin) return 0.0;
    double e = 0.0;
    for (int k = lo_bin; k <= hi_bin; ++k) e += mag[k] * mag[k];
    return std::sqrt(e / (double)(hi_bin - lo_bin + 1));
}

- (std::map<std::string, double>)stemRMSFromMag:(const std::vector<double>&)mag
                                        fftSize:(int)N {
    std::map<std::string, double> out;
    for (const auto& name : stem_names_cpp) {
        const std::vector<StemBandDef>* bands;
        auto it = STEM_BANDS.find(name);
        bands = (it != STEM_BANDS.end()) ? &it->second : &DEFAULT_BAND;

        double energy = 0.0;
        for (const auto& b : *bands)
            energy += [self computeBandEnergy:mag
                                        lowHz:b.low_hz
                                       highHz:b.high_hz
                                      fftSize:N
                                   sampleRate:sample_rate] * b.weight;

        double& ema = stem_ema[name];
        ema = ema_seeded ? (ema_alpha * energy + (1.0 - ema_alpha) * ema) : energy;
        out[name] = ema;
    }
    ema_seeded = true;
    return out;
}

- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length {

    processblock_entry_count++;
    if (!tracker || !matcher || length == 0) return @{};

    int N = fft_size_val;

    // Write into ring buffer
    for (int i = 0; i < (int)length; ++i) {
        ring_buf[ring_write_pos] = samples[i];
        ring_write_pos = (ring_write_pos + 1) % N;
        samples_accumulated++;
    }

    NSDictionary* lastResult = nil;

    while (samples_accumulated >= hop_size) {
        samples_accumulated -= hop_size;
        hop_count++;

        // Linearise: read oldest→newest from ring
        std::vector<double> frame_buf(N);
        for (int i = 0; i < N; ++i)
            frame_buf[i] = ring_buf[(ring_write_pos + i) % N];

        // STFT
        auto stft_frame = stft->analyze_frame(frame_buf.data(), N, 0.0);
        std::vector<double> mag(stft_frame.magnitude.begin(),
                                stft_frame.magnitude.end());

        // Chroma
        auto chroma_arr = chroma_ext->analyze_frame(mag);
        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());

        // Chroma energy (for silence detection inside tracker)
        double chroma_energy = 0.0;
        for (double v : live_chroma) chroma_energy += v * v;

        // Position tracking
        auto [pos, conf] = tracker->process(live_chroma);

        // Log every 43 hops (~1 second)
        if (hop_count % 43 == 0) {
            double ref_secs    = pos * (hop_size / (double)sample_rate);
            double total_secs  = n_frames_total * (hop_size / (double)sample_rate);
            NSLog(@"[HOP %4d] pos=%4d/%d (%.1f/%.1fs)  conf=%.3f  calib=%s",
                  hop_count, pos, n_frames_total,
                  ref_secs, total_secs,
                  conf,
                  level_calibrated ? "YES" : "no");
        }

        // Live band RMS
        std::map<std::string, double> live_rms = [self stemRMSFromMag:mag fftSize:N];

        // Reference RMS at tracked position
        std::map<std::string, double> ref_rms;
        for (const auto& name : stem_names_cpp) {
            const auto& v = reference_rms_map[name];
            ref_rms[name] = (pos >= 0 && pos < (int)v.size()) ? v[pos] : 0.0;
        }

        // Level calibration
        if (!level_calibrated && conf > 0.35) {
            double live_a = live_rms.count(anchor_stem) ? live_rms.at(anchor_stem) : 0.0;
            double ref_a  = ref_rms.count(anchor_stem)  ? ref_rms.at(anchor_stem)  : 0.0;
            if (live_a > 1e-6 && ref_a > 1e-6) {
                level_calib_live_sum += live_a;
                level_calib_ref_sum  += ref_a;
                level_calib_count++;
                if (level_calib_count >= LEVEL_CALIB_FRAMES) {
                    level_scalar     = level_calib_ref_sum / level_calib_live_sum;
                    level_calibrated = true;
                    NSLog(@"[LEVEL CALIB] anchor=%s  scalar=%.4f (%.1f dB)",
                          anchor_stem.c_str(), level_scalar,
                          20.0 * std::log10(level_scalar));
                }
            }
        }

        // Apply level scalar
        std::map<std::string, double> live_rms_scaled;
        for (const auto& name : stem_names_cpp)
            live_rms_scaled[name] = live_rms.at(name) * level_scalar;

        matcher->compute_gains(ref_rms, live_rms_scaled, conf);

        NSMutableDictionary *gD = [NSMutableDictionary new];
        NSMutableDictionary *lD = [NSMutableDictionary new];
        NSMutableDictionary *rD = [NSMutableDictionary new];
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

    return lastResult ?: @{};
}

@end







//#import "ASEWrapper.h"
//#include "position_tracker.h"
//#include "rms_matcher.h"
//#include "dft_engine.h"
//#include <map>
//#include <string>
//#include <vector>
//#include <numeric>
//#include <cmath>
//
//struct StemBandDef {
//    double low_hz;
//    double high_hz;
//    double weight;
//};
//
//static std::map<std::string, std::vector<StemBandDef>> STEM_BANDS = {
//    {"drums",  {{60.0,   250.0,  1.0}, {5000.0, 16000.0, 0.5}}},
//    {"bass",   {{40.0,   300.0,  1.0}}},
//    {"vocals", {{250.0,  3500.0, 1.0}}},
//    {"other",  {{500.0,  8000.0, 1.0}}},
//    {"guitar", {{150.0,  5000.0, 1.0}}},
//    {"piano",  {{120.0,  4000.0, 1.0}}},
//    {"keys",   {{120.0,  4000.0, 1.0}}},
//};
//
//static std::vector<StemBandDef> DEFAULT_BAND = {{100.0, 8000.0, 1.0}};
//
//@implementation ASEWrapper {
//    ase::PositionTracker* tracker;
//    ase::RMSMatcher*      matcher;
//    ase::STFT*            stft;
//    ase::ChromaExtractor* chroma_ext;
//
//    std::map<std::string, std::vector<double>> reference_rms_map;
//    std::vector<std::string> stem_names_cpp;
//    std::string anchor_stem;   // "drums" if present, else first stem
//
//    std::vector<double> ring_buf;
//    int ring_write_pos;
//    int samples_since_process;
//
//    int n_frames_total;
//    int hop_size;
//    int sample_rate;
//    int fft_size_val;
//
//    // Short-window EMA — alpha=0.6 means ~2-frame time constant
//    // This only smooths frame-to-frame jitter, not long-term drift
//    std::map<std::string, double> stem_ema;
//    bool ema_seeded;
//    double ema_alpha;   // higher = faster response, less smoothing
//
//    // Level calibration using anchor stem only
//    // scalar = mean_ref_anchor / mean_live_anchor over first N confident frames
//    double level_scalar;
//    bool   level_calibrated;
//    int    level_calib_count;
//    double level_calib_live_sum;
//    double level_calib_ref_sum;
//
//    int diag_count;
//    int processblock_entry_count;
//}
//
//static const int LEVEL_CALIB_FRAMES = 32;
//
//- (instancetype)init {
//    self = [super init];
//    if (self) {
//        hop_size     = 1024;
//        sample_rate  = 44100;
//        fft_size_val = 4096;
//
//        stft       = new ase::STFT(fft_size_val, hop_size, sample_rate, ase::Window::HANN);
//        chroma_ext = new ase::ChromaExtractor(fft_size_val, hop_size, sample_rate);
//
//        ring_buf.assign(fft_size_val, 0.0);
//        ring_write_pos        = 0;
//        samples_since_process = 0;
//
//        // Fast EMA — responds in ~2-3 frames, only kills frame-to-frame jitter
//        ema_alpha  = 0.6;
//        ema_seeded = false;
//
//        level_scalar        = 1.0;
//        level_calibrated    = false;
//        level_calib_count   = 0;
//        level_calib_live_sum = 0.0;
//        level_calib_ref_sum  = 0.0;
//
//        diag_count              = 0;
//        processblock_entry_count = 0;
//
//        tracker = nullptr;
//        matcher = nullptr;
//
//        NSLog(@"[ASEWrapper init] created");
//    }
//    return self;
//}
//
//- (void)dealloc {
//    delete tracker;
//    delete matcher;
//    delete stft;
//    delete chroma_ext;
//}
//
//- (void)resetTracker {
//    NSLog(@"[ASEWrapper resetTracker]");
//    if (tracker) tracker->reset(0);
//
//    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
//    ring_write_pos        = 0;
//    samples_since_process = 0;
//    ema_seeded            = false;
//
//    level_scalar         = 1.0;
//    level_calibrated     = false;
//    level_calib_count    = 0;
//    level_calib_live_sum = 0.0;
//    level_calib_ref_sum  = 0.0;
//
//    diag_count               = 0;
//    processblock_entry_count = 0;
//
//    for (auto& kv : stem_ema) kv.second = 0.0;
//}
//
//- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
//                  stemNames:(NSArray<NSString *> *)stemNames
//                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS {
//
//    NSLog(@"[loadReferenceChroma] chroma=%lu  stems=%@",
//          (unsigned long)chroma.count, stemNames);
//
//    std::vector<std::vector<double>> ref_chroma;
//    for (NSArray<NSNumber *> *row in chroma) {
//        std::vector<double> c_row;
//        c_row.reserve(row.count);
//        for (NSNumber *val in row) c_row.push_back(val.doubleValue);
//        ref_chroma.push_back(c_row);
//    }
//    n_frames_total = (int)ref_chroma.size();
//
//    stem_names_cpp.clear();
//    for (NSString *name in stemNames)
//        stem_names_cpp.push_back(std::string([name UTF8String]));
//
//    // Determine anchor stem
//    anchor_stem = stem_names_cpp.empty() ? "" : stem_names_cpp[0];
//    for (const auto& n : stem_names_cpp)
//        if (n == "drums") { anchor_stem = "drums"; break; }
//    NSLog(@"[loadReferenceChroma] anchor stem: %s", anchor_stem.c_str());
//
//    reference_rms_map.clear();
//    for (NSString *key in stemRMS) {
//        std::string k([key UTF8String]);
//        std::vector<double> v;
//        for (NSNumber *n in stemRMS[key]) v.push_back(n.doubleValue);
//        reference_rms_map[k] = v;
//    }
//
//    for (const auto& name : stem_names_cpp) {
//        const auto& rms = reference_rms_map[name];
//        if (!rms.empty()) {
//            double sum = 0.0, mx = 0.0;
//            for (double v : rms) { sum += v; mx = std::max(mx, v); }
//            NSLog(@"[REF STATS] %s  mean=%.5f  peak=%.5f  frames=%d",
//                  name.c_str(), sum / (double)rms.size(), mx, (int)rms.size());
//        } else {
//            NSLog(@"[REF STATS] %s  *** EMPTY ***", name.c_str());
//        }
//    }
//
//    stem_ema.clear();
//    for (const auto& name : stem_names_cpp) stem_ema[name] = 0.0;
//    ema_seeded = false;
//
//    level_scalar         = 1.0;
//    level_calibrated     = false;
//    level_calib_count    = 0;
//    level_calib_live_sum = 0.0;
//    level_calib_ref_sum  = 0.0;
//    diag_count               = 0;
//    processblock_entry_count = 0;
//
//    if (tracker) { delete tracker; tracker = nullptr; }
//    tracker = new ase::PositionTracker(ref_chroma);
//
//    if (matcher) { delete matcher; matcher = nullptr; }
//    matcher = new ase::RMSMatcher(stem_names_cpp, anchor_stem);
//
//    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
//    ring_write_pos        = 0;
//    samples_since_process = 0;
//
//    NSLog(@"[loadReferenceChroma] DONE  tracker=%s  matcher=%s  frames=%d",
//          tracker ? "ok" : "null",
//          matcher ? "ok" : "null",
//          n_frames_total);
//}
//
//// Compute sqrt(mean(mag^2)) over [low_hz, high_hz].
//// Identical formula to Python export_ios_map.py.
//- (double)computeBandEnergy:(const std::vector<double>&)mag
//                     lowHz:(double)lo
//                    highHz:(double)hi
//                   fftSize:(int)N
//                sampleRate:(int)sr {
//    int lo_bin = std::max(1,         (int)std::ceil (lo * N / (double)sr));
//    int hi_bin = std::min((int)mag.size() - 1,
//                          (int)std::floor(hi * N / (double)sr));
//    if (lo_bin > hi_bin) return 0.0;
//    double e = 0.0;
//    for (int k = lo_bin; k <= hi_bin; ++k) e += mag[k] * mag[k];
//    return std::sqrt(e / (double)(hi_bin - lo_bin + 1));
//}
//
//// Compute instantaneous band energy per stem, then apply light EMA.
//- (std::map<std::string, double>)stemRMSFromMag:(const std::vector<double>&)mag
//                                        fftSize:(int)N {
//    std::map<std::string, double> out;
//
//    for (const auto& name : stem_names_cpp) {
//        const std::vector<StemBandDef>* bands;
//        auto it = STEM_BANDS.find(name);
//        bands = (it != STEM_BANDS.end()) ? &it->second : &DEFAULT_BAND;
//
//        double energy = 0.0;
//        for (const auto& b : *bands)
//            energy += [self computeBandEnergy:mag
//                                       lowHz:b.low_hz
//                                      highHz:b.high_hz
//                                     fftSize:N
//                                  sampleRate:sample_rate] * b.weight;
//
//        // Light EMA — seed on first frame to avoid ramp from zero
//        double& ema = stem_ema[name];
//        ema = ema_seeded ? (ema_alpha * energy + (1.0 - ema_alpha) * ema)
//                         : energy;
//        out[name] = ema;
//    }
//    ema_seeded = true;
//    return out;
//}
//
//- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
//                                        length:(NSInteger)length {
//
//    processblock_entry_count++;
//    if (processblock_entry_count <= 3 || processblock_entry_count % 200 == 0) {
//        NSLog(@"[processBlock #%d] tracker=%s  matcher=%s  len=%ld",
//              processblock_entry_count,
//              tracker ? "ok" : "null",
//              matcher ? "ok" : "null",
//              (long)length);
//    }
//
//    if (!tracker || !matcher || length == 0) return @{};
//
//    int N = fft_size_val;
//
//    for (int i = 0; i < (int)length; ++i) {
//        ring_buf[ring_write_pos] = samples[i];
//        ring_write_pos = (ring_write_pos + 1) % N;
//    }
//    samples_since_process += (int)length;
//
//    NSDictionary* lastResult = nil;
//
//    while (samples_since_process >= hop_size) {
//        samples_since_process -= hop_size;
//
//        // Linearise ring buffer
//        std::vector<double> linear(N);
//        for (int i = 0; i < N; ++i)
//            linear[i] = ring_buf[(ring_write_pos + i) % N];
//
//        // STFT
//        auto frame = stft->analyze_frame(linear.data(), N, 0.0);
//        std::vector<double> mag(frame.magnitude.begin(), frame.magnitude.end());
//
//        // Chroma + position tracking
//        auto chroma_arr = chroma_ext->analyze_frame(mag);
//        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());
//        auto [pos, conf] = tracker->process(live_chroma);
//
//        double chroma_energy = 0.0;
//        for (double v : live_chroma) chroma_energy += v * v;
//
//        // Live band RMS (instantaneous + light EMA)
//        std::map<std::string, double> live_rms = [self stemRMSFromMag:mag fftSize:N];
//
//        // Reference RMS at current position
//        std::map<std::string, double> ref_rms;
//        for (const auto& name : stem_names_cpp) {
//            const auto& v = reference_rms_map[name];
//            ref_rms[name] = (pos >= 0 && pos < (int)v.size()) ? v[pos] : 0.0;
//        }
//
//        // ── Level calibration (anchor stem only) ──────────────────────
//        // We calibrate by matching the live anchor band energy to the
//        // reference anchor band energy. This corrects the global level
//        // offset between the Python-computed reference and iOS live signal.
//        // Only runs during the first LEVEL_CALIB_FRAMES confident frames
//        // where ref is non-zero.
//        if (!level_calibrated && conf > 0.3) {
//            double live_a = live_rms.count(anchor_stem) ? live_rms[anchor_stem] : 0.0;
//            double ref_a  = ref_rms.count(anchor_stem)  ? ref_rms[anchor_stem]  : 0.0;
//            if (live_a > 1e-6 && ref_a > 1e-6) {
//                level_calib_live_sum += live_a;
//                level_calib_ref_sum  += ref_a;
//                level_calib_count++;
//                if (level_calib_count >= LEVEL_CALIB_FRAMES) {
//                    double mean_live = level_calib_live_sum / level_calib_count;
//                    double mean_ref  = level_calib_ref_sum  / level_calib_count;
//                    level_scalar     = mean_ref / mean_live;
//                    level_calibrated = true;
//                    NSLog(@"[LEVEL CALIB] anchor=%s  mean_live=%.5f  mean_ref=%.5f  "
//                          @"scalar=%.4f (%.1f dB)",
//                          anchor_stem.c_str(), mean_live, mean_ref,
//                          level_scalar, 20.0 * std::log10(level_scalar));
//                }
//            }
//        }
//
//        // Apply global level scalar to all live stems
//        std::map<std::string, double> live_rms_scaled;
//        for (const auto& name : stem_names_cpp)
//            live_rms_scaled[name] = live_rms[name] * level_scalar;
//
//        // ── Diagnostic ────────────────────────────────────────────────
//        if (diag_count < 30) {
//            NSLog(@"[DIAG %d] pos=%d conf=%.3f calib=%s scalar=%.3f",
//                  diag_count, pos, conf,
//                  level_calibrated ? "YES" : "no",
//                  level_scalar);
//            for (const auto& name : stem_names_cpp) {
//                double lv = live_rms[name];
//                double ls = live_rms_scaled[name];
//                double rv = ref_rms[name];
//                NSLog(@"  %s  live=%.5f  live_scaled=%.5f  ref=%.5f  err=%.1f dB",
//                      name.c_str(), lv, ls, rv,
//                      20.0 * std::log10((rv + 1e-10) / (ls + 1e-10)));
//            }
//            diag_count++;
//        }
//
//        matcher->compute_gains(ref_rms, live_rms_scaled, conf);
//
//        NSMutableDictionary *gD = [NSMutableDictionary new];
//        NSMutableDictionary *lD = [NSMutableDictionary new];
//        NSMutableDictionary *rD = [NSMutableDictionary new];
//        for (const auto& name : stem_names_cpp) {
//            NSString *n = [NSString stringWithUTF8String:name.c_str()];
//            gD[n] = @(matcher->gains[name]);
//            lD[n] = @(matcher->live_rel_db[name]);
//            rD[n] = @(matcher->ref_rel_db[name]);
//        }
//
//        lastResult = @{
//            @"position":        @(pos),
//            @"confidence":      @(conf),
//            @"overall_gain_db": @(matcher->overall_gain_db),
//            @"ref_time":        @(pos * (hop_size / (double)sample_rate)),
//            @"progress":        @((double)pos / (double)std::max(1, n_frames_total - 1)),
//            @"stemGains":       gD,
//            @"stemLive":        lD,
//            @"stemRef":         rD,
//            @"chroma_energy":   @(chroma_energy),
//        };
//    }
//
//    return lastResult ?: @{};
//}
//
//@end
