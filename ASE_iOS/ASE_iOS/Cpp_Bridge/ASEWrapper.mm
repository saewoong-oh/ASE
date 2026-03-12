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
    int ring_buf_size;
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
    int hop_count;
}

static const int LEVEL_CALIB_FRAMES = 48;

// 🌟 THE FIX: Re-add the default init so Swift doesn't bypass our setup
- (instancetype)init {
    return [self initWithSampleRate:44100.0]; // Safe fallback
}

- (instancetype)initWithSampleRate:(double)incomingSampleRate {
    self = [super init];
    if (self) {
        hop_size     = 1024;
        sample_rate  = (int)incomingSampleRate;
        fft_size_val = 4096;

        stft       = new ase::STFT(fft_size_val, hop_size, sample_rate, ase::Window::HANN);
        chroma_ext = new ase::ChromaExtractor(fft_size_val, hop_size, sample_rate);

        ring_buf_size = 32768;
        ring_buf.assign(ring_buf_size, 0.0);
        
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

    // Only fill if the buffer has been properly allocated
    if (ring_buf.size() > 0) {
        std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    }
    
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
        300,    // history_frames
        100,     // search_radius
        10,    // sigma
        0.15,   // confidence_threshold
        1.10,   // max_tempo
        0.90    // min_tempo
    );
    matcher = new ase::RMSMatcher(stem_names_cpp, anchor_stem);

    if (ring_buf.size() > 0) {
        std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    }
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
    
    // 🌟 Safety net: Do not process if the ring buffer somehow isn't allocated
    if (!tracker || !matcher || length == 0 || ring_buf_size == 0) return @{};

    int N = fft_size_val;
    int M = ring_buf_size;

    // Write all incoming samples to the massive buffer safely
    for (int i = 0; i < (int)length; ++i) {
        ring_buf[ring_write_pos] = samples[i];
        ring_write_pos = (ring_write_pos + 1) % M;
        samples_accumulated++;
    }

    NSDictionary* lastResult = nil;

    while (samples_accumulated >= hop_size) {
        samples_accumulated -= hop_size;
        hop_count++;

        // Calculate exactly where this specific hop ends in the past
        int hop_end_pos = ring_write_pos - samples_accumulated;
        while (hop_end_pos < 0) hop_end_pos += M;
        hop_end_pos %= M;

        // Calculate where the N-sized STFT window starts
        int read_start_pos = hop_end_pos - N;
        while (read_start_pos < 0) read_start_pos += M;
        read_start_pos %= M;

        std::vector<double> frame_buf(N);
        for (int i = 0; i < N; ++i) {
            frame_buf[i] = ring_buf[(read_start_pos + i) % M];
        }

        // STFT
        auto stft_frame = stft->analyze_frame(frame_buf.data(), N, 0.0);
        std::vector<double> mag(stft_frame.magnitude.begin(),
                                stft_frame.magnitude.end());

        // Chroma
        auto chroma_arr = chroma_ext->analyze_frame(mag);
        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());

        double chroma_energy = 0.0;
        for (double v : live_chroma) chroma_energy += v * v;

        // Position tracking
        auto [pos, conf] = tracker->process(live_chroma);

        if (hop_count % 43 == 0) {
            double ref_secs    = pos * (hop_size / (double)sample_rate);
            double total_secs  = n_frames_total * (hop_size / (double)sample_rate);
            NSLog(@"[HOP %4d] pos=%4d/%d (%.1f/%.1fs)  conf=%.3f",
                  hop_count, pos, n_frames_total,
                  ref_secs, total_secs, conf);
        }

        std::map<std::string, double> live_rms = [self stemRMSFromMag:mag fftSize:N];
        std::map<std::string, double> ref_rms;
        for (const auto& name : stem_names_cpp) {
            const auto& v = reference_rms_map[name];
            ref_rms[name] = (pos >= 0 && pos < (int)v.size()) ? v[pos] : 0.0;
        }

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
                }
            }
        }

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




