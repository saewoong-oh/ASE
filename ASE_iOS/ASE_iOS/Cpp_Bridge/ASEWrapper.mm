#import "ASEWrapper.h"
#include "position_tracker.h"
#include "rms_matcher.h"
#include "dft_engine.h"
#include <map>
#include <string>
#include <vector>
#include <numeric>
#include <cmath>

// Frequency band definitions per stem type
// Each stem has a primary frequency band [low_hz, high_hz] and a weight
struct StemBandDef {
    double low_hz;
    double high_hz;
    double weight;
};

static std::map<std::string, StemBandDef> STEM_BANDS = {
    {"drums",  {60.0,   250.0,  1.0}},   // kick/snare body
    {"bass",   {40.0,   300.0,  1.0}},   // bass fundamental
    {"vocals", {250.0,  3500.0, 1.0}},   // vocal range
    {"other",  {500.0,  8000.0, 1.0}},   // guitars, keys, etc.
    {"guitar", {150.0,  5000.0, 1.0}},
    {"piano",  {120.0,  4000.0, 1.0}},
    {"keys",   {120.0,  4000.0, 1.0}},
};

// Secondary band for drums (hi-hats/cymbals)
static std::map<std::string, StemBandDef> STEM_BANDS_2 = {
    {"drums", {5000.0, 16000.0, 0.5}},
};

@implementation ASEWrapper {
    ase::PositionTracker* tracker;
    ase::RMSMatcher*      matcher;
    ase::STFT*            stft;
    ase::ChromaExtractor* chroma_ext;

    std::map<std::string, std::vector<double>> reference_rms_map;
    std::vector<std::string> stem_names_cpp;

    // Ring buffer — always 4096 samples, filled incrementally
    std::vector<double> ring_buf;
    int ring_write_pos;
    int samples_since_process;

    int n_frames_total;
    int hop_size;
    int sample_rate;
    
    // Per-stem band energy smoothing (EMA)
    std::map<std::string, double> stem_band_ema;
    double ema_alpha; // smoothing factor
}

- (instancetype)init {
    self = [super init];
    if (self) {
        hop_size    = 1024;
        sample_rate = 44100;
        int fft_size = 4096;

        stft       = new ase::STFT(fft_size, hop_size, sample_rate, ase::Window::HANN);
        chroma_ext = new ase::ChromaExtractor(fft_size, hop_size, sample_rate);

        ring_buf.assign(fft_size, 0.0);
        ring_write_pos        = 0;
        samples_since_process = 0;
        ema_alpha             = 0.15; // smoothing

        tracker = nullptr;
        matcher = nullptr;
    }
    return self;
}

- (void)dealloc {
    delete tracker;
    delete matcher;
    delete stft;
    delete chroma_ext;
}

- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS {

    std::vector<std::vector<double>> ref_chroma;
    for (NSArray<NSNumber *> *row in chroma) {
        std::vector<double> c_row;
        c_row.reserve(row.count);
        for (NSNumber *val in row) c_row.push_back(val.doubleValue);
        ref_chroma.push_back(c_row);
    }
    n_frames_total = (int)ref_chroma.size();
    NSLog(@"Loaded %d chroma frames", n_frames_total);

    stem_names_cpp.clear();
    for (NSString *name in stemNames)
        stem_names_cpp.push_back(std::string([name UTF8String]));

    reference_rms_map.clear();
    for (NSString *key in stemRMS) {
        std::string cpp_key([key UTF8String]);
        std::vector<double> rms_vals;
        for (NSNumber *val in stemRMS[key]) rms_vals.push_back(val.doubleValue);
        reference_rms_map[cpp_key] = rms_vals;
    }

    // Initialize EMA state for each stem
    stem_band_ema.clear();
    for (const auto& name : stem_names_cpp) {
        stem_band_ema[name] = 0.0;
    }

    if (tracker) { delete tracker; tracker = nullptr; }
    tracker = new ase::PositionTracker(ref_chroma);

    if (matcher) { delete matcher; matcher = nullptr; }
    matcher = new ase::RMSMatcher(stem_names_cpp, "drums");

    // Reset ring buffer
    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    ring_write_pos        = 0;
    samples_since_process = 0;

    NSLog(@"ASEWrapper ready. Stems: %zu", stem_names_cpp.size());
}

// Compute RMS energy of the spectrum within a frequency band [low_hz, high_hz]
- (double)computeBandEnergy:(const std::vector<double>&)magnitude
                     lowHz:(double)low_hz
                    highHz:(double)high_hz
                 fftSize:(int)fft_size
                sampleRate:(int)sr {
    int low_bin  = (int)std::ceil(low_hz  * fft_size / (double)sr);
    int high_bin = (int)std::floor(high_hz * fft_size / (double)sr);
    
    low_bin  = std::max(1, low_bin);
    high_bin = std::min((int)magnitude.size() - 1, high_bin);
    
    if (low_bin > high_bin) return 0.0;
    
    double energy = 0.0;
    for (int k = low_bin; k <= high_bin; ++k) {
        energy += magnitude[k] * magnitude[k];
    }
    // Normalize by bin count to make comparable across band widths
    int bin_count = high_bin - low_bin + 1;
    return std::sqrt(energy / (double)bin_count);
}

// Estimate per-stem live RMS using spectral band decomposition
// This gives INDEPENDENT estimates per stem, not derived from ref ratios
- (std::map<std::string, double>)estimateStemRMS:(const std::vector<double>&)magnitude
                                         fftSize:(int)fft_size {
    std::map<std::string, double> stem_rms;
    
    for (const auto& name : stem_names_cpp) {
        double band_energy = 0.0;
        
        // Primary band
        auto band_it = STEM_BANDS.find(name);
        if (band_it != STEM_BANDS.end()) {
            band_energy += [self computeBandEnergy:magnitude
                                           lowHz:band_it->second.low_hz
                                          highHz:band_it->second.high_hz
                                         fftSize:fft_size
                                      sampleRate:sample_rate] * band_it->second.weight;
        } else {
            // Unknown stem: use broadband
            band_energy += [self computeBandEnergy:magnitude
                                           lowHz:100.0
                                          highHz:8000.0
                                         fftSize:fft_size
                                      sampleRate:sample_rate];
        }
        
        // Secondary band (e.g., drums hi-hats)
        auto band2_it = STEM_BANDS_2.find(name);
        if (band2_it != STEM_BANDS_2.end()) {
            band_energy += [self computeBandEnergy:magnitude
                                           lowHz:band2_it->second.low_hz
                                          highHz:band2_it->second.high_hz
                                         fftSize:fft_size
                                      sampleRate:sample_rate] * band2_it->second.weight;
        }
        
        // Apply EMA smoothing
        double& ema = stem_band_ema[name];
        ema = ema_alpha * band_energy + (1.0 - ema_alpha) * ema;
        stem_rms[name] = ema;
    }
    
    return stem_rms;
}

- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length {
    if (!tracker || !matcher || length == 0) return @{};

    int fft_size = (int)ring_buf.size(); // 4096

    // Write incoming samples into the circular ring buffer
    for (int i = 0; i < (int)length; ++i) {
        ring_buf[ring_write_pos] = samples[i];
        ring_write_pos = (ring_write_pos + 1) % fft_size;
    }
    samples_since_process += (int)length;

    NSDictionary* lastResult = nil;

    while (samples_since_process >= hop_size) {
        samples_since_process -= hop_size;

        // Linearise the ring buffer
        std::vector<double> linear(fft_size);
        for (int i = 0; i < fft_size; ++i) {
            linear[i] = ring_buf[(ring_write_pos + i) % fft_size];
        }

        // STFT + Chroma
        auto frame = stft->analyze_frame(linear.data(), fft_size, 0.0);
        std::vector<double> mag(frame.magnitude.begin(), frame.magnitude.end());
        auto chroma_arr = chroma_ext->analyze_frame(mag);
        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());

        // Chroma energy for debugging
        double chroma_energy = 0.0;
        for (double v : live_chroma) chroma_energy += v * v;

        // Position tracking
        auto [pos, conf] = tracker->process(live_chroma);

        // --- LIVE RMS: independent spectral band estimate per stem ---
        // This is the KEY FIX: we estimate each stem's energy from
        // its characteristic frequency band, NOT from reference proportions
        std::map<std::string, double> live_rms_at_pos =
            [self estimateStemRMS:mag fftSize:fft_size];

        // --- REFERENCE RMS at current position ---
        std::map<std::string, double> ref_rms_at_pos;
        for (const auto& name : stem_names_cpp) {
            double r = (pos < (int)reference_rms_map[name].size())
                       ? reference_rms_map[name][pos] : 0.0;
            ref_rms_at_pos[name] = r;
        }

        // Compute gains
        matcher->compute_gains(ref_rms_at_pos, live_rms_at_pos, conf);

        // Pack result
        NSMutableDictionary *stemGains = [NSMutableDictionary new];
        NSMutableDictionary *stemLive  = [NSMutableDictionary new];
        NSMutableDictionary *stemRef   = [NSMutableDictionary new];

        for (const auto& name : stem_names_cpp) {
            NSString *nsName = [NSString stringWithUTF8String:name.c_str()];
            stemGains[nsName] = @(matcher->gains[name]);
            stemLive[nsName]  = @(matcher->live_rel_db[name]);
            stemRef[nsName]   = @(matcher->ref_rel_db[name]);
        }

        double ref_time = pos * (hop_size / 44100.0);
        double progress = (double)pos / (double)std::max(1, n_frames_total - 1);

        lastResult = @{
            @"position":        @(pos),
            @"confidence":      @(conf),
            @"overall_gain_db": @(matcher->overall_gain_db),
            @"ref_time":        @(ref_time),
            @"progress":        @(progress),
            @"stemGains":       stemGains,
            @"stemLive":        stemLive,
            @"stemRef":         stemRef,
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
//
//@implementation ASEWrapper {
//    ase::PositionTracker* tracker;
//    ase::RMSMatcher*      matcher;
//    ase::STFT*            stft;
//    ase::ChromaExtractor* chroma_ext;
//
//    std::map<std::string, std::vector<double>> reference_rms_map;
//    std::vector<std::string> stem_names_cpp;
//
//    // Ring buffer — always 4096 samples, filled incrementally
//    std::vector<double> ring_buf;
//    int ring_write_pos;        // next write position (circular)
//    int samples_since_process; // how many new samples since last chroma frame
//
//    int n_frames_total;
//    int hop_size;
//}
//
//- (instancetype)init {
//    self = [super init];
//    if (self) {
//        hop_size = 1024;
//        int fft_size = 4096;
//        int sr = 44100;
//
//        stft        = new ase::STFT(fft_size, hop_size, sr, ase::Window::HANN);
//        chroma_ext  = new ase::ChromaExtractor(fft_size, hop_size, sr);
//
//        ring_buf.assign(fft_size, 0.0);
//        ring_write_pos     = 0;
//        samples_since_process = 0;
//
//        tracker = nullptr;
//        matcher = nullptr;
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
//- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
//                  stemNames:(NSArray<NSString *> *)stemNames
//                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS {
//
//    std::vector<std::vector<double>> ref_chroma;
//    for (NSArray<NSNumber *> *row in chroma) {
//        std::vector<double> c_row;
//        c_row.reserve(row.count);
//        for (NSNumber *val in row) c_row.push_back(val.doubleValue);
//        ref_chroma.push_back(c_row);
//    }
//    n_frames_total = (int)ref_chroma.size();
//    NSLog(@"Loaded %d chroma frames", n_frames_total);
//
//    stem_names_cpp.clear();
//    for (NSString *name in stemNames)
//        stem_names_cpp.push_back(std::string([name UTF8String]));
//
//    reference_rms_map.clear();
//    for (NSString *key in stemRMS) {
//        std::string cpp_key([key UTF8String]);
//        std::vector<double> rms_vals;
//        for (NSNumber *val in stemRMS[key]) rms_vals.push_back(val.doubleValue);
//        reference_rms_map[cpp_key] = rms_vals;
//    }
//
//    if (tracker) { delete tracker; tracker = nullptr; }
//    tracker = new ase::PositionTracker(ref_chroma);
//
//    if (matcher) { delete matcher; matcher = nullptr; }
//    matcher = new ase::RMSMatcher(stem_names_cpp, "drums");
//
//    // Reset ring buffer
//    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
//    ring_write_pos        = 0;
//    samples_since_process = 0;
//
//    NSLog(@"ASEWrapper ready. Stems: %zu", stem_names_cpp.size());
//}
//
//- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
//                                        length:(NSInteger)length {
//    if (!tracker || !matcher || length == 0) return @{};
//
//    int fft_size = (int)ring_buf.size(); // 4096
//
//    // Write incoming samples into the circular ring buffer
//    for (int i = 0; i < (int)length; ++i) {
//        ring_buf[ring_write_pos] = samples[i];
//        ring_write_pos = (ring_write_pos + 1) % fft_size;
//    }
//    samples_since_process += (int)length;
//
//    // Only run chroma analysis every hop_size samples
//    // (process as many hops as have accumulated)
//    NSDictionary* lastResult = nil;
//
//    while (samples_since_process >= hop_size) {
//        samples_since_process -= hop_size;
//
//        // Linearise the ring buffer into a contiguous array for STFT
//        std::vector<double> linear(fft_size);
//        for (int i = 0; i < fft_size; ++i) {
//            linear[i] = ring_buf[(ring_write_pos + i) % fft_size];
//        }
//
//        // STFT + Chroma
//        auto frame      = stft->analyze_frame(linear.data(), fft_size, 0.0);
//        std::vector<double> mag(frame.magnitude.begin(), frame.magnitude.end());
//        auto chroma_arr = chroma_ext->analyze_frame(mag);
//        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());
//
//        // Debug: print chroma energy occasionally
//        double chroma_energy = 0.0;
//        for (double v : live_chroma) chroma_energy += v * v;
//
//        // Position tracking
//        auto [pos, conf] = tracker->process(live_chroma);
//
//        // Live RMS from the current window
//        double total_rms = ase::compute_rms(linear.data(), fft_size);
//
//        std::map<std::string, double> ref_rms_at_pos;
//        double total_ref = 0.0;
//        for (const auto& name : stem_names_cpp) {
//            double r = (pos < (int)reference_rms_map[name].size())
//                       ? reference_rms_map[name][pos] : 0.0;
//            ref_rms_at_pos[name] = r;
//            total_ref += r;
//        }
//
//        std::map<std::string, double> live_rms_at_pos;
//        if (total_ref > 1e-10) {
//            for (const auto& name : stem_names_cpp)
//                live_rms_at_pos[name] = total_rms * (ref_rms_at_pos[name] / total_ref);
//        } else {
//            double share = (stem_names_cpp.empty()) ? 0.0 : total_rms / stem_names_cpp.size();
//            for (const auto& name : stem_names_cpp)
//                live_rms_at_pos[name] = share;
//        }
//
//        matcher->compute_gains(ref_rms_at_pos, live_rms_at_pos, conf);
//
//        // Pack result
//        NSMutableDictionary *stemGains = [NSMutableDictionary new];
//        NSMutableDictionary *stemLive  = [NSMutableDictionary new];
//        NSMutableDictionary *stemRef   = [NSMutableDictionary new];
//
//        for (const auto& name : stem_names_cpp) {
//            NSString *nsName = [NSString stringWithUTF8String:name.c_str()];
//            stemGains[nsName] = @(matcher->gains[name]);
//            stemLive[nsName]  = @(matcher->live_rel_db[name]);
//            stemRef[nsName]   = @(matcher->ref_rel_db[name]);
//        }
//
//        double ref_time = pos * (hop_size / 44100.0);
//        double progress = (double)pos / (double)std::max(1, n_frames_total - 1);
//
//        lastResult = @{
//            @"position":        @(pos),
//            @"confidence":      @(conf),
//            @"overall_gain_db": @(matcher->overall_gain_db),
//            @"ref_time":        @(ref_time),
//            @"progress":        @(progress),
//            @"stemGains":       stemGains,
//            @"stemLive":        stemLive,
//            @"stemRef":         stemRef,
//            @"chroma_energy":   @(chroma_energy),
//            @"rms":             @(total_rms)
//        };
//    }
//
//    return lastResult ?: @{};
//}
//
//@end
