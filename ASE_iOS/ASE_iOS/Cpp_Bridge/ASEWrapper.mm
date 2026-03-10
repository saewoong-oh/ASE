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

static std::map<std::string, StemBandDef> STEM_BANDS = {
    {"drums",  {60.0,   250.0,  1.0}},
    {"bass",   {40.0,   300.0,  1.0}},
    {"vocals", {250.0,  3500.0, 1.0}},
    {"other",  {500.0,  8000.0, 1.0}},
    {"guitar", {150.0,  5000.0, 1.0}},
    {"piano",  {120.0,  4000.0, 1.0}},
    {"keys",   {120.0,  4000.0, 1.0}},
};

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

    std::vector<double> ring_buf;
    int ring_write_pos;
    int samples_since_process;

    int n_frames_total;
    int hop_size;
    int sample_rate;

    std::map<std::string, double> stem_band_ema;
    double ema_alpha;
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
        ema_alpha             = 0.15;

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

    stem_band_ema.clear();
    for (const auto& name : stem_names_cpp) {
        stem_band_ema[name] = 0.0;
    }

    if (tracker) { delete tracker; tracker = nullptr; }
    tracker = new ase::PositionTracker(ref_chroma);

    if (matcher) { delete matcher; matcher = nullptr; }
    matcher = new ase::RMSMatcher(stem_names_cpp, "drums");

    [self resetEngineState];

    NSLog(@"ASEWrapper ready. Stems: %zu", stem_names_cpp.size());
}

- (void)resetTracker {
    if (tracker) {
        tracker->reset(0);
    }
    [self resetEngineState];
}

- (void)resetEngineState {
    std::fill(ring_buf.begin(), ring_buf.end(), 0.0);
    ring_write_pos        = 0;
    samples_since_process = 0;
    for (auto& kv : stem_band_ema) {
        kv.second = 0.0;
    }
}

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
    for (int k = low_bin; k <= high_bin; ++k)
        energy += magnitude[k] * magnitude[k];
    int bin_count = high_bin - low_bin + 1;
    return std::sqrt(energy / (double)bin_count);
}

- (std::map<std::string, double>)estimateStemRMS:(const std::vector<double>&)magnitude
                                         fftSize:(int)fft_size {
    std::map<std::string, double> stem_rms;
    for (const auto& name : stem_names_cpp) {
        double band_energy = 0.0;
        auto band_it = STEM_BANDS.find(name);
        if (band_it != STEM_BANDS.end()) {
            band_energy += [self computeBandEnergy:magnitude
                                           lowHz:band_it->second.low_hz
                                          highHz:band_it->second.high_hz
                                         fftSize:fft_size
                                      sampleRate:sample_rate] * band_it->second.weight;
        } else {
            band_energy += [self computeBandEnergy:magnitude
                                           lowHz:100.0
                                          highHz:8000.0
                                         fftSize:fft_size
                                      sampleRate:sample_rate];
        }
        auto band2_it = STEM_BANDS_2.find(name);
        if (band2_it != STEM_BANDS_2.end()) {
            band_energy += [self computeBandEnergy:magnitude
                                           lowHz:band2_it->second.low_hz
                                          highHz:band2_it->second.high_hz
                                         fftSize:fft_size
                                      sampleRate:sample_rate] * band2_it->second.weight;
        }
        double& ema = stem_band_ema[name];
        ema = ema_alpha * band_energy + (1.0 - ema_alpha) * ema;
        stem_rms[name] = ema;
    }
    return stem_rms;
}

- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length {
    if (!tracker || !matcher || length == 0) return @{};

    int fft_size = (int)ring_buf.size();

    for (int i = 0; i < (int)length; ++i) {
        ring_buf[ring_write_pos] = samples[i];
        ring_write_pos = (ring_write_pos + 1) % fft_size;
    }
    samples_since_process += (int)length;

    NSDictionary* lastResult = nil;

    while (samples_since_process >= hop_size) {
        samples_since_process -= hop_size;

        std::vector<double> linear(fft_size);
        for (int i = 0; i < fft_size; ++i)
            linear[i] = ring_buf[(ring_write_pos + i) % fft_size];

        auto frame = stft->analyze_frame(linear.data(), fft_size, 0.0);
        std::vector<double> mag(frame.magnitude.begin(), frame.magnitude.end());
        auto chroma_arr = chroma_ext->analyze_frame(mag);
        std::vector<double> live_chroma(chroma_arr.begin(), chroma_arr.end());

        double chroma_energy = 0.0;
        for (double v : live_chroma) chroma_energy += v * v;

        auto [pos, conf] = tracker->process(live_chroma);

        std::map<std::string, double> live_rms_at_pos =
            [self estimateStemRMS:mag fftSize:fft_size];

        std::map<std::string, double> ref_rms_at_pos;
        for (const auto& name : stem_names_cpp) {
            double r = (pos < (int)reference_rms_map[name].size())
                       ? reference_rms_map[name][pos] : 0.0;
            ref_rms_at_pos[name] = r;
        }

        matcher->compute_gains(ref_rms_at_pos, live_rms_at_pos, conf);

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
