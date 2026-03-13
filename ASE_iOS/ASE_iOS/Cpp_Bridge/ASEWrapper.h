#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/// Objective-C++ bridge exposing the C++ ASE analysis engine to Swift.
///
/// Wraps the C++ PositionTracker, RMSMatcher, STFT, and ChromaExtractor
/// behind an Objective-C interface. Handles sample rate configuration,
/// reference data loading, and per-block audio processing.
@interface ASEWrapper : NSObject

/// Initialize the engine with a specific sample rate.
/// Use the actual hardware rate from AVAudioEngine to ensure correct
/// FFT bin frequencies and timing calculations.
- (instancetype)initWithSampleRate:(double)sampleRate;

/// Reset the position tracker and all internal state to the beginning.
- (void)resetTracker;

/// Load reference data (chroma, stem names, per-stem RMS) into the C++ engine.
/// This must be called after init and before processBlock: to provide the
/// reference fingerprint that the tracker matches against.
///
/// @param chroma    2D array of chroma vectors, shape [n_frames][12]
/// @param stemNames Ordered list of stem names (e.g., ["drums", "bass", "vocals", "other"])
/// @param stemRMS   Dict mapping each stem name to its per-frame RMS array
- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS;

/// Process a block of mono audio samples through the full analysis pipeline.
///
/// Internally accumulates samples in a ring buffer, runs STFT + chroma extraction
/// on each hop-aligned frame, feeds chroma into the position tracker, computes
/// per-stem RMS from frequency bands, and returns a dictionary of results
/// including position, confidence, and per-stem gain advisories.
///
/// @param samples  Pointer to mono Float64 audio samples
/// @param length   Number of samples in the block
/// @return Dictionary of analysis results, or empty dict if not enough data accumulated
- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length;

@end

NS_ASSUME_NONNULL_END


