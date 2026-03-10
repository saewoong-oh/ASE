#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ASEWrapper : NSObject

- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS;

/// Reset the position tracker to frame 0 without reloading reference data.
/// Call this before restarting file playback so tracking starts fresh.
- (void)resetTracker;

- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length;

@end

NS_ASSUME_NONNULL_END
