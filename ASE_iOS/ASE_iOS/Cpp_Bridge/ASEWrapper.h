#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ASEWrapper : NSObject

// 🌟 Expose the new dynamic sample rate initializer to Swift
- (instancetype)initWithSampleRate:(double)sampleRate;

- (void)resetTracker;

- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS;

- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length;

@end

NS_ASSUME_NONNULL_END







//#import <Foundation/Foundation.h>
//#import <AVFoundation/AVFoundation.h>
//
//NS_ASSUME_NONNULL_BEGIN
//
//@interface ASEWrapper : NSObject
//
//- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
//                  stemNames:(NSArray<NSString *> *)stemNames
//                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS;
//
//- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
//                                        length:(NSInteger)length;
//
//- (void)resetTracker;
//
//@end
//
//NS_ASSUME_NONNULL_END
