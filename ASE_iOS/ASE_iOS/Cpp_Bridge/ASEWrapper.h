#import <Foundation/Foundation.h>
#import <AVFoundation/AVFoundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ASEWrapper : NSObject

- (void)loadReferenceChroma:(NSArray<NSArray<NSNumber *> *> *)chroma
                  stemNames:(NSArray<NSString *> *)stemNames
                    stemRMS:(NSDictionary<NSString *, NSArray<NSNumber *> *> *)stemRMS;

// Return type is explicit: NSDictionary with NSString keys and id values
- (NSDictionary<NSString *, id> *)processBlock:(const double *)samples
                                        length:(NSInteger)length;

@end

NS_ASSUME_NONNULL_END
