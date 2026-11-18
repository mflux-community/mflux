#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <stdexcept>
#include "ane_bridge.h"

struct AneModelImpl {
  MLModel* model = nil;
};

AneModel::AneModel(const std::string& path, int T_, int I_, int O_)
    : T(T_), I(I_), O(O_), impl_(new AneModelImpl) {
  @autoreleasepool {
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
    NSError* err = nil;
    if ([url.pathExtension isEqualToString:@"mlpackage"]) {
      url = [MLModel compileModelAtURL:url error:&err];
      if (!url) throw std::runtime_error("compile failed: " + std::string(err.localizedDescription.UTF8String));
    }
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    impl_->model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!impl_->model) throw std::runtime_error("load failed: " + std::string(err.localizedDescription.UTF8String));
  }
}

AneModel::~AneModel() = default;

void AneModel::predict(const void* x, void* out) {
  @try {
  @autoreleasepool {
    NSError* err = nil;
    NSArray<NSNumber*>* xs = @[@(T), @(I)];
    NSArray<NSNumber*>* xst = @[@(I), @1];
    NSArray<NSNumber*>* ys = @[@(T), @(O)];
    NSArray<NSNumber*>* yst = @[@(O), @1];
    // Zero-copy wrappers around MLX's buffers.
    MLMultiArray* xin = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(x)
                                                            shape:xs
                                                         dataType:MLMultiArrayDataTypeFloat16
                                                          strides:xst
                                                      deallocator:nil
                                                            error:&err];
    if (!xin) throw std::runtime_error("input wrap failed");
    MLMultiArray* y_wrap = [[MLMultiArray alloc] initWithDataPointer:out
                                                             shape:ys
                                                          dataType:MLMultiArrayDataTypeFloat16
                                                           strides:yst
                                                       deallocator:nil
                                                             error:&err];
    if (!y_wrap) throw std::runtime_error("output wrap failed");
    MLDictionaryFeatureProvider* in =
        [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x" : [MLFeatureValue featureValueWithMultiArray:xin]} error:&err];
    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = @{@"y" : y_wrap};
    id<MLFeatureProvider> res = [impl_->model predictionFromFeatures:in options:opts error:&err];
    if (!res) throw std::runtime_error("predict failed: " + std::string(err.localizedDescription.UTF8String));
    // Core ML may ignore the backing and return its own buffer; copy back if so.
    MLMultiArray* got = [res featureValueForName:@"y"].multiArrayValue;
    if (got.dataPointer != out) {
      if (got.dataType != MLMultiArrayDataTypeFloat16 || got.shape.count != 2 || got.shape[0].intValue != T ||
          got.shape[1].intValue != O)
        throw std::runtime_error("unexpected output type or shape");
      const long s0 = got.strides[0].longValue, s1 = got.strides[1].longValue;
      const uint16_t* src = static_cast<const uint16_t*>(got.dataPointer);
      uint16_t* dst = static_cast<uint16_t*>(out);
      for (long t = 0; t < T; ++t)
        for (long o = 0; o < O; ++o) dst[t * O + o] = src[t * s0 + o * s1];   // stride-aware copy
    }
  }
  } @catch (NSException* e) {
    throw std::runtime_error(std::string("ObjC exception: ") + e.name.UTF8String + ": " + e.reason.UTF8String);
  }
}
