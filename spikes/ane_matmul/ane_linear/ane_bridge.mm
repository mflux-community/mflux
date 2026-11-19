#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <algorithm>
#include <stdexcept>
#include <vector>
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

void AneModel::predict(const void* x, void* out, int rows) {
  @try {
  @autoreleasepool {
    NSError* err = nil;
    NSArray<NSNumber*>* xs = @[@(rows), @(I)];
    NSArray<NSNumber*>* xst = @[@(I), @1];
    NSArray<NSNumber*>* ys = @[@(rows), @(O)];
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
      copybacks_.fetch_add(1, std::memory_order_relaxed);
      if (got.dataType != MLMultiArrayDataTypeFloat16 || got.shape.count != 2 || got.shape[0].intValue != rows ||
          got.shape[1].intValue != O)
        throw std::runtime_error("unexpected output type or shape");
      const long s0 = got.strides[0].longValue, s1 = got.strides[1].longValue;
      const uint16_t* src = static_cast<const uint16_t*>(got.dataPointer);
      uint16_t* dst = static_cast<uint16_t*>(out);
      for (long t = 0; t < rows; ++t)
        for (long o = 0; o < O; ++o) dst[t * O + o] = src[t * s0 + o * s1];   // stride-aware copy
    }
  }
  } @catch (NSException* e) {
    throw std::runtime_error(std::string("ObjC exception: ") + e.name.UTF8String + ": " + e.reason.UTF8String);
  }
}

void AneModel::predict_q8(const void* x_i8_cf, const float* tok, void* out, int rows) {
  @try {
  @autoreleasepool {
    NSError* err = nil;
    // fp32 row scales converted once per call for the "tok" input (fp16, [1, 1, 1, rows]).
    std::vector<_Float16> h(rows);
    for (int r = 0; r < rows; ++r) h[r] = static_cast<_Float16>(tok[r]);
    // Enumerated int8 inputs must arrive channels-first (1, I, 1, rows): a token-major
    // int8 boundary fails the CoreML plan builder (stage3b/diag_q8plan.py, arms C vs D).
    // int8 arrays cannot wrap an external pointer. The binding already transposed and
    // materialised (I, rows) on MLX's device, so this is a straight memcpy.
    MLMultiArray* xin = [[MLMultiArray alloc] initWithShape:@[@1, @(I), @1, @(rows)]
                                                    dataType:MLMultiArrayDataTypeInt8
                                                       error:&err];
    if (!xin) throw std::runtime_error("int8 input alloc failed");
    memcpy(xin.dataPointer, x_i8_cf, (size_t)rows * I);
    MLMultiArray* tin = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(rows)]
                                                  dataType:MLMultiArrayDataTypeFloat16
                                                     error:&err];
    if (!tin) throw std::runtime_error("tok alloc failed");
    memcpy(tin.dataPointer, h.data(), (size_t)rows * sizeof(_Float16));
    MLMultiArray* y_wrap = [[MLMultiArray alloc] initWithDataPointer:out
                                                             shape:@[@(rows), @(O)]
                                                         dataType:MLMultiArrayDataTypeFloat16
                                                          strides:@[@(O), @1]
                                                      deallocator:nil
                                                            error:&err];
    if (!y_wrap) throw std::runtime_error("output wrap failed");
    MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc]
        initWithDictionary:@{
          @"x" : [MLFeatureValue featureValueWithMultiArray:xin],
          @"tok" : [MLFeatureValue featureValueWithMultiArray:tin],
        }
                       error:&err];
    if (!in) throw std::runtime_error("feature provider failed");
    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = @{@"y" : y_wrap};
    id<MLFeatureProvider> res = [impl_->model predictionFromFeatures:in options:opts error:&err];
    if (!res) throw std::runtime_error("predict_q8 failed: " + std::string(err.localizedDescription.UTF8String));
    MLMultiArray* got = [res featureValueForName:@"y"].multiArrayValue;
    if (got.dataPointer != out) {
      copybacks_.fetch_add(1, std::memory_order_relaxed);
      if (got.dataType != MLMultiArrayDataTypeFloat16 || got.shape.count != 2 || got.shape[0].intValue != rows ||
          got.shape[1].intValue != O)
        throw std::runtime_error("unexpected q8 output type or shape");
      const long s0 = got.strides[0].longValue, s1 = got.strides[1].longValue;
      const uint16_t* src = static_cast<const uint16_t*>(got.dataPointer);
      uint16_t* dst = static_cast<uint16_t*>(out);
      for (long t = 0; t < rows; ++t)
        for (long o = 0; o < O; ++o) dst[t * O + o] = src[t * s0 + o * s1];
    }
  }
  } @catch (NSException* e) {
    throw std::runtime_error(std::string("ObjC exception: ") + e.name.UTF8String + ": " + e.reason.UTF8String);
  }
}
