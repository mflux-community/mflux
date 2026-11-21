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

void AneModel::set_multi(const std::vector<std::string>& in_names, const std::vector<int>& in_Is,
                         const std::vector<int>& in_int8, const std::vector<std::string>& tok_names,
                         const std::vector<std::string>& out_names) {
  in_names_ = in_names;
  in_Is_ = in_Is;
  in_int8_ = in_int8;
  tok_names_ = tok_names;
  out_names_ = out_names;
}

void AneModel::predict_multi(const std::vector<const void*>& in_ptrs, const std::vector<const float*>& tok_ptrs,
                             const std::vector<void*>& out_ptrs, const std::vector<int>& out_cols, int rows) {
  @try {
  @autoreleasepool {
    NSError* err = nil;
    NSMutableDictionary* d = [NSMutableDictionary dictionary];
    for (size_t k = 0; k < in_names_.size(); ++k) {
      NSString* nm = [NSString stringWithUTF8String:in_names_[k].c_str()];
      const int Ik = in_Is_[k];
      if (!in_int8_[k]) {
        MLMultiArray* a = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(in_ptrs[k])
                                                              shape:@[@(rows), @(Ik)]
                                                           dataType:MLMultiArrayDataTypeFloat16
                                                            strides:@[@(Ik), @1]
                                                        deallocator:nil
                                                              error:&err];
        if (!a) throw std::runtime_error("multi input wrap failed: " + in_names_[k]);
        d[nm] = [MLFeatureValue featureValueWithMultiArray:a];
      } else {
        MLMultiArray* a = [[MLMultiArray alloc] initWithShape:@[@1, @(Ik), @1, @(rows)]
                                                      dataType:MLMultiArrayDataTypeInt8
                                                         error:&err];
        if (!a) throw std::runtime_error("multi int8 alloc failed: " + in_names_[k]);
        memcpy(a.dataPointer, in_ptrs[k], (size_t)rows * Ik);
        d[nm] = [MLFeatureValue featureValueWithMultiArray:a];
        NSString* tn = [NSString stringWithUTF8String:tok_names_[k].c_str()];
        MLMultiArray* t = [[MLMultiArray alloc] initWithShape:@[@1, @1, @1, @(rows)]
                                                     dataType:MLMultiArrayDataTypeFloat16
                                                        error:&err];
        if (!t) throw std::runtime_error("multi tok alloc failed: " + tok_names_[k]);
        _Float16* th = static_cast<_Float16*>(t.dataPointer);
        const float* sf = tok_ptrs[k];
        for (int r = 0; r < rows; ++r) th[r] = static_cast<_Float16>(sf[r]);
        d[tn] = [MLFeatureValue featureValueWithMultiArray:t];
      }
    }
    MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:d error:&err];
    if (!in) throw std::runtime_error("multi provider failed");
    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    NSMutableDictionary* bk = [NSMutableDictionary dictionary];
    for (size_t j = 0; j < out_names_.size(); ++j) {
      MLMultiArray* y = [[MLMultiArray alloc] initWithDataPointer:out_ptrs[j]
                                                            shape:@[@(rows), @(out_cols[j])]
                                                         dataType:MLMultiArrayDataTypeFloat16
                                                          strides:@[@(out_cols[j]), @1]
                                                      deallocator:nil
                                                            error:&err];
      if (!y) throw std::runtime_error("multi output wrap failed: " + out_names_[j]);
      bk[[NSString stringWithUTF8String:out_names_[j].c_str()]] = y;
    }
    opts.outputBackings = bk;
    id<MLFeatureProvider> res = [impl_->model predictionFromFeatures:in options:opts error:&err];
    if (!res) throw std::runtime_error("predict_multi failed: " + std::string(err.localizedDescription.UTF8String));
    for (size_t j = 0; j < out_names_.size(); ++j) {
      NSString* nm = [NSString stringWithUTF8String:out_names_[j].c_str()];
      MLMultiArray* got = [res featureValueForName:nm].multiArrayValue;
      void* op = out_ptrs[j];
      if (got.dataPointer != op) {
        copybacks_.fetch_add(1, std::memory_order_relaxed);
        const int C = out_cols[j];
        if (got.dataType != MLMultiArrayDataTypeFloat16 || got.shape.count != 2 || got.shape[0].intValue != rows ||
            got.shape[1].intValue != C)
          throw std::runtime_error("unexpected multi output type or shape: " + out_names_[j]);
        const long s0 = got.strides[0].longValue, s1 = got.strides[1].longValue;
        const uint16_t* src = static_cast<const uint16_t*>(got.dataPointer);
        uint16_t* dst = static_cast<uint16_t*>(op);
        for (int t = 0; t < rows; ++t)
          for (int c = 0; c < C; ++c) dst[(size_t)t * C + c] = src[(size_t)t * s0 + c * s1];
      }
    }
  }
  } @catch (NSException* e) {
    throw std::runtime_error(std::string("ObjC exception: ") + e.name.UTF8String + ": " + e.reason.UTF8String);
  }
}

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

// ---- WLModel: weightless programs (Stage 3c) ----
// Design follows ~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md
// (liuliu/ccv, BSD-3-Clause); written from that description, no source copied.

WLModel::WLModel(const std::string& path, const std::string& compiled_dir, int Mpad_, int N_, int K_)
    : Mpad(Mpad_), N(N_), K(K_), impl_(new AneModelImpl) {
  @autoreleasepool {
    NSError* err = nil;
    NSURL* url = nil;
    NSString* cached = [NSString stringWithUTF8String:compiled_dir.c_str()];
    if ([[NSFileManager defaultManager] fileExistsAtPath:[cached stringByAppendingPathComponent:@"model.mil"]]) {
      url = [NSURL fileURLWithPath:cached];  // persistent compile cache hit
    } else {
      NSURL* src = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
      url = [MLModel compileModelAtURL:src error:&err];
      if (!url) throw std::runtime_error("wl compile failed: " + std::string(err.localizedDescription.UTF8String));
      if (compiled_dir.size()) {
        [[NSFileManager defaultManager] removeItemAtURL:[NSURL fileURLWithPath:cached] error:nil];
        [[NSFileManager defaultManager] copyItemAtURL:url toURL:[NSURL fileURLWithPath:cached] error:&err];
        if (!err) url = [NSURL fileURLWithPath:cached];  // load from the fresh copy
      }
    }
    MLModelConfiguration* cfg = [[MLModelConfiguration alloc] init];
    cfg.computeUnits = MLComputeUnitsCPUAndNeuralEngine;
    impl_->model = [MLModel modelWithContentsOfURL:url configuration:cfg error:&err];
    if (!impl_->model) throw std::runtime_error("wl load failed: " + std::string(err.localizedDescription.UTF8String));
  }
}

WLModel::~WLModel() = default;

void WLModel::predict(const void* x, const void* w, void* out) {
  @try {
  @autoreleasepool {
    NSError* err = nil;
    MLMultiArray* xa = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(x)
                                                shape:@[@1, @1, @(K), @(Mpad)]
                                             dataType:MLMultiArrayDataTypeInt8
                                              strides:@[@(K * Mpad), @(K * Mpad), @(Mpad), @1]
                                          deallocator:nil error:&err];
    if (!xa) throw std::runtime_error("wl x wrap failed: " + std::string(err.localizedDescription.UTF8String));
    MLMultiArray* wa = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(w)
                                                shape:@[@1, @1, @(N), @(K)]
                                             dataType:MLMultiArrayDataTypeInt8
                                              strides:@[@(N * K), @(N * K), @(K), @1]
                                          deallocator:nil error:&err];
    if (!wa) throw std::runtime_error("wl w wrap failed: " + std::string(err.localizedDescription.UTF8String));
    MLMultiArray* ya = [[MLMultiArray alloc] initWithDataPointer:out
                                                shape:@[@1, @1, @(N), @(Mpad)]
                                             dataType:MLMultiArrayDataTypeFloat16
                                              strides:@[@(N * Mpad), @(N * Mpad), @(Mpad), @1]
                                          deallocator:nil error:&err];
    if (!ya) throw std::runtime_error("wl y wrap failed: " + std::string(err.localizedDescription.UTF8String));
    MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:@{
      @"x" : [MLFeatureValue featureValueWithMultiArray:xa],
      @"w" : [MLFeatureValue featureValueWithMultiArray:wa],
    } error:&err];
    if (!in) throw std::runtime_error("wl provider failed: " + std::string(err.localizedDescription.UTF8String));
    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = @{@"y" : ya};
    id<MLFeatureProvider> res = [impl_->model predictionFromFeatures:in options:opts error:&err];
    if (!res) throw std::runtime_error("wl predict failed: " + std::string(err.localizedDescription.UTF8String));
    MLMultiArray* got = [res featureValueForName:@"y"].multiArrayValue;
    if (got.dataPointer != out) {  // backing ignored: copy back so the epilogue sees the values
      copybacks_.fetch_add(1, std::memory_order_relaxed);
      memcpy(out, got.dataPointer, (size_t)N * Mpad * 2);
    }
  }
  } @catch (NSException* e) {
    throw std::runtime_error(std::string("ObjC exception: ") + e.name.UTF8String + ": " + e.reason.UTF8String);
  }
}

// ---- WLModel fused form: Stage 3d row-split MLP ----

static NSMutableArray* shape_array(const std::vector<int64_t>& shape) {
  NSMutableArray* shp = [NSMutableArray arrayWithCapacity:shape.size()];
  for (int64_t v : shape) [shp addObject:@(v)];
  return shp;
}

static NSArray<NSNumber*>* row_major_strides(const std::vector<int64_t>& shape) {
  NSMutableArray* st = [NSMutableArray arrayWithCapacity:shape.size()];
  long run = 1;
  for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
    [st insertObject:@(run) atIndex:0];
    run *= (*it > 0 ? *it : 1);
  }
  return st;
}

// Copy a returned MLMultiArray into a caller buffer, honouring whatever strides Core ML chose.
static void copy_strided(MLMultiArray* got, void* dst) {
  const int rank = (int)got.shape.count;
  std::vector<long> gs(rank), es(rank);
  long total = 1;
  for (int i = rank - 1; i >= 0; --i) {
    gs[i] = total;  // row-major destination
    total *= got.shape[i].longValue;
  }
  for (int i = 0; i < rank; ++i) es[i] = got.strides[i].longValue;
  const uint16_t* src = static_cast<const uint16_t*>(got.dataPointer);
  uint16_t* d = static_cast<uint16_t*>(dst);
  std::vector<int> idx(rank, 0);
  for (long n = 0; n < total; ++n) {  // odometer, in case Core ML returns permuted strides
    long so = 0, dd = 0;
    for (int i = 0; i < rank; ++i) {
      so += (long)idx[i] * es[i];
      dd += (long)idx[i] * gs[i];
    }
    d[dd] = src[so];
    for (int i = rank - 1; i >= 0; --i) {
      if (++idx[i] < got.shape[i].intValue) break;
      idx[i] = 0;
    }
  }
}

void WLModel::set_fused(const std::vector<WLTensor>& ins, const std::vector<WLTensor>& outs) {
  fused_ins_ = ins;
  fused_outs_ = outs;
}

void WLModel::predict_fused(const std::vector<const void*>& in_ptrs, const std::vector<void*>& out_ptrs) {
  @try {
  @autoreleasepool {
    NSError* err = nil;
    if (in_ptrs.size() != fused_ins_.size())
      throw std::runtime_error("wl fused: input count mismatch");
    if (out_ptrs.size() != fused_outs_.size())
      throw std::runtime_error("wl fused: output count mismatch");
    NSMutableDictionary* d = [NSMutableDictionary dictionary];
    for (size_t k = 0; k < fused_ins_.size(); ++k) {
      const WLTensor& t = fused_ins_[k];
      MLMultiArray* a = [[MLMultiArray alloc] initWithDataPointer:const_cast<void*>(in_ptrs[k])
                                                            shape:shape_array(t.shape)
                                                         dataType:(t.int8 ? MLMultiArrayDataTypeInt8
                                                                           : MLMultiArrayDataTypeFloat16)
                                                          strides:row_major_strides(t.shape)
                                                      deallocator:nil
                                                            error:&err];
      if (!a)
        throw std::runtime_error("wl fused input wrap failed: " + t.name + ": " +
                                 (err.localizedDescription ? err.localizedDescription.UTF8String : "?"));
      d[[NSString stringWithUTF8String:t.name.c_str()]] = [MLFeatureValue featureValueWithMultiArray:a];
    }
    NSMutableDictionary* backings = [NSMutableDictionary dictionary];
    std::vector<NSString*> out_names;
    for (size_t j = 0; j < fused_outs_.size(); ++j) {
      const WLTensor& t = fused_outs_[j];
      NSString* nm = [NSString stringWithUTF8String:t.name.c_str()];
      out_names.push_back(nm);
      MLMultiArray* ya = [[MLMultiArray alloc] initWithDataPointer:out_ptrs[j]
                                                             shape:shape_array(t.shape)
                                                          dataType:(t.int8 ? MLMultiArrayDataTypeInt8
                                                                             : MLMultiArrayDataTypeFloat16)
                                                           strides:row_major_strides(t.shape)
                                                       deallocator:nil
                                                             error:&err];
      if (!ya)
        throw std::runtime_error(std::string("wl fused output wrap failed: ") + t.name + ": " +
                                 (err.localizedDescription ? err.localizedDescription.UTF8String : "?"));
      backings[nm] = ya;
    }
    MLDictionaryFeatureProvider* in = [[MLDictionaryFeatureProvider alloc] initWithDictionary:d error:&err];
    if (!in) throw std::runtime_error("wl fused provider failed");
    MLPredictionOptions* opts = [[MLPredictionOptions alloc] init];
    opts.outputBackings = backings;
    id<MLFeatureProvider> res = [impl_->model predictionFromFeatures:in options:opts error:&err];
    if (!res)
      throw std::runtime_error("wl fused predict failed: " +
                               std::string(err.localizedDescription.UTF8String));
    for (size_t j = 0; j < fused_outs_.size(); ++j) {
      MLMultiArray* got = [res featureValueForName:out_names[j]].multiArrayValue;
      if (!got) throw std::runtime_error(std::string("wl fused: no output named ") + fused_outs_[j].name);
      if (got.dataPointer != out_ptrs[j]) {  // backing ignored: the caller would read a stale buffer
        copybacks_.fetch_add(1, std::memory_order_relaxed);
        copy_strided(got, out_ptrs[j]);
      }
    }
  }
  } @catch (NSException* e) {
    throw std::runtime_error(std::string("ObjC exception: ") + e.name.UTF8String + ": " + e.reason.UTF8String);
  }
}
