// Core ML bridge: runs one enumerated-shape linear model on the ANE.
#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct AneModelImpl;

class AneModel {
 public:
  // path: compiled .mlmodelc or .mlpackage. W8A16: input "x" fp16 [rows, I]; output "y" fp16 [rows, O].
  // rows must match one of the model's enumerated token counts (T holds the default).
  AneModel(const std::string& path, int T, int I, int O);
  ~AneModel();
  // x and out are fp16, row-major. Both buffers are wrapped without copying.
  void predict(const void* x, void* out, int rows);
  // W8A8 pre form: x is int8 laid out channels-first (I, rows) contiguous, tok is fp32 per-row scales [rows].
  // Model inputs "x" (1, I, 1, rows) and "tok" (1, 1, 1, rows); the binding transposes.
  void predict_q8(const void* x_i8, const float* tok, void* out, int rows);
  int T, I, O;
  // Predictions whose outputBackings were ignored and needed a copy. Nonzero means
  // the model silently rejected the backing; timing with such a model pays the copy.
  long copybacks() const { return copybacks_.load(); }
  // Fused multi-input/multi-output block models (3b-2 wired-reduction probe).
  // fp16 inputs arrive as MLX's (rows, I) row-major buffer; int8 inputs as the
  // binding's channels-first (I, rows) materialisation, each paired by name with
  // its tok input. All outputs are fp16 (rows, cols[j]).
  void set_multi(const std::vector<std::string>& in_names, const std::vector<int>& in_Is,
                 const std::vector<int>& in_int8, const std::vector<std::string>& tok_names,
                 const std::vector<std::string>& out_names);
  void predict_multi(const std::vector<const void*>& in_ptrs, const std::vector<const float*>& tok_ptrs,
                     const std::vector<void*>& out_ptrs, const std::vector<int>& out_cols, int rows);
  size_t multi_in_count() const { return in_names_.size(); }
  // A failed prediction fills the output with NaN, records the message, and rethrows.
  void set_error(const std::string& e) {
    std::lock_guard<std::mutex> g(mu_);
    error_ = e;
  }
  void clear_error() {
    std::lock_guard<std::mutex> g(mu_);
    error_.clear();
  }
  std::string last_error() {
    std::lock_guard<std::mutex> g(mu_);
    return error_;
  }

 private:
  std::mutex mu_;
  std::string error_;
  std::atomic<long> copybacks_{0};
  std::vector<std::string> in_names_, tok_names_, out_names_;
  std::vector<int> in_Is_, in_int8_;
  std::unique_ptr<AneModelImpl> impl_;
};

// Weightless program model (Stage 3c): one program per (Mpad, N, K), int8
// weights and activations as graph inputs, constant 1/sqrt(K) dequantise
// inside, real scales applied in the MLX epilogue. Design follows
// ~/zCode_26_11/drawthings-ane-research/DRAWTHINGS-ANE-FINDINGS.md
// (liuliu/ccv, BSD-3-Clause); written from that description, no source copied.
// All buffers wrap MLX memory zero-copy, including int8.
// Stage 3d: one fused weightless program for a whole SwiGLU MLP (W8A16). The program has no
// baked weights, so it declares N graph inputs of arbitrary rank and dtype; all of them wrap
// caller-owned host buffers zero-copy, and the single output wraps a caller-owned fp16 buffer.
struct WLTensor {
  std::string name;
  std::vector<int64_t> shape;  // row-major, contiguous
  bool int8 = false;           // else fp16
};

class WLModel {
 public:
  // path: .mlpackage. compiled_dir: optional persistent .mlmodelc cache,
  // created on first load and reused after.
  WLModel(const std::string& path, const std::string& compiled_dir, int Mpad, int N, int K);
  ~WLModel();
  // x: int8 contiguous [K, Mpad] (tokens last); w: int8 [N, K]; out: fp16 [N, Mpad].
  void predict(const void* x, const void* w, void* out);
  // Fused form (3d): register the program's inputs and outputs once, then call with the payload
  // pointers. in_ptrs parallels the inputs passed to set_fused, out_ptrs parallels the outputs.
  // >= 1 output is supported: the a+b range fix returns the SwiGLU row max as a second output so
  // the fp32 GPU epilogue can restore it -- multiplying it back inside the program overflows fp16.
  void set_fused(const std::vector<WLTensor>& ins, const std::vector<WLTensor>& outs);
  void predict_fused(const std::vector<const void*>& in_ptrs, const std::vector<void*>& out_ptrs);
  // Needed by the MLX primitive: output shapes must be knowable when the graph is built, before
  // any call happens, because mx.compile cannot ask the model at eval time.
  size_t fused_out_count() const { return fused_outs_.size(); }
  std::vector<int64_t> fused_out_shape(size_t j) const { return fused_outs_[j].shape; }
  int Mpad, N, K;
  // One reused host buffer for the ANE's x rows. Core ML wires the pages of whatever buffer it
  // wraps, so wrapping a fresh MLX allocation per block wired ~31 MB x 34 and paged the run.
  // Calls are serial on the ANE stream, so one buffer per model is safe.
  std::vector<uint8_t> x_stage;
  std::string last_error() {
    std::lock_guard<std::mutex> g(mu_);
    return error_;
  }
  void clear_error() {
    std::lock_guard<std::mutex> g(mu_);
    error_.clear();
  }
  void set_error(const std::string& e) {
    std::lock_guard<std::mutex> g(mu_);
    error_ = e;
  }
  long copybacks() const { return copybacks_.load(); }

 private:
  std::mutex mu_;
  std::string error_;
  std::atomic<long> copybacks_{0};
  std::vector<WLTensor> fused_ins_;
  std::vector<WLTensor> fused_outs_;
  std::unique_ptr<AneModelImpl> impl_;
};
