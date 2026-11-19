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
