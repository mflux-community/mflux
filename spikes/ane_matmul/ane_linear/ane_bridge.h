// Core ML bridge: runs one enumerated-shape linear model on the ANE.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

struct AneModelImpl;

class AneModel {
 public:
  // path: compiled .mlmodelc or .mlpackage. W8A16: input "x" fp16 [rows, I]; output "y" fp16 [rows, O].
  // rows must match one of the model's enumerated token counts (T holds the default).
  AneModel(const std::string& path, int T, int I, int O);
  ~AneModel();
  // x and out are fp16, row-major. Both buffers are wrapped without copying.
  void predict(const void* x, void* out, int rows);
  // W8A8 pre form: x is int8 [rows, I], tok is fp32 per-row scales [rows]. Model inputs "x" and "tok".
  void predict_q8(const void* x_i8, const float* tok, void* out, int rows);
  int T, I, O;
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
  std::unique_ptr<AneModelImpl> impl_;
};
