// Core ML bridge: runs one fixed-shape fp16 linear model on the ANE.
#pragma once
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

struct AneModelImpl;

class AneModel {
 public:
  // path: compiled .mlmodelc or .mlpackage. Input "x" fp16 [T, I]; output "y" fp16 [T, O].
  AneModel(const std::string& path, int T, int I, int O);
  ~AneModel();
  // x and out are fp16, row-major. Both buffers are wrapped without copying.
  void predict(const void* x, void* out);
  int T, I, O;
  // A failed prediction fills the output with NaN and records the message. A later good call clears it.
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
