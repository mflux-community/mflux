// Core ML bridge: runs one fixed-shape fp16 linear model on the ANE.
#pragma once
#include <cstdint>
#include <memory>
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
  // Set when a prediction fails on the worker thread. Output is then filled with NaN.
  std::string last_error;

 private:
  std::unique_ptr<AneModelImpl> impl_;
};
