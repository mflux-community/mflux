#include <algorithm>
#include <cstdio>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include "mlx/mlx.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/primitives.h"
#include "ane_bridge.h"

namespace mx = mlx::core;
namespace nb = nanobind;

class AneLinear : public mx::Primitive {
 public:
  AneLinear(mx::Stream s, std::shared_ptr<AneModel> m, bool q8) : mx::Primitive(s), model_(std::move(m)), q8_(q8) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& x = inputs[0];
    auto& out = outputs[0];
    // W8A8 input 0 is the channels-first (I, rows) buffer built by the binding.
    const int rows = q8_ ? x.shape(1) : x.shape(0);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& enc = mx::cpu::get_command_encoder(stream());
    enc.set_input_array(x);
    enc.set_output_array(out);
    auto model = model_;
    const float* tokp = nullptr;
    if (q8_) {
      enc.set_input_array(inputs[1]);
      tokp = inputs[1].data<float>();
    }
    // x and out are kept alive by MLX until the task completes (same pattern as MLX CPU primitives).
    enc.dispatch([model, q8 = q8_, rows, xp = x.data<void>(), tokp, op = out.data<void>()]() {
      // Fail loud: record the error, fill the output with NaN so an unobserved
      // buffer cannot pass as valid data, then rethrow. The scheduler stores
      // the exception and raises it on the next main-thread enqueue (mx.eval).
      try {
        if (q8)
          model->predict_q8(xp, tokp, op, rows);
        else
          model->predict(xp, op, rows);
        model->clear_error();
      } catch (const std::exception& e) {
        model->set_error(e.what());
        auto* h = static_cast<uint16_t*>(op);
        std::fill(h, h + (size_t)rows * model->O, uint16_t(0x7E00));   // fp16 NaN: the failure is visible
        throw;
      }
    });
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("ane_linear must run on the CPU stream");
  }
  const char* name() const override { return "AneLinear"; }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const AneLinear&>(other);
    return model_.get() == o.model_.get() && q8_ == o.q8_;
  }
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    const int rows = q8_ ? inputs[0].shape(1) : inputs[0].shape(0);
    return {{rows, model_->O}};
  }

 private:
  std::shared_ptr<AneModel> model_;
  bool q8_;
};

// One fused block model with k inputs and m outputs (3b-2 wired-reduction probe).
// Inputs: fp16 arms pass (rows, I) arrays; q8 arms pass q1..qk int8 [rows, I] then
// tok1..tokk float32 [rows] (the binding transposes each q to channels-first).
class AneMulti : public mx::Primitive {
 public:
  AneMulti(mx::Stream s, std::shared_ptr<AneModel> m, bool q8, std::vector<int> cols, int n_in)
      : mx::Primitive(s), model_(std::move(m)), q8_(q8), cols_(std::move(cols)), n_in_(n_in) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    if (getenv("ANE_DEBUG"))
      fprintf(stderr, "[anemulti] inputs=%zu outputs=%zu cols=%zu n_in=%d q8=%d\n", inputs.size(),
              outputs.size(), cols_.size(), n_in_, (int)q8_);
    const int rows = q8_ ? inputs[0].shape(1) : inputs[0].shape(0);
    std::vector<const void*> iptrs;
    std::vector<const float*> tptrs;
    std::vector<void*> optrs;
    auto& enc = mx::cpu::get_command_encoder(stream());
    for (int k = 0; k < n_in_; ++k) {
      enc.set_input_array(inputs[k]);
      iptrs.push_back(inputs[k].data<void>());
    }
    if (q8_) {
      for (int k = 0; k < n_in_; ++k) {
        enc.set_input_array(inputs[n_in_ + k]);
        tptrs.push_back(inputs[n_in_ + k].data<float>());
      }
    }
    for (auto& out : outputs) {
      out.set_data(mx::allocator::malloc(out.nbytes()));
      enc.set_output_array(out);
      optrs.push_back(out.data<void>());
    }
    auto model = model_;
    auto cols = cols_;
    enc.dispatch([model, q8 = q8_, rows, iptrs, tptrs, optrs, cols]() {
      try {
        if (!q8) {
          std::vector<const float*> no_toks;
          model->predict_multi(iptrs, no_toks, optrs, cols, rows);
        } else {
          model->predict_multi(iptrs, tptrs, optrs, cols, rows);
        }
        model->clear_error();
      } catch (const std::exception& e) {
        model->set_error(e.what());
        for (size_t j = 0; j < optrs.size(); ++j) {
          auto* h = static_cast<uint16_t*>(optrs[j]);
          std::fill(h, h + (size_t)rows * cols[j], uint16_t(0x7E00));
        }
        throw;
      }
    });
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("ane_multi must run on the CPU stream");
  }
  const char* name() const override { return "AneMulti"; }
  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& o = static_cast<const AneMulti&>(other);
    return model_.get() == o.model_.get() && q8_ == o.q8_ && cols_ == o.cols_;
  }
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    const int rows = q8_ ? inputs[0].shape(1) : inputs[0].shape(0);
    std::vector<mx::Shape> shapes;
    for (int c : cols_) shapes.push_back({rows, c});
    return shapes;
  }

 private:
  std::shared_ptr<AneModel> model_;
  bool q8_;
  std::vector<int> cols_;
  int n_in_;
};

NB_MODULE(_ane_linear, m) {
  nb::class_<AneModel>(m, "AneModel")   // held by shared_ptr: no GIL needed when MLX frees it
      .def(nb::init<const std::string&, int, int, int>(), nb::arg("path"), nb::arg("tokens"), nb::arg("in_features"), nb::arg("out_features"))
      .def_ro("tokens", &AneModel::T)
      .def_ro("in_features", &AneModel::I)
      .def_ro("out_features", &AneModel::O)
      .def("last_error", &AneModel::last_error)
      .def("copybacks", &AneModel::copybacks);
  // ane_linear(x[rows, I] fp16, model) -> y[rows, O] fp16. rows must match a model tier.
  m.def(
      "ane_linear",
      [](const mx::array& x, std::shared_ptr<AneModel> model) {
        if (x.dtype() != mx::float16 || x.ndim() != 2 || x.shape(1) != model->I)
          throw std::invalid_argument("ane_linear: x must be fp16 [rows, in_features]");
        if (model->T > 0 && x.shape(0) > model->T)
          throw std::invalid_argument("ane_linear: rows exceeds the model's largest tier");
        auto xc = mx::contiguous(x);
        return mx::array(
            {x.shape(0), model->O}, mx::float16,
            std::make_shared<AneLinear>(mx::default_stream(mx::Device::cpu), model, false), {xc});
      },
      nb::arg("x"), nb::arg("model"));
  // ane_linear_q8(x int8 [rows, I], tok float32 [rows], model) -> y[rows, O] fp16 (W8A8 pre form).
  // The channels-first transpose the enumerated int8 boundary requires runs as an MLX
  // op (materialised on the MLX device), so the bridge only ever memcpy-copies.
  m.def(
      "ane_linear_q8",
      [](const mx::array& x, const mx::array& tok, std::shared_ptr<AneModel> model) {
        if (x.dtype() != mx::int8 || x.ndim() != 2 || x.shape(1) != model->I)
          throw std::invalid_argument("ane_linear_q8: x must be int8 [rows, in_features]");
        if (tok.dtype() != mx::float32 || tok.ndim() != 1 || tok.shape(0) != x.shape(0))
          throw std::invalid_argument("ane_linear_q8: tok must be float32 [rows]");
        const int rows = x.shape(0);
        if (model->T > 0 && rows > model->T)
          throw std::invalid_argument("ane_linear_q8: rows exceeds the model's largest tier");
        auto xt = mx::contiguous(mx::transpose(x, {1, 0}));  // (I, rows), contiguous int8
        auto tc = mx::contiguous(tok);
        return mx::array(
            {rows, model->O}, mx::float16,
            std::make_shared<AneLinear>(mx::default_stream(mx::Device::cpu), model, true), {xt, tc});
      },
      nb::arg("x"), nb::arg("tok"), nb::arg("model"));

  // make_multi_model(path, max_rows, in_names, in_Is, in_int8, tok_names, out_names)
  m.def(
      "make_multi_model",
      [](const std::string& path, int max_rows, const std::vector<std::string>& in_names,
         const std::vector<int>& in_Is, const std::vector<int>& in_int8,
         const std::vector<std::string>& tok_names, const std::vector<std::string>& out_names) {
        auto model = std::make_shared<AneModel>(path, max_rows, 0, 0);
        model->set_multi(in_names, in_Is, in_int8, tok_names, out_names);
        return model;
      },
      nb::arg("path"), nb::arg("max_rows"), nb::arg("in_names"), nb::arg("in_Is"), nb::arg("in_int8"),
      nb::arg("tok_names"), nb::arg("out_names"));
  // ane_multi_16(xs fp16 [rows, I_k], model, out_cols) -> [fp16 [rows, c] for c in out_cols]
  m.def(
      "ane_multi_16",
      [](const std::vector<mx::array>& xs, std::shared_ptr<AneModel> model, const std::vector<int>& out_cols) {
        if (xs.empty() || xs.size() != model->multi_in_count()) throw std::invalid_argument("ane_multi_16: input count mismatch");
        const int rows = xs[0].shape(0);
        std::vector<mx::array> ins;
        for (const auto& x : xs) {
          if (x.dtype() != mx::float16 || x.ndim() != 2 || x.shape(0) != rows)
            throw std::invalid_argument("ane_multi_16: all inputs must be fp16 [rows, I]");
          ins.push_back(mx::contiguous(x));
        }
        auto prim = std::make_shared<AneMulti>(mx::default_stream(mx::Device::cpu), model, false, out_cols,
                                               (int)xs.size());
        std::vector<mx::array> outs;
        for (int c : out_cols) outs.push_back(mx::array({rows, c}, mx::float16, prim, ins));
        return outs;
      },
      nb::arg("xs"), nb::arg("model"), nb::arg("out_cols"));
  // ane_multi_q8(qs int8 [rows, I_k], toks fp32 [rows], model, out_cols) -> list of fp16 [rows, c]
  m.def(
      "ane_multi_q8",
      [](const std::vector<mx::array>& qs, const std::vector<mx::array>& toks, std::shared_ptr<AneModel> model,
         const std::vector<int>& out_cols) {
        const size_t n = model->multi_in_count();
        if (qs.size() != n || toks.size() != n) throw std::invalid_argument("ane_multi_q8: q/tok count mismatch");
        const int rows = qs[0].shape(0);
        std::vector<mx::array> ins;
        for (const auto& q : qs) {
          if (q.dtype() != mx::int8 || q.ndim() != 2 || q.shape(0) != rows)
            throw std::invalid_argument("ane_multi_q8: q must be int8 [rows, in_features]");
          ins.push_back(mx::contiguous(mx::transpose(q, {1, 0})));  // (I, rows) contiguous
        }
        for (const auto& t : toks) {
          if (t.dtype() != mx::float32 || t.ndim() != 1 || t.shape(0) != rows)
            throw std::invalid_argument("ane_multi_q8: tok must be float32 [rows]");
          ins.push_back(mx::contiguous(t));
        }
        auto prim = std::make_shared<AneMulti>(mx::default_stream(mx::Device::cpu), model, true, out_cols, (int)n);
        std::vector<mx::array> outs;
        for (int c : out_cols) outs.push_back(mx::array({rows, c}, mx::float16, prim, ins));
        return outs;
      },
      nb::arg("qs"), nb::arg("toks"), nb::arg("model"), nb::arg("out_cols"));
  // Debug path: run one predict_multi synchronously on the main thread (isolates the
  // bridge from MLX primitive plumbing). Not for production use.
  m.def(
      "predict_multi_sync",
      [](std::vector<mx::array> xs, std::vector<mx::array> toks, std::shared_ptr<AneModel> model,
         std::vector<int> out_cols) {
        std::vector<const void*> iptrs;
        std::vector<const float*> tptrs;
        for (auto& x : xs) {
          mx::eval(x);
          iptrs.push_back(x.data<void>());
        }
        for (auto& t : toks) {
          mx::eval(t);
          tptrs.push_back(t.data<float>());
        }
        const int rows = xs[0].shape(0);
        std::vector<std::vector<uint16_t>> outs;
        for (int c : out_cols) outs.emplace_back((size_t)rows * c);
        std::vector<void*> optrs;
        for (auto& v : outs) optrs.push_back(v.data());
        model->predict_multi(iptrs, tptrs, optrs, out_cols, rows);
        model->clear_error();
        nb::list out;
        for (auto& v : outs)
          out.append(nb::bytes(reinterpret_cast<const char*>(v.data()), v.size() * 2));
        return out;
      },
      nb::arg("xs"), nb::arg("toks"), nb::arg("model"), nb::arg("out_cols"));
}
