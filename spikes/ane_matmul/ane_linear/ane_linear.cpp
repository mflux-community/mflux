#include <algorithm>
#include <cstdio>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
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
    const int rows = x.shape(0);
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
    return {{inputs[0].shape(0), model_->O}};
  }

 private:
  std::shared_ptr<AneModel> model_;
  bool q8_;
};

NB_MODULE(_ane_linear, m) {
  nb::class_<AneModel>(m, "AneModel")   // held by shared_ptr: no GIL needed when MLX frees it
      .def(nb::init<const std::string&, int, int, int>(), nb::arg("path"), nb::arg("tokens"), nb::arg("in_features"), nb::arg("out_features"))
      .def_ro("tokens", &AneModel::T)
      .def_ro("in_features", &AneModel::I)
      .def_ro("out_features", &AneModel::O)
      .def("last_error", &AneModel::last_error);
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
  // ane_linear_q8(x[rows, I] int8, tok[rows] float32, model) -> y[rows, O] fp16 (W8A8 pre form).
  m.def(
      "ane_linear_q8",
      [](const mx::array& x, const mx::array& tok, std::shared_ptr<AneModel> model) {
        if (x.dtype() != mx::int8 || x.ndim() != 2 || x.shape(1) != model->I)
          throw std::invalid_argument("ane_linear_q8: x must be int8 [rows, in_features]");
        if (tok.dtype() != mx::float32 || tok.ndim() != 1 || tok.shape(0) != x.shape(0))
          throw std::invalid_argument("ane_linear_q8: tok must be float32 [rows]");
        auto xc = mx::contiguous(x);
        auto tc = mx::contiguous(tok);
        return mx::array(
            {x.shape(0), model->O}, mx::float16,
            std::make_shared<AneLinear>(mx::default_stream(mx::Device::cpu), model, true), {xc, tc});
      },
      nb::arg("x"), nb::arg("tok"), nb::arg("model"));
}
