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
  AneLinear(mx::Stream s, std::shared_ptr<AneModel> m) : mx::Primitive(s), model_(std::move(m)) {}

  void eval_cpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    auto& x = inputs[0];
    auto& out = outputs[0];
    out.set_data(mx::allocator::malloc(out.nbytes()));
    auto& enc = mx::cpu::get_command_encoder(stream());
    enc.set_input_array(x);
    enc.set_output_array(out);
    auto model = model_;
    // x and out are kept alive by MLX until the task completes (same pattern as MLX CPU primitives).
    enc.dispatch([model, xp = x.data<void>(), op = out.data<void>()]() {
      // An exception must not leave this worker thread: it would terminate the process.
      try {
        model->predict(xp, op);
        model->clear_error();
      } catch (const std::exception& e) {
        model->set_error(e.what());
        std::fprintf(stderr, "ane_linear: prediction failed: %s\n", e.what());
        auto* h = static_cast<uint16_t*>(op);
        std::fill(h, h + (size_t)model->T * model->O, uint16_t(0x7E00));   // fp16 NaN: the failure is visible
      }
    });
  }
  void eval_gpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("ane_linear must run on the CPU stream");
  }
  const char* name() const override { return "AneLinear"; }
  bool is_equivalent(const mx::Primitive& other) const override {
    return model_.get() == static_cast<const AneLinear&>(other).model_.get();
  }
  std::vector<mx::Shape> output_shapes(const std::vector<mx::array>& inputs) override {
    return {{inputs[0].shape(0), model_->O}};
  }

 private:
  std::shared_ptr<AneModel> model_;
};

NB_MODULE(_ane_linear, m) {
  nb::class_<AneModel>(m, "AneModel")   // held by shared_ptr: no GIL needed when MLX frees it
      .def(nb::init<const std::string&, int, int, int>(), nb::arg("path"), nb::arg("tokens"), nb::arg("in_features"), nb::arg("out_features"))
      .def_ro("tokens", &AneModel::T)
      .def_ro("in_features", &AneModel::I)
      .def_ro("out_features", &AneModel::O)
      .def("last_error", &AneModel::last_error);
  // ane_linear(x[T, I] fp16, model) -> y[T, O] fp16. Build the graph node on the CPU stream.
  m.def(
      "ane_linear",
      [](const mx::array& x, std::shared_ptr<AneModel> model) {
        if (x.dtype() != mx::float16 || x.ndim() != 2 || x.shape(0) != model->T || x.shape(1) != model->I)
          throw std::invalid_argument("ane_linear: x must be fp16 [tokens, in_features] matching the model");
        auto xc = mx::contiguous(x);
        return mx::array(
            {x.shape(0), model->O}, mx::float16,
            std::make_shared<AneLinear>(mx::default_stream(mx::Device::cpu), model), {xc});
      },
      nb::arg("x"), nb::arg("model"));
}
