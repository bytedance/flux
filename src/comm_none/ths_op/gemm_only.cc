//===- gemm_only.cc ----------------------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include <ATen/core/List.h>
#include <ATen/core/jit_type.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <pybind11/pytypes.h>
#include <utility>
#include "flux/args/comm_none.h"

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

class GemmOnly : public torch::CustomClassHolder {
 private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const bool transpose_weight;
  const bool use_fp8_gemm;
  torch::Tensor gemm_buffer;

 public:
  auto
  get_gemm_meta(bool has_bias, bool fast_accum) {
    auto arch = get_arch();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype);

    // FP8 GEMM RRR layout is not supported, details can be viewed in issue #43
    auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
    auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
    UnifiedImplMeta impl_spec = None{};

    bool use_fast_accum = fast_accum and dt_conf.is_input_fp8();
    if (impl == _GemmV2{}) {
      impl_spec = make_gemm_v2_meta(use_fast_accum);
    } else if (impl == _GemmV3{}) {
      impl_spec = make_gemm_v3_meta(use_fast_accum);
    }
    auto meta = make_gemm_meta(dt_conf, arch, _CommNone{}, gemm_layout, impl, impl_spec);
    return meta;
  };

  RuntimeConfig
  get_rt_conf(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);

    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = transpose_weight ? weight.size(1) : weight.size(0);

    if (bias.has_value()) {
      CHECK_INPUT(bias.value(), this->output_dtype);
      TORCH_CHECK(bias->dim() == 2, "bias dim is not 2");
      if (use_fp8_gemm) {
        TORCH_CHECK(
            1 == bias->size(0), "bias dim0 != 1: " + std::to_string(bias->size(0)) + " vs 1");
      } else {
        TORCH_CHECK(
            m == bias->size(0),
            "bias dim0 != m: " + std::to_string(bias->size(0)) + " vs " + std::to_string(m));
      }

      TORCH_CHECK(
          n == bias->size(1),
          "bias dim1 != n: " + std::to_string(bias->size(1)) + " vs " + std::to_string(n));
    }
    if (output_buf.has_value()) {
      CHECK_INPUT(output_buf.value(), this->output_dtype);
      TORCH_CHECK(output_buf->dim() == 2, "output_buf dim is not 2");
      TORCH_CHECK(
          m == output_buf->size(0),
          "output_buf dim0 != m: " + std::to_string(output_buf->size(0)) + " vs " +
              std::to_string(m));
      TORCH_CHECK(
          n == output_buf->size(1),
          "output_buf dim1 != n: " + std::to_string(output_buf->size(1)) + " vs " +
              std::to_string(n));
    }

    int32_t wk = transpose_weight ? weight.size(0) : weight.size(1);
    TORCH_CHECK(
        wk == k, "weight k-dim mismatch: " + std::to_string(wk) + " != " + std::to_string(k));
    return make_runtime_config(m, n, k);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    auto rt_conf = get_rt_conf(input, weight, bias, output_buf);
    int m = rt_conf.m();
    int n = rt_conf.n();
    int k = rt_conf.k();

    torch::Tensor output;
    if (output_buf.has_value()) {
      output = output_buf.value();
    } else {
      output = torch::empty({m, n}, input.options());
    }

    auto meta = get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum);
    OpRegistry::OpPtr gemm_op;

    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    if (this->use_fp8_gemm) {
      torch::Tensor in_scale = get_optional_scale_tensor(input_scale);
      torch::Tensor w_scale = get_optional_scale_tensor(weight_scale);
      torch::Tensor out_scale = get_optional_scale_tensor(output_scale);

      const GemmFP8Arguments args{
          .m = m,
          .n = n,
          .k = k,
          .alpha = 1.0,
          .beta = 0.0,
          .A = input.data_ptr(),
          .B = weight.data_ptr(),
          .C = nullptr,
          .Aux = nullptr,
          .D = output.data_ptr(),
          .Vector = bias.has_value() ? bias->data_ptr() : nullptr,
          .abs_max_Aux = nullptr,
          .abs_max_D = nullptr,
          .scaleA = (float *)in_scale.data_ptr(),
          .scaleB = (float *)w_scale.data_ptr(),
          .scaleC = nullptr,
          .scaleD = (float *)out_scale.data_ptr(),
          .scaleAux = nullptr};

      int64_t workspace_size = gemm_op->get_workspace_size(args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
      gemm_op->run(args, workspace, stream);
    } else {
      // initialize mnk for streamk get_workspace_size
      const GemmOnlyArguments args{
          .m = m,
          .n = n,
          .k = k,
          .alpha = 1.0,
          .beta = bias.has_value() ? 1.0f : 0.0f,
          .input = input.data_ptr(),
          .weight = weight.data_ptr(),
          .bias = bias.has_value() ? bias->data_ptr() : nullptr,
          .output = output.data_ptr()};

      int64_t workspace_size = gemm_op->get_workspace_size(args);
      this->lazy_init_gemm_buffer(input, workspace_size);
      void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
      gemm_op->run(args, workspace, stream);
    }

    return output;
  }

 public:
  GemmOnly(
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      bool use_fp8_gemm)
      : input_dtype(input_dtype),
        output_dtype(output_dtype),
        transpose_weight(transpose_weight),
        use_fp8_gemm(is_fp8_torch_dtype(input_dtype) && use_fp8_gemm) {
    FLUX_CHECK(!(transpose_weight == true && use_fp8_gemm == true))
        << "FP8 GEMM does not support transpose weight";
  }

  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0)
      return;
    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      this->gemm_buffer = torch::empty({buffer_size}, input.options().dtype(at::ScalarType::Byte));
    }
  }

  torch::Tensor
  get_optional_scale_tensor(c10::optional<torch::Tensor> optional_scale_tensor) {
    torch::Tensor scale_tensor;
    if (optional_scale_tensor.has_value()) {
      scale_tensor = optional_scale_tensor.value();
    } else {
      scale_tensor = torch::empty(
          {1},
          at::TensorOptions()
              .dtype(c10::ScalarType::Float)
              .device(at::kCUDA)
              .device_index(c10::cuda::current_device()));
    }
    return scale_tensor;
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum) {
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output_buf),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        c10::nullopt);
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output_buf,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    auto meta =
        unify_type(this->get_gemm_meta(/*has_bias=*/bias.has_value(), /*fast_accum=*/fast_accum));
    auto rt_conf = this->get_rt_conf(input, weight, bias, output_buf);

    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          auto stream = c10::cuda::getCurrentCUDAStream();
          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto output [[maybe_unused]] = this->forward_impl(
                input,
                weight,
                bias,
                output_buf,
                input_scale,
                weight_scale,
                output_scale,
                fast_accum,
                hparams);
            timer.stop();
            if (iter >= warm_iters) {
              total_elapsed += timer.elapsed_millis();
            }
          }

          float avg_elapsed = int(total_elapsed / iters * 1000) / 1000.0;
          ctx->add(meta, rt_conf, hparams, avg_elapsed);
        },
        meta);

    auto best_hparams = ctx->record_best(meta, rt_conf);

    return this->forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output_buf),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(output_scale),
        fast_accum,
        std::move(best_hparams));
  }
};

namespace py = pybind11;

static int _register_gemm_only_ops [[maybe_unused]] = []() {
  ThsOpsInitRegistry::instance().register_one("gemm_only", [](py::module &m) {
    py::class_<GemmOnly>(m, "GemmOnly")
        .def(
            py::init([](py::object py_input_dtype,
                        py::object py_output_dtype,
                        bool transpose_weight,
                        bool use_fp8_gemm) {
              auto input_dtype = torch::python::detail::py_object_to_dtype(py_input_dtype);
              auto output_dtype = py_output_dtype.is(py::none())
                                      ? input_dtype
                                      : torch::python::detail::py_object_to_dtype(py_output_dtype);
              return new GemmOnly(input_dtype, output_dtype, transpose_weight, use_fp8_gemm);
            }),
            py::arg("input_dtype"),
            py::arg("output_dtype") = py::none(),
            py::arg("transpose_weight") = false,
            py::arg("use_fp8_gemm") = false)
        .def(
            "forward",
            &GemmOnly::forward,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false)
        .def(
            "profiling",
            &GemmOnly::profiling,
            py::arg("input"),
            py::arg("weight"),
            py::arg("bias") = py::none(),
            py::arg("output_buf") = py::none(),
            py::arg("input_scale") = py::none(),
            py::arg("weight_scale") = py::none(),
            py::arg("output_scale") = py::none(),
            py::arg("fast_accum") = false,
            py::arg("prof_ctx") = nullptr);
  });
  return 0;
}();

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
