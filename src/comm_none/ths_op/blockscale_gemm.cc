//===- blockscale_gemm.cc ----------------------------------------- C++ ---===//
//
// Copyright 2025 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include "comm_none/ths_op/blockscale_gemm.h"
#include "comm_none/cutlass_blockscale_gemm_impl.h"
#include "flux/args/comm_none.h"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <utility>

namespace bytedance {
namespace flux {
namespace ths_op {
using torch::Tensor;

class BlockScaleGemm::BlockScaleGemmImpl {
 private:
  const c10::ScalarType input_dtype;
  const c10::ScalarType output_dtype;
  const int32_t num_streams;
  torch::Tensor gemm_buffer;

  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> events;
  cudaEvent_t ready_event;

  static constexpr int32_t BLOCK_M = 128;
  static constexpr int32_t BLOCK_N = 128;
  static constexpr int32_t BLOCK_K = 128;
  static constexpr int32_t ScaleMsPerTile = 128;

 private:
  auto
  get_gemm_meta(bool has_bias) {
    auto arch = get_arch();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    DataTypeEnum accum_type = _FP32{}();

    auto dt_conf = make_gemm_dtype_config(
        input_dtype, input_dtype, has_bias ? output_dtype : _Void{}(), output_dtype, accum_type);

    auto gemm_layout = _RCR{}();
    auto impl = _GemmV3{}();
    auto impl_spec = make_gemm_v3_meta(false, true);

    auto meta = make_gemm_meta(dt_conf, arch, _CommNone{}, gemm_layout, impl, impl_spec);
    return meta;
  };

  RuntimeConfig
  get_rt_conf(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    CHECK_INPUT(input, this->input_dtype);
    CHECK_INPUT(weight, this->input_dtype);
    TORCH_CHECK(input.dim() == 2, "input shape is not 2");
    TORCH_CHECK(weight.dim() == 2, "weight dim is not 2");
    int32_t m = input.size(0);
    int32_t k = input.size(1);
    int32_t n = weight.size(0);

    FLUX_CHECK(!bias.has_value());
    FLUX_CHECK(input_scale.has_value());
    FLUX_CHECK(weight_scale.has_value());

    if (output.has_value()) {
      CHECK_INPUT(output.value(), this->output_dtype);
      FLUX_CHECK_EQ(output->dim(), 2);
      FLUX_CHECK_EQ(m, output->size(0));
      FLUX_CHECK_EQ(n, output->size(1));
    }

    int32_t wk = weight.size(1);
    FLUX_CHECK_EQ(wk, k) << "weight k-dim mismatch";
    return make_runtime_config(m, n, k);
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<UnifiedGemmHParams> const &hparams,
      cudaStream_t stream) {
    auto rt_conf = get_rt_conf(input, weight, bias, output, input_scale, weight_scale);
    int m = rt_conf.m();
    int n = rt_conf.n();
    int k = rt_conf.k();

    torch::Tensor out;
    if (output.has_value()) {
      out = output.value();
    } else {
      out = torch::empty({m, n}, weight.options().dtype(output_dtype));
    }

    if (m == 0) {
      return out;
    }

    auto meta = get_gemm_meta(bias.has_value());
    OpRegistry::OpPtr gemm_op;

    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }

    auto data_ptr_or = [](auto &&optional_tensor, void *other) -> void * {
      if (optional_tensor.has_value()) {
        return optional_tensor->data_ptr();
      } else {
        return other;
      }
    };

    // initialize mnk for streamk get_workspace_size
    const BlockScaleGemmArguments args{
        .m = m,
        .n = n,
        .k = k,
        .l = 1,
        .A = input.data_ptr(),
        .B = weight.data_ptr(),
        .mma_promotion_interval = 4,
        .blockscale_A = data_ptr_or(input_scale, nullptr),
        .blockscale_B = data_ptr_or(weight_scale, nullptr),
        .C = nullptr,
        .D = out.data_ptr(),
        .alpha = 1.0f,
        .beta = 0.0f,
        .scale_a = 1.0f,
        .scale_b = 1.0f,
        .scale_c = 1.0f,
        .scale_d = 1.0f,
        .scale_aux = 1.0f,
        .bias = data_ptr_or(bias, nullptr)};

    auto unified_hparams = gemm_op->get_runtime_gemm_hparams();
    auto tile_shape = unified_hparams.tile_shape();
    auto [tile_M, tile_N, tile_K] = tile_shape;
    FLUX_CHECK(tile_M == BLOCK_M && tile_N == BLOCK_N && tile_K == BLOCK_K)
        << "tile size of gemm is not match block size of scale.";
    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
    gemm_op->run(args, workspace, stream);

    return out;
  }

  torch::Tensor
  forward_cutlass_impl(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      cudaStream_t stream) {
    FLUX_CHECK(at::ScalarType::Float8_e4m3fn == input_dtype) << "reference only support e4m3.";
    FLUX_CHECK(!bias.has_value()) << "reference does not support bias.";

    auto rt_conf = get_rt_conf(input, weight, bias, output, input_scale, weight_scale);
    int m = rt_conf.m();
    int n = rt_conf.n();
    int k = rt_conf.k();

    torch::Tensor out;
    if (output.has_value()) {
      out = output.value();
    } else {
      out = torch::empty({m, n}, weight.options().dtype(output_dtype));
    }

    if (m == 0) {
      return out;
    }

    auto data_ptr_or = [](auto &&optional_tensor, void *other) -> void * {
      if (optional_tensor.has_value()) {
        return optional_tensor->data_ptr();
      } else {
        return other;
      }
    };

    // initialize mnk for streamk get_workspace_size
    const BlockScaleGemmArguments args{
        .m = m,
        .n = n,
        .k = k,
        .l = 1,
        .A = input.data_ptr(),
        .B = weight.data_ptr(),
        .mma_promotion_interval = 4,
        .blockscale_A = data_ptr_or(input_scale, nullptr),
        .blockscale_B = data_ptr_or(weight_scale, nullptr),
        .C = nullptr,
        .D = out.data_ptr(),
        .alpha = 1.0f,
        .beta = 0.0f,
        .scale_a = 1.0f,
        .scale_b = 1.0f,
        .scale_c = 1.0f,
        .scale_d = 1.0f,
        .scale_aux = 1.0f,
        .bias = data_ptr_or(bias, nullptr)};

    int64_t workspace_size = CutlassBlockScaleGemm::get_workspace_size(args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
    CutlassBlockScaleGemm::run(args, workspace, stream);

    return out;
  }

  void
  lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
    if (buffer_size <= 0) {
      return;
    }

    buffer_size = (buffer_size + 127) / 128 * 128;
    if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
      this->gemm_buffer = torch::empty({buffer_size}, input.options().dtype(at::ScalarType::Byte));
    }
  }

 public:
  BlockScaleGemmImpl(
      c10::ScalarType input_dtype, c10::ScalarType output_dtype, int32_t num_streams)
      : input_dtype(input_dtype), output_dtype(output_dtype), num_streams(num_streams) {
    // init stream pool
    FLUX_CHECK(num_streams > 0);
    this->streams.resize(num_streams);
    this->events.resize(num_streams);
    for (int32_t i = 0; i < num_streams; ++i) {
      CUDA_CHECK(cudaStreamCreateWithPriority(
          &this->streams[i], cudaStreamNonBlocking, get_highest_cuda_stream_priority()));
      CUDA_CHECK(cudaEventCreateWithFlags(&this->events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&this->ready_event, cudaEventDisableTiming));
  }

  ~BlockScaleGemmImpl() {
    for (auto &stream : streams) {
      cudaStreamDestroy(stream);
    }
    for (auto &event : events) {
      cudaEventDestroy(event);
    }
    cudaEventDestroy(this->ready_event);
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        c10::nullopt,
        stream);
  }

  torch::Tensor
  forward_multistream_impl(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      cudaStream_t stream) {
    CHECK_NDIM(input_list, 1);
    CHECK_NDIM(input, 2);
    FLUX_CHECK(input_list.device().type() == torch::kCPU) << "input splits is not on cpu\n";
    FLUX_CHECK(input_list.size(0) == weight.size(0));
    if (input_scale.has_value())
      CHECK_NDIM(input_scale.value(), 3);

    CHECK_NDIM(weight, 3);
    if (weight_scale.has_value())
      CHECK_NDIM(weight_scale.value(), 3);

    int m = input.size(0);
    int n = weight.size(1);
    int k = weight.size(2);
    int32_t blockscale_k = (k + BLOCK_K - 1) / BLOCK_K;
    torch::Tensor out;
    if (output.has_value()) {
      out = output.value();
    } else {
      out = torch::empty({m, n}, weight.options().dtype(output_dtype));
    }

    auto slice_optional_tensor = [](c10::optional<torch::Tensor> optional_tensor,
                                    int32_t dim,
                                    int32_t start,
                                    int32_t end) -> c10::optional<torch::Tensor> {
      if (optional_tensor.has_value()) {
        return optional_tensor->slice(dim, start, end);
      } else {
        return c10::nullopt;
      }
    };

    CUDA_CHECK(cudaEventRecord(this->ready_event, stream));
    for (int32_t i = 0; i < num_streams; ++i) {
      CUDA_CHECK(cudaStreamWaitEvent(this->streams[i], this->ready_event));
    }

    int32_t num_experts = weight.size(0);

    int32_t pre = 0;
    for (int32_t i = 0; i < num_experts; ++i) {
      int32_t cur_split = input_list[i].item<int32_t>();

      int32_t nxt = pre + cur_split;
      auto cur_out = out.slice(0, pre, nxt);
      auto cur_bias = slice_optional_tensor(bias, 0, pre, nxt);

      auto cur_weight = weight.slice(0, i, i + 1).reshape({n, k});
      auto cur_weight_scale = slice_optional_tensor(weight_scale, 0, i, i + 1);
      if (cur_weight_scale.has_value()) {
        int32_t blockscale_n = cur_weight_scale->size(1);
        int32_t blockscale_k = cur_weight_scale->size(2);
        cur_weight_scale = cur_weight_scale->reshape({blockscale_n, blockscale_k});
      }

      auto cur_input = input.slice(0, pre, nxt);
      int32_t input_scale_start = (pre + BLOCK_M - 1) / BLOCK_M;
      int32_t input_scale_end = (nxt + BLOCK_M - 1) / BLOCK_M;
      auto cur_input_scale =
          slice_optional_tensor(input_scale, 0, input_scale_start, input_scale_end);
      // TODO(zhengxuegui.0): workload aware and more fine grain schedule
      auto cur_stream = this->streams[i % num_streams];
      auto cur_event = this->events[i % num_streams];

      forward_impl(
          cur_input,
          cur_weight,
          cur_bias,
          cur_out,
          cur_input_scale,
          cur_weight_scale,
          c10::nullopt,
          cur_stream);
      CUDA_CHECK(cudaEventRecord(cur_event, cur_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, cur_event));
      pre = nxt;
    }

    return out;
  }

  torch::Tensor
  forward_multistream(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return forward_multistream_impl(
        std::move(input),
        std::move(input_list),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        stream);
  }

  torch::Tensor
  reference(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return forward_cutlass_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        stream);
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    auto meta = unify_type(this->get_gemm_meta(bias.has_value()));
    auto rt_conf = this->get_rt_conf(input, weight, bias, output, input_scale, weight_scale);

    ProfilingContext tmp_ctx("__tmp__");
    ProfilingContext *ctx = opt_ctx == nullptr ? &tmp_ctx : opt_ctx.get();

    auto stream = c10::cuda::getCurrentCUDAStream();

    OpRegistry::instance().visit_hparams(
        [&](UnifiedGemmHParams const &hparams) {
          constexpr int warm_iters = 5;
          constexpr int iters = 10;
          float total_elapsed = 0;

          for (int iter = 0; iter < warm_iters + iters; ++iter) {
            GpuTimer timer;
            timer.start(stream);
            auto out [[maybe_unused]] = this->forward_impl(
                input, weight, bias, output, input_scale, weight_scale, hparams, stream);
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
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        std::move(best_hparams),
        stream);
  }
};

BlockScaleGemm::BlockScaleGemm(
    c10::ScalarType input_dtype, c10::ScalarType output_dtype, int32_t num_streams)
    : impl_(new BlockScaleGemm::BlockScaleGemmImpl(input_dtype, output_dtype, num_streams)) {}

BlockScaleGemm::~BlockScaleGemm() { delete impl_; }

torch::Tensor
BlockScaleGemm::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->forward(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale));
}

torch::Tensor
BlockScaleGemm::forward_multistream(
    torch::Tensor input,
    torch::Tensor input_splits,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->forward_multistream(
      std::move(input),
      std::move(input_splits),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale));
}

torch::Tensor
BlockScaleGemm::reference(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->reference(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale));
}

torch::Tensor
BlockScaleGemm::profiling(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->profiling(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      std::move(opt_ctx));
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
