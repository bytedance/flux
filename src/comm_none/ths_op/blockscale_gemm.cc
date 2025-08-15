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
#include <c10/cuda/CUDAGuard.h>
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
  const c10::ScalarType weight_dtype;
  const c10::ScalarType output_dtype;
  const int32_t num_streams;
  torch::Tensor gemm_buffer;
  torch::Tensor workspace;

  std::vector<cudaStream_t> streams;
  std::vector<cudaEvent_t> events;
  cudaEvent_t ready_event;

  static constexpr int32_t BLOCK_M = 128;
  static constexpr int32_t BLOCK_N = 128;
  static constexpr int32_t BLOCK_K = 128;

 private:
  auto
  get_gemm_meta(bool has_bias) {
    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto weight_dtype = from_torch_dtype(this->weight_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    DataTypeEnum accum_type = _FP32{}();

    auto dt_conf = make_gemm_dtype_config(
        input_dtype, weight_dtype, has_bias ? output_dtype : _Void{}(), output_dtype, accum_type);

    auto gemm_layout = _RCR{}();
    auto impl = _GemmV3{}();
    auto impl_spec = make_gemm_v3_meta(false, true);

    auto meta = make_gemm_meta(dt_conf, arch, sm_core, _CommNone{}, gemm_layout, impl, impl_spec);
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
    CHECK_INPUT(weight, this->weight_dtype);
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
    check_io_dtype(input, weight, bias, output, input_scale, weight_scale);
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
      GemmBlockScaleNEnum scale_type_b,
      cudaStream_t stream) {
    check_io_dtype(input, weight, bias, output, input_scale, weight_scale);

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

    auto meta = unify_type(this->get_gemm_meta(bias.has_value()));
    auto cutlass_op = CutlassBlockScaleGemm(meta, scale_type_b);
    int64_t workspace_size = cutlass_op.get_workspace_size(args);
    this->lazy_init_gemm_buffer(input, workspace_size);
    void *workspace = this->gemm_buffer.defined() ? this->gemm_buffer.data_ptr() : nullptr;
    cutlass_op.run(args, workspace, stream);

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
      c10::ScalarType input_dtype,
      c10::ScalarType weight_dtype,
      c10::ScalarType output_dtype,
      int32_t num_streams)
      : input_dtype(input_dtype),
        weight_dtype(weight_dtype),
        output_dtype(output_dtype),
        num_streams(num_streams) {
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

  void
  check_io_dtype(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    CHECK_TYPE(input, this->input_dtype);
    CHECK_TYPE(weight, this->weight_dtype);
    if (bias.has_value())
      CHECK_TYPE(bias.value(), this->output_dtype);
    if (output.has_value())
      CHECK_TYPE(output.value(), this->output_dtype);
    if (input_scale.has_value())
      CHECK_TYPE(input_scale.value(), at::ScalarType::Float);
    if (weight_scale.has_value())
      CHECK_TYPE(weight_scale.value(), at::ScalarType::Float);
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
  wgrad(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    auto cur_meta = get_gemm_meta(bias.has_value());
    auto cur_rt_conf = get_rt_conf(input, weight, bias, output, input_scale, weight_scale);
    auto cur_hparams = OpRegistry::instance().get_hparams(cur_meta, cur_rt_conf);
    auto &cur_v3_hparams = std::get<unified_type_t<GemmV3HParams>>(cur_hparams.impl_spec());
    cur_v3_hparams.blockscale_M() = _BlockScaleMPerRow();
    cur_v3_hparams.blockscale_N() = _BlockScaleNPerCol();

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return forward_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        cur_hparams,
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
    check_io_dtype(input, weight, bias, output, input_scale, weight_scale);
    CHECK_NDIM(input_list, 1);
    CHECK_NDIM(input, 2);
    FLUX_CHECK(input_list.device().type() == torch::kCPU) << "input splits is not on cpu\n";
    FLUX_CHECK(input_list.size(0) == weight.size(0));
    if (input_scale.has_value())
      CHECK_NDIM(input_scale.value(), 2);

    CHECK_NDIM(weight, 3);
    if (weight_scale.has_value())
      CHECK_NDIM(weight_scale.value(), 3);

    int m = input.size(0);
    int n = weight.size(1);
    int k = weight.size(2);

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
    std::vector<at::Tensor> input_scales(num_experts);

    for (int32_t i = 0; i < num_experts; ++i) {
      int32_t cur_split = input_list[i].item<int32_t>();
      auto cur_stream = this->streams[i % num_streams];
      auto cur_event = this->events[i % num_streams];
      int32_t nxt = pre + cur_split;
      auto cur_out = out.slice(0, pre, nxt);
      auto cur_bias = slice_optional_tensor(bias, 0, pre, nxt);

      auto cur_weight = weight.slice(0, i, i + 1).reshape({n, k});
      auto cur_weight_scale = slice_optional_tensor(weight_scale, 0, i, i + 1);
      if (cur_weight_scale.has_value()) {
        cur_weight_scale = cur_weight_scale->squeeze(0);
      }

      auto cur_input = input.slice(0, pre, nxt);
      auto cur_input_scale = slice_optional_tensor(input_scale, 0, pre, nxt);  // stride: (1, t)
      FLUX_CHECK(cur_input_scale.has_value()) << "cur_input_scale is required but missing!";
      input_scales[i] = cur_input_scale->squeeze(0)
                            .contiguous()
                            .transpose(0, 1)
                            .contiguous()
                            .transpose(0, 1)
                            .clone();

      //   cudaDeviceSynchronize(); Here, we need sync
      cudaEvent_t input_ready_event;
      cudaEventCreate(&input_ready_event);
      cudaEventRecord(input_ready_event, 0);  // default stream
      CUDA_CHECK(cudaStreamWaitEvent(cur_stream, input_ready_event));

      // TODO(zhengxuegui.0): workload aware and more fine grain schedule
      auto cur_meta = get_gemm_meta(cur_bias.has_value());
      auto cur_rt_conf =
          get_rt_conf(cur_input, cur_weight, cur_bias, cur_out, input_scales[i], cur_weight_scale);
      auto cur_hparams = OpRegistry::instance().get_hparams(cur_meta, cur_rt_conf);
      auto &cur_v3_hparams = std::get<unified_type_t<GemmV3HParams>>(cur_hparams.impl_spec());
      cur_v3_hparams.blockscale_M() = _BlockScaleMPerRow();
      cur_v3_hparams.blockscale_N() = _BlockScaleNPerBlock();

      at::cuda::CUDAStream stream_ = at::cuda::getStreamFromExternal(
          (cudaStream_t)this->streams[i % num_streams], at::cuda::current_device());
      at::cuda::CUDAStreamGuard guard(stream_);

      forward_impl(
          cur_input,
          cur_weight,
          cur_bias,
          cur_out,
          input_scales[i],
          cur_weight_scale,
          cur_hparams,
          cur_stream);
      CUDA_CHECK(cudaEventRecord(cur_event, cur_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, cur_event));
      pre = nxt;
    }

    return out;
  }

  torch::Tensor
  wgrad_multistream_impl(
      torch::Tensor input,                // [M, K]
      torch::Tensor input_list,           // [num_experts]
      torch::Tensor weight,               // [N, K]
      c10::optional<torch::Tensor> bias,  // [E, M, N]
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,   // [groupscale_m, blockscale_k]
      c10::optional<torch::Tensor> weight_scale,  // [groupscale_n, blockscale_k]
      cudaStream_t stream) {
    check_io_dtype(input, weight, bias, output, input_scale, weight_scale);
    CHECK_NDIM(input_list, 1);
    CHECK_NDIM(input, 2);
    FLUX_CHECK(input_list.device().type() == torch::kCPU) << "input splits is not on cpu\n";
    FLUX_CHECK(input_list.sum().item<int>() == input.size(1))
        << "Sum of input_list must be equal to K (input's second dimension)\n";
    if (input_scale.has_value())
      CHECK_NDIM(input_scale.value(), 2);

    CHECK_NDIM(weight, 2);
    if (weight_scale.has_value())
      CHECK_NDIM(weight_scale.value(), 2);

    int32_t num_experts = input_list.size(0);
    int m = input.size(0);
    int n = weight.size(0);
    int k = input.size(1);

    torch::Tensor out;
    if (output.has_value()) {
      out = output.value();
    } else {
      out = torch::empty({num_experts, m, n}, weight.options().dtype(output_dtype));
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

    int32_t pre = 0;
    int32_t pre_scale_offset = 0;
    for (int32_t i = 0; i < num_experts; ++i) {
      at::cuda::CUDAStream stream_ = at::cuda::getStreamFromExternal(
          (cudaStream_t)this->streams[i % num_streams], at::cuda::current_device());
      at::cuda::CUDAStreamGuard guard(stream_);
      int32_t cur_split = input_list[i].item<int32_t>();

      int32_t nxt = pre + cur_split;
      auto cur_out = out.slice(0, i, i + 1);
      cur_out = cur_out.squeeze(0);
      auto cur_bias = slice_optional_tensor(bias, 0, i, i + 1);

      int32_t K_pad = ((cur_split + BLOCK_K - 1) / BLOCK_K) * BLOCK_K;

      auto cur_weight = weight.slice(1, pre, nxt);  // [M, K1]
      auto cur_input = input.slice(1, pre, nxt);    // [N, K1]

      if (cur_split < K_pad) {
        auto cur_weight_padded = torch::zeros({cur_weight.size(0), K_pad}, cur_weight.options());
        auto cur_input_padded = torch::zeros({cur_input.size(0), K_pad}, cur_input.options());
        cur_weight_padded.slice(1, 0, cur_split).copy_(cur_weight);
        cur_input_padded.slice(1, 0, cur_split).copy_(cur_input);
        cur_weight = cur_weight_padded;
        cur_input = cur_input_padded;
      }
      cur_weight = cur_weight.contiguous();
      cur_input = cur_input.contiguous();

      int32_t input_scale_start = pre_scale_offset;
      int32_t input_scale_end = pre_scale_offset + (cur_split + BLOCK_K - 1) / BLOCK_K;
      auto cur_input_scale =
          slice_optional_tensor(input_scale, 1, input_scale_start, input_scale_end);
      auto cur_weight_scale =
          slice_optional_tensor(weight_scale, 1, input_scale_start, input_scale_end);
      // TODO(zhengxuegui.0): workload aware and more fine grain schedule
      auto cur_stream = this->streams[i % num_streams];
      auto cur_event = this->events[i % num_streams];

      auto cur_meta = get_gemm_meta(cur_bias.has_value());
      auto cur_rt_conf =
          get_rt_conf(cur_input, cur_weight, cur_bias, cur_out, cur_input_scale, cur_weight_scale);
      auto cur_hparams = OpRegistry::instance().get_hparams(cur_meta, cur_rt_conf);
      auto &cur_v3_hparams = std::get<unified_type_t<GemmV3HParams>>(cur_hparams.impl_spec());
      cur_v3_hparams.blockscale_M() = _BlockScaleMPerRow();
      cur_v3_hparams.blockscale_N() = _BlockScaleNPerCol();

      forward_impl(
          cur_input,
          cur_weight,
          cur_bias,
          cur_out,
          cur_input_scale,
          cur_weight_scale,
          cur_hparams,
          cur_stream);
      CUDA_CHECK(cudaEventRecord(cur_event, cur_stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, cur_event));
      pre = nxt;
      pre_scale_offset = input_scale_end;
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
  wgrad_multistream(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return wgrad_multistream_impl(
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
  forward_grouped_impl(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      torch::Tensor input_scale,
      torch::Tensor weight_scale,
      cudaStream_t stream) {
    check_io_dtype(input, weight, bias, output, input_scale, weight_scale);
    CHECK_NDIM(input_list, 1);
    CHECK_NDIM(input, 2);
    FLUX_CHECK(input_list.device().type() == torch::kCPU) << "input splits is not on cpu\n";
    FLUX_CHECK(input_list.size(0) == weight.size(0));

    int M = input.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);

    CHECK(weight.size(2) == K) << "weight 2-dim mismatch: " << weight.size(2) << " != " << K;
    CHECK(input_list.size(0) == E)
        << "splits 0-dim mismatch: " << input_list.size(0) << " != " << E;

    torch::Tensor out;
    if (output.has_value()) {
      out = output.value();
    } else {
      out = torch::empty({M, N}, weight.options().dtype(output_dtype));
    }

    CHECK(out.device().type() == torch::DeviceType::CUDA)
        << "output device: " << out.device().type() << " != torch::DeviceType::CUDA";
    // CHECK(M >= 8) << "M must be greater than or equal 8 for cutlass grouped gemm.";
    // CHECK(N >= 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    // CHECK(K >= 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";

    using UnderlyingProblemShape = cute::Shape<int, int, int>;
    std::vector<UnderlyingProblemShape> problem_sizes;
    std::vector<void const *> ptr_A;
    std::vector<void const *> ptr_B;
    std::vector<void const *> ptr_C;
    std::vector<void *> ptr_D;
    std::vector<void const *> ptr_scale_A;
    std::vector<void const *> ptr_scale_B;

    // TODO (hanshi.s): bias
    {
      // initialize args vectors
      uint8_t const *ptr_A_cur = reinterpret_cast<uint8_t const *>(input.data_ptr());
      uint8_t const *ptr_B_cur = reinterpret_cast<uint8_t const *>(weight.data_ptr());
      uint8_t *ptr_D_cur = reinterpret_cast<uint8_t *>(out.data_ptr());
      uint8_t const *ptr_scale_A_cur = reinterpret_cast<uint8_t const *>(input_scale.data_ptr());
      uint8_t const *ptr_scale_B_cur = reinterpret_cast<uint8_t const *>(weight_scale.data_ptr());

      for (int i = 0; i < E; ++i) {
        int Mi = input_list[i].item().toInt();
        if (Mi == 0) {
          continue;
        }
        problem_sizes.emplace_back(Mi, N, K);
        ptr_A.emplace_back(ptr_A_cur);
        ptr_A_cur += Mi * K * c10::elementSize(input.scalar_type());
        ptr_B.emplace_back(ptr_B_cur);
        ptr_B_cur += N * K * c10::elementSize(weight.scalar_type());
        ptr_C.emplace_back(nullptr);
        ptr_D.emplace_back(ptr_D_cur);
        ptr_D_cur += Mi * N * c10::elementSize(out.scalar_type());

        ptr_scale_A.emplace_back(ptr_scale_A_cur);
        ptr_scale_A_cur += Mi * K / BLOCK_K * c10::elementSize(input_scale.scalar_type());
        ptr_scale_B.emplace_back(ptr_scale_B_cur);
        ptr_scale_B_cur +=
            N / BLOCK_N * K / BLOCK_K * c10::elementSize(weight_scale.scalar_type());
      }
    }

    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto weight_dtype = from_torch_dtype(this->weight_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    DataTypeEnum accum_type = _FP32{}();
    auto dt_conf = make_gemm_dtype_config(  // bias.has_value() ? output_dtype : _Void{}()
        input_dtype,
        weight_dtype,
        output_dtype,
        output_dtype,
        accum_type);
    auto gemm_layout = _RCR{}();
    auto impl = _GemmGroupedV3{}();
    auto impl_spec = make_gemm_v3_meta(false, true);
    auto meta = make_gemm_meta(dt_conf, arch, sm_core, _CommNone{}, gemm_layout, impl, impl_spec);
    auto rt_conf = make_runtime_config(M, N, K);

    auto hparams = OpRegistry::instance().get_hparams(meta, rt_conf);
    auto &v3_hparams = std::get<unified_type_t<GemmV3HParams>>(hparams.impl_spec());
    v3_hparams.blockscale_M() = _BlockScaleMPerRow();
    v3_hparams.blockscale_N() = _BlockScaleNPerBlock();

    OpRegistry::OpPtr gemm_op;
    gemm_op = OpRegistry::instance().get_op(meta, hparams);

    // initialize mnk for streamk get_workspace_size
    const BlockScaleGroupedGemmV3Arguments args{
        .problem_count = static_cast<int>(problem_sizes.size()),
        .problem_sizes = problem_sizes.data(),
        .ptr_A = ptr_A.data(),
        .ptr_B = ptr_B.data(),
        .ptr_C = ptr_C.data(),
        .ptr_D = ptr_D.data(),
        .ptr_blockscale_A = ptr_scale_A.data(),
        .ptr_blockscale_B = ptr_scale_B.data(),
        .alpha = 1.0f,
        .beta = 0.0f,
    };

    auto unified_hparams = gemm_op->get_runtime_gemm_hparams();
    auto tile_shape = unified_hparams.tile_shape();
    auto [tile_M, tile_N, tile_K] = tile_shape;
    FLUX_CHECK(tile_M == BLOCK_M && tile_N == BLOCK_N && tile_K == BLOCK_K)
        << "tile size of gemm is not match block size of scale.";
    int64_t workspace_size = gemm_op->get_workspace_size(args);
    workspace_size = (workspace_size + 127) / 128 * 128;
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      this->workspace =
          torch::empty({workspace_size}, weight.options().dtype(at::ScalarType::Byte));
    }
    void *workspace_ptr = this->workspace.defined() ? this->workspace.data_ptr() : nullptr;
    gemm_op->run(args, workspace_ptr, stream);

    return out;
  }

  torch::Tensor
  forward_grouped(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      torch::Tensor input_scale,
      torch::Tensor weight_scale) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return forward_grouped_impl(
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
  wgrad_grouped_impl(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      torch::Tensor input_scale,
      torch::Tensor weight_scale,
      cudaStream_t stream) {
    check_io_dtype(input, weight, bias, output, input_scale, weight_scale);
    CHECK_NDIM(input_list, 1);
    // CHECK_NDIM(input, 2);
    // CHECK_NDIM(weight, 2);
    FLUX_CHECK(input_list.device().type() == torch::kCPU) << "input splits is not on cpu\n";

    int M = input.size(0);
    int N = weight.size(0);
    int K = input_list.sum().item<int>();
    int E = input_list.size(0);

    torch::Tensor out;
    if (output.has_value()) {
      out = output.value();
    } else {
      out = torch::empty({E, M, N}, weight.options().dtype(output_dtype));
    }

    CHECK(out.device().type() == torch::DeviceType::CUDA)
        << "output device: " << out.device().type() << " != torch::DeviceType::CUDA";
    // CHECK(M >= 8) << "M must be greater than or equal 8 for cutlass grouped gemm.";
    // CHECK(N >= 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    // CHECK(K >= 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";

    using UnderlyingProblemShape = cute::Shape<int, int, int>;
    std::vector<UnderlyingProblemShape> problem_sizes;
    std::vector<void const *> ptr_A;
    std::vector<void const *> ptr_B;
    std::vector<void const *> ptr_C;
    std::vector<void *> ptr_D;
    std::vector<void const *> ptr_scale_A;
    std::vector<void const *> ptr_scale_B;

    // TODO (hanshi.s): K need to be padded to 128
    {
      // initialize args vectors
      uint8_t const *ptr_A_cur = reinterpret_cast<uint8_t *>(input.data_ptr());
      uint8_t const *ptr_B_cur = reinterpret_cast<uint8_t *>(weight.data_ptr());
      uint8_t *ptr_D_cur = reinterpret_cast<uint8_t *>(out.data_ptr());
      uint8_t const *ptr_scale_A_cur = reinterpret_cast<uint8_t *>(input_scale.data_ptr());
      uint8_t const *ptr_scale_B_cur = reinterpret_cast<uint8_t *>(weight_scale.data_ptr());

      for (int i = 0; i < E; ++i) {
        int Ki = input_list[i].item().toInt();
        //   if (Ki == 0) { // TODO (hanshi.s): some errors
        //     continue;
        //   }
        problem_sizes.emplace_back(M, N, Ki);
        ptr_A.emplace_back(ptr_A_cur);
        ptr_A_cur += M * Ki * c10::elementSize(input.scalar_type());
        ptr_B.emplace_back(ptr_B_cur);
        ptr_B_cur += N * Ki * c10::elementSize(weight.scalar_type());
        ptr_C.emplace_back(nullptr);
        ptr_D.emplace_back(ptr_D_cur);
        ptr_D_cur += M * N * c10::elementSize(out.scalar_type());
        ptr_scale_A.emplace_back(ptr_scale_A_cur);
        ptr_scale_A_cur += M * Ki / BLOCK_K * c10::elementSize(input_scale.scalar_type());
        ptr_scale_B.emplace_back(ptr_scale_B_cur);
        ptr_scale_B_cur += N * Ki / BLOCK_K * c10::elementSize(weight_scale.scalar_type());
      }
    }

    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto input_dtype = from_torch_dtype(this->input_dtype);
    auto weight_dtype = from_torch_dtype(this->weight_dtype);
    auto output_dtype = from_torch_dtype(this->output_dtype);
    DataTypeEnum accum_type = _FP32{}();
    auto dt_conf = make_gemm_dtype_config(  // bias.has_value() ? output_dtype : _Void{}()
        input_dtype,
        weight_dtype,
        output_dtype,
        output_dtype,
        accum_type);
    auto gemm_layout = _RCR{}();
    auto impl = _GemmGroupedV3{}();
    auto impl_spec = make_gemm_v3_meta(false, true);
    auto meta = make_gemm_meta(dt_conf, arch, sm_core, _CommNone{}, gemm_layout, impl, impl_spec);
    auto rt_conf = make_runtime_config(M, N, K);

    auto hparams = OpRegistry::instance().get_hparams(meta, rt_conf);
    auto &v3_hparams = std::get<unified_type_t<GemmV3HParams>>(hparams.impl_spec());
    v3_hparams.blockscale_M() = _BlockScaleMPerRow();
    v3_hparams.blockscale_N() = _BlockScaleNPerCol();

    OpRegistry::OpPtr gemm_op;
    gemm_op = OpRegistry::instance().get_op(meta, hparams);

    // initialize mnk for streamk get_workspace_size
    const BlockScaleGroupedGemmV3Arguments args{
        .problem_count = static_cast<int>(problem_sizes.size()),
        .problem_sizes = problem_sizes.data(),
        .ptr_A = ptr_A.data(),
        .ptr_B = ptr_B.data(),
        .ptr_C = ptr_C.data(),
        .ptr_D = ptr_D.data(),
        .ptr_blockscale_A = ptr_scale_A.data(),
        .ptr_blockscale_B = ptr_scale_B.data(),
        .alpha = 1.0f,
        .beta = 0.0f,
    };

    auto unified_hparams = gemm_op->get_runtime_gemm_hparams();
    auto tile_shape = unified_hparams.tile_shape();
    auto [tile_M, tile_N, tile_K] = tile_shape;
    FLUX_CHECK(tile_M == BLOCK_M && tile_N == BLOCK_N && tile_K == BLOCK_K)
        << "tile size of gemm is not match block size of scale.";
    int64_t workspace_size = gemm_op->get_workspace_size(args);
    workspace_size = (workspace_size + 127) / 128 * 128;
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      this->workspace =
          torch::empty({workspace_size}, weight.options().dtype(at::ScalarType::Byte));
    }
    void *workspace_ptr = this->workspace.defined() ? this->workspace.data_ptr() : nullptr;
    gemm_op->run(args, workspace_ptr, stream);

    return out;
  }

  torch::Tensor
  wgrad_grouped(
      torch::Tensor input,
      torch::Tensor input_list,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      torch::Tensor input_scale,
      torch::Tensor weight_scale) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    return wgrad_grouped_impl(
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
      c10::optional<torch::Tensor> weight_scale,
      bool is_groupwise_b) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    auto scale_type_b = is_groupwise_b ? GemmBlockScaleNEnum::Col : GemmBlockScaleNEnum::Block;
    return forward_cutlass_impl(
        std::move(input),
        std::move(weight),
        std::move(bias),
        std::move(output),
        std::move(input_scale),
        std::move(weight_scale),
        scale_type_b,
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
    c10::ScalarType input_dtype,
    c10::ScalarType weight_dtype,
    c10::ScalarType output_dtype,
    int32_t num_streams)
    : impl_(new BlockScaleGemm::BlockScaleGemmImpl(
          input_dtype, weight_dtype, output_dtype, num_streams)) {}

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
BlockScaleGemm::wgrad(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->wgrad(
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
BlockScaleGemm::wgrad_multistream(
    torch::Tensor input,
    torch::Tensor input_splits,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->wgrad_multistream(
      std::move(input),
      std::move(input_splits),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale));
}

torch::Tensor
BlockScaleGemm::forward_grouped(
    torch::Tensor input,
    torch::Tensor input_splits,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    torch::Tensor input_scale,
    torch::Tensor weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->forward_grouped(
      std::move(input),
      std::move(input_splits),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale));
}

torch::Tensor
BlockScaleGemm::wgrad_grouped(
    torch::Tensor input,
    torch::Tensor input_splits,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    torch::Tensor input_scale,
    torch::Tensor weight_scale) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->wgrad_grouped(
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
    c10::optional<torch::Tensor> weight_scale,
    bool is_groupwise_b) {
  FLUX_CHECK(impl_ != nullptr) << "BlockScaleGemm is not initialized";
  return impl_->reference(
      std::move(input),
      std::move(weight),
      std::move(bias),
      std::move(output),
      std::move(input_scale),
      std::move(weight_scale),
      is_groupwise_b);
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
