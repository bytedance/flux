//===- gemm_with_barrier.cc --------------------------------------- C++ ---===//
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
#include "ag_gemm/ths_op/gemm_with_barrier.h"

#include <c10/cuda/CUDAStream.h>

#include "coll/ths_op/all_gather_types.h"
#include "flux/args/ag_gemm.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"

namespace bytedance::flux {

AGGemmMeta
get_gemm_meta(
    at::ScalarType input_torch_dtype,
    at::ScalarType output_torch_dtype,
    bool transpose_weight,
    bool has_bias,
    bool fast_accum) {
  ArchEnum arch = get_arch();
  SMCoreEnum sm_core = get_sm_core();
  auto input_dtype = ths_op::from_torch_dtype(input_torch_dtype);
  auto output_dtype = ths_op::from_torch_dtype(output_torch_dtype);

  bool is_fp8_gemm = ths_op::is_fp8_torch_dtype(input_torch_dtype);
  bool is_s8_gemm = ths_op::is_s8_torch_dtype(input_torch_dtype);
  DataTypeEnum accum_type = is_s8_gemm ? _S32{}() : _FP32{}();
  DataTypeEnum block_scale_type = _FP32{}();
  auto dtype_config = make_gemm_dtype_config(
      input_dtype,
      input_dtype,
      has_bias ? output_dtype : _Void{}(),
      output_dtype,
      accum_type,
      block_scale_type);

  auto gemm_layout = transpose_weight ? _RRR{}() : _RCR{}();
  UnifiedImplMeta impl_spec = None{};

  bool use_fast_accum = fast_accum && is_fp8_gemm;
  auto impl = ((int)arch < (int)_Sm90{}()) ? _GemmV2{}() : _GemmV3{}();
  if (impl == _GemmV2{}) {
    impl_spec = make_gemm_v2_meta(use_fast_accum);
  } else if (impl == _GemmV3{}) {
    impl_spec = unify_type(make_gemm_v3_meta(use_fast_accum));
  }

  auto meta =
      make_gemm_meta(dtype_config, arch, sm_core, _AGKernel{}, gemm_layout, impl, impl_spec);
  return meta;
}

RuntimeConfig
get_rt_config(
    int world_size,
    int nnodes,
    int m,
    int n,
    int k,
    AGRingMode ring_mode) {  // TODO(houqi.1993) what about ring mode
  return make_runtime_config(
      m, n, k, make_all_gather_runtime_config(world_size, nnodes, (int)ring_mode));
}

void
GemmWithBarirer::lazy_init_gemm_buffer(torch::Tensor input, int64_t buffer_size) {
  if (buffer_size <= 0)
    return;
  buffer_size = (buffer_size + 127) / 128 * 128;
  if (!this->gemm_buffer.defined() || buffer_size > this->gemm_buffer.numel()) {
    auto options = input.options().dtype(c10::ScalarType::Byte);
    this->gemm_buffer = torch::empty({buffer_size}, options);
  }
}

GemmWithBarirer::GemmWithBarirer(int rank, int world_size, int32_t nnodes)
    : nnodes(nnodes), world_size(world_size), rank(rank) {}

torch::Tensor
GemmWithBarirer::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    torch::Tensor barrier,
    bool fast_accum,
    int32_t *producer_signal,
    bool transpose_weight) {
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  return forward(
      input,
      weight,
      bias,
      output,
      input_scale,
      weight_scale,
      output_scale,
      barrier,
      fast_accum,
      transpose_weight,
      c10::nullopt,  // use default hparams
      producer_signal,
      stream);
}

torch::Tensor
GemmWithBarirer::forward(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    torch::Tensor barrier,
    bool fast_accum,
    bool transpose_weight,
    c10::optional<UnifiedGemmHParams> const &hparams,
    int32_t *producer_signal,
    cudaStream_t stream) {
  auto output_tensor = this->initialize(
      input,
      weight,
      bias,
      output,
      input_scale,
      weight_scale,
      output_scale,
      barrier,
      fast_accum,
      transpose_weight,
      hparams,
      stream);

  // if not a nullptr, gemm need to wait producer kernel to be launch.
  if (producer_signal != nullptr) {
    CU_CHECK(
        CUStreamWaitValue(stream, (CUdeviceptr)(producer_signal), 1, CU_STREAM_WAIT_VALUE_EQ));
  }

  /// GEMM
  this->run(stream, /*launch_with_pdl=*/false);
  return output_tensor;
}

torch::Tensor
GemmWithBarirer::initialize(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    c10::optional<torch::Tensor> output,
    c10::optional<torch::Tensor> input_scale,
    c10::optional<torch::Tensor> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    torch::Tensor barrier,
    bool fast_accum,
    bool transpose_weight,
    c10::optional<UnifiedGemmHParams> const &hparams,
    cudaStream_t stream) {
  at::ScalarType input_dtype = input.scalar_type();
  bool is_fp8_gemm = ths_op::is_fp8_torch_dtype(input_dtype);
  bool is_s8_gemm = ths_op::is_s8_torch_dtype(input_dtype);
  at::ScalarType output_dtype =
      (is_fp8_gemm || is_s8_gemm) ? at::ScalarType::BFloat16 : input_dtype;

  // verify all kinds of shapes
  FLUX_CHECK(!(transpose_weight && is_fp8_gemm)) << "FP8 GEMM does not support transpose weight";

  CHECK_NDIM(input, 2);
  CHECK_CUDA(input);
  CHECK_NDIM(weight, 2);
  CHECK_CUDA(weight);

  int m = input.size(0);
  int n = transpose_weight ? weight.size(1) : weight.size(0);
  int k = transpose_weight ? weight.size(0) : weight.size(1);
  CHECK_2D(input, m, k);
  CHECK_TYPE(weight, input_dtype);

  if (bias.has_value()) {
    CHECK_2D(bias.value(), (is_fp8_gemm || is_s8_gemm) ? 1 : m, n);
    CHECK_TYPE(bias.value(), output_dtype);
    CHECK_CUDA(bias.value());
  }
  if (!is_s8_gemm && !is_fp8_gemm) {
    FLUX_CHECK(!input_scale.has_value());
    FLUX_CHECK(!weight_scale.has_value());
  }

  auto meta = get_gemm_meta(
      input_dtype,
      output_dtype,
      transpose_weight,
      /*has_bias=*/bias.has_value(),
      /*fast_accum=*/fast_accum);
  auto rt_config = get_rt_config(
      this->world_size,
      this->nnodes,
      m,
      n,
      k,
      AGRingMode::All2All);  // TODO(houqi.1993) set this later
  if (hparams.has_value()) {
    this->cutlass_op = OpRegistry::instance().get_op(meta, hparams.value());
  } else {
    auto params = OpRegistry::instance().get_hparams(meta, rt_config);
    this->cutlass_op = OpRegistry::instance().get_op(meta, params);
  }
  torch::Tensor output_tensor;

  if (output.has_value()) {
    output_tensor = output.value();
    CHECK_2D(output_tensor, m, n);
    CHECK_TYPE(output_tensor, output_dtype);
  } else {
    output_tensor = torch::empty(
        {m, n},
        at::TensorOptions(output_dtype)
            .device(at::kCUDA)
            .device_index(c10::cuda::current_device()));
  }
  std::any gemm_args;
  auto data_ptr_or = [](auto &&t, void *other) -> void * {
    return t.has_value() ? t->data_ptr() : other;
  };
  if (is_fp8_gemm) {
    // TODO(houqi.1993) what about output_scale?
    if (output_scale.has_value()) {
      CHECK_CUDA(output_scale.value());
      CHECK_TYPE(output_scale.value(), at::ScalarType::Float);
    }
    gemm_args = AGFP8KernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .rank = rank,
        .world_size = world_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = 0.0f,
        .A = input.data_ptr(),
        .B = weight.data_ptr(),
        .C = nullptr,
        .Aux = nullptr,
        .D = output_tensor.data_ptr(),
        .barrier_buffer = barrier.data_ptr(),
        .Vector = data_ptr_or(bias, nullptr),
        .abs_max_Aux = nullptr,
        .abs_max_D = nullptr,
        .scaleA = (float *)data_ptr_or(input_scale, nullptr),
        .scaleB = (float *)data_ptr_or(weight_scale, nullptr),
        .scaleC = nullptr,
        .scaleD = (float *)data_ptr_or(output_scale, nullptr),
        .scaleAux = nullptr};
  } else if (is_s8_gemm) {
    // check input_scale
    FLUX_CHECK(input_scale.has_value());
    torch::Tensor input_scale_t = input_scale.value();
    CHECK_2D(input_scale_t, m, 1);
    CHECK_TYPE(input_scale_t, at::ScalarType::Float);
    CHECK_CUDA(input_scale_t);
    // check weight_scale
    FLUX_CHECK(weight_scale.has_value());
    torch::Tensor weight_scale_t = weight_scale.value();
    CHECK_2D(weight_scale_t, 1, n);
    CHECK_TYPE(weight_scale_t, at::ScalarType::Float);
    CHECK_CUDA(weight_scale_t);

    gemm_args = AGS8KernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .rank = rank,
        .world_size = world_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .A = input.data_ptr(),
        .B = weight.data_ptr(),
        .bias = data_ptr_or(bias, nullptr),
        .output = output_tensor.data_ptr(),
        .scale_A = (float *)input_scale_t.data_ptr(),
        .scale_B = (float *)weight_scale_t.data_ptr(),
        .barrier_buffer = barrier.data_ptr()};
  } else {
    // AG GEMM Arguments
    gemm_args = AGKernelArguments{
        .m = m,
        .n = n,
        .k = k,
        .rank = rank,
        .world_size = world_size,
        .nnodes = nnodes,
        .alpha = 1.0f,
        .beta = bias.has_value() ? 1.0f : 0.0f,
        .input = input.data_ptr(),
        .weight = weight.data_ptr(),
        .bias = data_ptr_or(bias, nullptr),
        .output = output_tensor.data_ptr(),
        .barrier_buffer = barrier.data_ptr()};
  }

  // AG Gemm Workspace
  int64_t workspace_size = this->cutlass_op->get_workspace_size(gemm_args);
  this->lazy_init_gemm_buffer(input, workspace_size);

  /// GEMM initialize
  this->cutlass_op->initialize(
      gemm_args, workspace_size ? this->gemm_buffer.data_ptr() : nullptr, stream);
  return output_tensor;
}

void
GemmWithBarirer::run(cudaStream_t stream, bool launch_with_pdl) {
  this->cutlass_op->run(stream, launch_with_pdl);
}

}  // namespace bytedance::flux
