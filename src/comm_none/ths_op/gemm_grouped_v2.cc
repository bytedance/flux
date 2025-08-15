//===- gemm_grouped_v2.cc ----------------------------------------- C++ ---===//
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

#include "comm_none/ths_op/gemm_grouped_v2.h"
#include "flux/args/comm_none.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/ths_op/util.h"
#include <ATen/core/jit_type.h>
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/DeviceType.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <torch/all.h>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <cutlass/gemm_coord.h>

namespace bytedance {
namespace flux {
namespace ths_op {

using torch::Tensor;

class GemmGroupedV2::GemmGroupedV2Impl {
 private:
  torch::Tensor weight;
  const int num_experts;
  const at::ScalarType in_type;
  const at::ScalarType out_type;
  const bool is_fp8_gemm;

  torch::Tensor workspace;

 private:
  void
  init_workspace(int64_t workspace_size) {
    if (workspace_size <= 0) {
      return;
    }
    workspace_size = (workspace_size + 127) / 128 * 128;
    if (!this->workspace.defined() || workspace_size > this->workspace.numel()) {
      this->workspace =
          torch::empty({workspace_size}, this->weight.options().dtype(at::ScalarType::Byte));
    }
  }

 public:
  GemmGroupedV2Impl(
      torch::Tensor weight, int64_t num_experts, at::ScalarType in_type, at::ScalarType out_type)
      : weight(weight),
        num_experts(num_experts),
        in_type(in_type),
        out_type(out_type),
        is_fp8_gemm(is_fp8_torch_dtype(in_type)) {
    CHECK_INPUT(weight, in_type);
    FLUX_CHECK_EQ(weight.dim(), 3) << "gemm grouped weight shape is not 3";
    FLUX_CHECK_EQ(weight.size(0), num_experts) << "gemm grouped 0-dim is not equal to num_experts";
  }

  torch::Tensor
  forward(
      torch::Tensor input,
      torch::Tensor splits_cpu,
      c10::optional<std::vector<torch::Tensor>> input_scale,
      c10::optional<std::vector<torch::Tensor>> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int64_t sm_margin) {
    // input: [M, K], weight: [E, N, K], output: [M, N]
    CHECK_INPUT(input, this->in_type);
    FLUX_CHECK_EQ(input.dim(), 2) << "input shape is not 2";
    FLUX_CHECK_EQ(splits_cpu.dim(), 1) << "splits_cpu shape is not 1";
    FLUX_CHECK_EQ(splits_cpu.device(), torch::DeviceType::CPU);
    CHECK(
        splits_cpu.scalar_type() == at::ScalarType::Int ||
        splits_cpu.scalar_type() == at::ScalarType::Long)
        << "splits_cpu type can be int or long";

    int M = input.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);

    FLUX_CHECK_EQ(weight.size(2), K);
    FLUX_CHECK_EQ(splits_cpu.size(0), E);

    auto output = torch::empty(
        {M, N},
        at::TensorOptions().dtype(out_type).device(at::kCUDA).device_index(
            at::cuda::current_device()));

    FLUX_CHECK_GE(M, 8) << "M must be greater than or equal 8 for cutlass grouped gemm.";
    FLUX_CHECK_GE(N, 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    FLUX_CHECK_GE(K, 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";

    std::vector<cutlass::gemm::GemmCoord> problem_sizes;
    std::vector<void *> ptr_A;
    std::vector<void *> ptr_B;
    std::vector<void *> ptr_C;
    std::vector<void *> ptr_D;
    std::vector<void *> ptr_Aux;
    std::vector<void *> ptr_Vector;

    // Initialize args vectors
    uint8_t *ptr_A_cur = reinterpret_cast<uint8_t *>(input.data_ptr());
    uint8_t *ptr_B_cur = reinterpret_cast<uint8_t *>(weight.data_ptr());
    uint8_t *ptr_D_cur = reinterpret_cast<uint8_t *>(output.data_ptr());

    int problem_count = this->num_experts;
    for (int i = 0; i < problem_count; ++i) {
      int Mi = splits_cpu[i].item().toInt();
      if (Mi == 0) {
        continue;
      }
      problem_sizes.emplace_back(cutlass::gemm::GemmCoord(Mi, N, K));
      ptr_A.emplace_back(ptr_A_cur);
      ptr_A_cur += Mi * K * c10::elementSize(input.scalar_type());
      ptr_B.emplace_back(ptr_B_cur);
      ptr_B_cur += N * K * c10::elementSize(weight.scalar_type());
      ptr_C.emplace_back(nullptr);  // null
      ptr_D.emplace_back(ptr_D_cur);
      ptr_D_cur += Mi * N * c10::elementSize(output.scalar_type());
      ptr_Aux.emplace_back(nullptr);     // null
      ptr_Vector.emplace_back(nullptr);  // null
    }

    std::vector<float *> input_scale_vec(problem_count, nullptr);
    if (input_scale.has_value() && !input_scale->empty()) {
      bool has_one_scale = input_scale->size() == 1;
      for (int i = 0; i < problem_count; ++i) {
        input_scale_vec[i] = (float *)(input_scale->at(has_one_scale ? 0 : i).data_ptr());
      }
    }
    std::vector<float *> weight_scale_vec(problem_count, nullptr);
    if (weight_scale.has_value() && !weight_scale->empty()) {
      bool has_one_scale = weight_scale->size() == 1;
      for (int i = 0; i < problem_count; ++i) {
        weight_scale_vec[i] = (float *)(weight_scale->at(has_one_scale ? 0 : i).data_ptr());
      }
    }
    float *output_scale_ptr =
        (float *)(output_scale.has_value() ? output_scale->data_ptr() : nullptr);

    auto args = GemmGroupedV2Arguments{
        .problem_sizes = problem_sizes.data(),
        .problem_count = static_cast<int>(problem_sizes.size()),
        .alpha = 1.0f,
        .beta = 0.0f,
        .ptr_A = ptr_A.data(),
        .ptr_B = ptr_B.data(),
        .ptr_C = ptr_C.data(),
        .ptr_D = ptr_D.data(),
        .ptr_Aux = ptr_Aux.data(),
        .ptr_Vector = ptr_Vector.data(),
        .abs_max_Aux = nullptr,
        .abs_max_D = nullptr,
        .scaleA = (float const **)input_scale_vec.data(),
        .scaleB = (float const **)weight_scale_vec.data(),
        .scaleC = nullptr,
        .scaleD = output_scale_ptr,
        .scaleAux = nullptr,
        .sm_margin = sm_margin};

    auto arch = get_arch();
    auto sm_core = get_sm_core();
    auto input_dtype = from_torch_dtype(this->in_type);
    auto output_dtype = from_torch_dtype(this->out_type);
    auto dtype_c = is_fp8_gemm ? _BF16{} : input_dtype;
    auto dtype_config = make_gemm_dtype_config(input_dtype, input_dtype, dtype_c, output_dtype);
    FLUX_CHECK(is_fp8_gemm || !fast_accum) << "only FP8 support fast accum";
    auto v2_meta = make_gemm_v2_meta(fast_accum);

    // Here use RCR layout, but support both RCR and RCC
    auto meta = make_gemm_meta(
        dtype_config, arch, sm_core, _CommNone{}, _RCR{}, _GemmGroupedV2{}, v2_meta);
    auto rt_conf = make_runtime_config(cute::ceil_div(M, args.problem_count), N, K);
    auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->init_workspace(workspace_size);
    void *workspace_ptr = this->workspace.defined() ? this->workspace.data_ptr() : nullptr;
    gemm_op->run(args, workspace_ptr, stream);
    return output;
  }

  std::tuple<torch::Tensor, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->weight, this->num_experts);
  }
};

GemmGroupedV2::GemmGroupedV2(
    torch::Tensor weight, int64_t num_experts, at::ScalarType in_type, at::ScalarType out_type)
    : impl_(new GemmGroupedV2::GemmGroupedV2Impl(weight, num_experts, in_type, out_type)) {}
GemmGroupedV2::~GemmGroupedV2() { delete impl_; }
torch::Tensor
GemmGroupedV2::forward(
    torch::Tensor input,
    torch::Tensor splits_cpu,
    c10::optional<std::vector<torch::Tensor>> input_scale,
    c10::optional<std::vector<torch::Tensor>> weight_scale,
    c10::optional<torch::Tensor> output_scale,
    bool fast_accum,
    int64_t sm_margin) {
  FLUX_CHECK(impl_ != nullptr) << "GemmGroupedV2 impl is nullptr";
  return impl_->forward(
      input, splits_cpu, input_scale, weight_scale, output_scale, fast_accum, sm_margin);
}

}  // namespace ths_op
}  // namespace flux
}  // namespace bytedance
