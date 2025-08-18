//===- gemm_grouped_v3.cc ----------------------------------------- C++ ---===//
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

#include "comm_none/ths_op/gemm_grouped_v3.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/ths_op/ths_op.h"
#include "flux/cuda/cuda_common.h"
#include "flux/args/comm_none.h"
#include "flux/flux.h"
#include "flux/ths_op/util.h"
#include "torch/all.h"
#include "cutlass/gemm/gemm.h"
#include <ATen/core/List.h>
#include <ATen/ops/empty.h>
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <ATen/core/jit_type.h>
#include <c10/cuda/CUDAStream.h>
#include <numa.h>
#include <cstdlib>
#include "cutlass/util/packed_stride.hpp"

namespace bytedance::flux::ths_op {

/// This class only runs the basic grouped_gemm, it is mainly used for testing
class GemmGroupedV3::GemmGroupedV3Impl {
 private:
  const at::ScalarType _st;
  torch::Tensor weight;
  const int num_experts;
  torch::Tensor workspace;

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
  GemmGroupedV3Impl(torch::Tensor weight, int64_t num_experts)
      : _st(weight.scalar_type()), weight(weight), num_experts(num_experts) {
    CHECK_INPUT(weight, _st);
    CHECK(weight.dim() == 3) << "gemm grouped weight shape is not 3";
    CHECK(weight.size(0) == num_experts) << "gemm grouped 0-dim is not equal to num_experts";
  }

  torch::Tensor
  forward_impl(
      torch::Tensor input,
      torch::Tensor splits_cpu,
      c10::optional<UnifiedGemmHParams> const &hparams) {
    CHECK_INPUT(input, this->_st);
    CHECK(input.dim() == 2) << "input shape is not 2";

    CHECK(splits_cpu.dim() == 1) << "splits_cpu shape is not 1";
    CHECK(
        splits_cpu.scalar_type() == at::ScalarType::Int ||
        splits_cpu.scalar_type() == at::ScalarType::Long)
        << "splits_cpu type can be int or long";

    int M = input.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);

    CHECK(weight.size(2) == K) << "weight 2-dim mismatch: " << weight.size(2) << " != " << K;
    CHECK(splits_cpu.size(0) == E)
        << "splits 0-dim mismatch: " << splits_cpu.size(0) << " != " << E;

    auto output_type = this->_st;
    bool is_fp8 = c10::isFloat8Type(this->_st);
    if (is_fp8) {
      output_type = at::ScalarType::BFloat16;
    }
    auto output = torch::empty(
        {M, N},
        at::TensorOptions()
            .dtype(output_type)
            .device(at::kCUDA)
            .device_index(c10::cuda::current_device()));

    CHECK(output.device().type() == torch::DeviceType::CUDA)
        << "output device: " << output.device().type() << " != torch::DeviceType::CUDA";
    CHECK(M >= 8) << "M must be greater than or equal 8 for cutlass grouped gemm.";
    CHECK(N >= 8) << "N must be greater than or equal 8 for cutlass grouped gemm.";
    CHECK(K >= 8) << "K must be greater than or equal 8 for cutlass grouped gemm.";

    using UnderlyingProblemShape = cute::Shape<int, int, int>;
    std::vector<UnderlyingProblemShape> problem_sizes;
    std::vector<void const *> ptr_A;
    std::vector<void const *> ptr_B;
    std::vector<void const *> ptr_C;
    std::vector<void *> ptr_D;
    {
      // initialize args vectors
      uint8_t const *ptr_A_cur = reinterpret_cast<uint8_t const *>(input.const_data_ptr());
      uint8_t const *ptr_B_cur = reinterpret_cast<uint8_t const *>(weight.const_data_ptr());
      uint8_t *ptr_D_cur = reinterpret_cast<uint8_t *>(output.data_ptr());

      for (int i = 0; i < this->num_experts; ++i) {
        int Mi = splits_cpu[i].item().toInt();
        if (Mi == 0) {
          continue;
        }
        problem_sizes.emplace_back(N, Mi, K);
        ptr_A.emplace_back(ptr_A_cur);
        ptr_A_cur += Mi * K * c10::elementSize(input.scalar_type());
        ptr_B.emplace_back(ptr_B_cur);
        ptr_B_cur += N * K * c10::elementSize(weight.scalar_type());
        ptr_C.emplace_back(nullptr);
        ptr_D.emplace_back(ptr_D_cur);
        ptr_D_cur += Mi * N * c10::elementSize(output.scalar_type());
      }
    }

    auto args = GemmGroupedV3Arguments{
        .problem_count = static_cast<int>(problem_sizes.size()),
        .alpha = 1.0f,
        .beta = 0.0f,
        .problem_sizes = problem_sizes.data(),
        // swap AB to enable RCC layout
        .ptr_A = ptr_B.data(),
        .ptr_B = ptr_A.data(),
        .ptr_C = ptr_C.data(),
        .ptr_D = ptr_D.data()};

    ArchEnum arch = get_arch();
    auto dtype = from_torch_dtype(this->_st);
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(dtype));
    auto v3_meta = make_gemm_v3_meta(_False{});
    auto meta = make_gemm_meta(dt_conf, arch, _CommNone{}, _RCC{}, _GemmGroupedV3{}(), v3_meta);
    auto rt_conf = make_runtime_config(N, cute::ceil_div(M, args.problem_count), K);
    OpRegistry::OpPtr gemm_op;
    if (hparams.has_value()) {
      gemm_op = OpRegistry::instance().get_op(meta, hparams.value());
    } else {
      gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    }
    // auto gemm_op = OpRegistry::instance().get_op(meta, rt_conf);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    int64_t workspace_size = gemm_op->get_workspace_size(args);
    this->init_workspace(workspace_size);
    void *workspace_ptr = this->workspace.defined() ? this->workspace.data_ptr() : nullptr;
    gemm_op->run(args, workspace_ptr, stream);
    return output;
  }

  torch::Tensor
  forward(torch::Tensor input, torch::Tensor splits_cpu) {
    return forward_impl(input, splits_cpu, c10::nullopt);
  }

  torch::Tensor
  profiling(
      torch::Tensor input,
      torch::Tensor splits_cpu,
      c10::intrusive_ptr<ProfilingContext> opt_ctx) {
    int M = input.size(0);
    int K = input.size(1);
    int E = weight.size(0);
    int N = weight.size(1);
    ArchEnum arch = get_arch();
    auto dtype = from_torch_dtype(this->_st);
    auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(dtype));
    auto v3_meta = make_gemm_v3_meta(_False{});
    auto meta = unify_type(
        make_gemm_meta(dt_conf, arch, _CommNone{}, _RCC{}, _GemmGroupedV3{}(), v3_meta));
    auto rt_conf = make_runtime_config(N, cute::ceil_div(M, this->num_experts), K);

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
            auto output [[maybe_unused]] = this->forward_impl(input, splits_cpu, hparams);
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

    return this->forward_impl(std::move(input), std::move(splits_cpu), std::move(best_hparams));
  }

  std::tuple<torch::Tensor, int64_t>
  get_pickle_info() const {
    return std::make_tuple(this->weight, this->num_experts);
  }
};
GemmGroupedV3::GemmGroupedV3(torch::Tensor weight, int64_t num_experts)
    : impl_(new GemmGroupedV3::GemmGroupedV3Impl(weight, num_experts)) {}
GemmGroupedV3::~GemmGroupedV3() { delete impl_; }
torch::Tensor
GemmGroupedV3::forward(torch::Tensor input, torch::Tensor splits_cpu) {
  return this->impl_->forward(input, splits_cpu);
}
torch::Tensor
GemmGroupedV3 ::profiling(
    torch::Tensor input, torch::Tensor splits_cpu, c10::intrusive_ptr<ProfilingContext> opt_ctx) {
  return this->impl_->profiling(input, splits_cpu, opt_ctx);
}

}  // namespace bytedance::flux::ths_op
