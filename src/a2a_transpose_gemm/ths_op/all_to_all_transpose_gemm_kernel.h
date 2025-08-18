//===- all_to_all_transpose_gemm_kernel.h ------------------------- C++ ---===//
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

#pragma once
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/ths_op/ths_op.h"
#include "a2a_transpose_gemm/ths_op/all_to_all_types.h"

namespace bytedance::flux::ths_op {

class AllToAllTransposeGemmOp {
 public:
  AllToAllTransposeGemmOp(
      std::shared_ptr<Group> pg_world,
      int32_t nnodes,
      int32_t sp_size,
      int32_t bs,
      int32_t num_head,
      int32_t seq,
      int32_t head_dim,
      int32_t max_num_comm_buf,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool a2a_only);

  ~AllToAllTransposeGemmOp();

  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> seq_lens_cpu,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllToAllOptionWithOptional option,
      c10::optional<torch::Tensor> a2a_transpose_output,
      int32_t num_comm_sm,
      int32_t sm_margin);

  torch::Tensor post_attn_a2a(
      torch::Tensor input,
      c10::optional<torch::Tensor> seq_lens_cpu,
      AllToAllOptionWithOptional opt,
      int32_t num_comm_sm);

  torch::Tensor post_attn_a2a_no_cpy(
      torch::Tensor input,
      c10::optional<torch::Tensor> seq_lens_cpu,
      AllToAllOptionWithOptional opt,
      int32_t num_comm_sm,
      int32_t comm_buf_idx);

  void sp_group_barrier_all();

  torch::Tensor gemm_only(
      torch::Tensor input,  // this should be the full input
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,  // this should be the full scale
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      int32_t num_comm_sm,
      int32_t sm_margin);

  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> output,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      bool transpose_weight,
      AllToAllOptionWithOptional option,
      c10::optional<torch::Tensor> a2a_transpose_output,
      int32_t num_comm_sm,
      int32_t sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class AllToAllTransposeGemmOpImpl;
  AllToAllTransposeGemmOpImpl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
