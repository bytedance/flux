//===- gemm_all2all_transpose.h ----------------------------------- C++ ---===//
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
#include "gemm_a2a_transpose/ths_op/pre_attn_a2a_types.h"
#include <c10/core/ScalarType.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {

class GemmAllToAllTransposeOp {
 public:
  GemmAllToAllTransposeOp(
      std::shared_ptr<Group> pg_world,
      int32_t nnodes,
      int32_t sp_size,
      int32_t bs,
      int32_t seq,
      int32_t hidden_dim,
      int32_t head_dim,
      int32_t n_dim,
      c10::ScalarType input_dtype,
      c10::ScalarType output_dtype,
      bool transpose_weight,
      int32_t gqa,
      PreAttnAllToAllCommOp comm_op,
      int32_t max_num_comm_buf);

  ~GemmAllToAllTransposeOp();

  std::vector<torch::Tensor> forward(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> seq_lens_cpu,
      c10::optional<torch::Tensor> bias,
      c10::optional<std::vector<torch::Tensor>> outputs,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int32_t num_comm_sm,
      int32_t sm_margin);

  std::vector<torch::Tensor> pre_attn_qkv_pack_a2a(
      torch::Tensor qkv, c10::optional<torch::Tensor> seq_lens_cpu, int32_t num_comm_sm);

  void sp_group_barrier_all();

  torch::Tensor pre_attn_a2a(
      torch::Tensor input, c10::optional<torch::Tensor> seq_lens_cpu, int32_t num_comm_sm);

  std::vector<torch::Tensor> pre_attn_qkv_pack_a2a_no_cpy(
      torch::Tensor qkv,
      c10::optional<torch::Tensor> seq_lens_cpu,
      int32_t num_comm_sm,
      int32_t comm_buf_idx);

  torch::Tensor pre_attn_a2a_no_cpy(
      torch::Tensor input,
      c10::optional<torch::Tensor> seq_lens_cpu,
      int32_t num_comm_sm,
      int32_t comm_buf_idx);

  torch::Tensor get_input_comm_buf(torch::Tensor input, int32_t comm_buf_idx);

  std::vector<torch::Tensor> profiling(
      torch::Tensor input,
      torch::Tensor weight,
      c10::optional<torch::Tensor> bias,
      c10::optional<torch::Tensor> input_scale,
      c10::optional<torch::Tensor> weight_scale,
      c10::optional<torch::Tensor> output_scale,
      bool fast_accum,
      int32_t num_comm_sm,
      int32_t sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class GemmAllToAllTransposeOpImpl;
  GemmAllToAllTransposeOpImpl *impl_ = nullptr;
};
}  // namespace bytedance::flux::ths_op
