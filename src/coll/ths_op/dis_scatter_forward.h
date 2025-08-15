//===- dis_scatter_forward.h -------------------------------- C++ ---===//
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
#include <torch/all.h>
#include "flux/ths_op/ths_op.h"

namespace bytedance::flux::ths_op {

class DisScatterForward {
 public:
  DisScatterForward(
      int64_t total_num_experts,
      int64_t max_m,
      int64_t n_dim,
      int64_t topk,
      int64_t rank, /** rank in team  */
      int64_t tp_world_size,
      int64_t ep_world_size,
      int64_t local_world_size,
      float moe_capacity_ratio,
      int64_t duplicate_comm_buffer,
      int nvshmem_team = 0 /** NVSHMEM_TEAM_WORLD = 0 */);

  ~DisScatterForward();
  torch::Tensor forward(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int64_t sm_margin,
      bool copy_to_local_tensor);
  torch::Tensor forward_gpu(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      torch::Tensor output_tensor,
      int64_t sm_margin,
      bool copy_to_local_tensor);
  std::vector<torch::Tensor> pre_comm_index(
      torch::Tensor ep_token_counts_cpu,
      torch::Tensor cur_topk_indices,
      torch::Tensor cur_topk_values,
      int64_t sm_margin);
  std::vector<torch::Tensor> pre_comm_index_gpu(
      torch::Tensor ep_token_counts_gpu,
      torch::Tensor cur_topk_indices,
      torch::Tensor cur_topk_values,
      torch::Tensor ag_topk_indices,
      torch::Tensor ag_topk_values,
      int64_t sm_margin);
  void ep_barrier_all();
  torch::Tensor copy_from_output_comm_buffer(torch::Tensor output, int64_t comm_buffer_id);
  void copy_to_input_comm_buffer(torch::Tensor input, int64_t comm_buffer_id);
  torch::Tensor get_input_comm_buffer(
      std::vector<int64_t> shape, at::ScalarType data_type, int64_t comm_buffer_id);
  torch::Tensor forward_gpu_no_cpy(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits,
      torch::Tensor ep_token_counts,
      torch::Tensor output_tensor,
      int64_t sm_margin,
      int64_t comm_buffer_id);
  torch::Tensor forward_no_cpy(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int64_t sm_margin,
      int64_t comm_buffer_id);
  torch::Tensor profiling(
      torch::Tensor input,
      torch::Tensor ag_exp_indices,
      torch::Tensor ag_scatter_idx,
      torch::Tensor splits_cpu,
      torch::Tensor ep_token_counts,
      int64_t sm_margin,
      c10::intrusive_ptr<ProfilingContext> opt_ctx);

 private:
  class DisScatterForwardOpImpl;
  DisScatterForwardOpImpl *impl_;
};

}  // namespace bytedance::flux::ths_op
