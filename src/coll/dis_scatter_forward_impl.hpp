//===- dis_scatter_forward_impl.hpp --------------------------------------- C++ ---===//
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
#include "flux/flux.h"
#include <nvshmemx.h>
#include <nvshmem.h>
#define ALL2ALL_DISPATCH_BLOCK 8
#define ALL2ALL_DISPATCH_SPLITS 4
namespace bytedance {
namespace flux {

struct DisScatterForwardBuildIndexParams {
  int32_t *ag_exp_indices;
  int32_t *ag_scatter_idx;
  int32_t total_num_experts;
  int32_t n_nodes;
  int32_t n_tokens_cur_ep;
  int32_t max_token_per_rank;
  int32_t topk;
  // rank / ep_ranks_per_node
  int32_t cur_node_id;
  // global_token_start/end stores the offset the first and last token
  // of the corresponding rank who has the same local rank as current gpu
  int32_t global_token_start[kMaxNodes];
  int32_t global_token_end[kMaxNodes];

  // for forward_gpu, we can calculate global_token_start/end through ep_token_cum_sum_gpu_ptr
  int32_t *ep_token_cum_sum_gpu_ptr;
  int32_t local_rank;
  int32_t local_world_size;

  // following are outputs
  // [n_nodes * N_DISPATCH_SPLITS]
  int32_t *token_count_send;
  // [n_nodes * max_token * N_DISPATCH_SPLITS]
  int32_t *block_idx_to_send;
  // [n_nodes * max_token * N_DISPATCH_SPLITS]
  int32_t *token_count_in_block;
  // [n_nodes * max_token * N_DISPATCH_SPLITS]
  int32_t *sorted_token_idx_send;
  // [n_nodes * max_token * N_DISPATCH_SPLITS]
  int32_t *token_count_recv;
  // [n_nodes * max_token * N_DISPATCH_SPLITS]
  int32_t *sorted_token_idx_recv;

  __device__ __forceinline__ int32_t
  get_global_token_start(int32_t node_id) const {
    if (ep_token_cum_sum_gpu_ptr != nullptr) {
      int32_t target_rank = local_rank + node_id * local_world_size;
      return ep_token_cum_sum_gpu_ptr[target_rank];
    } else {
      return global_token_start[node_id];
    }
  }

  __device__ __forceinline__ int32_t
  get_global_token_end(int32_t node_id) const {
    if (ep_token_cum_sum_gpu_ptr != nullptr) {
      int32_t target_rank = local_rank + node_id * local_world_size;
      return ep_token_cum_sum_gpu_ptr[target_rank + 1];
    } else {
      return global_token_end[node_id];
    }
  }
};

struct DisScatterForwardParams {
  void *internal_ptrs[kMaxLocalWorldSize];  // input_ptrs[rank]: (m * world_size, n)
  void *output_ptrs[kMaxLocalWorldSize];
  void *barrier_ptrs[kMaxLocalWorldSize];
  int32_t ep_cum_sum[kMaxLocalWorldSize];
  int32_t *local_barrier;
  // has the same meaning as the ep_cum_sum arrary
  // when calling forward_gpu, ep_cum_sum_gpu_ptr
  // saves the effective cum sum shape information,
  // in contrast, ep_cum_sum array stores the cum sum
  // informations
  int32_t *ep_cum_sum_gpu_ptr;
  int32_t world_size;
  int32_t rank;
  int32_t local_world_size;
  int32_t n_tokens_cur_ep;
  int32_t hidden_dim;
  int32_t n_threadblocks;
  DisScatterForwardBuildIndexParams index_args;
  int nvshmem_team;
};

struct DisScatterPreCommIndexParams {
  // input
  int32_t ep_cum_sum[kMaxWorldSize];
  int32_t *ep_cum_sum_gpu_ptr;
  int32_t topk;
  int32_t rank;
  int32_t world_size;
  int32_t total_num_experts;
  int32_t n_threadblocks;
  int index_element_size;
  int value_element_size;
  int indice_n_dim;
  int values_n_dim;
  int32_t *local_barrier;
  void *cur_topk_idx;
  void *cur_topk_val;
  void *ag_topk_idx_comm;
  void *ag_topk_val_comm;
  // output
  void *ag_topk_idx_re;
  void *ag_topk_val_re;
  int nvshmem_team;
};

void dis_scatter_forward_flatten_impl(const DisScatterForwardParams &params, cudaStream_t stream);
void dis_scatter_forward_build_index_flatten_impl(
    const DisScatterForwardBuildIndexParams &params, cudaStream_t stream);
void dis_scatter_pre_comm_index_impl(
    const DisScatterPreCommIndexParams &params, cudaStream_t stream);

void shape_info_cum_sum_impl(
    int *splits,
    int *token_per_rank,
    int *splits_cum_sum,
    int *splits_cum_sum_per_rank,
    int *token_per_rank_cum_sum,
    int total_experts,
    int rank,
    int world_size,
    int local_world_size,
    cudaStream_t stream);

}  // namespace flux
}  // namespace bytedance
