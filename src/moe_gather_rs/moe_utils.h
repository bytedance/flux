//===- moe_utils.h ------------------------------------------------ C++ ---===//
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

namespace bytedance::flux {

void ep_index_filter_impl(
    int32_t *scatter_idx,
    int32_t *pos_filtered,
    int32_t *token_idx_filtered,
    int32_t *total_token_acc,
    int32_t *ep_n_token_cum_sum,
    int32_t *splits_gpu_cum_sum,
    int32_t *reduce_token_idx,
    int32_t max_token_per_rank,
    int32_t topk,
    int32_t total_num_experts,
    int32_t world_size,
    int32_t ep_world_size,
    int32_t tp_world_size,
    int32_t cur_rank,
    cudaStream_t stream);

void ep_topk_reduce_impl(
    void *input,
    void *output,
    int32_t *reduce_token_idx,
    int M,
    int N,
    int topk,
    cudaStream_t stream);

void topk_reduce_scatter_impl(
    void **ptrs,
    int groups,
    DataTypeEnum dtype,
    int32_t *scatter_idx,
    int32_t topk,
    void *output_ptr,
    int M,
    int N);

void sort_impl(
    int64_t num_elems,
    uint64_t *key_in,
    uint64_t *key_out,
    uint64_t *val_in,
    uint64_t *val_out,
    cudaStream_t stream);

void index_put_impl(
    int64_t num_tokens, int64_t topk, index_t *index, bool *data, bool value, cudaStream_t stream);

void calculate_prepared_nums_impl(
    int64_t num_experts,
    int64_t num_tokens,
    int32_t *splits,
    int32_t *offsets,
    index_t *last_src,
    int32_t *prepared_nums,
    cudaStream_t stream);

void calculate_transport_info_impl(
    int64_t num_experts,
    int64_t world_size,
    index_t *prepared_order,
    index_t *prepared_offsets,
    index_t *transport_nums,
    cudaStream_t stream);

void scatter_add_impl(
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    index_t *index,
    void *input,
    void *output,
    cudaStream_t stream);

void transport_impl(
    int64_t src_rank,
    int64_t world_size,
    int64_t tokens_per_rank,
    int64_t hidden_size,
    void *send_buffer,
    index_t *transport_offsets,
    index_t *transport_nbytes,
    void *dev_recv_buffer_ptrs[],
    cudaStream_t stream);

/**
 * @brief A CUDA-implementation to calculate blocked gather A
 *
 * @param[in] splits: full splits with all experts
 * @param[in] ep_start: EP offset. usually ep_rank * ep_nexperts
 * @param[in] ep_nexperts: experts per EP.
 * @param[in] block_size_m: block size of m. splits is paded by block_size_m
 * @param[out] gather_a_index: gather index with pad
 * @param[out] expert_index: expert index
 */
void calc_moe_triton_blocked_gather_a(
    const int32_t *splits,
    int32_t ep_start,
    int32_t ep_nexperts,
    int32_t block_size_m,
    int32_t *gather_a_index,
    int32_t *expert_index,
    int num_blocks,
    int num_threads,
    cudaStream_t stream);

}  // namespace bytedance::flux
