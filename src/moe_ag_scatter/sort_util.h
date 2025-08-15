//===- sort_util.h --------------------------------------------- C++ ------===//
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
#include "flux/utils.h"

namespace bytedance::flux {

// given scatter_index, compute gather_index.
// scatter_index maps [ntokens] to [ntokens*topk], gather_index is the reverse operation.
// consider ep, we only need the gather_index for partial experts,
// we just store the gather_index for rows whose index is in [rows_offset, rows_offset+total_rows)
void calc_gather_index_impl(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t expert_idx_start,
    int32_t expert_idx_end,
    int32_t const *splits,
    int32_t const *scatter_index,
    int32_t *gather_index_ep,
    int32_t *total_nrows_ep_gpu,  // scalar
    cudaStream_t stream);

void calc_gather_index_impl_v2(
    int32_t nexperts,
    int32_t ntokens,
    int32_t topk,
    int32_t rows_start,  // let's hope this won't overflow: not as many as 2**31 tokens
    int32_t rows_end,
    int32_t const *scatter_index,
    int32_t *gather_index_ep,
    cudaStream_t stream);

// The original computing flow:
//   input (shard) -> (ag) -> input (full) -> (scatter) -> mat A -> (gemm) -> mat D
// We sort matrix A so that the dependant data from
// input (shard) is as contiguous as possible.
// The new flow is:
//   input (shard) -> (ag) -> input (full) -> (scatter&sort) -> sorted mat A
//    -> (gemm) -> sorted mat D -> (scatter) -> mat D
// The original gemm is #nexperts problems, sort the tokens by a paired key:
// (the rank it is from, expert id), constructing #nexperts * #tp_size new problems.
// This is used to make overlapingg all-gather possible. By overlapping computing tokens
// from a rank whose data is ready, with fetching data from the next rank.
//
// splits: [nexperts]
// gather_index: [ntokens*topk]
// scatter_index: [ntokens*topk]
// sorted_splits: [nexperts*tp_size]
// sorted_gather_index: [ntokens*topk]
//   row index of `sorted mat A` -> row index of `input (full)`
// sorted_scatter_index: [nexperts*tp_size]
//   row index of `sorted mat D` -> row index of `mat D`
struct AGScatterSortOpArguments {
  DistEnv dist_env;
  int32_t ntokens;
  int32_t nexperts_ep;
  int32_t const *splits_ep;
  int32_t const *gather_index_ep;
  int32_t *sorted_splits;
  int32_t *sorted_scatter_index;
  int32_t *sorted_gather_index;
};

void ag_scatter_sort_impl(AGScatterSortOpArguments const &args, cudaStream_t stream);

struct AGScatterSortOpArgumentsV2 {
  int rank;  // not used.
  int world_size;
  int32_t ntokens;
  int32_t nexperts_ep;
  int32_t const *splits_ep;
  int32_t const *gather_index_ep;
  int32_t *sorted_splits;
  int32_t *sorted_splits_cumsum;
  int32_t *sorted_scatter_index;
  int32_t *sorted_gather_index;
};
void ag_scatter_sort_impl_v2(AGScatterSortOpArgumentsV2 const &args, cudaStream_t stream);

void sort_scatter_index_to_per_expert(
    int *sorted_scatter_index,
    int *splits_gpu,
    int ep_start,
    int ep_nexperts,
    cudaStream_t stream);

struct ProblemSchedule {
  int32_t expert_id;
  int32_t m_start;
  int32_t m_end;
  int32_t source_rank_start;
  int32_t source_rank_end;

  friend std::ostream &
  operator<<(std::ostream &os, ProblemSchedule const &sched) {
    os << "expert_id:" << sched.expert_id;
    os << ",m_start:" << sched.m_start;
    os << ",m_end:" << sched.m_end;
    os << ",source_rank_start:" << sched.source_rank_start;
    os << ",source_rank_end:" << sched.source_rank_end;
    return os;
  }
};

// Obtain the ordering of tiles in the m dimension such that the number of ranks
// dependent on data from other ranks is minimized for the same amount of computation.
//
// tile_size: tile_M
// ntiles: how many tiles we prefer to choose in a split
// Returns: <expert_id, m_start, m_end>
std::vector<ProblemSchedule> get_sorted_problem_schedule(
    std::vector<int32_t> const &sorted_splits_cpu,
    DistEnv const &dist_env,
    int32_t nexperts_ep,
    int32_t tile_size,
    int32_t ntiles = 4);

std::vector<ProblemSchedule> get_sorted_problem_schedule_v2(
    const int32_t *const splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups);

std::vector<ProblemSchedule> get_relax_sorted_problem_schedule_v2(
    std::vector<int32_t> const &splits,
    int rank,
    int tp_size,
    const int *split_accum_per_rank_ptr,
    const int expert_idx_offset,
    const int nexperts_ep,
    const int tiled_m_size,
    const int num_weight_groups,
    const int nfold);

std::vector<ProblemSchedule> get_sorted_problem_schedule_v2_with_ntiles_limit(
    std::vector<int32_t> const &splits,
    int rank,
    int tp_size,
    const int *split_accum_per_rank_ptr,
    const int expert_idx_offset,
    const int nexperts_ep,
    const int tiled_m_size,
    const int num_weight_groups,
    const int ntiles_limit);

// we shift the computation rank order to comply with the order of gathering data.
// for ranks of the same nodes, circular shift the ranks to make the current local rank
// to be processed first. for different nodes, do the same shifting strategy as the local ranks.
// e.g. two nodes with ranks [0,1,2,3], [4,5,6,7]:
//  for rank #1, the order is: (1,2,3,0,5,6,7,4)
//  for rank #6, the order is: (6,7,4,5,2,3,0,1)
CUTLASS_HOST_DEVICE
int
shift_rank_to_order(int rank, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(rank);
  int node_idx_shift = (node_idx - dist_env.node_idx + dist_env.nnodes) % dist_env.nnodes;
  int local_rank_shift =
      (local_rank - dist_env.local_rank + dist_env.local_world_size) % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_shift, node_idx_shift);
}

CUTLASS_HOST_DEVICE
int
revert_order_to_rank(int order, DistEnv const &dist_env) {
  auto [node_idx, local_rank] = dist_env.global_rank_to_node_idx_local_rank(order);
  int node_idx_origin = (node_idx + dist_env.node_idx) % dist_env.nnodes;
  int local_rank_origin = (local_rank + dist_env.local_rank) % dist_env.local_world_size;
  return dist_env.local_rank_to_global_rank(local_rank_origin, node_idx_origin);
}

}  // namespace bytedance::flux
