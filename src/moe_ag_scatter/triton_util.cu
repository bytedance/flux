//===- triton_util.cu ------------------------------------------ C++ ------===//
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
#include <climits>
#include "cutlass/fast_math.h"
#include "sort_util.h"
#include "triton_util.h"
#include "workspace_util.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/reduce_utils.cuh"
#include "flux/cuda/cuda_common_device.hpp"

namespace bytedance::flux {
namespace {
__device__ __forceinline__ int64_t
pad_to(int64_t sz, int64_t pad) {
  return (sz + pad - 1) / pad * pad;
}

__global__ void
get_moe_ag_scatter_args_kernel(
    const int32_t *__restrict__ splits,
    const int32_t *__restrict__ splits_per_rank_acc,
    void *problem_schedule_ptr_,
    int num_scheds,
    int32_t *gather_index_ptr,
    int32_t *scatter_index_ptr,
    int ep_start,
    int ep_nexperts,
    int world_size,
    int tile_size_m,
    int32_t *m_pad_ptr,
    int32_t *gather_a_ptr,
    int32_t *scatter_d_ptr,
    int32_t *expert_idx_ptr,
    int32_t *rank_start_ptr,
    int32_t *rank_end_ptr) {
  extern __shared__ char shared_storage[];
  struct SchedudleTile {
    int16_t expert_idx;
    int16_t tile_m_idx;
  };
  auto rank_id = [=](int eid, int m) {
    const int *ptr = splits_per_rank_acc + eid * world_size;
    return std::distance(ptr, upper_bound_kernel(ptr, ptr + world_size, m));
  };

  const int *ep_splits = splits + ep_start;
  int *ep_splits_acc = (int *)shared_storage;
  int *ep_splits_acc_pad = (int *)ep_splits_acc + ep_nexperts;
  block_prefix_sum_and_sync(ep_splits, ep_splits_acc, ep_nexperts);
  aligned_block_prefix_sum_and_sync(ep_splits, ep_splits_acc_pad, ep_nexperts, tile_size_m);

  SchedudleTile *sched_tile = (SchedudleTile *)(ep_splits_acc_pad + ep_nexperts);
  static_assert(sizeof(SchedudleTile) == sizeof(int));

  int warp_idx = threadIdx.x / kWarpSize;
  int lane_idx = threadIdx.x % kWarpSize;

  int count = num_scheds;
  const ProblemSchedV2 *scheds = (const ProblemSchedV2 *)problem_schedule_ptr_;

  // calculate expert_index and tiled_m_index
  if (warp_idx == 0) {
    int cur_offset = 0;
    int count_pad = (count + kWarpSize - 1) / kWarpSize * kWarpSize;
    for (int i = lane_idx; i < count_pad; i += kWarpSize) {
      bool has_sched = i < count && scheds[i].tile_m_size > 0;
      int len = has_sched ? scheds[i].tile_m_size : 0;
      int temp_offset = warp_prefix_sum(threadIdx.x, len);
      if (has_sched) {
        for (int m = cur_offset + temp_offset - len, j = 0; j < scheds[i].tile_m_size; j++, m++) {
          sched_tile[m].expert_idx = scheds[i].problem_idx;
          sched_tile[m].tile_m_idx = scheds[i].tile_m_start + j;
        }
      }
      cur_offset += __shfl_sync(0xffffffff, temp_offset, kWarpSize - 1);
    }
  }
  __syncthreads();
  // splits a thread block into multiple groups, each group process a tile
  int group_size = cutlass::round_up(tile_size_m, kWarpSize);
  int num_groups_per_tb = blockDim.x / group_size;
  int num_groups = num_groups_per_tb * gridDim.x;
  int group_id = threadIdx.x / group_size + blockIdx.x * num_groups_per_tb;
  int gid = threadIdx.x % group_size;
  // fill problem_info
  int num_tiles = ep_splits_acc_pad[ep_nexperts - 1] / tile_size_m;
  for (int i = group_id; i < num_tiles; i += num_groups) {
    int eid = sched_tile[i].expert_idx;
    int m_start = sched_tile[i].tile_m_idx * tile_size_m;  // inner expert `eid`
    int m_end = min(ep_splits[eid], m_start + tile_size_m);
    int m_offset = ep_splits_acc[eid] - ep_splits[eid];
    int m = m_start + gid;
    int m_pad = i * tile_size_m + gid;
    gather_a_ptr[m_pad] = m < m_end ? gather_index_ptr[m + m_offset] : INT_MAX;
    scatter_d_ptr[m_pad] = m < m_end ? scatter_index_ptr[m + m_offset] : INT_MAX;
    if (gid == 0) {
      expert_idx_ptr[i] = eid;
      rank_start_ptr[i] = rank_id(eid, m_start);
      rank_end_ptr[i] = rank_id(eid, m_end - 1);
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *m_pad_ptr = ep_splits_acc_pad[ep_nexperts - 1];
  }
}

}  // namespace

void
get_moe_ag_scatter_args(
    const int32_t *splits_gpu_ptr,
    const int32_t *cumsum_per_rank_gpu_ptr,
    void *problem_schedule_ptr,
    int num_scheds,
    int32_t *gather_index_ptr,
    int32_t *scatter_index_ptr,
    int ep_start,
    int ep_nexperts,
    int world_size,
    int M_this_ep,
    int tile_size_m,
    int *m_pad_ptr,
    int32_t *gather_a_ptr,
    int32_t *scatter_d_ptr,
    int32_t *expert_idx_ptr,
    int32_t *rank_start_ptr,
    int32_t *rank_end_ptr,
    cudaStream_t stream) {
  int tiled_m_max = (M_this_ep + tile_size_m - 1) / tile_size_m + ep_nexperts;
  int shared_memory_size = sizeof(int) * ep_nexperts * 2  // for accumulation of splits/splits_pad
                           + sizeof(int) * tiled_m_max;   // for tiled_m sched info
  get_moe_ag_scatter_args_kernel<<<16, 1024, shared_memory_size, stream>>>(
      splits_gpu_ptr,
      cumsum_per_rank_gpu_ptr,
      problem_schedule_ptr,
      num_scheds,
      gather_index_ptr,
      scatter_index_ptr,
      ep_start,
      ep_nexperts,
      world_size,
      tile_size_m,
      m_pad_ptr,
      gather_a_ptr,
      scatter_d_ptr,
      expert_idx_ptr,
      rank_start_ptr,
      rank_end_ptr);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace bytedance::flux
