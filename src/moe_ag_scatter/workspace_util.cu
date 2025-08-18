//===- workspace_util.cu --------------------------------------- C++ ------===//
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
#include "flux/ths_op/util.h"
#include "workspace_util.h"
#include <cutlass/gemm_coord.h>
#include "flux/args/moe_ag_scatter.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/reduce_utils.cuh"
#include "flux/flux.h"
namespace bytedance::flux {

__device__ __forceinline__ void *
ptr_with_offset(void *ptr, intptr_t offset) {
  return (void *)((char *)ptr + offset);
}

static_assert(sizeof(ProblemSchedV2) == 6);

__device__ __forceinline__ void
calc_sorted_problem_schedule_v2(
    const GemmGroupedV2AGScatterArguments &args, ProblemSchedV2 *problem_schedules) {
  const int tp_size = args.world_size;
  const int rank = args.rank;
  const int num_groups = args.num_groups;
  const int ep_nexperts = args.ep_nexperts;
  const int tile_size_m = args.tile_size_m;

  const int32_t *ep_splits = args.splits + args.ep_start;
  // tiled_m range [tiled_m * tiled_m_size, (tiled_m + 1) * tiled_m_size), which may cross multiple
  // segments, or multiple stage. this function calculate in which stage should `tiled_m` be placed
  auto get_stage_for_tile = [=](int eid, int tiled_m) {
    const int *cumsum_this_rank = args.accum_per_rank_ptr + eid * tp_size;

    auto get_rank_id = [=](int m) {
      auto iter = upper_bound_kernel(cumsum_this_rank, cumsum_this_rank + tp_size, m);
      return std::distance(cumsum_this_rank, iter);
    };

    int m_start_this_tile = tiled_m * tile_size_m;
    int m_end_this_tile = std::min(ep_splits[eid], (tiled_m + 1) * tile_size_m) - 1;
    int segment_start_this_tile = get_rank_id(m_start_this_tile);
    int segment_end_this_tile = get_rank_id(m_end_this_tile);
    int stage_max = 0;
    for (int sid = segment_start_this_tile; sid <= segment_end_this_tile; sid++) {
      int stage = (sid - rank + tp_size) % tp_size;
      int ntokens_this_rank =
          sid == 0 ? cumsum_this_rank[0] : (cumsum_this_rank[sid] - cumsum_this_rank[sid - 1]);
      if (ntokens_this_rank != 0) {
        stage_max = std::max(stage, stage_max);
      }
    }
    return stage_max;
  };

  int sched_idx = 0;
  for (int i = threadIdx.x; i < tp_size * num_groups * ep_nexperts; i += blockDim.x) {
    int eid = i % ep_nexperts;
    int gid = (i / ep_nexperts) % num_groups;
    int stage = (i / ep_nexperts / num_groups) % tp_size;
    int segment = (stage + rank) % tp_size;
    const int *cumsum_this_rank = args.accum_per_rank_ptr + eid * tp_size;
    auto get_cumsum_this_rank_with_zero_pad = [=](int segment) {
      return segment == 0 ? 0 : cumsum_this_rank[segment - 1];
    };
    const int m_start = get_cumsum_this_rank_with_zero_pad(segment);
    const int m_end = cumsum_this_rank[segment];
    int tiled_m_start = m_start / tile_size_m;
    int tiled_m_end = (m_end - 1) / tile_size_m;
    bool is_valid_sched = m_end > m_start;  // has tiles to schedule from this stage
    if (is_valid_sched) {                   // guard in case invalid memory read
      int start_stage = get_stage_for_tile(eid, tiled_m_start);
      int end_stage = get_stage_for_tile(eid, tiled_m_end);
      bool own_start = stage == start_stage;
      bool own_end = stage == end_stage;
      if (!own_start)
        tiled_m_start++;
      if (!own_end)
        tiled_m_end--;
    }
    is_valid_sched &= (tiled_m_start <= tiled_m_end);

    ProblemSchedV2 &sched = problem_schedules[i];
    sched.problem_idx = eid + ep_nexperts * gid;
    sched.tile_m_start = tiled_m_start;
    sched.tile_m_size = is_valid_sched ? (tiled_m_end - tiled_m_start + 1) : -1;
    sched_idx++;
  }
}

__device__ __forceinline__ void
fill_problem_info(
    const GemmGroupedV2AGScatterArguments &args,
    const int tiled_m,
    const int tiled_n,
    const int threadblock_count,
    ProblemInfo *problem_info,  // this is what ProblemVisitor host mode wants.
    void *shared_storage) {
  struct SchedudleTile {
    int16_t expert_idx;
    int16_t tile_m_idx;
  };
  SchedudleTile *sched_tile = (SchedudleTile *)shared_storage;
  static_assert(sizeof(SchedudleTile) == sizeof(int));

  int warp_idx = threadIdx.x / kWarpSize;
  int lane_idx = threadIdx.x % kWarpSize;

  int count = args.num_problem_schedules;
  const ProblemSchedV2 *scheds = (const ProblemSchedV2 *)args.problem_schedules;

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
  // fill problem_info
  int num_tiles = tiled_m * tiled_n * args.num_groups;
  int tiles_per_tb = (num_tiles + threadblock_count - 1) / threadblock_count;
  for (int i = threadIdx.x; i < num_tiles; i += blockDim.x) {
    int tile_idx_m = i / tiled_n;
    int tile_idx_n = i % tiled_n;

    int tid = i % threadblock_count;
    int index = i / threadblock_count;

    auto &info = problem_info[tid * tiles_per_tb + index];
    auto &tile = sched_tile[tile_idx_m];
    info.problem_idx = tile.expert_idx;
    info.problem_tile_idx = tile_idx_n + tile.tile_m_idx * tiled_n;
  }
}

template <typename GemmLayoutA, typename GemmLayoutB, typename GemmLayoutC, typename GemmLayoutD>
__global__ void
prepare_workspace_kernel(
    GemmGroupedV2AGScatterArguments args,
    int input_elem_size,
    int output_elem_size,
    int threadblock_count,
    void *workspace) {
  const int kAlignment = 128;
  extern __shared__ char shared_storage[];
  int *ep_splits_acc = (int *)shared_storage;
  int problem_count = args.ep_nexperts * args.num_groups;
  int N = args.N;
  int K = args.K;
  int *ep_splits = args.splits + args.ep_start;
  block_prefix_sum_and_sync(ep_splits, ep_splits_acc, args.ep_nexperts);
  // the offsets
  int offset_problem_sizes = 0;
  int offset_ptr_A =
      pad_to(offset_problem_sizes + problem_count * sizeof(cutlass::gemm::GemmCoord), kAlignment);
  int offset_ptr_B = pad_to(offset_ptr_A + problem_count * sizeof(void *), kAlignment);
  int offset_ptr_C = pad_to(offset_ptr_B + problem_count * sizeof(void *), kAlignment);
  int offset_ptr_D = pad_to(offset_ptr_C + problem_count * sizeof(void *), kAlignment);
  int offset_scale_D = pad_to(offset_ptr_D + problem_count * sizeof(void *), kAlignment);
  int offset_lda = pad_to(offset_scale_D + problem_count * sizeof(float *), kAlignment);
  int offset_ldb = pad_to(offset_lda + problem_count * sizeof(int64_t), kAlignment);
  int offset_ldc = pad_to(offset_ldb + problem_count * sizeof(int64_t), kAlignment);
  int offset_ldd = pad_to(offset_ldc + problem_count * sizeof(int64_t), kAlignment);
  int offset_ldr = pad_to(offset_ldd + problem_count * sizeof(int64_t), kAlignment);
  int offset_gather_A = pad_to(offset_ldr + problem_count * sizeof(int64_t), kAlignment);
  int offset_scatter_D = pad_to(offset_gather_A + problem_count * sizeof(int *), kAlignment);
  int offset_tile_count = pad_to(offset_scatter_D + problem_count * sizeof(int *), kAlignment);
  int offset_problem_info = pad_to(offset_tile_count + 1 * sizeof(int), kAlignment);
  // the ptrs
  cutlass::gemm::GemmCoord *problem_sizes =
      (cutlass::gemm::GemmCoord *)((char *)workspace + offset_problem_sizes);
  void **ptr_A = (void **)((char *)workspace + offset_ptr_A);
  void **ptr_B = (void **)((char *)workspace + offset_ptr_B);
  void **ptr_C = (void **)((char *)workspace + offset_ptr_C);
  void **ptr_D = (void **)((char *)workspace + offset_ptr_D);
  float **scale_D = (float **)((char *)workspace + offset_scale_D);
  int64_t *lda = (int64_t *)((char *)workspace + offset_lda);
  int64_t *ldb = (int64_t *)((char *)workspace + offset_ldb);
  int64_t *ldc = (int64_t *)((char *)workspace + offset_ldc);
  int64_t *ldd = (int64_t *)((char *)workspace + offset_ldd);
  int64_t *ldr = (int64_t *)((char *)workspace + offset_ldr);
  int **gather_A = (int **)((char *)workspace + offset_gather_A);
  int **scatter_D = (int **)((char *)workspace + offset_scatter_D);
  int *tile_count = (int *)((char *)workspace + offset_tile_count);
  ProblemInfo *problem_info = (ProblemInfo *)((char *)workspace + offset_problem_info);

  for (int i = threadIdx.x; i < problem_count; i += blockDim.x) {
    int gid = i / args.ep_nexperts;
    int eid = i % args.ep_nexperts;
    int Mi = ep_splits[eid];
    int M_acc = ep_splits_acc[eid] - Mi;

    problem_sizes[i] = cutlass::gemm::GemmCoord(Mi, N, K);
    ptr_A[i] = args.input;
    ptr_B[i] = ptr_with_offset(args.weight[gid], eid * N * K * input_elem_size);
    ptr_C[i] = nullptr;
    ptr_D[i] = ptr_with_offset(args.output[gid], M_acc * N * output_elem_size);
    lda[i] = GemmLayoutA::packed({Mi, K}).stride(0);
    ldb[i] = GemmLayoutB::packed({K, N}).stride(0);
    ldc[i] = GemmLayoutC::packed({Mi, N}).stride(0);
    ldd[i] = GemmLayoutD::packed({Mi, N}).stride(0);
    ldr[i] = 0;
    scale_D[i] = args.scaleD[gid] ? (args.scaleD[gid] + eid) : nullptr;
    gather_A[i] = args.gather_A + M_acc;
    scatter_D[i] = args.scatter_D + M_acc;
  }

  __syncthreads();

  calc_sorted_problem_schedule_v2(args, (ProblemSchedV2 *)args.problem_schedules);
  __syncthreads();
  // calculate padded tiled_m
  aligned_block_prefix_sum_and_sync(ep_splits, ep_splits_acc, args.ep_nexperts, args.tile_size_m);
  int M_pad_this_ep = ep_splits_acc[args.ep_nexperts - 1];
  const int tiled_m = M_pad_this_ep / args.tile_size_m;
  const int tiled_n = (args.N + args.tile_size_n - 1) / args.tile_size_n;
  fill_problem_info(
      args, tiled_m, tiled_n, threadblock_count, problem_info, (void *)shared_storage);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *tile_count = tiled_m * tiled_n * args.num_groups;
  }
}

/**
 * @brief fill in workspace with right pointer/values. and return MoeAgScatterArguments with device
 * pointers
 *
 * @param args
 * @param workspace
 * @param workspace_size
 * @param threadblock_count
 * @param num_blocks
 * @param num_threads
 * @param stream
 */
void
make_workspace_async(
    const GemmGroupedV2AGScatterArguments &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    int threadblock_count,
    void *workspace,
    cudaStream_t stream) {
  FLUX_CHECK_EQ(layout, GemmLayoutEnum::RCR);
  int tiled_m = (args.M_this_ep + args.tile_size_m - 1) / args.tile_size_m + args.ep_nexperts;
  int shared_memory_size = std::max(
      args.ep_nexperts * sizeof(int),            // for prefix sum of ep_splits
      args.num_groups * tiled_m * sizeof(int));  // for problem schedule tile_idx_m
  shared_memory_size = pad_to(shared_memory_size, 1024 * sizeof(int));
  prepare_workspace_kernel<
      cutlass::layout::RowMajor,
      cutlass::layout::ColumnMajor,
      cutlass::layout::RowMajor,
      cutlass::layout::RowMajor><<<1, 768, shared_memory_size, stream>>>(
      args, input_elem_size, output_elem_size, threadblock_count, workspace);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void
get_sorted_problem_schedule_v2_kernel(
    GemmGroupedV2AGScatterArguments args, ProblemSchedV2 *problem_schedules) {
  calc_sorted_problem_schedule_v2(args, problem_schedules);
}

void
get_sorted_problem_schedule_cuda_v2(
    const int32_t *const splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups,
    ProblemSchedV2 *problem_schedules,
    cudaStream_t stream) {
  GemmGroupedV2AGScatterArguments args;
  args.rank = rank;
  args.world_size = tp_size;
  args.num_groups = num_weight_groups;
  args.ep_start = ep_start;
  args.ep_nexperts = ep_nexperts;
  args.splits = (int *)splits;
  args.problem_schedules = (void *)problem_schedules;
  args.accum_per_rank_ptr = (int *)cumsum_per_rank_ptr;
  args.tile_size_m = tiled_m_size;
  get_sorted_problem_schedule_v2_kernel<<<1, 1024, 0, stream>>>(args, problem_schedules);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace bytedance::flux
