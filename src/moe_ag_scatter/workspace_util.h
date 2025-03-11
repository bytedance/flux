//===- workspace_util.h ---------------------------------------- C++ ------===//
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
#include "flux/args/moe_ag_scatter.h"
namespace bytedance::flux {

struct ProblemInfo {
  int32_t problem_idx;       // problem index
  int32_t problem_tile_idx;  // tile index inner problem. tile_idx_m = problem_tile_idx / tiled_n.
                             // tile_idx_n = problem_tile_idx % tiled_n
};

// save some memory
struct ProblemSchedV2 {
  int16_t problem_idx;
  int16_t tile_m_start;
  int16_t tile_m_size;
};

/* all ptrs pointer to device memory.
workspace structure:
  problem_sizes, cutlass::gemm::GemmCoord, problem_count
  ptr_A, void *, problem_count
  ptr_B, void *, problem_count
  ptr_C, void *, problem_count
  ptr_D, void *, problem_count
  scale_D, float *, problem_count
  lda, int64_t, problem_count
  ldb, int64_t, problem_count
  ldc, int64_t, problem_count
  ldd, int64_t, problem_count
  ldr, int64_t, problem_count
  gather_A, int *, problem_count
  scatter_D, int *, problem_count
  tile_count, int, 1
  problem_info, ProblemInfo, pad_to(num_tiles, threadblock_count)
*/
struct MoeAgScatterWorkspaceArgumements {
  void *problem_sizes;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  float **scale_D;
  int64_t *lda;
  int64_t *ldb;
  int64_t *ldc;
  int64_t *ldd;
  int64_t *ldr;
  int32_t **gather_A;
  int32_t **scatter_D;
  int *tile_count;  // keep it in device to avoid device sync
  ProblemInfo *problem_info;
};

void make_workspace_async(
    const GemmGroupedV2AGScatterArguments &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    int threadblock_count,
    void *workspace,
    cudaStream_t stream);

/**
 * @brief Get the sorted problem schedule cuda v2 object
 *
 * @param splits
 * @param rank
 * @param tp_size
 * @param cumsum_per_rank_ptr
 * @param ep_start
 * @param ep_nexperts
 * @param tiled_m_size
 * @param num_weight_groups
 * @return std::vector<ProblemSchedule>
 */
void get_sorted_problem_schedule_cuda_v2(
    const int32_t *const splits,
    int rank,
    int tp_size,
    const int *cumsum_per_rank_ptr,
    const int ep_start,
    const int ep_nexperts,
    const int tiled_m_size,
    const int num_weight_groups,
    ProblemSchedV2 *problem_schedules,
    cudaStream_t stream);

}  // namespace bytedance::flux
