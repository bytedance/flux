//===- moe_utils.cu ---------------------------------------------- C++ ---===//
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

#include "flux/cuda/reduce_utils.cuh"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/moe_utils.h"
namespace bytedance::flux {

__global__ void
calc_scatter_index_kernel(
    const int *rank, const int *count, int *scatter_index, const int total_num) {
  constexpr unsigned FULL_MASK = 0xffffffff;
  __shared__ int s_offset[1024];
  const int expert_rank = blockIdx.x;
  const int expert_num = expert_rank + 1;
  if (threadIdx.x < 32) {
    int cur_offset = 0;
    int expert_num_pad = ((expert_num + 31) >> 5) << 5;
    for (int i = threadIdx.x; i < expert_num_pad; i += 32) {
      int len = i < expert_num ? count[i] : 0;
      int temp_offset = warp_prefix_sum(threadIdx.x, len);
      if (i < expert_num)
        s_offset[i] = cur_offset + temp_offset - len;
      cur_offset += __shfl_sync(FULL_MASK, temp_offset, 31);
    }
  }
  __syncthreads();

  const int warp_tid = threadIdx.x & 0x1F;
  const unsigned int t_mask = (1 << warp_tid) - 1;

  int *s_expert_offset = s_offset + blockIdx.x;
  int total_num_pad = ((total_num + blockDim.x - 1) / blockDim.x) * blockDim.x;
  for (int tid = threadIdx.x; tid < total_num_pad; tid += blockDim.x) {
    int rank_id = tid < total_num ? __ldg(&rank[tid]) : -1;
    const bool match = (rank_id == expert_rank);
    int active_mask = __ballot_sync(FULL_MASK, match);

    int warp_expert_offset = 0;
    if (warp_tid == 0)
      warp_expert_offset = atomicAdd(s_expert_offset, __popc(active_mask));
    warp_expert_offset = __shfl_sync(FULL_MASK, warp_expert_offset, 0);

    int warp_offset = __popc(active_mask & t_mask);
    if (match)
      scatter_index[tid] = warp_expert_offset + warp_offset;
  }
}

void
calc_scatter_index(
    const int *choosed_experts,  // of total_num
    const int *count,            // of expert_num
    int *scatter_index,          // of total_num
    const int total_num,         // topk * ntokens
    int expert_num,
    cudaStream_t stream) {
  calc_scatter_index_kernel<<<expert_num, 1024, 0, stream>>>(
      choosed_experts, count, scatter_index, total_num);
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace bytedance::flux
