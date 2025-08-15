//===- all2all_impl.cu --------------------------------------- C++ ---===//
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
#include <cooperative_groups.h>
#include <type_traits>
#include "all2all_impl.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"
#include <nvshmemx.h>
#include <nvshmem.h>

namespace bytedance {
namespace flux {

__global__ void
all2all(const All2allParams params) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ndim = params.ndim;
  int rank = params.rank;
  int expert_per_rank = params.expert_per_rank;
  int exp_start = bid * expert_per_rank;
  int exp_end = exp_start + expert_per_rank;
  int m_start = params.input_splits_cumsum[exp_start];
  int m_end = params.input_splits_cumsum[exp_end];
  char *dst_ptr = reinterpret_cast<char *>(params.output_ptr) +
                  params.max_token * ndim * rank * params.element_size;
  char *src_ptr =
      reinterpret_cast<char *>(params.input_ptr) + m_start * ndim * params.element_size;
  float *scale_dst_ptr = params.scale_output_buffer + params.max_token * rank;
  float *scale_src_ptr = params.scale_input_buffer + m_start;
  int *splits_src_ptr = params.splits_input_buffer + bid * expert_per_rank;
  ;
  int *splits_dst_ptr = params.splits_output_buffer;
  if (tid < expert_per_rank) {
    int tgt_exp = tid + bid * expert_per_rank;
    splits_src_ptr[tid] =
        params.input_splits_cumsum[tgt_exp + 1] - params.input_splits_cumsum[tgt_exp];
  }
  __syncthreads();
  if (params.with_scale) {
    nvshmemx_putmem_nbi_block(
        scale_dst_ptr, scale_src_ptr, sizeof(float) * (m_end - m_start), bid);
  }
  nvshmemx_putmem_nbi_block(
      splits_dst_ptr + rank * expert_per_rank, splits_src_ptr, sizeof(int) * expert_per_rank, bid);
  nvshmemx_putmem_nbi_block(dst_ptr, src_ptr, params.element_size * (m_end - m_start) * ndim, bid);
  // Element * input = params.inputs_
}

__global__ void
all2all_v2(const All2allParams params) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int ndim = params.ndim;
  int rank = params.rank;
  int expert_per_rank = params.expert_per_rank;
  int exp_start = bid * expert_per_rank;
  int exp_end = exp_start + expert_per_rank;
  int m_start = params.input_splits_cumsum[exp_start];
  int m_end = params.input_splits_cumsum[exp_end];
  char *dst_ptr = reinterpret_cast<char *>(params.output_ptr) +
                  params.max_token * ndim * rank * params.element_size;
  char *src_ptr =
      reinterpret_cast<char *>(params.input_ptr) + m_start * ndim * params.element_size;
  float *scale_dst_ptr = params.scale_output_buffer + params.max_token * rank;
  float *scale_src_ptr = params.scale_input_buffer + m_start;
  bool with_scale = params.with_scale;
  int *splits_src_ptr = params.splits_input_buffer + bid * expert_per_rank;
  int *splits_dst_ptr = params.splits_output_buffer;
  if (tid < expert_per_rank) {
    int tgt_exp = tid + bid * expert_per_rank;
    splits_src_ptr[tid] =
        params.input_splits_cumsum[tgt_exp + 1] - params.input_splits_cumsum[tgt_exp];
  }
  __syncthreads();
  nvshmemx_putmem_nbi_block(dst_ptr, src_ptr, params.element_size * (m_end - m_start) * ndim, bid);

  nvshmemx_putmem_nbi_block(
      splits_dst_ptr + rank * expert_per_rank, splits_src_ptr, sizeof(int) * expert_per_rank, bid);

  nvshmem_fence();
  if (with_scale) {
    nvshmemx_putmem_signal_nbi_block(
        scale_dst_ptr,
        scale_src_ptr,
        sizeof(float) * (m_end - m_start),
        params.signal_buffer + rank,
        params.signal_to_wait,
        NVSHMEM_SIGNAL_SET,
        bid);
  }
  if (tid == 0) {
    if (!with_scale)
      nvshmemx_signal_op(
          params.signal_buffer + rank, params.signal_to_wait, NVSHMEM_SIGNAL_SET, bid);
    nvshmem_signal_wait_until(params.signal_buffer + bid, NVSHMEM_CMP_EQ, params.signal_to_wait);
  }
  // __syncthreads();
}

void
all2all_impl(const All2allParams &params, cudaStream_t stream) {
  int world_size = params.world_size;
  dim3 grid_dim(world_size);
  dim3 block_dim(256);
  all2all_v2<<<grid_dim, block_dim, 0, stream>>>(params);
  // nvshmemx_barrier_all_on_stream(stream);
}

}  // namespace flux
}  // namespace bytedance