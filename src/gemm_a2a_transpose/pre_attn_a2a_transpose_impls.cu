//===- pre_attn_a2a_transpose_impls.cu ---------------------------- C++ ---===//
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
#include "pre_attn_a2a_transpose_impls.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/system_barrier.hpp"

#include "cutlass/kernel_hardware_info.h"

namespace bytedance {
namespace flux {
namespace {

#define kMaxThreadsPerBlock 1024

using SystemBarrier = cutlass::detail::SystemBarrier;

template <typename T, int32_t PACK_SIZE>
bool
isAligned(void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % (sizeof(T) * PACK_SIZE) == 0;
}

// output tile
struct All2AllTransposeTileInfo {
  int32_t bs_idx;
  int32_t seq_begin;
  int32_t seq_end;
  int32_t nh_begin;
  int32_t nh_end;
};

__device__ __forceinline__ int32_t
ceil_div(int32_t a, int32_t b) {
  return (a + b - 1) / b;
}

__device__ __forceinline__ static int
ld_acquire(int *ptr) {
  int state = 0;
  // Acquire pattern using acquire modifier
  asm volatile("ld.global.acquire.sys.b32 %0, [%1];\n" : "=r"(state) : "l"(ptr));
  return state;
}

__device__ __forceinline__ void
wait_ready(
    const int32_t remote_rank,
    const All2AllTransposeTileInfo &tile_info,
    const PreAttnAll2AllTransposeParam &param) {
  int32_t producer_m_begin = tile_info.bs_idx * param.local_seq_len + tile_info.seq_begin -
                             remote_rank * param.local_seq_len;
  int32_t producer_m_end = tile_info.bs_idx * param.local_seq_len + tile_info.seq_end -
                           remote_rank * param.local_seq_len;
  int32_t producer_n_begin =
      (tile_info.nh_begin + param.rank * param.local_nheads) * param.head_dim;
  int32_t producer_n_end = (tile_info.nh_end + param.rank * param.local_nheads) * param.head_dim;
  int32_t barrier_m_begin = producer_m_begin / param.m_per_barrier;
  int32_t barrier_m_end = (producer_m_end - 1) / param.m_per_barrier;
  int32_t barrier_n_begin = producer_n_begin / param.n_per_barrier;
  int32_t barrier_n_end = (producer_n_end - 1) / param.n_per_barrier;
  int32_t num_barriers_per_row =
      ceil_div(param.local_nheads * param.head_dim * param.world_size, param.n_per_barrier);
  int32_t *barrier_ptr = (int32_t *)param.barrier_ptrs[remote_rank];

  int32_t num_barriers =
      (barrier_m_end - barrier_m_begin + 1) * (barrier_n_end - barrier_n_begin + 1);
  for (int32_t i = threadIdx.x; i < num_barriers; i += blockDim.x) {
    int32_t m_idx = i / (barrier_n_end - barrier_n_begin + 1);
    int32_t n_idx = i % (barrier_n_end - barrier_n_begin + 1);
    int32_t *flag_ptr =
        barrier_ptr + (m_idx + barrier_m_begin) * num_barriers_per_row + n_idx + barrier_n_begin;

#pragma unroll 1
    while (ld_acquire(flag_ptr) != 1) {
    }
  }

  __syncthreads();
}

template <typename Element, typename ElementVec>
__device__ __forceinline__ void
pull_tile_to_local(
    const int32_t group_id,
    const int32_t num_block_per_group,
    const int32_t remote_rank,
    const All2AllTransposeTileInfo &tile_info,
    const PreAttnAll2AllTransposeParam &param) {
  wait_ready(remote_rank, tile_info, param);

  constexpr int32_t VEC_SIZE = sizeof(ElementVec) / sizeof(Element);
  const int32_t &local_seq_len = param.local_seq_len;
  const int32_t &world_size = param.world_size;
  const int32_t &head_dim = param.head_dim;
  const int32_t &local_nheads = param.local_nheads;
  const int32_t &rank = param.rank;

  const int32_t tid = threadIdx.x;
  const int32_t num_threads = blockDim.x;

  const int32_t &bs_idx = tile_info.bs_idx;
  const int32_t &seq_begin = tile_info.seq_begin;
  const int32_t &seq_end = tile_info.seq_end;
  const int32_t &nh_begin = tile_info.nh_begin;
  const int32_t &nh_end = tile_info.nh_end;

  const int32_t nheads = local_nheads * world_size;
  const int32_t seq_len = local_seq_len * world_size;
  const int32_t vec_head_dim = head_dim / VEC_SIZE;
  const int32_t vec_hidden_dim = vec_head_dim * nheads;

  const int32_t tile_seq = seq_end - seq_begin;
  const int32_t tile_nh = nh_end - nh_begin;
  const int32_t nelems = tile_nh * tile_seq * vec_head_dim;

  ElementVec *vec_input_ptr = reinterpret_cast<ElementVec *>(param.input_ptrs[remote_rank]) +
                              bs_idx * local_seq_len * vec_hidden_dim;
  ElementVec *vec_output_ptr =
      reinterpret_cast<ElementVec *>(param.output_ptr) + bs_idx * local_seq_len * vec_hidden_dim;

  // [tile_seq, tile_nh, head_dim]
  for (int32_t i = tid; i < nelems; i += num_threads) {
    int32_t seq_idx = i / (vec_head_dim * tile_nh);
    int32_t nh_idx = (i - seq_idx * (vec_head_dim * tile_nh)) / vec_head_dim;
    int32_t hd_idx = i - nh_idx * vec_head_dim - seq_idx * vec_head_dim * tile_nh;

    const int32_t out_nh_idx = nh_idx + nh_begin;
    const int32_t &out_seq_idx = seq_idx + seq_begin;
    const int32_t input_seq_idx = out_seq_idx % local_seq_len;
    const int32_t input_nh_idx = out_nh_idx + rank * local_nheads;
    const int32_t input_offset =
        input_seq_idx * vec_hidden_dim + input_nh_idx * vec_head_dim + hd_idx;
    const int32_t out_offset =
        out_nh_idx * seq_len * vec_head_dim + out_seq_idx * vec_head_dim + hd_idx;
    vec_output_ptr[out_offset] = vec_input_ptr[input_offset];
  }
}

__device__ __forceinline__ All2AllTransposeTileInfo
get_tile_info(int32_t remote_rank, int32_t tile_idx, const PreAttnAll2AllTransposeParam &param) {
  const int32_t num_tiles_n = ceil_div(param.local_nheads * param.head_dim, param.TILE_N);
  const int32_t num_tiles_per_bs = ceil_div(param.local_seq_len, param.TILE_M) * num_tiles_n;
  const int32_t bs_idx = tile_idx / num_tiles_per_bs;
  tile_idx -= bs_idx * num_tiles_per_bs;

  const int32_t seq_idx = tile_idx / num_tiles_n;
  const int32_t n_idx = tile_idx % num_tiles_n;
  const int32_t n_begin = n_idx * param.TILE_N;

  const int32_t seq_begin = seq_idx * param.TILE_M;
  const int32_t seq_end = seq_begin + param.TILE_M > param.local_seq_len
                              ? param.local_seq_len
                              : seq_begin + param.TILE_M;
  const int32_t nh_begin = n_begin / param.head_dim;
  const int32_t head_per_tile = param.TILE_N / param.head_dim;
  const int32_t nh_end = nh_begin + head_per_tile >= param.local_nheads ? param.local_nheads
                                                                        : nh_begin + head_per_tile;
  const int32_t offset_seq = remote_rank * param.local_seq_len;
  return {bs_idx, seq_begin + offset_seq, seq_end + offset_seq, nh_begin, nh_end};
}

}  // namespace

template <typename Element, bool kAligned>
__global__ void
__launch_bounds__(kMaxThreadsPerBlock)
    pre_attn_all2all_transpose_kernel(const PreAttnAll2AllTransposeParam param) {
  int32_t num_tiles_per_rank = param.bs * ceil_div(param.local_seq_len, param.TILE_M) *
                               ceil_div(param.local_nheads * param.head_dim, param.TILE_N);

  int32_t num_block_per_group = param.NUM_COMM_SM / param.world_size;
  int32_t group_id = blockIdx.x % num_block_per_group;
  int32_t remote_rank = blockIdx.x / num_block_per_group;
  for (int i = group_id; i < num_tiles_per_rank; i += num_block_per_group) {
    using ElementVec = std::conditional_t<kAligned, uint4, Element>;
    auto tile_info = get_tile_info(remote_rank, i, param);
    pull_tile_to_local<Element, ElementVec>(
        group_id, num_block_per_group, remote_rank, tile_info, param);
  }
}

void
pre_attn_all2all_transpose_impl(
    const PreAttnAll2AllTransposeParam param, DataTypeEnum input_dtype, cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  FLUX_CHECK(param.NUM_COMM_SM % param.world_size == 0);
  // FLUX_CHECK(param.local_seq_len % param.TILE_M == 0);
  FLUX_CHECK(param.TILE_N % param.head_dim == 0);

  int32_t grid_size = param.NUM_COMM_SM;
  tuple_return_if(
      cute::make_tuple(_BF16{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        constexpr int32_t PACK_SIZE = 16 / sizeof(Element);
        bool align =
            isAligned<Element, PACK_SIZE>(param.output_ptr) && param.head_dim % PACK_SIZE == 0;
        if (align) {
          pre_attn_all2all_transpose_kernel<Element, true>
              <<<grid_size, kThreadsPerBlock, 0, stream>>>(param);
        } else {
          pre_attn_all2all_transpose_kernel<Element, false>
              <<<grid_size, kThreadsPerBlock, 0, stream>>>(param);
        }
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
}

}  // namespace flux
}  // namespace bytedance