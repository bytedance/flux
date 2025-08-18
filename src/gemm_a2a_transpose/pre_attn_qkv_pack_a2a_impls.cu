//===- pre_attn_qkv_pack_a2a_impls.cu ----------------------------- C++ ---===//
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
#include "pre_attn_qkv_pack_a2a_impls.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/system_barrier.hpp"
#include "cutlass/kernel_hardware_info.h"

namespace bytedance {
namespace flux {
namespace {
#define BOOL_SWITCH(CONDITION, NAME, ...) \
  [&] {                                   \
    if (CONDITION) {                      \
      constexpr bool NAME = true;         \
      return __VA_ARGS__();               \
    } else {                              \
      constexpr bool NAME = false;        \
      return __VA_ARGS__();               \
    }                                     \
  }()

#define kMaxThreadsPerBlock 1024

using SystemBarrier = cutlass::detail::SystemBarrier;

struct PreAttnQKVPackA2ATileInfo {
  int32_t bs_idx;
  int32_t seq_begin;
  int32_t seq_end;
  int32_t nh_begin;
  int32_t nh_end;
};

template <typename T, int32_t PACK_SIZE>
bool
isAligned(void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % (sizeof(T) * PACK_SIZE) == 0;
}

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

__device__ __forceinline__ int32_t
linear_out_nh_idx_to_input_nh_idx(
    const int32_t linear_out_nh_idx,
    const int32_t local_q_nheads,
    const int32_t local_k_nheads,
    const int32_t local_v_nheads,
    const int32_t rank,
    const int32_t world_size) {
  bool is_q_head = linear_out_nh_idx < local_q_nheads;
  bool is_k_head = (!is_q_head) && (linear_out_nh_idx < local_q_nheads + local_k_nheads);
  int32_t local_out_nheads =
      is_q_head ? local_q_nheads : (is_k_head ? local_k_nheads : local_v_nheads);
  int32_t local_nheads_offset =
      is_q_head ? 0 : (is_k_head ? local_q_nheads : local_q_nheads + local_k_nheads);
  int32_t out_nh_idx = linear_out_nh_idx - local_nheads_offset;
  int32_t input_nh_idx = out_nh_idx + rank * local_out_nheads + local_nheads_offset * world_size;
  return input_nh_idx;
}

__device__ __forceinline__ void
wait_ready(
    const int32_t remote_rank,
    const PreAttnQKVPackA2ATileInfo &tile_info,
    const PreAttnQKVPackA2AParams &params) {
  const int32_t local_seq_len_remote_rank =
      params.cusum_seq_lens[remote_rank + 1] - params.cusum_seq_lens[remote_rank];
  const int32_t &local_q_nheads = params.local_q_nheads;
  const int32_t &local_k_nheads = params.local_k_nheads;
  const int32_t &local_v_nheads = params.local_v_nheads;
  const int32_t local_nheads = local_q_nheads + local_k_nheads + local_v_nheads;
  const int32_t seq_len_offset_remote_rank = params.cusum_seq_lens[remote_rank];
  int32_t producer_m_begin = tile_info.bs_idx * local_seq_len_remote_rank + tile_info.seq_begin -
                             seq_len_offset_remote_rank;
  int32_t producer_m_end = tile_info.bs_idx * local_seq_len_remote_rank + tile_info.seq_end -
                           seq_len_offset_remote_rank;
  int32_t input_nh_begin = linear_out_nh_idx_to_input_nh_idx(
      tile_info.nh_begin,
      local_q_nheads,
      local_k_nheads,
      local_v_nheads,
      params.rank,
      params.world_size);
  int32_t input_nh_end = linear_out_nh_idx_to_input_nh_idx(
      tile_info.nh_end,
      local_q_nheads,
      local_k_nheads,
      local_v_nheads,
      params.rank,
      params.world_size);

  int32_t producer_n_begin = input_nh_begin * params.head_dim;
  int32_t producer_n_end = input_nh_end * params.head_dim;
  int32_t barrier_m_begin = producer_m_begin / params.m_per_barrier;
  int32_t barrier_m_end = (producer_m_end - 1) / params.m_per_barrier;
  int32_t barrier_n_begin = producer_n_begin / params.n_per_barrier;
  int32_t barrier_n_end = (producer_n_end - 1) / params.n_per_barrier;
  int32_t num_barriers_per_row =
      ceil_div(local_nheads * params.head_dim * params.world_size, params.n_per_barrier);
  int32_t *barrier_ptr = (int32_t *)params.barrier_ptrs[remote_rank];

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

template <typename Element, typename ElementVec, bool kSkipBarrier>
__device__ __forceinline__ void
pull_tile_to_local(
    const int32_t remote_rank,
    const PreAttnQKVPackA2ATileInfo &tile_info,
    const PreAttnQKVPackA2AParams &params) {
  if constexpr (!kSkipBarrier) {
    wait_ready(remote_rank, tile_info, params);
  }
  constexpr int32_t VEC_SIZE = sizeof(ElementVec) / sizeof(Element);
  const int32_t &world_size = params.world_size;
  const int32_t &rank = params.rank;

  const int32_t &head_dim = params.head_dim;
  const int32_t local_seq_len_remote_rank =
      params.cusum_seq_lens[remote_rank + 1] - params.cusum_seq_lens[remote_rank];
  const int32_t seq_len_offset_remote_rank = params.cusum_seq_lens[remote_rank];
  const int32_t &local_q_nheads = params.local_q_nheads;
  const int32_t &local_k_nheads = params.local_k_nheads;
  const int32_t &local_v_nheads = params.local_v_nheads;
  const int32_t local_nheads = local_q_nheads + local_k_nheads + local_v_nheads;

  const int32_t tid = threadIdx.x;
  const int32_t num_threads = blockDim.x;

  const int32_t &bs_idx = tile_info.bs_idx;
  const int32_t &seq_begin = tile_info.seq_begin;
  const int32_t &seq_end = tile_info.seq_end;
  const int32_t &nh_begin = tile_info.nh_begin;
  const int32_t &nh_end = tile_info.nh_end;

  const int32_t nheads = local_nheads * world_size;
  const int32_t seq_len = params.cusum_seq_lens[world_size];
  const int32_t vec_head_dim = head_dim / VEC_SIZE;
  const int32_t vec_hidden_dim = vec_head_dim * nheads;

  const int32_t tile_seq = seq_end - seq_begin;
  const int32_t tile_nh = nh_end - nh_begin;
  const int32_t nelems = tile_nh * tile_seq * vec_head_dim;

  ElementVec *vec_input_ptr = reinterpret_cast<ElementVec *>(params.input_ptrs[remote_rank]) +
                              bs_idx * local_seq_len_remote_rank * vec_hidden_dim;
  ElementVec *vec_q_ptr = reinterpret_cast<ElementVec *>(params.q_ptr) +
                          bs_idx * seq_len * local_q_nheads * vec_head_dim;
  ElementVec *vec_k_ptr = reinterpret_cast<ElementVec *>(params.k_ptr) +
                          bs_idx * seq_len * local_k_nheads * vec_head_dim;
  ElementVec *vec_v_ptr = reinterpret_cast<ElementVec *>(params.v_ptr) +
                          bs_idx * seq_len * local_v_nheads * vec_head_dim;

  // [tile_seq, tile_nh, head_dim]
  for (int32_t i = tid; i < nelems; i += num_threads) {
    int32_t seq_idx = i / (vec_head_dim * tile_nh);
    int32_t nh_idx = (i - seq_idx * (vec_head_dim * tile_nh)) / vec_head_dim;
    int32_t hd_idx = i - nh_idx * vec_head_dim - seq_idx * vec_head_dim * tile_nh;

    // index of seq dim
    const int32_t &out_seq_idx = seq_idx + seq_begin;
    const int32_t input_seq_idx = out_seq_idx - seq_len_offset_remote_rank;

    // index of nh dim
    const int32_t linear_out_nh_idx = nh_idx + nh_begin;
    bool is_q_head = linear_out_nh_idx < local_q_nheads;
    bool is_k_head = (!is_q_head) && (linear_out_nh_idx < local_q_nheads + local_k_nheads);

    int32_t out_nh_idx, input_nh_idx, out_seq_stride;
    ElementVec *vec_output_ptr;

    int32_t local_out_nheads =
        is_q_head ? local_q_nheads : (is_k_head ? local_k_nheads : local_v_nheads);
    int32_t local_nheads_offset =
        is_q_head ? 0 : (is_k_head ? local_q_nheads : local_q_nheads + local_k_nheads);

    vec_output_ptr = is_q_head ? vec_q_ptr : (is_k_head ? vec_k_ptr : vec_v_ptr);
    out_seq_stride = local_out_nheads * vec_head_dim;
    out_nh_idx = linear_out_nh_idx - local_nheads_offset;
    input_nh_idx = out_nh_idx + rank * local_out_nheads + local_nheads_offset * world_size;
    const int32_t input_offset =
        input_seq_idx * vec_hidden_dim + input_nh_idx * vec_head_dim + hd_idx;
    const int32_t out_offset = out_seq_idx * out_seq_stride + out_nh_idx * vec_head_dim + hd_idx;
    vec_output_ptr[out_offset] = vec_input_ptr[input_offset];
  }
}

__device__ __forceinline__ PreAttnQKVPackA2ATileInfo
get_tile_info(
    int32_t remote_rank, int32_t linear_tile_idx, const PreAttnQKVPackA2AParams &params) {
  const int32_t local_nheads =
      params.local_q_nheads + params.local_k_nheads + params.local_v_nheads;
  const int32_t local_seq_len_remote_rank =
      params.cusum_seq_lens[remote_rank + 1] - params.cusum_seq_lens[remote_rank];
  const int32_t num_tiles_per_bs =
      ceil_div(local_seq_len_remote_rank, params.TILE_S) * ceil_div(local_nheads, params.TILE_NH);
  const int32_t bs_idx = linear_tile_idx / num_tiles_per_bs;
  linear_tile_idx = linear_tile_idx - bs_idx * num_tiles_per_bs;

  const int32_t num_tiles_nh = ceil_div(local_nheads, params.TILE_NH);
  const int32_t major_idx = linear_tile_idx / num_tiles_nh;
  const int32_t minor_idx = linear_tile_idx - major_idx * num_tiles_nh;
  const int32_t major_begin = major_idx * params.TILE_S;
  const int32_t minor_begin = minor_idx * params.TILE_NH;

  const int32_t seq_begin = major_begin;
  const int32_t seq_end = seq_begin + params.TILE_S >= local_seq_len_remote_rank
                              ? local_seq_len_remote_rank
                              : seq_begin + params.TILE_S;
  const int32_t nh_begin = minor_begin;
  const int32_t nh_end =
      nh_begin + params.TILE_NH >= local_nheads ? local_nheads : nh_begin + params.TILE_NH;
  const int32_t offset_seq = params.cusum_seq_lens[remote_rank];
  return {bs_idx, seq_begin + offset_seq, seq_end + offset_seq, nh_begin, nh_end};
}

__device__ __forceinline__ int32_t
get_num_tiles_for_target_rank(int32_t target_rank, const PreAttnQKVPackA2AParams &params) {
  const int32_t local_nheads =
      params.local_q_nheads + params.local_k_nheads + params.local_v_nheads;
  int32_t local_seq_len =
      params.cusum_seq_lens[target_rank + 1] - params.cusum_seq_lens[target_rank];
  int32_t num_tiles =
      params.bs * ceil_div(local_seq_len, params.TILE_S) * ceil_div(local_nheads, params.TILE_NH);
  return num_tiles;
}

}  // namespace

template <typename Element, bool kAligned, bool kSkipBarrier>
__global__
__launch_bounds__(kMaxThreadsPerBlock) void pre_attn_qkv_pack_a2a_kernel(
    const PreAttnQKVPackA2AParams params) {
  if (params.num_comm_sm >= params.world_size) {
    int32_t num_block_per_group = params.num_comm_sm / params.world_size;
    int32_t group_id = blockIdx.x % num_block_per_group;
    int32_t remote_rank = blockIdx.x / num_block_per_group;
    int32_t num_tiles_remote_rank = get_num_tiles_for_target_rank(remote_rank, params);
    for (int i = group_id; i < num_tiles_remote_rank; i += num_block_per_group) {
      auto tile_info = get_tile_info(remote_rank, i, params);
      using ElementVec = std::conditional_t<kAligned, uint4, Element>;
      pull_tile_to_local<Element, ElementVec, kSkipBarrier>(remote_rank, tile_info, params);
    }
  } else {
    // 8 is enough to single node
    int32_t num_tiles_per_rank[8];
    int32_t max_num_tiles = 0;
    for (int remote_rank = blockIdx.x, j = 0; remote_rank < params.world_size;
         remote_rank += params.num_comm_sm, j += 1) {
      num_tiles_per_rank[j] = get_num_tiles_for_target_rank(remote_rank, params);
      max_num_tiles =
          max_num_tiles < num_tiles_per_rank[j] ? num_tiles_per_rank[j] : max_num_tiles;
    }
    for (int i = 0; i < max_num_tiles; i++) {
      for (int remote_rank = blockIdx.x, j = 0; remote_rank < params.world_size;
           remote_rank += params.num_comm_sm, j += 1) {
        if (i < num_tiles_per_rank[j]) {
          auto tile_info = get_tile_info(remote_rank, i, params);
          using ElementVec = std::conditional_t<kAligned, uint4, Element>;
          pull_tile_to_local<Element, ElementVec, kSkipBarrier>(remote_rank, tile_info, params);
        }
      }
    }
  }
}

void
pre_attn_qkv_pack_a2a_impl(
    const PreAttnQKVPackA2AParams params, DataTypeEnum input_dtype, cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);

  FLUX_CHECK(
      (params.num_comm_sm % params.world_size == 0 ||
       params.world_size % params.num_comm_sm == 0) &&
      params.num_comm_sm > 0);

  int32_t grid_size = params.num_comm_sm;
  tuple_return_if(
      cute::make_tuple(_BF16{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        constexpr int32_t PACK_SIZE = 16 / sizeof(Element);
        bool align = params.head_dim % PACK_SIZE == 0;
        align &= isAligned<Element, PACK_SIZE>(params.q_ptr);
        align &= isAligned<Element, PACK_SIZE>(params.k_ptr);
        align &= isAligned<Element, PACK_SIZE>(params.v_ptr);

        BOOL_SWITCH(align, kAlign, [&]() {
          BOOL_SWITCH(params.skip_barrier, kSkipBarrier, [&] {
            pre_attn_qkv_pack_a2a_kernel<Element, kAlign, kSkipBarrier>
                <<<grid_size, kThreadsPerBlock, 0, stream>>>(params);
          });
        });
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
}

}  // namespace flux
}  // namespace bytedance