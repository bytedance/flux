//===- post_attn_a2a_transpose_impls.cu --------------------------- C++ ---===//
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
#include "cute/layout.hpp"
#include "cute/tensor_impl.hpp"
#include <type_traits>
#include "post_attn_a2a_transpose_impls.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/arch/reg_reconfig.h"

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

#define SYNC_METHOD_SWITCH(SYNC_METHOD, ...)                                        \
  [&] {                                                                             \
    if (SYNC_METHOD == SyncMethod::SyncNone) {                                      \
      constexpr int32_t kSyncMethod = static_cast<int32_t>(SyncMethod::SyncNone);   \
      return __VA_ARGS__();                                                         \
    } else if (SYNC_METHOD == SyncMethod::SyncAtomic) {                             \
      constexpr int32_t kSyncMethod = static_cast<int32_t>(SyncMethod::SyncAtomic); \
      return __VA_ARGS__();                                                         \
    }                                                                               \
  }()

struct AttnAll2AllTileInfo {
  int32_t bs_idx;
  int32_t seq_begin;
  int32_t seq_end;
};

template <typename T, int32_t PACK_SIZE>
bool
isAligned(void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % (sizeof(T) * PACK_SIZE) == 0;
}

// TODO(houqi.1993) implement a sync_peers with real multiple grids support.
// use a cudaLaunchCooperativeLaunch-like implementation but implement ourself
__device__ __forceinline__ void
sync_peers_atomic(int32_t **sync_barriers, int32_t bid, int32_t rank, int32_t world_size) {
  if (threadIdx.x < world_size) {
    int *remote_ptr = sync_barriers[threadIdx.x] + bid * world_size + rank;
#pragma unroll 1
    while (atomicCAS_system(remote_ptr, 0, 1) != 0) {
    }
    int *local_ptr = sync_barriers[rank] + bid * world_size + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(local_ptr, 1, 0) != 1) {
    }
  }
}

template <int kSyncMethod>
__device__ __forceinline__ void
sync_peers(int32_t **sync_barriers, int32_t bid, int32_t rank, int32_t world_size) {
  if constexpr (kSyncMethod == static_cast<int32_t>(SyncMethod::SyncAtomic)) {
    sync_peers_atomic(sync_barriers, bid, rank, world_size);
    __syncthreads();
  }
}

template <typename Element, typename ElementVec>
__device__ __forceinline__ void
push_tile_to_dst(
    int32_t bid,
    int32_t block_per_tile,
    const AttnAll2AllTileInfo &tile_info,
    const PostAttnAll2AllParams &params) {
  const int32_t group_id = bid % block_per_tile;  // a group of CTA copy a tile

  const int32_t &world_size = params.world_size;
  const int32_t &seq_len = params.seq_len;
  const int32_t &nheads = params.nheads;
  const int32_t &head_dim = params.head_dim;

  const int32_t &bs_idx = tile_info.bs_idx;

  const int32_t local_seq_len = seq_len / world_size;

  const int32_t local_nheads = nheads / world_size;
  const int32_t dst_rank = tile_info.seq_begin / local_seq_len;

  const int32_t vec_head_dim = head_dim / (sizeof(ElementVec) / sizeof(Element));
  const int32_t vec_hidden_dim = vec_head_dim * nheads;
  const int32_t tile_seq_len = tile_info.seq_end - tile_info.seq_begin;
  const int32_t nelems = vec_head_dim * local_nheads * tile_seq_len;
  const int32_t num_threads_group = block_per_tile * blockDim.x;

  for (int32_t i = group_id * blockDim.x + threadIdx.x; i < nelems; i += num_threads_group) {
    int32_t src_hd_idx = i % vec_head_dim;
    int32_t src_seq_idx = (i / vec_head_dim) % tile_seq_len + tile_info.seq_begin;
    int32_t src_nh_idx = i / (vec_head_dim * tile_seq_len);
    int32_t dst_seq_idx = src_seq_idx % local_seq_len;

    // add offset in bs/seq dim
    ElementVec *vec_input_ptr = reinterpret_cast<ElementVec *>(params.input_ptr) +
                                bs_idx * local_seq_len * vec_hidden_dim +
                                src_seq_idx * vec_head_dim;
    ElementVec *vec_output_ptr = reinterpret_cast<ElementVec *>(params.output_ptrs[dst_rank]) +
                                 bs_idx * local_seq_len * vec_hidden_dim +
                                 dst_seq_idx * vec_hidden_dim;
    vec_output_ptr[(params.rank * local_nheads + src_nh_idx) * vec_head_dim + src_hd_idx] =
        vec_input_ptr[src_nh_idx * seq_len * vec_head_dim + src_hd_idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t num_tiles_per_local_seq = (local_seq_len + params.TILE_M - 1) / params.TILE_M;
    int32_t dst_tile_idx =
        bs_idx * num_tiles_per_local_seq + (tile_info.seq_begin % local_seq_len) / params.TILE_M;

    asm volatile("fence.acq_rel.sys;\n");
    int32_t *dst_barrier = static_cast<int32_t *>(params.barrier_ptrs[dst_rank]) + dst_tile_idx;
    int32_t rank_cnt = atomicAdd_system(dst_barrier, -1);
    // last tile in nheads dim
    if (rank_cnt - 1 == -1 * world_size * block_per_tile) {
      asm volatile("st.global.release.sys.b32 [%0], 1;\n" : : "l"(dst_barrier));
    }
  }
}

__device__ __forceinline__ AttnAll2AllTileInfo
get_tile_info(int32_t tile_idx, const PostAttnAll2AllParams &params) {
  int32_t local_seq_len = params.seq_len / params.world_size;
  int32_t num_tiles_per_local_seq = (local_seq_len + params.TILE_M - 1) / params.TILE_M;
  int32_t num_tiles_per_seq = num_tiles_per_local_seq * params.world_size;

  int32_t bs_idx = tile_idx / num_tiles_per_seq;
  int32_t tile_idx_seq = tile_idx % num_tiles_per_seq;
  int32_t dst_rank = tile_idx_seq % params.world_size;

  int32_t seq_begin = tile_idx_seq / params.world_size * params.TILE_M + dst_rank * local_seq_len;
  int32_t seq_end =
      seq_begin + params.TILE_M >= params.seq_len ? params.seq_len : seq_begin + params.TILE_M;
  return {bs_idx, seq_begin, seq_end};
}
}  // namespace

template <typename Element, bool kAligned, int32_t kSyncMethod>
__global__ void
post_attn_all2all_transpose_kernel(const PostAttnAll2AllParams params) {
  int32_t total_tiles = (params.seq_len * params.bs + params.TILE_M - 1) / params.TILE_M;
  int32_t empty_blocks = gridDim.x - params.num_comm_sm;
  if (blockIdx.x < empty_blocks)
    return;
  int32_t bid = blockIdx.x - empty_blocks;

  sync_peers<kSyncMethod>(params.sync_barriers, bid, params.rank, params.world_size);

  // notify the a2a kernel has been launched.
  if (threadIdx.x == 0 && bid == 0) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], 1;" : : "l"(params.a2a_signal));
  }

  if (params.num_comm_sm > params.world_size) {
    const int32_t block_per_tile = params.num_comm_sm / params.world_size;
    // cudaTriggerProgrammaticLaunchCompletion();
    for (int32_t i = bid / block_per_tile; i < total_tiles; i += params.world_size) {
      auto tile_info = get_tile_info(i, params);
      using ElementVec = std::conditional_t<kAligned, uint4, Element>;
      push_tile_to_dst<Element, ElementVec>(bid, block_per_tile, tile_info, params);
    }
  } else {
    for (int32_t i = bid; i < total_tiles; i += params.num_comm_sm) {
      auto tile_info = get_tile_info(i, params);
      using ElementVec = std::conditional_t<kAligned, uint4, Element>;
      push_tile_to_dst<Element, ElementVec>(bid, 1, tile_info, params);
    }
  }
}

template <typename Element, bool kAligned, bool kSkipBarrier>
__device__ __forceinline__ void
post_attn_all2all_tile_comm(
    const PostAttnAll2AllParams &params,
    int32_t bid,
    int32_t remote_rank,
    int32_t num_block_per_group) {
  using namespace cute;
  int32_t thread_id = threadIdx.x;
  int32_t local_nheads = params.nheads / params.world_size;
  int32_t local_seq_len = params.seq_len / params.world_size;
  int32_t local_hidden_dim = local_nheads * params.head_dim;
  auto inputGmemLayout = make_layout(
      make_shape(params.bs, params.world_size, local_seq_len, local_hidden_dim),
      cute::LayoutRight{});
  auto outputGmemLayout = make_layout(
      make_shape(params.bs, local_seq_len, params.world_size, local_hidden_dim),
      cute::LayoutRight{});
  auto mInput = make_tensor(
      make_gmem_ptr<Element>(params.input_ptr),
      inputGmemLayout);  // (bs, world_size, local_seq_len, local_hidden_dim)

  auto mOutput = make_tensor(
      make_gmem_ptr<Element>(params.output_ptrs[remote_rank]),
      outputGmemLayout);  // (bs, local_seq_len, world_size, local_hidden_dim)
  auto gInput_per_group = mInput(_, remote_rank, _, _);    // (bs, local_seq_len, local_hidden_dim)
  auto gOutput_per_group = mOutput(_, _, params.rank, _);  // (bs, local_seq_len, local_hidden_dim)

  using ElementVec = std::conditional_t<kAligned, uint4, Element>;

  Tensor gInput_vec =
      recast<ElementVec>(gInput_per_group);  // (bs, local_seq_len, local_hidden_dim_vec)
  Tensor gOutput_vec =
      recast<ElementVec>(gOutput_per_group);  // (bs, local_seq_len, local_hidden_dim_vec)

  int32_t bs = params.bs;
  int32_t num_threads = blockDim.x * num_block_per_group;
  int32_t tid_in_group = thread_id + bid % num_block_per_group * blockDim.x;
  int32_t *dst_barrier = static_cast<int32_t *>(params.barrier_ptrs[remote_rank]);
  int32_t num_tiles_per_group = (bs * local_seq_len + params.TILE_M - 1) / params.TILE_M;
  for (int32_t i = 0; i < num_tiles_per_group; ++i) {
    int32_t m_begin = i * params.TILE_M;
    int32_t m_end = m_begin + params.TILE_M < bs * local_seq_len ? m_begin + params.TILE_M
                                                                 : bs * local_seq_len;
    int32_t num_elems = size<2>(gInput_vec) * (m_end - m_begin);
    auto tile_shape = cute::make_shape(
        size<2>(gInput_vec), m_end - m_begin);  // (local_hidden_dim_vec, m_end - m_begin)
    for (int32_t t = tid_in_group; t < num_elems; t += num_threads) {
      auto coord = idx2crd(t, tile_shape);
      int32_t hidden_idx = get<0>(coord);
      int32_t m_idx = get<1>(coord) + m_begin;
      int32_t bs_idx = m_idx / local_seq_len;
      int32_t seq_idx = m_idx % local_seq_len;
      gOutput_vec(bs_idx, seq_idx, hidden_idx) = gInput_vec(bs_idx, seq_idx, hidden_idx);
    }

    if constexpr (!kSkipBarrier) {
      __syncthreads();

      if (thread_id == 0) {
        int32_t *tile_barrier = dst_barrier + i;
        asm volatile("fence.acq_rel.sys;\n");
        int32_t rank_cnt = atomicAdd_system(tile_barrier, -1);
        // last tile in nheads dim
        if (rank_cnt - 1 == -1 * params.world_size * num_block_per_group) {
          asm volatile("st.global.release.sys.b32 [%0], 1;\n" : : "l"(tile_barrier));
        }
      }
    }
  }
}

template <typename Element, bool kAligned, bool kSkipBarrier, int32_t kSyncMethod>
__global__ void
post_attn_all2all_kernel(const PostAttnAll2AllParams params) {
  using namespace cute;
  int32_t empty_blocks = gridDim.x - params.num_comm_sm;
  if (blockIdx.x < empty_blocks)
    return;
  int32_t bid = blockIdx.x - empty_blocks;
  sync_peers<kSyncMethod>(params.sync_barriers, bid, params.rank, params.world_size);

  // notify that the a2a kernel has been launched.
  if (threadIdx.x == 0 && bid == 0) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], 1;" : : "l"(params.a2a_signal));
  }
  if (params.num_comm_sm < params.world_size) {
    for (int32_t r = bid; r < params.world_size; r += params.num_comm_sm) {
      post_attn_all2all_tile_comm<Element, kAligned, kSkipBarrier>(params, bid, r, 1);
    }
  } else {
    int32_t num_block_per_group = params.num_comm_sm / params.world_size;
    int32_t remote_rank = bid / num_block_per_group;
    if (bid < params.world_size * num_block_per_group) {
      post_attn_all2all_tile_comm<Element, kAligned, kSkipBarrier>(
          params, bid, remote_rank, num_block_per_group);
    }
  }
}

template <typename Element, bool kAligned, bool kSkipBarrier>
__device__ __forceinline__ void
post_attn_all2all_dyn_tile_comm(
    const PostAttnAll2AllParams &params,
    int32_t bid,
    int32_t remote_rank,
    int32_t num_block_per_group) {
  using namespace cute;
  int32_t thread_id = threadIdx.x;

  int32_t local_nheads = params.nheads / params.world_size;
  int32_t total_seq_len = params.cusum_seq_lens[params.world_size];
  int32_t local_hidden_dim = local_nheads * params.head_dim;
  int32_t remote_rank_local_seq_start = params.cusum_seq_lens[remote_rank];
  int32_t remote_rank_local_seq_len =
      params.cusum_seq_lens[remote_rank + 1] - remote_rank_local_seq_start;
  int64_t stride_input_bs = (int64_t)total_seq_len * local_hidden_dim;
  int64_t stride_input_seq = (int64_t)local_hidden_dim;
  auto inputPerGroupGmemLayout = make_layout(
      make_shape(params.bs, remote_rank_local_seq_len, local_hidden_dim),
      make_stride(stride_input_bs, stride_input_seq, Int<1>{}));

  int64_t stride_output_bs =
      (int64_t)remote_rank_local_seq_len * local_hidden_dim * params.world_size;
  int64_t stride_output_seq = (int64_t)local_hidden_dim * params.world_size;
  auto outputPerGroupGmemLayout = make_layout(
      make_shape(params.bs, remote_rank_local_seq_len, local_hidden_dim),
      make_stride(stride_output_bs, stride_output_seq, Int<1>{}));

  int64_t output_offset = params.rank * local_hidden_dim;
  auto gOutput_per_group = make_tensor(
      make_gmem_ptr<Element>(params.output_ptrs[remote_rank]) + output_offset,
      outputPerGroupGmemLayout);

  int64_t input_offset = remote_rank_local_seq_start * local_hidden_dim;
  auto gInput_per_group = make_tensor(
      make_gmem_ptr<Element>(params.input_ptr) + input_offset, inputPerGroupGmemLayout);

  using ElementVec = std::conditional_t<kAligned, uint4, Element>;

  Tensor gInput_vec = recast<ElementVec>(
      gInput_per_group);  // (bs, remote_rank_local_seq_len, local_hidden_dim_vec)
  Tensor gOutput_vec = recast<ElementVec>(
      gOutput_per_group);  // (bs, remote_rank_local_seq_len, local_hidden_dim_vec)

  int32_t bs = params.bs;
  int32_t num_threads = blockDim.x * num_block_per_group;
  int32_t tid_in_group = thread_id + bid % num_block_per_group * blockDim.x;
  int32_t *dst_barrier = static_cast<int32_t *>(params.barrier_ptrs[remote_rank]);
  int32_t num_tiles_per_group =
      (bs * remote_rank_local_seq_len + params.TILE_M - 1) / params.TILE_M;
  for (int32_t i = 0; i < num_tiles_per_group; ++i) {
    int32_t m_begin = i * params.TILE_M;
    int32_t m_end = m_begin + params.TILE_M < bs * remote_rank_local_seq_len
                        ? m_begin + params.TILE_M
                        : bs * remote_rank_local_seq_len;
    int32_t num_elems = size<2>(gInput_vec) * (m_end - m_begin);
    auto tile_shape = cute::make_shape(
        size<2>(gInput_vec), m_end - m_begin);  // (local_hidden_dim_vec, m_end - m_begin)
    for (int32_t t = tid_in_group; t < num_elems; t += num_threads) {
      auto coord = idx2crd(t, tile_shape);
      int32_t hidden_idx = get<0>(coord);
      int32_t m_idx = get<1>(coord) + m_begin;
      int32_t bs_idx = m_idx / remote_rank_local_seq_len;
      int32_t seq_idx = m_idx % remote_rank_local_seq_len;
      gOutput_vec(bs_idx, seq_idx, hidden_idx) = gInput_vec(bs_idx, seq_idx, hidden_idx);
    }

    if constexpr (!kSkipBarrier) {
      __syncthreads();

      if (thread_id == 0) {
        int32_t *tile_barrier = dst_barrier + i;
        asm volatile("fence.acq_rel.sys;\n");
        int32_t rank_cnt = atomicAdd_system(tile_barrier, -1);
        // last tile in nheads dim
        if (rank_cnt - 1 == -1 * params.world_size * num_block_per_group) {
          asm volatile("st.global.release.sys.b32 [%0], 1;\n" : : "l"(tile_barrier));
        }
      }
    }
  }
}

template <typename Element, bool kAligned, bool kSkipBarrier, int32_t kSyncMethod>
__global__ void
post_attn_all2all_dyn_seq_kernel(const PostAttnAll2AllParams params) {
  using namespace cute;
  int32_t empty_blocks = gridDim.x - params.num_comm_sm;
  if (blockIdx.x < empty_blocks)
    return;
  int32_t bid = blockIdx.x - empty_blocks;
  sync_peers<kSyncMethod>(params.sync_barriers, bid, params.rank, params.world_size);

  // notify that the a2a kernel has been launched.
  if (threadIdx.x == 0 && bid == 0) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], 1;" : : "l"(params.a2a_signal));
  }

  if (params.num_comm_sm < params.world_size) {
    for (int32_t r = bid; r < params.world_size; r += params.num_comm_sm) {
      post_attn_all2all_dyn_tile_comm<Element, kAligned, kSkipBarrier>(params, bid, r, 1);
    }
  } else {
    int32_t num_block_per_group = params.num_comm_sm / params.world_size;
    int32_t remote_rank = bid / num_block_per_group;
    if (bid < params.world_size * num_block_per_group) {
      post_attn_all2all_dyn_tile_comm<Element, kAligned, kSkipBarrier>(
          params, bid, remote_rank, num_block_per_group);
    }
  }
}

int32_t
get_post_attn_all2all_transpose_block_num(int32_t bs, int32_t seq_len, int32_t tile_m) {
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  return sm_count;
}

// [bs, local_nheads, seq_len, head_dim] => [bs, local_seq_len, nheads, head_dim]
void
post_attn_a2a_transpose_impl(
    const PostAttnAll2AllParams &params,
    DataTypeEnum input_dtype,
    SyncMethod sync_method,
    cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  FLUX_CHECK_LE(params.num_comm_sm, sm_count);
  int32_t local_seq_len = params.seq_len / params.world_size;
  FLUX_CHECK(local_seq_len % params.TILE_M == 0);
  FLUX_CHECK(params.num_comm_sm > 0);
  int32_t grid_size = sm_count;

  tuple_return_if(
      cute::make_tuple(_BF16{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        constexpr int32_t PACK_SIZE = 16 / sizeof(Element);
        bool align =
            isAligned<Element, PACK_SIZE>(params.input_ptr) && params.head_dim % PACK_SIZE == 0;
        SYNC_METHOD_SWITCH(sync_method, [&] {
          BOOL_SWITCH(align, kAligned, [&] {
            post_attn_all2all_transpose_kernel<Element, kAligned, kSyncMethod>
                <<<grid_size, kThreadsPerBlock, 0, stream>>>(params);
          });
        });
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
}

// [bs, seq_len, local_nheads, head_dim] => [bs, local_seq_len, nheads, head_dim]
void
post_attn_a2a_impl(
    const PostAttnAll2AllParams &params,
    DataTypeEnum input_dtype,
    SyncMethod sync_method,
    cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  FLUX_CHECK_LE(params.num_comm_sm, sm_count);
  FLUX_CHECK(params.num_comm_sm > 0);
  int32_t grid_size = sm_count;

  tuple_return_if(
      cute::make_tuple(_BF16{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        constexpr int32_t PACK_SIZE = 16 / sizeof(Element);
        bool align =
            isAligned<Element, PACK_SIZE>(params.input_ptr) && params.head_dim % PACK_SIZE == 0;
        SYNC_METHOD_SWITCH(sync_method, [&] {
          BOOL_SWITCH(align, kAligned, [&] {
            BOOL_SWITCH(params.skip_barrier, kSkipBarrier, [&] {
              post_attn_all2all_kernel<Element, kAligned, kSkipBarrier, kSyncMethod>
                  <<<grid_size, kThreadsPerBlock, 0, stream>>>(params);
            });
          });
        });
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
}

// [bs, seq_len, local_nheads, head_dim] => [bs, local_seq_len, nheads, head_dim]
void
post_attn_a2a_dyn_impl(
    const PostAttnAll2AllParams &params,
    DataTypeEnum input_dtype,
    SyncMethod sync_method,
    cudaStream_t stream) {
  static constexpr int32_t kThreadsPerBlock = 1024;
  int sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  FLUX_CHECK_LE(params.num_comm_sm, sm_count);
  FLUX_CHECK(params.num_comm_sm > 0);
  int32_t grid_size = sm_count;

  tuple_return_if(
      cute::make_tuple(_BF16{}),
      [&](auto cdtype) { return cdtype == input_dtype; },
      [&](auto cdtype) {
        using Element = decltype(to_cuda_dtype(cdtype));
        constexpr int32_t PACK_SIZE = 16 / sizeof(Element);
        bool align =
            isAligned<Element, PACK_SIZE>(params.input_ptr) && params.head_dim % PACK_SIZE == 0;
        SYNC_METHOD_SWITCH(sync_method, [&] {
          BOOL_SWITCH(align, kAligned, [&] {
            BOOL_SWITCH(params.skip_barrier, kSkipBarrier, [&] {
              post_attn_all2all_dyn_seq_kernel<Element, kAligned, kSkipBarrier, kSyncMethod>
                  <<<grid_size, kThreadsPerBlock, 0, stream>>>(params);
            });
          });
        });
      },
      [&]() { FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype << "\n"; });
}

}  // namespace flux
}  // namespace bytedance
