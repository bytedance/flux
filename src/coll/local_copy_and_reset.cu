//===- local_copy_and_reset.cu ------------------------------------ C++ ---===//
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

#include "coll/local_copy_and_reset.hpp"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/flux.h"

namespace bytedance {
namespace flux {

constexpr int32_t kMaxPackBytes = 128 / 8;
constexpr int32_t kLocalCopyThreadPerBlock = 128;

template <typename T, int32_t PACK_SIZE>
struct alignas(sizeof(T) * PACK_SIZE) PackedData {
  __device__
  PackedData() {}

  T elem[PACK_SIZE];
};

template <typename T, int32_t PACK_SIZE>
bool
isAligned(void *ptr) {
  return reinterpret_cast<std::uintptr_t>(ptr) % sizeof(PackedData<T, PACK_SIZE>) == 0;
}

// TODO(houqi.1993) implement a sync_peers with real multiple grids support.
// use a cudaLaunchCooperativeLaunch-like implementation but implement ourself
__device__ __forceinline__ void
sync_peers_atomic(int32_t **sync_barriers, int32_t rank, int32_t world_size) {
  if (threadIdx.x < world_size) {
    int *remote_ptr = sync_barriers[threadIdx.x] + blockIdx.x * world_size + rank;
#pragma unroll 1
    while (atomicCAS_system(remote_ptr, 0, 1) != 0) {
    }
    int *local_ptr = sync_barriers[rank] + blockIdx.x * world_size + threadIdx.x;
#pragma unroll 1
    while (atomicCAS_system(local_ptr, 1, 0) != 1) {
    }
  }
}

// TODO(houqi.1993) implement a sync_peers with real multiple grids support.
// copy of cudaIpc_barrier_all.cu CudaIpcBarrierAllRingModeKernel but support multiple grids
__device__ __forceinline__ void
sync_peers_ring(int32_t **sync_barriers, int32_t rank, int32_t world_size) {
  int next_peer = (rank + 1) % world_size;
  int *ptr_next_peer = sync_barriers[next_peer] + blockIdx.x * world_size;
  int *ptr_cur_rank = sync_barriers[rank] + blockIdx.x * world_size;
  if (threadIdx.x != 0)
    return;
  if (rank == 0) {
    // PCIE do not support automic
    atomic_store_release_sys(ptr_next_peer, 1);
  } else {
#pragma unroll 1
    while (ld_acquire_sys(ptr_cur_rank) != 1) {
    }
    atomic_store_release_sys(ptr_next_peer, 1);
  }
  // quit in a reversed order
  if (rank != world_size - 1) {
#pragma unroll 1
    while (ld_acquire_sys(ptr_next_peer) != 0) {
    }
  }
  ptr_cur_rank[0] = 0;
}

template <bool kSyncInRingMode>
__device__ __forceinline__ void
sync_peers(int32_t **sync_barriers, int32_t rank, int32_t world_size) {
  if constexpr (kSyncInRingMode) {
    sync_peers_ring(sync_barriers, rank, world_size);
    __syncthreads();
  } else {
    sync_peers_atomic(sync_barriers, rank, world_size);
    __syncthreads();
  }
}

template <typename PackedInputType, typename ScaleType, bool kSyncInRingMode>
__global__ void
local_copy_and_reset_kernel(
    PackedInputType *input_src,
    PackedInputType *input_dst,
    ScaleType *scale_src,
    ScaleType *scale_dst,
    int32_t *counter,
    int32_t *ag_barrier,
    int32_t world_size,
    int32_t rank,
    size_t input_nele,
    size_t scale_nele,
    bool has_scale,
    int32_t **sync_barriers) {
  int32_t copy_blocks = gridDim.x - 2;
  if (sync_barriers != nullptr) {
    sync_peers<kSyncInRingMode>(sync_barriers, rank, world_size);
  }

  if (blockIdx.x < copy_blocks) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = blockDim.x * copy_blocks;
    for (size_t i = tid; i < input_nele; i += step) {
      input_dst[i] = input_src[i];
    }

    if (has_scale) {
      for (size_t i = tid; i < scale_nele; i += step) {
        scale_dst[i] = scale_src[i];
      }
    }
  } else if (blockIdx.x == copy_blocks) {
    // reset ag barrier
    int32_t tid = threadIdx.x;
    for (int32_t i = tid; i < world_size; i += blockDim.x) {
      ag_barrier[i] = (i == rank ? 1 : 0);
    }
  } else if (blockIdx.x == copy_blocks + 1) {
    // reset counter
    int32_t tid = threadIdx.x;
    // The extra plus one is to reset ag_launched_signal.
    for (int32_t i = tid; i < world_size + 1; i += blockDim.x) {
      counter[i] = 0;
    }
  }
  if (sync_barriers != nullptr) {
    sync_peers<kSyncInRingMode>(sync_barriers, rank, world_size);
  }
}

size_t
get_local_copy_max_block_num(size_t num_input, int32_t pack_size) {
  size_t total_blocks =
      (num_input / pack_size + kLocalCopyThreadPerBlock - 1) / kLocalCopyThreadPerBlock + 2;
  return total_blocks;
}

void
local_copy_and_reset_impl(
    void *input_src,
    void *input_dst,
    void *scale_src,
    void *scale_dst,
    int32_t *counter,
    int32_t *ag_barrier,
    int32_t world_size,
    int32_t rank,
    int32_t m,
    int32_t n,
    int32_t **sync_barriers,
    DataTypeEnum input_dtype,
    DataTypeEnum scale_dtype,
    bool sync_ring_mode,
    cudaStream_t stream) {
  dim3 block_size(kLocalCopyThreadPerBlock);
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_S8{}),
          cute::make_tuple(_FP32{}),
          cute::make_tuple(cute::_0{}, cute::_1{})),
      [&](auto tup) {
        auto [candidate_input_dtype, candidate_scale_dtype, candidate_sync_ring_mode] = tup;
        return input_dtype == candidate_input_dtype && scale_dtype == candidate_scale_dtype &&
               sync_ring_mode == candidate_sync_ring_mode;
      },
      [&](auto tup) {
        auto [candidate_input_dtype, candidate_scale_dtype, candidate_sync_ring_mode] = tup;
        using ElementInput = decltype(to_cuda_dtype(candidate_input_dtype));
        using ElementScale = decltype(to_cuda_dtype(candidate_scale_dtype));
        constexpr int32_t kMaxPackSize = kMaxPackBytes / sizeof(ElementInput);
        constexpr int kSyncInRingMode = decltype(candidate_sync_ring_mode){};
        size_t num_input = (size_t)m * n;
        bool isAlignedInputs = isAligned<ElementInput, kMaxPackSize>(input_src) &&
                               isAligned<ElementInput, kMaxPackSize>(input_dst) &&
                               num_input % kMaxPackSize == 0;
        bool has_scale = (scale_src != nullptr && scale_dst != nullptr);
        if (isAlignedInputs) {
          size_t total_blocks = get_local_copy_max_block_num(num_input, kMaxPackSize);
          local_copy_and_reset_kernel<
              PackedData<ElementInput, kMaxPackSize>,
              ElementScale,
              kSyncInRingMode><<<total_blocks, block_size, 0, stream>>>(
              reinterpret_cast<PackedData<ElementInput, kMaxPackSize> *>(input_src),
              reinterpret_cast<PackedData<ElementInput, kMaxPackSize> *>(input_dst),
              reinterpret_cast<ElementScale *>(scale_src),
              reinterpret_cast<ElementScale *>(scale_dst),
              counter,
              ag_barrier,
              world_size,
              rank,
              static_cast<size_t>(num_input / kMaxPackSize),
              static_cast<size_t>(m),
              has_scale,
              sync_barriers);
        } else {
          size_t total_blocks = get_local_copy_max_block_num(num_input, 1);
          local_copy_and_reset_kernel<PackedData<ElementInput, 1>, ElementScale, kSyncInRingMode>
              <<<total_blocks, block_size, 0, stream>>>(
                  reinterpret_cast<PackedData<ElementInput, 1> *>(input_src),
                  reinterpret_cast<PackedData<ElementInput, 1> *>(input_dst),
                  reinterpret_cast<ElementScale *>(scale_src),
                  reinterpret_cast<ElementScale *>(scale_dst),
                  counter,
                  ag_barrier,
                  world_size,
                  rank,
                  num_input,
                  static_cast<size_t>(m),
                  has_scale,
                  sync_barriers);
        }
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype
                          << ",scale_dtype=" << scale_dtype << "\n";
      });
}
}  // namespace flux
}  // namespace bytedance
