//===- all_gather_impls.cu --------------------------------------- C++ ---===//
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
#include "all_gather_impls.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"

namespace bytedance {
namespace flux {
#define CP_CONFIG_SWITCH(CP_SIZE_PER_RANK, ...) \
  [&] {                                         \
    if (CP_SIZE_PER_RANK >= 64 * 8192) {        \
      constexpr int kBlocksPerRank = 24;        \
      constexpr int kCoRunRanks = 4;            \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr int kBlocksPerRank = 10;        \
      constexpr int kCoRunRanks = 4;            \
      return __VA_ARGS__();                     \
    }                                           \
  }()

constexpr int32_t kMaxPackBytes = 128 / 8;

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

template <
    typename Element,
    typename ScaleType,
    const int32_t PACK_SIZE,
    const int32_t BLOCKS_PER_RANK,
    const int32_t CO_RUN_RANKS>
__global__ void
ag_a2a_mode_kernel(AllGatherParams params) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
  // notify the ag kernel has been launched.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], 1;" : : "l"(params.ag_signal));
  }

  size_t input_nele = (size_t)params.m * params.n / PACK_SIZE;
  int32_t bid = blockIdx.x % BLOCKS_PER_RANK;
  size_t tid = bid * blockDim.x + threadIdx.x;
  size_t step = blockDim.x * BLOCKS_PER_RANK;
  for (int32_t i = params.rank + 1 + blockIdx.x / BLOCKS_PER_RANK;
       i < (params.world_size + params.rank);
       i += CO_RUN_RANKS) {
    int32_t idx = i % params.world_size;
    PackedData<Element, PACK_SIZE> *dst_input_ptr =
        reinterpret_cast<PackedData<Element, PACK_SIZE> *>(params.input_ptrs[params.rank]) +
        input_nele * idx;
    ScaleType *dst_scale_ptr =
        reinterpret_cast<ScaleType *>(params.scale_ptrs[params.rank]) + params.m * idx;

    PackedData<Element, PACK_SIZE> *src_input_ptr =
        reinterpret_cast<PackedData<Element, PACK_SIZE> *>(params.input_ptrs[idx]) +
        input_nele * idx;
    ScaleType *src_scale_ptr =
        reinterpret_cast<ScaleType *>(params.scale_ptrs[idx]) + params.m * idx;
    for (size_t j = tid; j < input_nele; j += step) {
      dst_input_ptr[j] = src_input_ptr[j];
    }

    if (params.has_scale) {
      for (int32_t j = tid; j < params.m; j += step) {
        dst_scale_ptr[j] = src_scale_ptr[j];
      }
    }

    // Only read the original input from other ranks (copied to the input buffer during the local
    // copy stage), and the written data(full input and ag barrier) is only used for local GEMM.
    // The written data only needs to be visible to threads on the same device, so __threadfence is
    // used.
    __threadfence();

    // Synchronize to make sure that all threads within block finish writing data
    __syncthreads();
    if (threadIdx.x == 0) {
      int32_t counter_val = atomicAdd(params.counter + idx, 1);
      if (counter_val == BLOCKS_PER_RANK - 1) {
        params.ag_barriers[params.rank][idx] = 1;
      }
    }
  }
}

void
ag_a2a_mode(
    const AllGatherParams &params,
    DataTypeEnum input_dtype,
    DataTypeEnum scale_dtype,
    cudaStream_t stream) {
  constexpr int32_t kThreadPerBlock = 512;
  dim3 block_size(kThreadPerBlock);
  tuple_return_if(
      tuple_cartesian_product(cute::make_tuple(_S8{}), cute::make_tuple(_FP32{})),
      [&](auto tup) {
        auto [candidate_input_dtype, candidate_scale_dtype] = tup;
        return input_dtype == candidate_input_dtype && scale_dtype == candidate_scale_dtype;
      },
      [&](auto tup) {
        auto [candidate_input_dtype, candidate_scale_dtype] = tup;
        using ElementInput = decltype(to_cuda_dtype(candidate_input_dtype));
        using ElementScale = decltype(to_cuda_dtype(candidate_scale_dtype));
        constexpr int32_t kMaxPackSize = kMaxPackBytes / sizeof(ElementInput);
        bool isAlignedInputs = true;
        for (size_t i = 0; i < params.world_size; ++i) {
          isAlignedInputs =
              isAlignedInputs && isAligned<ElementInput, kMaxPackSize>(params.input_ptrs[i]);
        }
        size_t num_input = (size_t)params.m * params.n;
        isAlignedInputs = isAlignedInputs && num_input % kMaxPackSize == 0;
        size_t cp_size_per_rank = num_input * sizeof(ElementInput);
        CP_CONFIG_SWITCH(cp_size_per_rank, [&] {
          dim3 grid_size(kCoRunRanks * kBlocksPerRank);
          if (isAlignedInputs) {
            ag_a2a_mode_kernel<
                ElementInput,
                ElementScale,
                kMaxPackSize,
                kBlocksPerRank,
                kCoRunRanks><<<grid_size, block_size, 0, stream>>>(params);
          } else {
            ag_a2a_mode_kernel<ElementInput, ElementScale, 1, kBlocksPerRank, kCoRunRanks>
                <<<grid_size, block_size, 0, stream>>>(params);
          }
        });
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for input_dtype=" << input_dtype
                          << ",scale_dtype=" << scale_dtype << "\n";
      });
}

template <typename T, bool kAligned>
CUTLASS_DEVICE void
copy_continous_kernel(T *__restrict__ dst_, const T *__restrict__ src_, int nelems) {
  using VecType = std::conditional_t<kAligned, uint4, T>;  // load as uint4
  nelems = nelems / (sizeof(VecType) / sizeof(T));
  VecType *dst = (VecType *)dst_;
  const VecType *src = (const VecType *)src_;
  CUTLASS_PRAGMA_UNROLL
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nelems; i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

template <typename T, typename ScaleType, bool kAligned>
CUTLASS_GLOBAL void
ag_ring1d_with_scale_kernel(AllGatherParams params) {
  // notify the ag kernel has been launched.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], 1;" : : "l"(params.ag_signal));
  }

  auto grid_group = cooperative_groups::this_grid();

  int rank_to = (params.rank + params.world_size - 1) % params.world_size;
  // int rank_from = (params.rank + 1) % params.world_size;
  int rank = params.rank;
  int m_per_rank = params.m;
  int elems_per_rank = m_per_rank * params.n;
  auto flags = [&](int rank_, int segment) -> int * {
    return params.ag_barriers[rank_] + segment;
  };
  auto data_ptr = [&](int rank_, int segment) -> T * {
    return (T *)params.input_ptrs[rank_] + segment * elems_per_rank;
  };
  auto wait_ready = [&](int *ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int flag = atomic_load_acquire_sys(ptr);
      while (flag != 1) {
        flag = atomic_load_sys(ptr);
      }
    }
    grid_group.sync();
  };
  auto set_ready = [&](int *ptr) {
    grid_group.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      atomic_store_release_sys(ptr, 1);
    }
  };
  // calculate
  for (int i = 0; i < params.world_size - 1; i++) {
    int segment_send = (rank + i) % params.world_size;
    if (i != 0) {
      wait_ready(flags(rank, segment_send));
    }
    copy_continous_kernel<T, kAligned>(
        data_ptr(rank_to, segment_send), data_ptr(rank, segment_send), elems_per_rank);
    if (params.has_scale) {
      ScaleType *dst_scale_ptr =
          reinterpret_cast<ScaleType *>(params.scale_ptrs[rank_to]) + m_per_rank * segment_send;
      ScaleType *src_scale_ptr =
          reinterpret_cast<ScaleType *>(params.scale_ptrs[rank]) + m_per_rank * segment_send;
      copy_continous_kernel<ScaleType, false>(dst_scale_ptr, src_scale_ptr, m_per_rank);
    }
    set_ready(flags(rank_to, segment_send));
  }
}

template <typename T, typename ScaleType, bool kAligned>
CUTLASS_GLOBAL void
ag_ring2d_with_scale_kernel(AllGatherParams params) {
  // notify the ag kernel has been launched.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.b32 [%0], 1;" : : "l"(params.ag_signal));
  }

  auto grid_group = cooperative_groups::this_grid();
  int numa_world_size = params.sub_world_size;

  int rank = params.rank;
  int nnode = rank / numa_world_size;
  int rank_to = (rank + params.world_size - 1) % params.world_size;
  int rank_to_sub = (rank - 1 + numa_world_size) % numa_world_size + nnode * numa_world_size;
  int m_per_rank = params.m;
  int elems_per_rank = m_per_rank * params.n;
  auto flags = [&](int rank_, int segment) -> int * {
    return params.ag_barriers[rank_] + segment;
  };
  auto data_ptr = [&](int rank_, int segment) -> T * {
    return (T *)params.input_ptrs[rank_] + segment * elems_per_rank;
  };
  auto wait_ready = [&](int *ptr) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      int flag = atomic_load_acquire_sys(ptr);
      while (flag != 1) {
        flag = atomic_load_sys(ptr);
      }
    }
    grid_group.sync();
  };
  auto set_ready = [&](int *ptr) {
    grid_group.sync();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      atomic_store_release_sys(ptr, 1);
    }
  };
  // calculate
  for (int i = 0; i < params.world_size - 1; i++) {
    int segment_send = (rank + i) % params.world_size;
    bool is_2d_step = i >= numa_world_size && rank % numa_world_size == 0;
    if (is_2d_step) {
      segment_send = (segment_send + numa_world_size) % params.world_size;
      rank_to = rank_to_sub;
    }
    // stage 0 is always ready.
    if (i != 0 && !is_2d_step) {
      wait_ready(flags(rank, segment_send));
    }
    copy_continous_kernel<T, kAligned>(
        data_ptr(rank_to, segment_send), data_ptr(rank, segment_send), elems_per_rank);
    if (params.has_scale) {
      ScaleType *dst_scale_ptr =
          reinterpret_cast<ScaleType *>(params.scale_ptrs[rank_to]) + m_per_rank * segment_send;
      ScaleType *src_scale_ptr =
          reinterpret_cast<ScaleType *>(params.scale_ptrs[rank]) + m_per_rank * segment_send;
      copy_continous_kernel<ScaleType, false>(dst_scale_ptr, src_scale_ptr, m_per_rank);
    }
    set_ready(flags(rank_to, segment_send));
  }
}

void
ag_ring_with_scale(
    const AllGatherParams &params,
    int input_elem_size,
    int scale_elem_size,
    int num_grids,
    bool use_2d_mode,
    cudaStream_t stream) {
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(cute::_1{}, cute::_2{}, cute::_4{}), cute::make_tuple(cute::_4{})),
      [&](auto tup) {
        auto [candidate_input_elem_size, candidate_scale_elem_size] = tup;
        return input_elem_size == candidate_input_elem_size &&
               scale_elem_size == candidate_scale_elem_size;
      },
      [&](auto tup) {
        auto [candidate_input_elem_size, candidate_scale_elem_size] = tup;
        auto _to_dtype = [](auto dtype) {
          if constexpr (std::is_same_v<decltype(dtype), cute::_1>) {
            return int8_t{};
          } else if constexpr (std::is_same_v<decltype(dtype), cute::_2>) {
            return int16_t{};
          } else if constexpr (std::is_same_v<decltype(dtype), cute::_4>) {
            return int32_t{};
          } else {
            static_assert(dtype != 1 && dtype != 2 && dtype != 4, "invalid dtype");
          }
        };
        using ElementInput = decltype(_to_dtype(candidate_input_elem_size));
        using ElementScale = decltype(_to_dtype(candidate_scale_elem_size));
        constexpr int32_t kMaxPackSize = kMaxPackBytes / sizeof(ElementInput);
        bool isAlignedInputs = true;
        for (size_t i = 0; i < params.world_size; ++i) {
          isAlignedInputs =
              isAlignedInputs && isAligned<ElementInput, kMaxPackSize>(params.input_ptrs[i]);
        }
        size_t num_input = (size_t)params.m * params.n;
        isAlignedInputs = isAlignedInputs && num_input % kMaxPackSize == 0;
        void *func =
            use_2d_mode
                ? (isAlignedInputs
                       ? (void *)ag_ring2d_with_scale_kernel<ElementInput, ElementScale, true>
                       : (void *)ag_ring2d_with_scale_kernel<ElementInput, ElementScale, false>)
                : (isAlignedInputs
                       ? (void *)ag_ring1d_with_scale_kernel<ElementInput, ElementScale, true>
                       : (void *)ag_ring1d_with_scale_kernel<ElementInput, ElementScale, false>);
        void *args[1] = {(void *)&params};
        CUDA_CHECK(
            cudaLaunchCooperativeKernel(func, dim3(num_grids), dim3(1024), &args[0], 0, stream));
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for input_elem_size=" << input_elem_size
                          << ",scale_elem_size=" << scale_elem_size << "\n";
      });
}

}  // namespace flux
}  // namespace bytedance
