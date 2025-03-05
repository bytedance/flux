//===- topk_gather_rs_v2.cu --------------------------------------- C++ ---===//
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

#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/cuda/reduce_utils.cuh"
#include "flux/cuda/system_barrier.hpp"
#include "flux/flux.h"
#include "moe_gather_rs/topk_gather_rs.hpp"
#include <cute/container/array_aligned.hpp>
#include <cute/numeric/int.hpp>
#include <type_traits>
#include <cutlass/arch/memory.h>
#include <cutlass/barrier.h>
#include <cutlass/numeric_conversion.h>
#include <sys/types.h>
#include <cute/underscore.hpp>
#include <cute/layout.hpp>
#include <cute/pointer.hpp>
#include <cute/tensor.hpp>

namespace bytedance::flux {
namespace {

constexpr int WorkerBarrierId = 1;
constexpr int FullBarrierId = 2;

CUTLASS_DEVICE int
to_tile_idx(int m_tile_idx, int n_tile_idx, int tiled_m, int tiled_n) {
  return m_tile_idx * tiled_n + n_tile_idx;
}

template <int BarrierId, int kNumThreads>
struct BarSync {
  CUTLASS_DEVICE
  static void
  sync() {
    sub_barrier_sync(BarrierId, kNumThreads);
  }
};

template <typename T>
constexpr static bool kIsFp16 = std::is_same_v<T, half> || std::is_same_v<T, cutlass::half_t>;
template <typename T>
constexpr static bool kIsBf16 =
    std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, cutlass::bfloat16_t>;

template <typename T>
using ToVecType =
    std::conditional_t<kIsFp16<T>, __half2, std::conditional_t<kIsBf16<T>, __nv_bfloat162, void>>;

template <typename T, typename PackType = uint4>
union Pack {
  using VecT = ToVecType<T>;
  static_assert(sizeof(PackType) >= sizeof(VecT));
  static_assert(sizeof(PackType) % sizeof(VecT) == 0);
  constexpr static int kElemsPerPack = sizeof(PackType) / sizeof(T);
  constexpr static int kVecElemsPerPack = sizeof(PackType) / sizeof(VecT);

  PackType data;
  T elems[kElemsPerPack];
  VecT elems_vec[kVecElemsPerPack];
};

/** Usage: add<__half>(uint4* src, uint4* dst);
 * template param:
 *   typename HalfType: __half/__nv_bfloat16
 *   typename T: actual type, such as uint4
 */
template <typename T, typename PackType>
CUTLASS_DEVICE PackType
addPack(const Pack<T, PackType> &lhs, const Pack<T, PackType> &rhs) {
  Pack<T> buffer;
#pragma unroll  // trust the compiler
  for (int i = 0; i < Pack<T>::kVecElemsPerPack; i++) {
    buffer.elems_vec[i] = __hadd2(lhs.elems_vec[i], rhs.elems_vec[i]);
  }
  return buffer.data;
}

CUTLASS_DEVICE void
storePack(void *ptr, uint4 data) {
  asm volatile(
      "{\n"
      "  st.global.v4.u32 [%0], {%1, %2, %3, %4};\n"
      "}\n"
      :
      : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
}

CUTLASS_DEVICE uint4
loadPack(void *ptr) {
  uint4 data;
  // .ca, .cg, .cs, .lu, .cv
  asm volatile(
      "{\n"
      "  mov.b32 %0, %5;\n"
      "  mov.b32 %1, %6;\n"
      "  mov.b32 %2, %7;\n"
      "  mov.b32 %3, %8;\n"
      "  ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      "}\n"
      : "=r"(data.x), "=r"(data.y), "=r"(data.z), "=r"(data.w)
      : "l"(ptr), "r"(data.x), "r"(data.y), "r"(data.z), "r"(data.w));
  return data;
}

CUTLASS_DEVICE void
atomic_store_relax_sys(int *ptr, int value) {
  atomic_ref_sys<int> ref(*ptr);
  ref.store(value, cuda::memory_order_relaxed);
}

template <
    typename T,
    const int kTiledM,
    const int kTiledN,
    const int kNumWorkerThreads,
    const bool kUseReadMode>
struct AllGatherOp {
  __device__ void operator()(
      TopKReduceGatherRSV2Arguments const &params, int blk_m, int blk_n, int32_t sid, int stage);
};

template <typename T, const int kTiledM, const int kTiledN, const int kNumWorkerThreads>
struct AllGatherOp<T, kTiledM, kTiledN, kNumWorkerThreads, false> {
  __device__ void
  operator()(
      TopKReduceGatherRSV2Arguments const &params, int blk_m, int blk_n, int32_t sid, int stage) {
    static_assert(
        std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "unsupported type");
    using VecT = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
    using PackT = Pack<T, uint4>;
    using WorkerBarSync = BarSync<WorkerBarrierId, kNumWorkerThreads>;
    using WorkerBarrier = cutlass::detail::GenericSystemBarrier<WorkerBarSync>;
    using FullBarSync = BarSync<FullBarrierId, kNumWorkerThreads + 32>;
    using FullBarrier = cutlass::detail::GenericSystemBarrier<FullBarSync>;
    constexpr int kElemsPerPack = sizeof(uint4) / sizeof(T);

    constexpr int kGroupSize = kTiledN / kElemsPerPack;  // each group is responsible for a row
    static_assert(kNumWorkerThreads % kGroupSize == 0);
    constexpr int kNumGroups = kNumWorkerThreads / kGroupSize;

    static_assert(kTiledN % (kGroupSize * kElemsPerPack) == 0);
    PackT pack;
    int wid = threadIdx.x / kGroupSize;
    int wtid = threadIdx.x % kGroupSize;
    int segment = (stage + params.rank) % params.world_size;
    int rank_to = (params.rank + params.world_size - 1) % params.world_size;  // ring to prev
    const int total_m_per_rank = params.m_full / params.world_size;
    // printf("total M :%d \n", totalM);
    const int ntokens_per_rank = total_m_per_rank / params.topk;
    const int N = params.n;
    const int n_split = N / params.n_split;
    const int row_start = blk_m * kTiledM + segment * ntokens_per_rank;
    const int row_end = min(row_start + kTiledM, (segment + 1) * ntokens_per_rank);
    const int col_g = wtid * kElemsPerPack  // inner group offset
                      + blk_n * kTiledN     // BLOCK offset
                      + sid * n_split;      // SPLIT_N offset

    int tiled_m_per_rank = (ntokens_per_rank + kTiledM - 1) / kTiledM;
    int tiled_m = tiled_m_per_rank * params.world_size;
    int tiled_n_per_split = (n_split + kTiledN - 1) / kTiledN;
    int tiled_n = tiled_n_per_split * params.n_split;
    int m_tile_idx = tiled_m_per_rank * segment + blk_m;
    int n_tile_idx = tiled_n_per_split * sid + blk_n;
    int tile_idx = to_tile_idx(m_tile_idx, n_tile_idx, tiled_m, tiled_n);
    bool is_worker = threadIdx.x < kNumWorkerThreads;
    if (is_worker) {
      if (stage != 0) {
        int *tile_barrier_ptr = params.tile_barrier_ptrs[params.rank];
        WorkerBarrier::wait_eq(tile_barrier_ptr, threadIdx.x, tile_idx, 2);
      }
      if (stage != params.world_size - 1) {  // wait only
        const T *src_ptr = (T *)(params.reduce_ptrs[params.rank]);
        T *dst_ptr = (T *)(params.reduce_ptrs[rank_to]);
        for (int64_t row_g = wid + row_start; row_g < row_end; row_g += kNumGroups) {
          int64_t offset = row_g * N + col_g;
          auto *pack_src = (PackT *)(src_ptr + offset);
          auto *pack_dst = (PackT *)(dst_ptr + offset);
          storePack(pack_dst, loadPack(pack_src));
        }
      }
      FullBarSync::sync();
    } else {
      int *tile_barrier_ptr = params.tile_barrier_ptrs[rank_to];
      int thread_idx = threadIdx.x - kNumWorkerThreads;
      FullBarSync::sync();
      if (thread_idx == 0) {
        atomic_store_release_sys(tile_barrier_ptr + tile_idx, 2);
      }
    }
  }
};

template <typename T, const int kTiledM, const int kTiledN, const int kNumWorkerThreads>
struct AllGatherOp<T, kTiledM, kTiledN, kNumWorkerThreads, true> {
  __device__ void
  operator()(
      TopKReduceGatherRSV2Arguments const &params, int blk_m, int blk_n, int32_t sid, int stage) {
    static_assert(
        std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "unsupported type");
    using VecT = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
    using PackT = Pack<T, uint4>;
    using WorkerBarSync = BarSync<WorkerBarrierId, kNumWorkerThreads>;
    using WorkerBarrier = cutlass::detail::GenericSystemBarrier<WorkerBarSync>;
    using FullBarSync = BarSync<FullBarrierId, kNumWorkerThreads + 32>;
    using FullBarrier = cutlass::detail::GenericSystemBarrier<FullBarSync>;
    constexpr int kElemsPerPack = sizeof(uint4) / sizeof(T);

    constexpr int kGroupSize = kTiledN / kElemsPerPack;  // each group is responsible for a row
    static_assert(kNumWorkerThreads % kGroupSize == 0);
    constexpr int kNumGroups = kNumWorkerThreads / kGroupSize;

    static_assert(kTiledN % (kGroupSize * kElemsPerPack) == 0);
    PackT pack;
    int wid = threadIdx.x / kGroupSize;
    int wtid = threadIdx.x % kGroupSize;
    int segment = (stage + params.rank + 1) % params.world_size;
    int rank_from = (params.rank + 1) % params.world_size;  // ring to next
    const int total_m_per_rank = params.m_full / params.world_size;
    const int ntokens_per_rank = total_m_per_rank / params.topk;
    const int N = params.n;
    const int n_split = N / params.n_split;
    const int row_start = blk_m * kTiledM + segment * ntokens_per_rank;
    const int row_end = min(row_start + kTiledM, (segment + 1) * ntokens_per_rank);
    const int col_g = wtid * kElemsPerPack  // inner group offset
                      + blk_n * kTiledN     // BLOCK offset
                      + sid * n_split;      // SPLIT_N offset

    int tiled_m_per_rank = (ntokens_per_rank + kTiledM - 1) / kTiledM;
    int tiled_m = tiled_m_per_rank * params.world_size;
    int tiled_n_per_split = (n_split + kTiledN - 1) / kTiledN;
    int tiled_n = tiled_n_per_split * params.n_split;
    int m_tile_idx = tiled_m_per_rank * segment + blk_m;
    int n_tile_idx = tiled_n_per_split * sid + blk_n;
    int tile_idx = to_tile_idx(m_tile_idx, n_tile_idx, tiled_m, tiled_n);
    bool is_worker = threadIdx.x < kNumWorkerThreads;
    if (is_worker) {
      int *tile_barrier_ptr = params.tile_barrier_ptrs[rank_from];
      WorkerBarrier::wait_eq(tile_barrier_ptr, threadIdx.x, tile_idx, stage != 0 ? 2 : 1);
      if (stage != params.world_size - 1) {  // wait only
        const T *src_ptr = (T *)(params.reduce_ptrs[rank_from]);
        T *dst_ptr = (T *)(params.reduce_ptrs[params.rank]);
        for (int64_t row_g = wid + row_start; row_g < row_end; row_g += kNumGroups) {
          int64_t offset = row_g * N + col_g;
          auto *pack_src = (PackT *)(src_ptr + offset);
          auto *pack_dst = (PackT *)(dst_ptr + offset);
          storePack(pack_dst, loadPack(pack_src));
        }
      }
      FullBarSync::sync();
    } else {
      int *tile_barrier_ptr = params.tile_barrier_ptrs[params.rank];
      int thread_idx = threadIdx.x - kNumWorkerThreads;
      FullBarSync::sync();
      if (thread_idx == 0) {
        atomic_store_release_dev(tile_barrier_ptr + tile_idx, 2);
      }
    }
  }
};

template <
    typename T,
    const int kTiledM,
    const int kTiledN,
    const int kTopk,
    const int kNumWeightGroups,
    const int kNumWorkerThreads,
    const bool kHasVecScale,
    const bool kUseReadMode>
struct TopkGatherRsOp {
  CUTLASS_DEVICE void operator()(
      TopKReduceGatherRSV2Arguments const &params,
      int32_t *smem_idx,
      const int blk_m,
      const int blk_n,
      const int sid,
      const int stage);
};

template <
    typename T,
    const int kTiledM,
    const int kTiledN,
    const int kTopk,
    const int kNumWeightGroups,
    const int kNumWorkerThreads,
    const bool kHasVecScale>
struct TopkGatherRsOp<
    T,
    kTiledM,
    kTiledN,
    kTopk,
    kNumWeightGroups,
    kNumWorkerThreads,
    kHasVecScale,
    false> {
  CUTLASS_DEVICE void
  operator()(
      TopKReduceGatherRSV2Arguments const &params,
      int32_t *smem_idx,
      const int blk_m,
      const int blk_n,
      const int sid,
      const int stage) {
    static_assert(
        std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "unsupported type");
    using VecT = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
    using PackT = Pack<T, uint4>;
    using WorkerBarSync = BarSync<WorkerBarrierId, kNumWorkerThreads>;
    using WorkerBarrier = cutlass::detail::GenericSystemBarrier<WorkerBarSync>;
    using FullBarSync = BarSync<FullBarrierId, kNumWorkerThreads + 32>;
    using FullBarrier = cutlass::detail::GenericSystemBarrier<FullBarSync>;
    constexpr int kElemsPerPack = sizeof(uint4) / sizeof(T);
    constexpr int kElemsVecPerPack = kElemsPerPack / sizeof(T);

    constexpr int kGroupSize = kTiledN / kElemsPerPack;  // each group is responsible for a row
    static_assert(kNumWorkerThreads % kGroupSize == 0);
    constexpr int kNumGroups = kNumWorkerThreads / kGroupSize;

    static_assert(kTiledN % (kGroupSize * kElemsPerPack) == 0);

    PackT pack;

    float acc[kElemsPerPack];
    int wid = threadIdx.x / kGroupSize;
    int wtid = threadIdx.x % kGroupSize;

    // load the routing_idx first to the shared memory
    constexpr int IDX_LOAD_UNROLL =
        (kTiledM * kTopk + kNumWorkerThreads - 1) / kNumWorkerThreads;  // 1 for most cases

    int segment = (stage + params.rank + 1) % params.world_size;
    int rank_to = (params.rank + params.world_size - 1) % params.world_size;  // ring to prev
    const int total_m_per_rank = params.m_full / params.world_size;
    const int ntokens_per_rank = total_m_per_rank / params.topk;
    const int64_t N = params.n;
    const int64_t N_split = N / params.n_split;
    const int row_start = blk_m * kTiledM + segment * ntokens_per_rank;
    const int row_end = min(row_start + kTiledM, (segment + 1) * ntokens_per_rank);
    const int routing_idx_start = row_start * kTopk;
    const int routing_idx_end =
        min(routing_idx_start + kTiledM * kTopk, (segment + 1) * total_m_per_rank);

    const int64_t col_g = wtid * kElemsPerPack  // inner group offset
                          + blk_n * kTiledN     // BLOCK offset
                          + sid * N_split;      // SPLIT_N offset

    int tiled_m_per_rank = (ntokens_per_rank + kTiledM - 1) / kTiledM;
    int tiled_m = tiled_m_per_rank * params.world_size;
    int tiled_n_per_split = (N_split + kTiledN - 1) / kTiledN;
    int tiled_n = tiled_n_per_split * params.n_split;
    int m_tile_idx = tiled_m_per_rank * segment + blk_m;
    int n_tile_idx = tiled_n_per_split * sid + blk_n;
    int tile_idx = to_tile_idx(m_tile_idx, n_tile_idx, tiled_m, tiled_n);
    bool is_worker = threadIdx.x < kNumWorkerThreads;

    if (is_worker) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0, row = threadIdx.x; iter < IDX_LOAD_UNROLL;
           iter++, row += kNumWorkerThreads) {
        if (routing_idx_start + row < routing_idx_end) {
          smem_idx[row] = params.routing_idx[routing_idx_start + row];
        }
      }
      WorkerBarSync::sync();

      if (stage != 0) {
        int *tile_barrier_ptr = params.tile_barrier_ptrs[params.rank];
        // wait for tile ready
        WorkerBarrier::wait_eq(tile_barrier_ptr, threadIdx.x, tile_idx, 1);
      }

      // _l for local, for inner block row index. _g for global, for total input row index
      for (int row_l = wid, row_g = wid + row_start; row_g < row_end;
           row_l += kNumGroups, row_g += kNumGroups) {
        for (int i = 0; i < kElemsPerPack; i++) {
          acc[i] = 0.f;
        }
        // for more input_groups
        for (int topk = 0; topk < kTopk; topk++) {
          int64_t row_g_from = smem_idx[topk + row_l * kTopk];
          for (int j = 0; j < kNumWeightGroups; j++) {
            pack.data =
                loadPack((T *)params.input_ptrs[j] + row_g_from * N + col_g);  // load with uint4
            if constexpr (kHasVecScale) {
              float output_vec_scale = params.output_vec_scale_ptrs[j][row_g_from];
              for (int i = 0; i < kElemsPerPack; i++) {
                acc[i] += element_to_float(pack.elems[i]) * output_vec_scale;
              }
            } else {
              for (int i = 0; i < kElemsPerPack; i++) {
                acc[i] += element_to_float(pack.elems[i]);
              }
            }
          }
        }
        for (int i = 0; i < kElemsVecPerPack; i++) {
          pack.elems_vec[i] = floats_to_element<VecT>(acc[i * 2], acc[i * 2 + 1]);
        }
        bool last_round = stage == params.world_size - 1;
        int64_t row_off = ((int64_t)row_g) * N + col_g;
        int64_t row_out_off = last_round ? (row_off - segment * ntokens_per_rank * N) : row_off;
        void *output_ptr =
            (T *)(last_round ? params.output_ptr : params.reduce_ptrs[rank_to]) + row_out_off;
        if (stage == 0) {                    // copy only
          storePack(output_ptr, pack.data);  // copy to output (last round)
        } else {                             // reduce
          PackT pack_lr;
          pack_lr.data = loadPack((T *)(params.reduce_ptrs[params.rank]) + row_off);
          storePack(output_ptr, addPack(pack_lr, pack));
        }
      }
      FullBarSync::sync();
    } else {
      int *tile_barrier_ptr = params.tile_barrier_ptrs[rank_to];
      FullBarSync::sync();
      int thread_idx = threadIdx.x - kNumWorkerThreads;
      if (thread_idx == 0) {
        atomic_store_release_sys(tile_barrier_ptr + tile_idx, 1);
      }
    }
  }
};

template <
    typename T,
    const int kTiledM,
    const int kTiledN,
    const int kTopk,
    const int kNumWeightGroups,
    const int kNumWorkerThreads,
    const bool kHasVecScale>
struct TopkGatherRsOp<
    T,
    kTiledM,
    kTiledN,
    kTopk,
    kNumWeightGroups,
    kNumWorkerThreads,
    kHasVecScale,
    true> {
  CUTLASS_DEVICE void
  operator()(
      TopKReduceGatherRSV2Arguments const &params,
      int32_t *smem_idx,
      const int blk_m,
      const int blk_n,
      const int sid,
      const int stage) {
    static_assert(
        std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "unsupported type");
    using VecT = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
    using PackT = Pack<T, uint4>;
    using WorkerBarSync = BarSync<WorkerBarrierId, kNumWorkerThreads>;
    using WorkerBarrier = cutlass::detail::GenericSystemBarrier<WorkerBarSync>;
    using FullBarSync = BarSync<FullBarrierId, kNumWorkerThreads + 32>;
    using FullBarrier = cutlass::detail::GenericSystemBarrier<FullBarSync>;
    constexpr int kElemsPerPack = sizeof(uint4) / sizeof(T);
    constexpr int kElemsVecPerPack = kElemsPerPack / sizeof(T);

    constexpr int kGroupSize = kTiledN / kElemsPerPack;  // each group is responsible for a row
    static_assert(kNumWorkerThreads % kGroupSize == 0);
    constexpr int kNumGroups = kNumWorkerThreads / kGroupSize;

    static_assert(kTiledN % (kGroupSize * kElemsPerPack) == 0);

    PackT pack;

    float acc[kElemsPerPack];
    int wid = threadIdx.x / kGroupSize;
    int wtid = threadIdx.x % kGroupSize;

    // load the routing_idx first to the shared memory
    constexpr int IDX_LOAD_UNROLL =
        (kTiledM * kTopk + kNumWorkerThreads - 1) / kNumWorkerThreads;  // 1 for most cases

    int segment = (stage + params.rank + 1) % params.world_size;
    int rank_from = (params.rank + 1) % params.world_size;  // ring to prev
    const int total_m_per_rank = params.m_full / params.world_size;
    const int ntokens_per_rank = total_m_per_rank / params.topk;
    const int64_t N = params.n;
    const int64_t N_split = N / params.n_split;
    const int row_start = blk_m * kTiledM + segment * ntokens_per_rank;
    const int row_end = min(row_start + kTiledM, (segment + 1) * ntokens_per_rank);
    const int routing_idx_start = row_start * kTopk;
    const int routing_idx_end =
        min(routing_idx_start + kTiledM * kTopk, (segment + 1) * total_m_per_rank);

    const int64_t col_g = wtid * kElemsPerPack  // inner group offset
                          + blk_n * kTiledN     // BLOCK offset
                          + sid * N_split;      // SPLIT_N offset

    int tiled_m_per_rank = (ntokens_per_rank + kTiledM - 1) / kTiledM;
    int tiled_m = tiled_m_per_rank * params.world_size;
    int tiled_n_per_split = (N_split + kTiledN - 1) / kTiledN;
    int tiled_n = tiled_n_per_split * params.n_split;
    int m_tile_idx = tiled_m_per_rank * segment + blk_m;
    int n_tile_idx = tiled_n_per_split * sid + blk_n;
    int tile_idx = to_tile_idx(m_tile_idx, n_tile_idx, tiled_m, tiled_n);
    bool is_worker = threadIdx.x < kNumWorkerThreads;

    if (is_worker) {
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0, row = threadIdx.x; iter < IDX_LOAD_UNROLL;
           iter++, row += kNumWorkerThreads) {
        if (routing_idx_start + row < routing_idx_end) {
          smem_idx[row] = params.routing_idx[routing_idx_start + row];
        }
      }
      WorkerBarSync::sync();

      if (stage != 0) {
        int *tile_barrier_ptr = params.tile_barrier_ptrs[rank_from];
        // wait for tile ready
        WorkerBarrier::wait_eq(tile_barrier_ptr, threadIdx.x, tile_idx, 1);
      }

      // _l for local, for inner block row index. _g for global, for total input row index
      for (int row_l = wid, row_g = wid + row_start; row_g < row_end;
           row_l += kNumGroups, row_g += kNumGroups) {
        for (int i = 0; i < kElemsPerPack; i++) {
          acc[i] = 0.f;
        }
        // for more input_groups
        for (int topk = 0; topk < kTopk; topk++) {
          int64_t row_g_from = smem_idx[topk + row_l * kTopk];
          for (int j = 0; j < kNumWeightGroups; j++) {
            pack.data =
                loadPack((T *)params.input_ptrs[j] + row_g_from * N + col_g);  // load with uint4
            if constexpr (kHasVecScale) {
              float output_vec_scale = params.output_vec_scale_ptrs[j][row_g_from];
              for (int i = 0; i < kElemsPerPack; i++) {
                acc[i] += element_to_float(pack.elems[i]) * output_vec_scale;
              }
            } else {
              for (int i = 0; i < kElemsPerPack; i++) {
                acc[i] += element_to_float(pack.elems[i]);
              }
            }
          }
        }
        for (int i = 0; i < kElemsVecPerPack; i++) {
          pack.elems_vec[i] = floats_to_element<VecT>(acc[i * 2], acc[i * 2 + 1]);
        }
        bool last_round = stage == params.world_size - 1;
        int64_t row_off = ((int64_t)row_g) * N + col_g;
        int64_t row_out_off = last_round ? (row_off - segment * ntokens_per_rank * N) : row_off;
        void *output_ptr =
            (T *)(last_round ? params.output_ptr : params.reduce_ptrs[params.rank]) + row_out_off;

        if (stage == 0) {                    // copy only
          storePack(output_ptr, pack.data);  // copy to output (last round)
        } else {                             // reduce
          PackT pack_r;
          pack_r.data = loadPack((T *)(params.reduce_ptrs[rank_from]) + row_off);
          storePack(output_ptr, addPack(pack_r, pack));
        }
      }
      FullBarSync::sync();
    } else {
      int *tile_barrier_ptr = params.tile_barrier_ptrs[params.rank];
      FullBarSync::sync();
      int thread_idx = threadIdx.x - kNumWorkerThreads;
      if (thread_idx == 0) {
        atomic_store_release_dev(tile_barrier_ptr + tile_idx, 1);
      }
    }
  }
};

template <
    typename T,
    const int kTopk,
    const int kNumWeightGroups,
    const int kTiledM,
    const int kTiledN,
    const int kNumWorkerThreads,
    const bool kHasVecScale,
    const bool kUseReadMode>
__global__
__launch_bounds__(kNumWorkerThreads + 32, 1) void topk_gather_rs_v2_kernel(
    TopKReduceGatherRSV2Arguments params) {
  using Barrier = cutlass::Barrier;
  // perform the reduction with float
  __shared__ int smem_buf[kTiledM * kTopk];
  const int ntokens = params.m_full / kTopk;
  const int n_per_split = params.n / params.n_split;
  if (params.do_all_reduce) {
    params.output_ptr = (void *)((T *)params.reduce_ptrs[params.rank] +
                                 ntokens / params.world_size * params.rank * params.n);
  }
  int m_tiles_per_rank = (ntokens / params.world_size + kTiledM - 1) / kTiledM;
  int n_tiles_per_split = n_per_split / kTiledN;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < params.n_split; sid++) {
    Barrier::wait_eq(params.barrier[params.rank], threadIdx.x, sid, 1);
    if (kUseReadMode) {
      int rank_to = (params.rank + params.world_size - 1) % params.world_size;  // ring to prev
      Barrier::wait_eq(params.barrier[rank_to], threadIdx.x, sid, 1);
    }
    for (int stage = 0; stage < params.world_size; stage++) {
      for (int blk_id = blockIdx.x; blk_id < m_tiles_per_rank * n_tiles_per_split;
           blk_id += gridDim.x) {
        int blk_m = blk_id / n_tiles_per_split;
        int blk_n = blk_id % n_tiles_per_split;
        TopkGatherRsOp<
            T,
            kTiledM,
            kTiledN,
            kTopk,
            kNumWeightGroups,
            kNumWorkerThreads,
            kHasVecScale,
            kUseReadMode>{}(params, &smem_buf[0], blk_m, blk_n, sid, stage);
      }
    }
    if (params.do_all_reduce) {
      for (int stage = 0; stage < params.world_size; stage++) {  // all gather stages
        for (int blk_id = blockIdx.x; blk_id < m_tiles_per_rank * n_tiles_per_split;
             blk_id += gridDim.x) {
          int blk_m = blk_id / n_tiles_per_split;
          int blk_n = blk_id % n_tiles_per_split;
          AllGatherOp<T, kTiledM, kTiledN, kNumWorkerThreads, kUseReadMode>{}(
              params, blk_m, blk_n, sid, stage);
        }
      }
    }
  }
}
}  // namespace

void
topk_gather_rs_v2(
    TopKReduceGatherRSV2Arguments const &args, DataTypeEnum dtype, cudaStream_t stream) {
  constexpr int kNumWorkerThreads = 768;
  constexpr int kNumSyncThreads = 32;
  constexpr int kNumThreads = kNumSyncThreads + kNumWorkerThreads;
  dim3 grid_dim(args.threadblock_count, 1, 1);
  dim3 block_dim(kNumThreads);

  constexpr int kTiledM = 128;
  constexpr int kTiledN = 1024;
  FLUX_CHECK_DIV(args.n / args.n_split, kTiledN);
  FLUX_CHECK_EQ(args.tile_size_m, kTiledM);
  FLUX_CHECK_EQ(args.tile_size_n, kTiledN);
  bool has_vec_scale = args.output_vec_scale_ptrs[0] != nullptr;
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(
              cute::_1{},
              cute::_2{},
              cute::_3{},
              cute::_4{},
              cute::_5{},
              cute::_6{},
              cute::_8{},
              cute::_10{}),
          cute::make_tuple(cute::_1{}, cute::_2{}),
          cute::make_tuple(cute::true_type{}, cute::false_type{}),
          cute::make_tuple(cute::true_type{}, cute::false_type{})),
      [&](auto tup) {
        auto [cdtype, ctopk, input_groups_, has_vec_scale_, use_read_mode_] = tup;
        return cdtype == dtype && ctopk == args.topk && has_vec_scale_ == has_vec_scale &&
               input_groups_ == args.input_groups && use_read_mode_ == args.use_read_mode;
      },
      [&](auto tup) {
        auto [dtype_, topk_, input_groups_, has_vec_scale_, use_read_mode_] = tup;
        constexpr int kTopk = decltype(topk_){};
        constexpr int kInputWeightGroups = decltype(input_groups_){};
        constexpr bool kHasVecScale = decltype(has_vec_scale_){};
        constexpr bool kUseReadMode = decltype(use_read_mode_){};
        using T = decltype(to_cuda_dtype(dtype_));
        topk_gather_rs_v2_kernel<
            T,
            kTopk,
            kInputWeightGroups,
            kTiledM,
            kTiledN,
            kNumWorkerThreads,
            kHasVecScale,
            kUseReadMode><<<grid_dim, block_dim, 0, stream>>>(args);
      },
      [&]() {
        FLUX_CHECK(false) << "unsupported for topk=" << args.topk << " dtype:" << dtype
                          << " input groups: " << args.input_groups;
      });
}

namespace {

template <
    typename T,
    const int kTiledM,
    const int kTiledN,
    const int kTopk,
    const int kNumWorkerThreads,
    const bool kHasVecScale>
__device__ void
ep_gather_rs_impl_v2(
    TopKReduceGatherRSV2Arguments const &params,
    int32_t *smem_idx,
    int blk_m,
    int blk_n,
    int sid,
    int stage,
    int ep_m_start,
    int ep_m_end) {
  static_assert(std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>, "unsupported type");
  using VecT = std::conditional_t<std::is_same_v<T, __half>, __half2, __nv_bfloat162>;
  using PackT = Pack<T, uint4>;
  using WorkerBarSync = BarSync<WorkerBarrierId, kNumWorkerThreads>;
  using WorkerBarrier = cutlass::detail::GenericSystemBarrier<WorkerBarSync>;
  using FullBarSync = BarSync<FullBarrierId, kNumWorkerThreads + 32>;
  using FullBarrier = cutlass::detail::GenericSystemBarrier<FullBarSync>;
  constexpr int kElemsPerPack = sizeof(uint4) / sizeof(T);
  constexpr int kElemsVecPerPack = kElemsPerPack / sizeof(T);

  constexpr int kGroupSize = kTiledN / kElemsPerPack;  // each group is responsible for a row
  static_assert(kNumWorkerThreads % kGroupSize == 0);
  constexpr int kNumGroups = kNumWorkerThreads / kGroupSize;

  static_assert(kTiledN % (kGroupSize * kElemsPerPack) == 0);

  PackT pack;
  float acc[kElemsPerPack];
  int wid = threadIdx.x / kGroupSize;
  int wtid = threadIdx.x % kGroupSize;

  constexpr int TOKEN_IDX_N = kTiledM * kTopk;
  constexpr int IDX_LOAD_UNROLL = (TOKEN_IDX_N + kNumWorkerThreads - 1) / kNumWorkerThreads;

  int segment = (stage + params.rank + 1) % params.world_size;
  int rank_to = (params.rank + params.world_size - 1) % params.world_size;  // ring to prev

  const int total_m_per_rank = params.m_full / params.world_size;
  // printf("total M :%d \n", totalM);
  const int ntokens_per_rank = total_m_per_rank / params.topk;

  const int N = params.n;
  const int n_split = N / params.n_split;
  const int row_start = blk_m * kTiledM + segment * ntokens_per_rank;
  const int row_end = min(row_start + kTiledM, (segment + 1) * ntokens_per_rank);
  const int routing_idx_start = row_start * kTopk;
  const int routing_idx_end =
      min(routing_idx_start + kTiledM * kTopk, (segment + 1) * total_m_per_rank);

  const int col_g = wtid * kElemsPerPack  // inner group offset
                    + blk_n * kTiledN     // BLOCK offset
                    + sid * n_split;      // SPLIT_N offset

  int tiled_m_per_rank = (ntokens_per_rank + kTiledM - 1) / kTiledM;
  int tiled_m = tiled_m_per_rank * params.world_size;
  int tiled_n_per_split = (n_split + kTiledN - 1) / kTiledN;
  int tiled_n = tiled_n_per_split * params.n_split;
  int m_tile_idx = tiled_m_per_rank * segment + blk_m;
  int n_tile_idx = tiled_n_per_split * sid + blk_n;
  int tile_idx = to_tile_idx(m_tile_idx, n_tile_idx, tiled_m, tiled_n);
  bool is_worker = threadIdx.x < kNumWorkerThreads;

  if (is_worker) {
    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0, row_l = threadIdx.x; iter < IDX_LOAD_UNROLL;
         iter++, row_l += kNumWorkerThreads) {
      if (routing_idx_start + row_l < routing_idx_end) {
        smem_idx[row_l] = params.routing_idx[routing_idx_start + row_l];
      }
    }
    WorkerBarSync::sync();

    if (stage != 0) {
      int *tile_barrier_ptr = params.tile_barrier_ptrs[params.rank];
      WorkerBarrier::wait_eq(tile_barrier_ptr, threadIdx.x, tile_idx, 1);
      // wait for tile ready
    }

    for (size_t row_l = wid, row_g = wid + row_start; row_g < row_end;
         row_l += kNumGroups, row_g += kNumGroups) {
      for (int i = 0; i < kElemsPerPack; i++) {
        acc[i] = 0.f;
      }

      for (int topk = 0; topk < kTopk; topk++) {
        int64_t row_g_from = smem_idx[topk + row_l * kTopk];
        if (row_g_from >= ep_m_start && row_g_from < ep_m_end) {
          row_g_from -= ep_m_start;
          for (int j = 0; j < params.input_groups; j++) {
            PackT &pack_D = *(PackT *)((T *)params.input_ptrs[j] + row_g_from * N + col_g);
            pack.data = pack_D.data;
            if constexpr (kHasVecScale) {
              float output_vec_scale = params.output_vec_scale_ptrs[j][row_g_from];
              for (int i = 0; i < kElemsPerPack; i++) {
                acc[i] += element_to_float(pack.elems[i]) * output_vec_scale;
              }
            } else {
              for (int i = 0; i < kElemsPerPack; i++) {
                acc[i] += element_to_float(pack.elems[i]);
              }
            }
          }
        }
      }
      for (int i = 0; i < kElemsVecPerPack; i++) {
        pack.elems_vec[i] = floats_to_element<VecT>(acc[i * 2], acc[i * 2 + 1]);
      }

      bool last_round = stage == params.world_size - 1;
      int64_t row_off = ((int64_t)row_g) * N + col_g;
      int64_t row_off_out = last_round ? (row_off - segment * ntokens_per_rank * N) : row_off;
      void *output_ptr =
          (T *)(last_round ? params.output_ptr : params.reduce_ptrs[rank_to]) + row_off_out;
      auto *pack_lr_ptr = (PackT *)((T *)(params.reduce_ptrs[params.rank]) + row_off);
      PackT pack_lr;
      pack_lr.data = pack_lr_ptr->data;

      if (stage == 0) {  // copy only
        storePack(output_ptr, pack.data);
      } else {  // reduce
        storePack(output_ptr, addPack(pack_lr, pack));
      }
    }
    FullBarSync::sync();
  } else {
    int *tile_barrier_ptr = params.tile_barrier_ptrs[rank_to];
    int thread_idx = threadIdx.x - kNumWorkerThreads;
    FullBarSync::sync();
    if (thread_idx == 0) {
      atomic_store_release_sys(tile_barrier_ptr + tile_idx, 1);
    }
  }
}

template <
    typename T,
    const int kTopk,
    const int kTiledM,
    const int kTiledN,
    const int kNumWorkerThreads,
    const bool kHasVecScale>
__global__
__launch_bounds__(kNumWorkerThreads + 32, 1) void ep_topk_gather_rs_kernel_v2(
    TopKReduceGatherRSV2Arguments params, int32_t ep_start, int32_t ep_nexperts) {
  extern __shared__ char shared_storage[];
  using Barrier = cutlass::Barrier;

  const int ntokens = params.m_full / kTopk;
  const int ntokens_per_rank = ntokens / params.world_size;
  const int n_per_split = params.n / params.n_split;
  int *splits_acc = (int *)shared_storage;
  block_prefix_sum_and_sync(params.splits, splits_acc, params.nexperts);
  int ep_m_start = ep_start == 0 ? 0 : splits_acc[ep_start - 1];
  int ep_m_end = splits_acc[ep_start + ep_nexperts - 1];
  __syncthreads();

  if (params.do_all_reduce) {
    params.output_ptr =
        (void *)((T *)params.reduce_ptrs[params.rank] + ntokens_per_rank * params.rank * params.n);
  }

  int m_tiles_per_rank = (ntokens_per_rank + kTiledM - 1) / kTiledM;
  int n_tiles_per_split = n_per_split / kTiledN;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int sid = 0; sid < params.n_split; sid++) {
    Barrier::wait_eq(params.barrier[params.rank], threadIdx.x, sid, 1);
    for (int stage = 0; stage < params.world_size; stage++) {  // reduce_scatter stages
      for (int blk_id = blockIdx.x; blk_id < m_tiles_per_rank * n_tiles_per_split;
           blk_id += gridDim.x) {
        int blk_m = blk_id / n_tiles_per_split;
        int blk_n = blk_id % n_tiles_per_split;
        ep_gather_rs_impl_v2<T, kTiledM, kTiledN, kTopk, kNumWorkerThreads, kHasVecScale>(
            params, (int32_t *)shared_storage, blk_m, blk_n, sid, stage, ep_m_start, ep_m_end);
      }
    }

    if (params.do_all_reduce) {
      for (int stage = 0; stage < params.world_size; stage++) {  // all gather stages
        for (int blk_id = blockIdx.x; blk_id < m_tiles_per_rank * n_tiles_per_split;
             blk_id += gridDim.x) {
          int blk_m = blk_id / n_tiles_per_split;
          int blk_n = blk_id % n_tiles_per_split;
          AllGatherOp<T, kTiledM, kTiledN, kNumWorkerThreads, false>{}(
              params, blk_m, blk_n, sid, stage);
        }
      }
    }
  }
}
}  // namespace
void
ep_topk_gather_rs_v2(
    TopKReduceGatherRSV2Arguments const &args,
    DataTypeEnum dtype,
    int32_t ep_start,
    int32_t ep_nexperts,
    cudaStream_t stream) {
  constexpr int kNumWorkerThreads = 768;
  constexpr int kNumSyncThreads = 32;
  constexpr int kNumThreads = kNumSyncThreads + kNumWorkerThreads;

  dim3 grid_dim(args.threadblock_count, 1, 1);
  dim3 block_dim(kNumThreads);

  constexpr int kTiledM = 128;
  constexpr int kTiledN = 1024;
  FLUX_CHECK_DIV(args.n / args.n_split, kTiledN);
  FLUX_CHECK_EQ(args.tile_size_m, kTiledM);
  FLUX_CHECK_EQ(args.tile_size_n, kTiledN);
  bool has_vec_scale = args.output_vec_scale_ptrs[0] != nullptr;
  tuple_return_if(
      tuple_cartesian_product(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(
              cute::_1{},
              cute::_2{},
              cute::_3{},
              cute::_4{},
              cute::_5{},
              cute::_6{},
              cute::_8{},
              cute::_10{}),  // topk
          cute::make_tuple(cute::true_type{}, cute::false_type{})),
      [&](auto tup) {
        auto [dtype_, topk_, has_vec_scale_] = tup;
        return dtype_ == dtype && topk_ == args.topk && has_vec_scale_ == has_vec_scale;
      },
      [&](auto tup) {
        auto [dtype_, topk_, has_vec_scale_] = tup;
        constexpr int kTopk = decltype(topk_){};
        constexpr bool kHasVecScale = decltype(has_vec_scale_){};
        using T = decltype(to_cuda_dtype(dtype_));
        int shared_mem_size = std::max(sizeof(int) * kTopk * kTiledM, sizeof(int) * args.nexperts);
        ep_topk_gather_rs_kernel_v2<T, kTopk, kTiledM, kTiledN, kNumWorkerThreads, kHasVecScale>
            <<<grid_dim, block_dim, shared_mem_size, stream>>>(args, ep_start, ep_nexperts);
      },
      [&]() { FLUX_CHECK(false) << "unsupported for topk=" << args.topk << " dtype:" << dtype; });
}

}  // namespace bytedance::flux
