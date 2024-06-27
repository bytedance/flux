//===- reduce_scatter_kernel.hpp ---------------------------------- C++ ---===//
//
// Copyright 2023 ByteDance Ltd. and/or its affiliates. All rights reserved.
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

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <cuda_runtime_api.h>
#include <cuda/std/atomic>
#include <utility>
#include "cutlass/detail/helper_macros.hpp"
#ifdef FLUX_SHM_USE_NVSHMEM
#include "host/nvshmem_api.h"
#include "host/nvshmemx_api.h"
#endif
#include "tile_scheduler/threadblock_swizzle_segment_util.hpp"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include "cute/tensor.hpp"
#include "cute/layout.hpp"
#include "cute/underscore.hpp"
#include "cutlass/fast_math.h"
#include "cutlass/barrier.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/cuda_stub.h"
#include "flux/cuda/memory_utils.hpp"
#include "flux/flux.h"
#include "flux/utils.h"
#include "flux/cuda/cuda_common_device.hpp"
#include "reduce_scatter/reduce_scatter_topos.hpp"
#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"
#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
#include <nccl.h>
#endif

// #define FLUX_DEBUG_RS

#define NextRank(rank_) (((rank_) + 1) % kLocalWorldSize)
#define NextLocalRank(rank_, node_) ((((rank_) + 1) % kNumaWorldSize + (node_) * kNumaWorldSize))
#define PrevLocalRank(rank_, node_) \
  ((((rank_) - 1 + kNumaWorldSize) % kNumaWorldSize + (node_) * kNumaWorldSize))
#define NextNodeRank(rank_) (((rank_) + kNumaWorldSize) % kLocalWorldSize)

namespace bytedance::flux {
namespace {

constexpr int kNumaWorldSize = 4;
constexpr int kNumaNodes = 2;

using VecType = uint4;
struct ReduceScatterParams {
  int rank;
  int world_size;
  int nnodes;
  int m;
  int n;
  int num_blocks;
  int sleep_ns;
#ifdef FLUX_DEBUG_RS
  bool run_local = true;
  bool run_remote = true;
  bool do_copy = true;
  bool do_wait = true;
#endif
  bool use_barrier_queue;
  bool use_gemmk;
  bool per_tile_flags;
  bool use_cudaMemcpyAsync;
  int n_split;
  int sub_world_size;
  void *opaque;
  bool use_1d_ring;
  bool use_p2p_read;
  void *args_workspace;
  void *scatter_ptr_aux[kMaxWorldSize];
  void *barrier_ptr[kMaxWorldSize];
  void *reduce_ptr[kMaxWorldSize];
};

std::ostream &
operator<<(std::ostream &os, const ReduceScatterParams &param) {
  os << "ReduceScatterParams rank:" << param.rank << " world_size:" << param.world_size;
  os << " m:" << param.m << " n:" << param.n;
  os << " num_blocks:" << param.num_blocks;
  os << " sleep_n: " << param.sleep_ns;
  os << " use_barrier_queue:" << param.use_barrier_queue;
  os << " use_gemmk:" << param.use_gemmk;
  os << " per_tile_flags: " << param.per_tile_flags;
  os << " use_cudaMemcpyAsync: " << param.use_cudaMemcpyAsync;
  os << " n_split: " << param.n_split;
  os << " sub_world_size: " << param.sub_world_size;
  os << " use_1d_ring: " << param.use_1d_ring;
  os << " use_p2p_read: " << param.use_p2p_read;
  os << " args_workspace: " << param.args_workspace;
  return os;
}

//// make ld_acquire and red_release public ////
// steal protected methods ld_acquire and red_release
template <template <typename T> class Barrier, class Sync = cutlass::detail::SyncthreadsSync>
struct GenericBarrierWithFallback : public Barrier<Sync> {
  using Base = Barrier<Sync>;
  CUTLASS_DEVICE static void
  wait_eq(void *lock_ptr, int thread_idx, int flag_idx, int val) {
    int *flag_ptr = static_cast<int *>(lock_ptr) + flag_idx;
    // clang-format off
    if (thread_idx == 0) {
      unsigned cnt = 1 << 6;
      #pragma unroll 1
      while (Base::ld_acquire(flag_ptr) != val) {
        nanosleep(cnt);
        cnt *= 2;
        cnt = min(cnt, 2 << 16); // in case cnt overflow or too long wait
      }
    }
    // clang-format on
    Sync::sync();
  }
};

using BarrierWithFallback = GenericBarrierWithFallback<cutlass::GenericBarrier>;
using SystemBarrierWithFallback =
    GenericBarrierWithFallback<cutlass::detail::GenericSystemBarrier>;

CUTLASS_DEVICE static void
wait_eq_sys(void *lock_ptr, int thread_idx, int flag_idx, int val = 1) {
  SystemBarrierWithFallback::wait_eq(lock_ptr, thread_idx, flag_idx, val);
}

template <typename T>
constexpr static bool kIsFp16 = std::is_same_v<T, half> || std::is_same_v<T, cutlass::half_t>;
template <typename T>
constexpr static bool kIsBf16 =
    std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, cutlass::bfloat16_t>;

template <typename T>
using ToVecType =
    std::conditional_t<kIsFp16<T>, __half2, std::conditional_t<kIsBf16<T>, __nv_bfloat162, void>>;

template <typename T, typename VecT = ToVecType<T>>
class VecAdd {
 public:
  CUTLASS_DEVICE
  VecT
  operator()(const VecT &lhs, const VecT &rhs) {
    static_assert(kIsBf16<T> || kIsFp16<T>, "only FP16 and BF16 is supported");
    return __hadd2(rhs, lhs);
  }
};

/** Usage: add<__half>(uint4* src, uint4* dst);
 * template param:
 *   typename HalfType: __half/__nv_bfloat16
 *   typename T: actual type, such as uint4
 */
template <typename HalfType, typename T>
CUTLASS_DEVICE T
add(const T *__restrict__ lhs, const T *__restrict__ rhs) {
  using VecT = ToVecType<HalfType>;
  static_assert(sizeof(T) >= sizeof(VecT));
  static_assert(sizeof(T) % sizeof(VecT) == 0);
  constexpr int kVecSize = sizeof(T) / sizeof(VecT);
  static_assert(kVecSize == 1 || kVecSize == 2 || kVecSize == 4);
  union {
    VecT values[kVecSize];
    T value;
  } lhs_packed, rhs_packed, res_packed;
  lhs_packed.value = *rhs;
  rhs_packed.value = *lhs;
  VecAdd<HalfType> op;
  if constexpr (kVecSize == 1) {
    res_packed.values[0] = op(rhs_packed.values[0], lhs_packed.values[0]);
  }
  if constexpr (kVecSize == 2) {
    res_packed.values[0] = op(rhs_packed.values[0], lhs_packed.values[0]);
    res_packed.values[1] = op(rhs_packed.values[1], lhs_packed.values[1]);
  }
  if constexpr (kVecSize == 4) {
    res_packed.values[0] = op(rhs_packed.values[0], lhs_packed.values[0]);
    res_packed.values[1] = op(rhs_packed.values[1], lhs_packed.values[1]);
    res_packed.values[2] = op(rhs_packed.values[2], lhs_packed.values[2]);
    res_packed.values[3] = op(rhs_packed.values[3], lhs_packed.values[3]);
  }
  return res_packed.value;
}

CUTLASS_GLOBAL void
sleep_async(int64_t sleep_ns) {
  nanosleep(sleep_ns);
}

CUTLASS_GLOBAL void
wait_async(void *ptr) {
  cutlass::Barrier::wait_eq(ptr, threadIdx.x, 0, 1);
}

template <typename T>
CUTLASS_DEVICE void
add_continous_kernel(T *dst_, const T *lhs_, const T *rhs_, int nelems) {
  using VecType = uint4;  // load as uint4
  nelems = nelems / sizeof(VecType) * sizeof(T);
  VecType *dst = (VecType *)dst_;
  const VecType *lhs = (const VecType *)lhs_;
  const VecType *rhs = (const VecType *)rhs_;
  CUTLASS_PRAGMA_UNROLL
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nelems; i += gridDim.x * blockDim.x) {
    dst[i] = add<T, VecType>(rhs + i, lhs + i);
  }
}

template <typename T>
CUTLASS_DEVICE void
add_continous_kernel(T *dst_, const T *lhs_, int nelems) {
  add_continous_kernel(dst_, lhs_, dst_, nelems);
}

template <typename T>
CUTLASS_DEVICE void
memset_continous_kernel(T *__restrict__ dst_, const T &value, int nelems) {
  using VecType = uint4;  // load as uint4
  nelems = nelems / sizeof(VecType) * sizeof(T);
  VecType *dst = (VecType *)dst_;
  VecType src_vec;
  T *src_ptr = (T *)&src_vec;
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < sizeof(VecType) / sizeof(T); ++i) {
    src_ptr[i] = value;
  }
  CUTLASS_PRAGMA_UNROLL
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nelems; i += gridDim.x * blockDim.x) {
    dst[i] = src_vec;
  }
}

template <typename T>
CUTLASS_DEVICE void
copy_continous_kernel(T *__restrict__ dst_, const T *__restrict__ src_, int nelems) {
  using VecType = uint4;  // load as uint4
  nelems = nelems / sizeof(VecType) * sizeof(T);
  VecType *dst = (VecType *)dst_;
  const VecType *src = (const VecType *)src_;
  CUTLASS_PRAGMA_UNROLL
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nelems; i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

template <typename T>
CUTLASS_GLOBAL void
add_continous(T *__restrict__ dst_, const T *__restrict__ src_, int nelems) {
  add_continous_kernel<T>(dst_, src_, nelems);
}

enum : int { kInitialized = 0, kGemmDone = 1, kAccumulatedLocal = 2 };

template <typename T, int kM, int kN, bool kFlattenTile = false>
struct ReduceScatterBase {
  using CtaShapeMNL = cute::Shape<cute::Int<kM>, cute::Int<kN>, cute::Int<1>>;
  using TileBarrier = BarrierWithFallback;
  using Params = ReduceScatterParams;
  static const int kVecLen = sizeof(VecType) / sizeof(T);
  static const int kMN = kM * kN;
  static const int kTileSizeVec = kMN / kVecLen;
  static const int kThreadsCol = kN / kVecLen;

  static_assert(sizeof(VecType) % sizeof(T) == 0, "");
  static_assert(!kFlattenTile || kN % kVecLen == 0, "Vectorize tile per line failed");

  struct SharedStorage {};

 public:
  CUTLASS_DEVICE
  ReduceScatterBase(const Params &params_)
      : params(params_),
        tiled_m((params_.m + kM - 1) / kM),
        tiled_n((params_.n + kN - 1) / kN),
        tiled_m_per_rank((params_.m / params_.world_size + kM - 1) / kM),
        num_tiles(tiled_m * tiled_n),
        tiles_per_rank(tiled_m_per_rank * tiled_n),
        problem_shape(params_.m, params_.n, cute::_1{}) {}

  const Params &params;
  int tiled_m, tiled_n, tiled_m_per_rank;
  int num_tiles, tiles_per_rank;
  cute::Shape<int, int, cute::_1> problem_shape;

 public:
  CUTLASS_DEVICE PerTileFlagsWrapper
  flags(int rank) {
    return PerTileFlagsWrapper(params.barrier_ptr[rank], num_tiles);
  }

  CUTLASS_DEVICE BarrierWorkQeueuFlagsWrapper
  work_queue_flags(int rank) {
    return BarrierWorkQeueuFlagsWrapper(flags(rank).extra_ptr(0), params.world_size);
  }

#ifdef FLUX_DEBUG_RS
#define DECLARE_DEBUG_FIELD(name) \
  CUTLASS_DEVICE bool name() const { return params.name; }
#else
#define DECLARE_DEBUG_FIELD(name) \
  CUTLASS_DEVICE bool name() const { return true; }
#endif

  // clang-format off
  DECLARE_DEBUG_FIELD(do_copy)
  DECLARE_DEBUG_FIELD(do_wait)
  DECLARE_DEBUG_FIELD(run_local)
  DECLARE_DEBUG_FIELD(run_remote)
#undef DECLARE_DEBUG_FIELD
  // clang-format on

  template <typename ThreadMapShape>
  CUTLASS_DEVICE auto
  buffer_tile(int m, int n, int tcol, int trow, const ThreadMapShape &thread_map_shape) {
    using cute::_1, cute::_0, cute::_;
    using X = cute::Underscore;
    auto tile = [&]() {
      if constexpr (kFlattenTile) {
        return cute::make_tensor(
            cute::make_gmem_ptr(buffer_ptr(to_tile_idx(m, n) * cute::size(CtaShapeMNL{}))),
            CtaShapeMNL{},
            cute::make_stride(cute::size<1>(CtaShapeMNL{}), _1{}, cute::size(CtaShapeMNL{})))(
            _, _, 0);
      } else {
        return cute::local_tile(
            cute::make_tensor(
                cute::make_gmem_ptr(buffer_ptr(0)),
                cute::make_shape(
                    params.m / params.world_size, params.n, cute::_1{}),  // less shape
                cute::make_stride(params.n, _1{}, params.m * params.n)),
            CtaShapeMNL{},
            make_coord(_, _, _),
            cute::Step<_1, _1, X>{})(_, _, m, n, 0);
      }
    }();
    auto tile_maped =
        cute::make_tensor(
            std::forward<decltype(tile)>(tile).data(),
            cute::make_layout(cute::get<1>(tile.layout()), cute::get<0>(tile.layout())))
            .compose(cute::make_layout(thread_map_shape));
    return cute::filter((cute::recast<VecType>(tile_maped(_, tcol, trow, _))));
  }

  template <typename ThreadMapShape>
  CUTLASS_DEVICE auto
  reduce_buffer_tile(
      int rank,
      int m,
      int n,
      int tcol,
      int trow,
      const ThreadMapShape &thread_map_shape,
      int offset = 0) {
    using cute::_1, cute::_0, cute::_;
    using cute::Tensor;
    using X = cute::Underscore;
    using cute::get;
    auto tile = [&]() {
      if constexpr (kFlattenTile) {
        return cute::make_tensor(
            cute::make_gmem_ptr(
                buffer_ptr(rank, offset + to_tile_idx(m, n) * cute::size(CtaShapeMNL{}))),
            CtaShapeMNL{},
            cute::make_stride(cute::size<1>(CtaShapeMNL{}), _1{}, cute::size(CtaShapeMNL{})))(
            _, _, 0);
      } else {
        return cute::local_tile(
            cute::make_tensor(
                cute::make_gmem_ptr(buffer_ptr(rank, offset)),
                problem_shape,
                cute::make_stride(params.n, _1{}, params.m * params.n)),
            CtaShapeMNL{},
            make_coord(_, _, _),
            cute::Step<_1, _1, X>{})(_, _, m, n, 0);
      }
    }();
    auto tile_maped = cute::make_tensor(
                          std::forward<decltype(tile)>(tile).data(),
                          cute::make_layout(get<1>(tile.layout()), get<0>(tile.layout())))
                          .compose(cute::make_layout(thread_map_shape));
    return cute::filter((cute::recast<VecType>(tile_maped(_, tcol, trow, _))));
  }

  template <typename ThreadMapShape>
  CUTLASS_DEVICE auto
  data_tile(
      int rank,
      int m,
      int n,
      int tcol,
      int trow,
      const ThreadMapShape &thread_map_shape,
      int offset = 0) {
    using cute::_1, cute::_0, cute::_;
    using cute::Tensor;
    using X = cute::Underscore;
    using cute::get;
    auto tile = [&]() {
      if constexpr (kFlattenTile) {
        return cute::make_tensor(
            cute::make_gmem_ptr(
                data_ptr(rank, offset + to_tile_idx(m, n) * cute::size(CtaShapeMNL{}))),
            CtaShapeMNL{},
            cute::make_stride(cute::size<1>(CtaShapeMNL{}), _1{}, cute::size(CtaShapeMNL{})))(
            _, _, 0);
      } else {
        return cute::local_tile(
            cute::make_tensor(
                cute::make_gmem_ptr(data_ptr(rank, offset)),
                problem_shape,
                cute::make_stride(params.n, _1{}, params.m * params.n)),
            CtaShapeMNL{},
            make_coord(_, _, _),
            cute::Step<_1, _1, X>{})(_, _, m, n, 0);
      }
    }();
    auto tile_maped = cute::make_tensor(
                          std::forward<decltype(tile)>(tile).data(),
                          cute::make_layout(get<1>(tile.layout()), get<0>(tile.layout())))
                          .compose(cute::make_layout(thread_map_shape));
    return cute::filter((cute::recast<VecType>(tile_maped(_, tcol, trow, _))));
  }

  CUTLASS_DEVICE int
  to_tile_idx(int m, int n) {
    return m * tiled_n + n;
  }

  template <typename ThreadMapShape>
  CUTLASS_DEVICE auto
  coord(int m, int n, int tcol, int trow, ThreadMapShape thread_map_shape) {
    using cute::_1, cute::_0, cute::_;
    using cute::Tensor;
    using X = cute::Underscore;
    using cute::get;
    Tensor cAux_xT = cute::local_tile(
        cute::make_identity_tensor(problem_shape),
        CtaShapeMNL{},
        make_coord(_, _, _),
        cute::Step<_1, _1, X>{})(_, _, m, n, 0);  // identity tiled_m * tiled_n
    Tensor cAux_nm = cute::make_tensor(
                         std::forward<decltype(cAux_xT)>(cAux_xT).data(),
                         make_layout(get<1>(cAux_xT.layout()), get<0>(cAux_xT.layout())))
                         .compose(make_layout(thread_map_shape))(
                             _, tcol, trow, _);  // transpose and remap to thread
    // vectorize
    return cute::filter(cute::outer_partition(cAux_nm, cute::Shape<cute::Int<kVecLen>>{}, (_0{})));
  }

 protected:
  CUTLASS_DEVICE T *
  data_ptr(int rank, int idx) {
    return (T *)params.scatter_ptr_aux[rank] + idx;
  }

  CUTLASS_DEVICE T *
  buffer_ptr(int idx) {
    return (T *)params.reduce_ptr[params.rank] + idx;
  }

  CUTLASS_DEVICE T *
  buffer_ptr(int rank, int idx) {
    return (T *)params.reduce_ptr[rank] + idx;
  }
};

#define USE_REDUCE_SCATTER_BASE                                                              \
 public:                                                                                     \
  using Base = ReduceScatterBase<T, kM, kN, kFlattenTile>;                                   \
  using Base::kThreadsCol, Base::kTileSizeVec, Base::kVecLen;                                \
  using SharedStorage = typename Base::SharedStorage;                                        \
  using TileBarrier = typename Base::TileBarrier;                                            \
  using Params = typename Base::Params;                                                      \
                                                                                             \
 private:                                                                                    \
  using Base::flags, Base::work_queue_flags;                                                 \
  using Base::params, Base::tiled_m, Base::tiled_n, Base::tiled_m_per_rank, Base::num_tiles, \
      Base::tiles_per_rank;                                                                  \
  using Base::data_tile, Base::buffer_tile, Base::coord, Base::to_tile_idx;                  \
  using Base::do_copy, Base::do_wait, Base::run_local, Base::run_remote

// no gemmk support
template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing2dPull : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing2dPull(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    const int rank = params.rank, lnode = rank / kNumaWorldSize;
    int m_per_rank = params.m / params.world_size;
    nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us

    int kNumThreads = blockDim.x;
    // force divided by caller: no assert for kernel
    int thread_rows = (kNumThreads + kThreadsCol - 1) / kThreadsCol;  // 512 / 32 = 16
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, thread_rows);
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},           // for vectorize
        cute::C<kThreadsCol>{},       // threads per line
        thread_rows,                  // num_threads / threads_per_line
        kTileSizeVec / kNumThreads);  // num_elements / num_threads / vectorize = 8
    auto [tcol, trow] = cute::idx2crd((int)threadIdx.x, thread_shape);

    const auto &topo = bytedance::flux::kTopologys[0];
    for (int s = 0; s < kNumaWorldSize; s++) {
      int rrank = topo.rank_from[s][rank];
      int rnode = rrank / kNumaWorldSize;
      bool is_inter_node = lnode != rnode;
      bool is_prev_inter_node = ((topo.rank_from[s][rrank]) / kNumaWorldSize) != rnode;
      int segment = topo.segments[s][rnode];
      bool skip_tile = (!run_remote() && is_inter_node) || (!run_local() && !is_inter_node);

      for (auto sid : {NextNodeRank(segment), segment}) {
        int m_start = m_per_rank * sid, m_end = m_start + m_per_rank;
        int tiled_m_start = m_start / kM, tiled_m_end = (m_end - 1) / kM + 1;  // open set
        int tile_m_per_seg = tiled_m_end - tiled_m_start;
        for (int bid = blockIdx.x; bid < tile_m_per_seg * tiled_n; bid += gridDim.x) {
          // int m = bid / tiled_n, n = bid % tiled_n;
          int m = bid % tile_m_per_seg, n = bid / tile_m_per_seg;
          int m0 = m;
          m += tiled_m_start;
          int tile_idx = to_tile_idx(m, n);
          int reduce_tile_idx = to_tile_idx(m0 + (tiled_m_per_rank + 1) * sid, n);
          if (do_wait() && !skip_tile) {
            if (is_prev_inter_node) {
              int *lock_ptr = flags(rrank).epilogue_ptr(tile_idx);
              print_per_block("remote epilogue %d[%d] = %d vs 1\n", rrank, tile_idx, *lock_ptr);
              TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
            } else {
              int *lock_ptr = flags(rrank).reduce_ptr(reduce_tile_idx);
              print_per_block(
                  "remote reduce %d[%d] = %d vs 1\n", rrank, reduce_tile_idx, *lock_ptr);
              TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
            }
            if (!is_inter_node) {  // wait for self gemm done
              int *lock_ptr = flags(rank).epilogue_ptr(tile_idx);
              print_per_block("local [%d] = %d vs 1\n", tile_idx, *lock_ptr);
              TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
            }
          }
          if (do_copy() && !skip_tile) {
            auto rdata = data_tile(rrank, m, n, tcol, trow, thread_map_shape);
            auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
            if (is_inter_node) {
              auto ldata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if ((cute::get<0>(coord_v(i)) >= m_start && cute::get<0>(coord_v(i)) < m_end)) {
                  ldata(i) = rdata(i);
                }
              }
            } else {
              auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if ((cute::get<0>(coord_v(i)) >= m_start && cute::get<0>(coord_v(i)) < m_end)) {
                  ldata(i) = add<T>(&ldata(i), &rdata(i));
                }
              }
            }
          }
          if (!is_inter_node) {  // no need for inter_node
            int *lock_ptr = flags(rank).reduce_ptr(reduce_tile_idx);
            TileBarrier::arrive_inc(lock_ptr, threadIdx.x, 0, (int)1);
            print_per_block("arrived: barrier[%d]=%d\n", reduce_tile_idx, *lock_ptr);
          }
        }
        if (is_inter_node) {
          break;
        }
      }
    }

    int m_start = m_per_rank * rank;
    int m_end = m_per_rank + m_start;  // open set
    int tiled_m_start = m_start / kM;
    int tiled_m_end = (m_end - 1) / kM + 1;  // open set
    int tile_m_per_seg = tiled_m_end - tiled_m_start;
    // reduce local
    for (int bid = blockIdx.x; bid < tile_m_per_seg * tiled_n; bid += gridDim.x) {
      int m = bid % tile_m_per_seg, n = bid / tile_m_per_seg;
      int m0 = m;  // for buffer reduce, range (0, tiled_m_per_rank)
      m += tiled_m_start;
      if (do_copy()) {
        auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape);
        auto rdata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
        auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(ldata); i++) {
          if ((cute::get<0>(coord_v(i)) >= m_start && cute::get<0>(coord_v(i)) < m_end)) {
            ldata(i) = add<T>(&ldata(i), &rdata(i));
          }
        }
      }
    }
    print_per_block("[%d] rskernel_inter_numa done\n", blockIdx.x);
  }

  CUTLASS_DEVICE
  static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing2dPull<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

// no gemmk support
template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing2dPullPerWarp : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing2dPullPerWarp(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    // 1024 / 32 = 32 >= (2 + 1) * 10 => 30 * 32 = 960
    // 1024 / 32 = 32 >= (4 + 1) * 6 => 30 * 32 = 960
    constexpr int kNumWorkersPerGroup = 64;
    constexpr int kNumThreadsPerGroup = kNumWorkersPerGroup + warpSize;

    int group = threadIdx.x / kNumThreadsPerGroup;  // group should in [0, 15). 1024 / 64 = 16
    int num_groups_local = blockDim.x / kNumThreadsPerGroup;
    int group_global = group + num_groups_local * blockIdx.x;
    int num_groups_total = num_groups_local * gridDim.x;
    int tid = threadIdx.x % kNumThreadsPerGroup;
    bool is_worker = tid < kNumWorkersPerGroup;

    const int rank = params.rank, lnode = rank / kNumaWorldSize;
    int m_per_rank = params.m / params.world_size;
    nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us

    // force divided by caller: no assert for kernel
    constexpr int kThreadsRow = kNumWorkersPerGroup / kThreadsCol;  // 64 / 32 = 2
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, cute::C<kThreadsRow>{});
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},      // for vectorize
        cute::C<kThreadsCol>{},  // threads per line
        cute::C<kThreadsRow>{},  // num_threads / threads_per_line
        cute::C<kTileSizeVec / kNumWorkersPerGroup>{});
    auto [tcol, trow] = cute::idx2crd(tid, thread_shape);

    const auto &topo = bytedance::flux::kTopologys[0];
    for (int s = 0; s < kNumaWorldSize; s++) {
      int rrank = topo.rank_from[s][rank];
      int rnode = rrank / kNumaWorldSize;
      bool is_inter_node = lnode != rnode;
      bool is_prev_inter_node = ((topo.rank_from[s][rrank]) / kNumaWorldSize) != rnode;
      int segment = topo.segments[s][rnode];
      bool skip_tile = (!run_remote() && is_inter_node) || (!run_local() && !is_inter_node);

      for (auto sid : {NextNodeRank(segment), segment}) {
        int m_start = m_per_rank * sid, m_end = m_start + m_per_rank;
        int tiled_m_start = m_start / kM, tiled_m_end = (m_end - 1) / kM + 1;  // open set
        int tile_m_per_seg = tiled_m_end - tiled_m_start;
        int bid = group_global;
        if (is_worker) {
          for (; bid < tile_m_per_seg * tiled_n; bid += num_groups_total) {
            // int m = bid / tiled_n, n = bid % tiled_n;
            int m = bid % tile_m_per_seg, n = bid / tile_m_per_seg;
            int m0 = m;
            m += tiled_m_start;
            int tile_idx = to_tile_idx(m, n);
            int reduce_tile_idx = to_tile_idx(m0 + (tiled_m_per_rank + 1) * sid, n);
            if (do_wait() && !skip_tile) {
              if (is_prev_inter_node) {
                int *lock_ptr = flags(rrank).epilogue_ptr(tile_idx);
                print_per_block("remote epilogue %d[%d] = %d vs 1\n", rrank, tile_idx, *lock_ptr);
                TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
              } else {
                int *lock_ptr = flags(rrank).reduce_ptr(reduce_tile_idx);
                print_per_block(
                    "remote reduce %d[%d] = %d vs 1\n", rrank, reduce_tile_idx, *lock_ptr);
                TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
              }
              if (!is_inter_node) {  // wait for self gemm done
                int *lock_ptr = flags(rank).epilogue_ptr(tile_idx);
                print_per_block("local [%d] = %d vs 1\n", tile_idx, *lock_ptr);
                TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
              }
            }

            if (do_copy() && !skip_tile) {
              auto rdata = data_tile(rrank, m, n, tcol, trow, thread_map_shape);
              auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
              if (is_inter_node) {
                auto ldata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(ldata); i++) {
                  if ((cute::get<0>(coord_v(i)) >= m_start && cute::get<0>(coord_v(i)) < m_end)) {
                    ldata(i) = rdata(i);
                  }
                }
              } else {
                auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape);
                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(ldata); i++) {
                  if ((cute::get<0>(coord_v(i)) >= m_start && cute::get<0>(coord_v(i)) < m_end)) {
                    ldata(i) = add<T>(&ldata(i), &rdata(i));
                  }
                }
              }
            }
            if (!is_inter_node) {  // no need for inter_node
              int *lock_ptr = flags(rank).reduce_ptr(reduce_tile_idx);
              TileBarrier::arrive_inc(lock_ptr, threadIdx.x, 0, (int)1);
              print_per_block("arrived: barrier[%d]=%d\n", reduce_tile_idx, *lock_ptr);
            }
          }
        } else {
          for (; bid < tile_m_per_seg * tiled_n; bid += num_groups_total) {
            // int m = bid / tiled_n, n = bid % tiled_n;
            int m = bid % tile_m_per_seg, n = bid / tile_m_per_seg;
            int m0 = m;
            m += tiled_m_start;
            int tile_idx = to_tile_idx(m, n);
            int reduce_tile_idx = to_tile_idx(m0 + (tiled_m_per_rank + 1) * sid, n);
            if (do_wait() && !skip_tile) {
              if (is_prev_inter_node) {
                int *lock_ptr = flags(rrank).epilogue_ptr(tile_idx);
                print_per_block("remote epilogue %d[%d] = %d vs 1\n", rrank, tile_idx, *lock_ptr);
                TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
              } else {
                int *lock_ptr = flags(rrank).reduce_ptr(reduce_tile_idx);
                print_per_block(
                    "remote reduce %d[%d] = %d vs 1\n", rrank, reduce_tile_idx, *lock_ptr);
                TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
              }
              if (!is_inter_node) {  // wait for self gemm done
                int *lock_ptr = flags(rank).epilogue_ptr(tile_idx);
                print_per_block("local [%d] = %d vs 1\n", tile_idx, *lock_ptr);
                TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
              }
            }

            if (!is_inter_node) {  // no need for inter_node
              int *lock_ptr = flags(rank).reduce_ptr(reduce_tile_idx);
              TileBarrier::arrive_inc(lock_ptr, threadIdx.x, 0, (int)1);
              print_per_block("arrived: barrier[%d]=%d\n", reduce_tile_idx, *lock_ptr);
            }
          }
        }
        if (is_inter_node) {
          break;
        }
      }
    }

    int m_start = m_per_rank * rank;
    int m_end = m_per_rank + m_start;  // open set
    int tiled_m_start = m_start / kM;
    int tiled_m_end = (m_end - 1) / kM + 1;  // open set
    int tile_m_per_seg = tiled_m_end - tiled_m_start;
    // reduce local
    for (int bid = blockIdx.x; bid < tile_m_per_seg * tiled_n; bid += gridDim.x) {
      int m = bid % tile_m_per_seg, n = bid / tile_m_per_seg;
      int m0 = m;  // for buffer reduce, range (0, tiled_m_per_rank)
      m += tiled_m_start;
      if (do_copy()) {
        auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape);
        auto rdata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
        auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(ldata); i++) {
          if ((cute::get<0>(coord_v(i)) >= m_start && cute::get<0>(coord_v(i)) < m_end)) {
            ldata(i) = add<T>(&ldata(i), &rdata(i));
          }
        }
      }
    }
    print_per_block("[%d] rskernel_inter_numa done\n", blockIdx.x);
  }

  CUTLASS_DEVICE
  static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing2dPullPerWarp<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing2dPullGemmk : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing2dPullGemmk(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    const int rank = params.rank, lnode = rank / kNumaWorldSize;
    int m_per_rank = params.m / params.world_size;
    nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us

    int num_threads = blockDim.x;
    // force divided by caller: no assert for kernel
    int thread_rows = (num_threads + kThreadsCol - 1) / kThreadsCol;
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, thread_rows);
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},           // for vectorize
        cute::C<kThreadsCol>{},       // threads per line
        thread_rows,                  // num_threads / threads_per_line
        kTileSizeVec / num_threads);  // num_elements / num_threads / vectorize = 8
    auto [tcol, trow] = cute::idx2crd((int)threadIdx.x, thread_shape);
    int offset_per_seg = (m_per_rank - tiled_m_per_rank * kM) * params.n;

    const auto &topo = bytedance::flux::kTopologys[0];
    for (int s = 0; s < kNumaWorldSize; s++) {
      int rrank = topo.rank_from[s][rank];
      int rnode = rrank / kNumaWorldSize;
      bool is_inter_node = lnode != rnode;
      bool is_prev_inter_node = ((topo.rank_from[s][rrank]) / kNumaWorldSize) != rnode;
      int segment = topo.segments[s][rnode];
      bool skip_tile = (!run_remote() && is_inter_node) || (!run_local() && !is_inter_node);
      for (auto sid : {NextNodeRank(segment), segment}) {
        int tiled_m_start = tiled_m_per_rank * sid;  // open set
        for (int bid = blockIdx.x; bid < tiles_per_rank; bid += gridDim.x) {
          int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
          int m0 = m;
          m += tiled_m_start;
          int tile_idx = to_tile_idx(m, n);

          if (do_wait() && !skip_tile) {
            int *lock_ptr = is_prev_inter_node ? flags(rrank).epilogue_ptr(tile_idx)
                                               : flags(rrank).reduce_ptr(tile_idx);
            print_per_block("remote %d[%d] = %d\n", rrank, tile_idx, *lock_ptr);
            TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
            if (!is_inter_node) {  // wait for self gemm done
              int *lock_ptr = flags(rank).epilogue_ptr(tile_idx);
              print_per_block("local [%d] = %d vs 1\n", tile_idx, *lock_ptr);
              TileBarrier::wait_eq(lock_ptr, threadIdx.x, 0, 1);
            }
          }
          int offset = offset_per_seg * sid;
          if (do_copy() && !skip_tile) {
            auto rdata = data_tile(rrank, m, n, tcol, trow, thread_map_shape, offset);
            auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
            int m_end = tiled_m_start * kM + m_per_rank;
            if (is_inter_node) {
              auto ldata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if (cute::get<0>(coord_v(i)) >= m_end) {
                  break;
                }
                ldata(i) = rdata(i);
              }
            } else {
              auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape, offset);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if (cute::get<0>(coord_v(i)) >= m_end) {
                  break;
                }
                ldata(i) = add<T>(&ldata(i), &rdata(i));
              }
            }
          }
          if (!is_inter_node) {  // no need for inter_node
            int *lock_ptr = flags(rank).reduce_ptr(tile_idx);
            TileBarrier::arrive_inc(lock_ptr, threadIdx.x, 0, (int)1);
            print_per_block("arrived: barrier[%d]=%d\n", tile_idx, *lock_ptr);
          }
        }
        if (is_inter_node) {
          break;
        }
      }
    }

    int tiled_m_start = tiled_m_per_rank * rank;
    int m_end = tiled_m_start * kM + m_per_rank;
    int offset = offset_per_seg * rank;
    // reduce local
    for (int bid = blockIdx.x; bid < tiles_per_rank; bid += gridDim.x) {
      int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
      int m0 = m;
      m += tiled_m_start;
      if (do_copy()) {
        auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape, offset);
        auto rdata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
        auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(ldata); i++) {
          if (cute::get<1>(coord_v(i)) < params.n && (cute::get<0>(coord_v(i)) < m_end)) {
            ldata(i) = add<T>(&ldata(i), &rdata(i));
          }
        }
      }
    }
    print_per_block("[%d] rskernel_inter_numa_gemmk done\n", blockIdx.x);
  }

  CUTLASS_DEVICE
  static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing2dPullGemmk<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing2dPushGemmk : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

  // seems cutlass::Barrier is not enough. but with Barrier it's a little slow
  using Barrier = cutlass::detail::GenericSystemBarrier<cutlass::detail::SyncthreadsSync>;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing2dPushGemmk(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    auto print_debug = [&](auto &...args) {
#ifdef FLUX_DEBUG
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf(args...);
      }
#endif
    };

    const int rank = params.rank, lnode = rank / kNumaWorldSize;
    int m_per_rank = params.m / params.world_size;

    int num_threads = blockDim.x;
    // force divided by caller: no assert for kernel
    int thread_rows = (num_threads + kThreadsCol - 1) / kThreadsCol;
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, thread_rows);
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},           // for vectorize
        cute::C<kThreadsCol>{},       // threads per line
        thread_rows,                  // num_threads / threads_per_line
        kTileSizeVec / num_threads);  // num_elements / num_threads / vectorize = 8
    auto [tcol, trow] = cute::idx2crd((int)threadIdx.x, thread_shape);
    int offset_per_seg = (m_per_rank - tiled_m_per_rank * kM) * params.n;

    const auto &topo = bytedance::flux::kTopologys[0];
    for (int s = 0; s < kNumaWorldSize; s++) {
      int to_rank = topo.rank_to[s][rank];
      int to_node = to_rank / kNumaWorldSize;
      bool is_inter_node = lnode != to_node;
      bool is_ring_start = ((topo.rank_from[s][rank]) / kNumaWorldSize) != lnode;
      int segment = topo.segments[s][lnode];
      bool skip_tile = (!run_remote() && is_inter_node) || (!run_local() && !is_inter_node);
      for (auto sid : {NextNodeRank(segment), segment}) {
        int tiled_m_start = tiled_m_per_rank * sid;  // open set
        for (int bid = blockIdx.x; bid < tiles_per_rank; bid += gridDim.x) {
          int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
          int tile_idx = to_tile_idx(m + tiled_m_start, n);
          int m0 = m;
          int m_free = m0 + topo.unused_segments_push[to_rank] * tiled_m_per_rank;
          int tile_idx_free = to_tile_idx(m_free, n);

          m += tiled_m_start;

          if (do_wait() && !skip_tile) {
            if (!is_ring_start) {
              int *flag_ptr = flags(rank).reduce_ptr(tile_idx);  // TODO(houqi.1993)
              Barrier::wait_eq(flag_ptr, threadIdx.x, 0, 1);
            }

            int *flag_ptr = flags(rank).epilogue_ptr(tile_idx);
            Barrier::wait_eq(flag_ptr, threadIdx.x, 0, 1);
          }
          int offset = offset_per_seg * sid;
          if (do_copy() && !skip_tile) {
            auto rdata = reduce_buffer_tile(
                to_rank, is_inter_node ? m_free : m, n, tcol, trow, thread_map_shape);
            auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape, offset);
            auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
            int m_end = tiled_m_start * kM + m_per_rank;
            if (is_ring_start) {  // copy only
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {  // TODO(houqi.1993) check size
                if (cute::get<1>(coord_v(i)) < params.n && (cute::get<0>(coord_v(i)) < m_end)) {
                  rdata(i) = ldata(i);
                }
              }
            } else {  // reduce and copy
              auto local_reduce_buffer =
                  reduce_buffer_tile(rank, m, n, tcol, trow, thread_map_shape);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if (cute::get<1>(coord_v(i)) < params.n && (cute::get<0>(coord_v(i)) < m_end)) {
                  rdata(i) = add<T>(&ldata(i), &local_reduce_buffer(i));
                }
              }
            }
          }
          int *flag_ptr = flags(to_rank).reduce_ptr(is_inter_node ? tile_idx_free : tile_idx);
          Barrier::arrive_inc(flag_ptr, threadIdx.x, 0, (int)1);
        }
        if (is_inter_node) {
          break;
        }
      }
    }

    int tiled_m_start = tiled_m_per_rank * rank;
    int m_end = tiled_m_start * kM + m_per_rank;
    int offset = offset_per_seg * rank;
    // reduce local
    for (int bid = blockIdx.x; bid < tiles_per_rank; bid += gridDim.x) {
      int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
      int m0 = m;
      m += tiled_m_start;
      int m_free = m0 + topo.unused_segments_push[rank] * tiled_m_per_rank;
      if (do_wait()) {
        int tile_idx = to_tile_idx(m, n);
        int *flag_ptr = flags(rank).epilogue_ptr(tile_idx);
        Barrier::wait_eq(flag_ptr, threadIdx.x, 0, 1);
        flag_ptr = flags(rank).reduce_ptr(tile_idx);
        Barrier::wait_eq(flag_ptr, threadIdx.x, 0, 1);

        int tile_idx_free = to_tile_idx(m_free, n);
        flag_ptr = flags(rank).reduce_ptr(tile_idx_free);
        Barrier::wait_eq(flag_ptr, threadIdx.x, 0, 1);
      }
      if (do_copy()) {
        auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape, offset);
        auto rdata = reduce_buffer_tile(rank, m, n, tcol, trow, thread_map_shape);
        auto rdata_next_numa = reduce_buffer_tile(rank, m_free, n, tcol, trow, thread_map_shape);
        auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(ldata); i++) {
          if (cute::get<1>(coord_v(i)) < params.n && (cute::get<0>(coord_v(i)) < m_end)) {
            ldata(i) = add<T>(&ldata(i), &rdata(i));
            ldata(i) = add<T>(&ldata(i), &rdata_next_numa(i));
          }
        }
      }
    }
    print_per_block("[%d] reduce_scatter_tp8_push_gemmk done\n", rank);
  }

  CUTLASS_DEVICE
  static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing2dPushGemmk<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

// no gemmk support. only support m % kM == 0 and (m / kM) % world_size == 0
// nearly the same as ReduceScatterRing2dPull with both m=124/12288
template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing2dPullWithQueue : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing2dPullWithQueue(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    const int rank = params.rank, lnode = rank / kNumaWorldSize;
    const int num_threads = blockDim.x;
    // force divided by caller: no assert for kernel
    const int thread_rows = (num_threads + kThreadsCol - 1) / kThreadsCol;  // 512 / 32 = 16
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, thread_rows);
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},           // for vectorize
        cute::C<kThreadsCol>{},       // threads per line
        thread_rows,                  // num_threads / threads_per_line
        kTileSizeVec / num_threads);  // num_elements / num_threads / vectorize = 8
    auto [tcol, trow] = cute::idx2crd((int)threadIdx.x, thread_shape);
    auto &problem_shape = Base::problem_shape;

    nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us
    cutlass::FastDivmod div_mod_tiled_n(tiled_n);
    __shared__ int svalue;

    const auto &topo = bytedance::flux::kTopologys[0];
    for (int s = 0; s < kNumaWorldSize; s++) {
      int rrank = topo.rank_from[s][rank];
      int rnode = rrank / kNumaWorldSize;
      bool is_inter_node = lnode != rnode;
      bool is_prev_inter_node = ((topo.rank_from[s][rrank]) / kNumaWorldSize) != rnode;
      int wait_value = (is_prev_inter_node ? kGemmDone : kAccumulatedLocal);

      // remote first
      for (int segment : {NextNodeRank(topo.segments[s][rnode]), topo.segments[s][rnode]}) {
        // for inter_node=true, no_use_segment will not be pulled.
        // it pulls from inter node, it's ring start in local mode.
        int segment_inter_node = is_inter_node ? NextNodeRank(topo.segments[s][lnode]) : segment;
        bool skip_tile = (!run_remote() && is_inter_node) || (!run_local() && !is_inter_node);

        for (int i = blockIdx.x; i < tiles_per_rank; i += gridDim.x) {
          if (threadIdx.x == 0) {
            int *ptr = is_prev_inter_node
                           ? flags(rrank).epilogue_queue_ptr(segment * tiles_per_rank + i)
                           : flags(rrank).reduce_queue_ptr(segment * tiles_per_rank + i);
            for (svalue = atomic_load_acquire_sys(ptr); svalue <= 0;
                 svalue = atomic_load_sys(ptr)) {
            }
          }
          __syncthreads();
          int tile_idx = svalue - 1;  // got tile_idx to copy
          int ts_tile_idx = tile_idx + (segment_inter_node - segment) * tiles_per_rank;

          if (do_wait() && !skip_tile) {
            print_per_block(
                "wait_remote %d[%d] = %d vs %d\n",
                rrank,
                tile_idx,
                flags(rrank).epilogue(tile_idx),
                wait_value);
            cutlass::Barrier::wait_eq(
                flags(rrank).epilogue_ptr(0), threadIdx.x, tile_idx, wait_value);
            if (!is_inter_node) {
              print_per_block(
                  "wait_local %d[%d] = %d vs %d\n",
                  rank,
                  tile_idx,
                  flags(rank).epilogue(tile_idx),
                  wait_value);
              cutlass::Barrier::wait_eq(
                  flags(rank).epilogue_ptr(0),
                  threadIdx.x,
                  tile_idx,
                  (int)1);  // wait for self gemm done
            }
          }
          if (do_copy() && !skip_tile) {
            int m, n;
            div_mod_tiled_n(m, n, tile_idx);
            auto rdata = data_tile(rrank, m, n, tcol, trow, thread_map_shape);
            auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
            if (is_inter_node) {
              auto ldata = buffer_tile(m, n, tcol, trow, thread_map_shape);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if (elem_less(coord_v(i), problem_shape)) {
                  ldata(i) = rdata(i);
                }
              }
            } else {
              auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape);
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(ldata); i++) {
                if (elem_less(coord_v(i), problem_shape)) {
                  ldata(i) = add<T>(&ldata(i), &rdata(i));
                }
              }
            }
          }
          // also
          if (threadIdx.x == 0) {
            int index = atomicAdd(work_queue_flags(rank).reduce_done_ptr(segment_inter_node), 1);
            atomicAdd_system(
                flags(rank).reduce_queue_ptr(segment * tiles_per_rank + index), tile_idx + 1);
          }
          int *barrier_ptr = is_inter_node ? flags(rank).reduce_ptr(tile_idx)
                                           : flags(rank).epilogue_ptr(tile_idx);
          cutlass::Barrier::arrive_inc(barrier_ptr, threadIdx.x, 0, (int)1);
          print_per_block("arrived: barrier[%d]=%d\n", tile_idx, *barrier_ptr);
        }
        if (is_inter_node) {
          // wait for other intra-node done
          break;
        }
      }
    }

    // check if other sm is still working
    for (int tile_idx = blockIdx.x + rank * tiles_per_rank; tile_idx < (rank + 1) * tiles_per_rank;
         tile_idx += gridDim.x) {
      if (do_wait()) {
        wait_eq_sys(flags(rank).reduce_ptr(tile_idx), threadIdx.x, 0, 1);
        wait_eq_sys(flags(rank).epilogue_ptr(tile_idx), threadIdx.x, 0, 2);
      }
      if (do_copy()) {
        int m, n;
        div_mod_tiled_n(m, n, tile_idx);
        auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape);
        auto rdata = buffer_tile(m % tiled_m_per_rank, n, tcol, trow, thread_map_shape);
        auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(ldata); i++) {
          if (elem_less(coord_v(i), problem_shape)) {
            ldata(i) = add<T>(&ldata(i), &rdata(i));
          }
        }
      }
    }
  }
  CUTLASS_DEVICE
  static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing2dPullWithQueue<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing1dPullGemmk : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing1dPullGemmk(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    const int num_threads = blockDim.x;
    // force divided by caller: no assert for kernel
    int thread_rows = (num_threads + kThreadsCol - 1) / kThreadsCol;
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, thread_rows);
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},           // for vectorize
        cute::C<kThreadsCol>{},       // threads per line
        thread_rows,                  // num_threads / threads_per_line
        kTileSizeVec / num_threads);  // num_elements / num_threads / vectorize = 8

    nanosleep(params.sleep_ns);
    auto [tcol, trow] = cute::idx2crd((int)threadIdx.x, thread_shape);
    const int rank = params.rank;
    const int m_per_rank = params.m / params.world_size;
    const int offset_per_seg = (m_per_rank - tiled_m_per_rank * kM) * params.n;
    const int offset = offset_per_seg * rank;
    const int tiled_m_start = tiled_m_per_rank * rank;
    const int m_end = tiled_m_start * kM + m_per_rank;
    const auto shape = cute::make_shape(m_end, params.n, cute::_1{});

    for (int i = 0; i < params.world_size - 1; i++) {
      int rrank = (rank - i + params.world_size - 1) % params.world_size;
      auto flags = PerTileFlagsWrapper(params.barrier_ptr[rrank], num_tiles);
      for (int bid = blockIdx.x; bid < tiles_per_rank; bid += gridDim.x) {
        int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
        int m0 = m;
        m += tiled_m_start;
        int tile_idx = to_tile_idx(m, n);
        if (do_wait()) {
          TileBarrier::wait_eq(flags.epilogue_ptr(tile_idx), threadIdx.x, 0, 1);
        }
        if (do_copy()) {
          auto ldata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
          auto rdata = data_tile(rrank, m, n, tcol, trow, thread_map_shape, offset);
          auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(ldata); i++) {
            if (cute::elem_less(coord_v(i), shape)) {
              ldata(i) = add<T>(&ldata(i), &rdata(i));
            }
          }
        }
      }
    }

    // remote reduced done. added to local
    auto local_flags = PerTileFlagsWrapper(params.barrier_ptr[rank], num_tiles);
    for (int bid = blockIdx.x; bid < tiles_per_rank; bid += gridDim.x) {
      int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
      int m0 = m;
      m += tiled_m_start;
      int tile_idx = to_tile_idx(m, n);
      if (do_wait()) {
        TileBarrier::wait_eq(local_flags.epilogue_ptr(tile_idx), threadIdx.x, 0, 1);
      }
      if (do_copy()) {
        auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape, offset);
        auto rdata = buffer_tile(m0, n, tcol, trow, thread_map_shape);
        auto coord_v = coord(m, n, tcol, trow, thread_map_shape);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(ldata); i++) {
          if (cute::elem_less(coord_v(i), shape)) {
            ldata(i) = add<T>(&ldata(i), &rdata(i));
          }
        }
      }
    }

    print_per_block("rskernel_inter_numa done\n");
  }
  CUTLASS_DEVICE
  static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing1dPullGemmk<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

template <typename T, int kM, int kN, bool kFlattenTile>
class ReduceScatterRing1dPushGemmk : public ReduceScatterBase<T, kM, kN, kFlattenTile> {
  USE_REDUCE_SCATTER_BASE;

  using BarrierType = cutlass::detail::SystemBarrier;

 public:
  CUTLASS_DEVICE
  ReduceScatterRing1dPushGemmk(const ReduceScatterParams &param) : Base(param) {}

  CUTLASS_DEVICE void
  run() {
    constexpr int kWarpSize = 32;
    constexpr int kNumWorkersPerGroup = kWarpSize * 4;
    constexpr int kNumThreadsPerGroup = kNumWorkersPerGroup + kWarpSize;

    // gid_local should in [0, 15). 1024 / 64 = 16
    int gid_local = threadIdx.x / kNumThreadsPerGroup;
    int num_groups_per_cta = blockDim.x / kNumThreadsPerGroup;
    int gid = gid_local + num_groups_per_cta * blockIdx.x;
    int num_groups = num_groups_per_cta * gridDim.x;
    int tid = threadIdx.x % kNumThreadsPerGroup;
    bool is_worker = tid < kNumWorkersPerGroup;
    // do nothing
    if (threadIdx.x >= num_groups_per_cta * kNumThreadsPerGroup)
      return;

    constexpr int num_threads = kNumWorkersPerGroup;
    // force divided by caller: no assert for kernel
    constexpr int thread_rows = (num_threads + kThreadsCol - 1) / kThreadsCol;
    auto thread_shape = cute::make_shape(cute::C<kThreadsCol>{}, cute::C<thread_rows>{});
    auto thread_map_shape = cute::make_shape(
        cute::C<kVecLen>{},                      // for vectorize
        cute::C<kThreadsCol>{},                  // threads per line
        cute::C<thread_rows>{},                  // num_threads / threads_per_line
        cute::C<kTileSizeVec / num_threads>{});  // num_elements / num_threads / vectorize = 8

    nanosleep(params.sleep_ns);
    auto [tcol, trow] = cute::idx2crd((int)tid, thread_shape);
    const int rank = params.rank;
    const int m_per_rank = params.m / params.world_size;
    const int offset_per_seg = (m_per_rank - tiled_m_per_rank * kM) * params.n;

    auto arrive_inc = [&](int *ptr) {
      sub_barrier_sync(gid_local, kNumThreadsPerGroup);
      if (tid == 0 || tid == kNumWorkersPerGroup) {
        atomic_ref_sys<int> ref(*ptr);
        ref.fetch_add(1, cuda::memory_order_release);
      }
    };
    auto wait_eq = [&](int *ptr) {
      if (tid == 0) {
        atomic_ref_sys<int> ref(*ptr);
        if (ref.load(cuda::memory_order_acquire) != 1) {
          while (ref.load(cuda::memory_order_relaxed) != 1) {
          }
        }
      }
      sub_barrier_sync(gid_local + num_groups_per_cta, kNumWorkersPerGroup);
    };

    auto print_worker = [&](const auto &...args) { print_if(tid == 0, args...); };
    auto print_waiter = [&](const auto &...args) {
      print_if(tid == kNumWorkersPerGroup, args...);
    };

    int rrank = (rank + params.world_size - 1) % params.world_size;
    for (int round = 0; round < params.world_size; round++) {
      //
      int segment = (round + rank + 1) % params.world_size;
      const int offset = offset_per_seg * segment;
      const int tiled_m_start = tiled_m_per_rank * segment;
      const int m_end = tiled_m_start * kM + m_per_rank;
      const auto shape = cute::make_shape(m_end, params.n, cute::_1{});
      // first rount: only wait gemm, no wait remote
      bool is_ring_start = round == 0;
      // last round: only reduce local, no push remote
      bool last_round = (round == (params.world_size - 1));
      for (int bid = gid; bid < tiles_per_rank; bid += num_groups) {
        int m = bid % tiled_m_per_rank, n = bid / tiled_m_per_rank;
        m += tiled_m_start;
        int tile_idx = to_tile_idx(m, n);
        if (is_worker) {  // doing copy here
          // wait for current gemm ready to push
          int *flag_ptr = flags(rank).epilogue_ptr(tile_idx);
          print_worker("%d:%d wait gemm %d to be 1\n", rank, gid, tile_idx);
          wait_eq(flag_ptr);
          if (!is_ring_start) {  // also wait for reduce buffer ready to do reduce
            int *flag_ptr = flags(rank).reduce_ptr(tile_idx);
            print_worker("%d:%d wait reduce %d to be 1\n", rank, gid, tile_idx);
            wait_eq(flag_ptr);
          }

          auto ldata = data_tile(rank, m, n, tcol, trow, thread_map_shape, offset);
          auto lrdata = reduce_buffer_tile(rank, m, n, tcol, trow, thread_map_shape);
          auto rdata = reduce_buffer_tile(rrank, m, n, tcol, trow, thread_map_shape);

          int start = m * kM + trow;
          int end = min(start + kM, m_end);
          int copy_counts = (end - start + thread_rows - 1) / thread_rows;  // size(ldata)

          if (last_round) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < copy_counts; i++) {
              ldata(i) = add<T>(&ldata(i), &lrdata(i));
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < copy_counts; i++) {
              rdata(i) = add<T>(&ldata(i), &lrdata(i));
            }
          }
          sub_barrier_sync(gid_local, kNumThreadsPerGroup);
        } else {  // doing wait here
          int *flag_ptr = flags(rrank).reduce_ptr(tile_idx);
          arrive_inc(flag_ptr);
          print_waiter(
              "%d:%d arrived at reduce %d[%d] to %d\n", rank, gid, rrank, tile_idx, *flag_ptr);
        }
      }
    }
  }
  CUTLASS_DEVICE static void
  invoke(const ReduceScatterParams &params, SharedStorage &) {
    ReduceScatterRing1dPushGemmk<T, kM, kN, kFlattenTile> op(params);
    op.run();
  }
};

template <typename T>
CUTLASS_GLOBAL void
run_per_segment_kernel(ReduceScatterParams params) {
  auto grid_group = cooperative_groups::this_grid();
  nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us

  int rank_to = (params.rank + params.world_size - 1) % params.world_size;
  // int rank_from = (params.rank + 1) % params.world_size;
  int rank = params.rank;
  int m_per_rank = params.m / params.world_size;
  int elems_per_rank = m_per_rank * params.n;
  int num_segments = params.n_split * params.world_size;
  auto flags = [&](int rank) {
    return PerRankFlagsWrapper(params.barrier_ptr[rank], num_segments);
  };
  auto data_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.scatter_ptr_aux[rank] + segment * elems_per_rank;
  };
  auto reduce_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.reduce_ptr[rank] + segment * elems_per_rank;
  };
  auto wait_ready = [&](int *ptr) {
#if defined(FLUX_DEBUG_RS)
    if (!params.do_wait)
      return;
#endif
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
      atomic_add_release_sys(ptr, 1);
    }
  };
  // calculate
  for (int i = 0; i < params.world_size; i++) {
    int segment_send = (rank + i + 1) % params.world_size;
    // replace with printf
    print_per_kernel("wait %d segment %d gemm done\n", rank, segment_send);
    T *src = data_ptr(rank, segment_send);
    wait_ready(flags(rank).gemm_done_ptr(segment_send));
    T *dst = reduce_ptr(rank_to, segment_send);
    if (i != 0) {
      print_per_kernel("wait %d segment %d copy done\n", rank, segment_send);
      wait_ready(flags(rank).copy_done_ptr(segment_send));
    }
    if (i == 0) {  // copy to remote only
      copy_continous_kernel(dst, src, elems_per_rank);
    } else if (i == params.world_size - 1) {  // local reduce
      add_continous_kernel<T>(src, reduce_ptr(rank, segment_send), elems_per_rank);
      break;
    } else {  // reduce to remote
      add_continous_kernel<T>(dst, src, reduce_ptr(rank, segment_send), elems_per_rank);
    }
    print_per_kernel("set %d segment %d copy done\n", rank_to, segment_send);
    set_ready(flags(rank_to).copy_done_ptr(segment_send));
  }
}

template <typename T>
CUTLASS_GLOBAL void
run_per_segment_kernel_tp8(ReduceScatterParams params) {
  nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us
  auto grid_group = cooperative_groups::this_grid();
  // false for 0 <- 1 <- 2 <- 3, true for 1 -> 2 -> 3
  int rank = params.rank;
  int rank_local = params.rank % kNumaWorldSize, numa_id = params.rank / kNumaWorldSize;
  bool ring_order_asc = numa_id == 1;

  int rank_prev = (rank_local + kNumaWorldSize - 1) % kNumaWorldSize + numa_id * kNumaWorldSize;
  int rank_next = (rank_local + 1) % kNumaWorldSize + numa_id * kNumaWorldSize;
  int rank_to = ring_order_asc ? rank_next : rank_prev;
  int rank_from = ring_order_asc ? rank_prev : rank_next;
  int rank_next_node = (rank + kNumaWorldSize) % kLocalWorldSize;
  int m_per_rank = params.m / params.world_size;
  int elems_per_rank = m_per_rank * params.n;
  int num_segments = params.n_split * params.world_size;
  auto flags = [&](int rank) {
    return PerRankFlagsWrapper(params.barrier_ptr[rank], num_segments);
  };
  auto data_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.scatter_ptr_aux[rank] + segment * elems_per_rank;
  };
  auto reduce_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.reduce_ptr[rank] + segment * elems_per_rank;
  };
  auto wait_ready = [&](int *ptr) {
#if defined(FLUX_DEBUG_RS)
    if (!params.do_wait)
      return;
#endif
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
      atomic_add_release_sys(ptr, 1);
    }
  };
  auto get_send_segment_inner = [&](int iter) -> int {
    return ring_order_asc ? (rank_local - 1 - iter + kNumaWorldSize) % kNumaWorldSize
                          : (rank_local + iter + 1) % kNumaWorldSize;
  };
  // calculate inner NUMA
  int numa_iter = 0;
  int segment_send_inter = (numa_id + numa_iter + 1) % kNumaNodes * kNumaWorldSize;
  for (int i = 0; i < kNumaWorldSize; i++) {
    int segment_send = segment_send_inter + get_send_segment_inner(i);
    // replace with printf
    print_per_kernel("[%d] wait segment %d gemm done\n", i, segment_send);
    wait_ready(flags(rank).gemm_done_ptr(segment_send));
    T *src = data_ptr(rank, segment_send);
    T *dst = reduce_ptr(rank_to, segment_send);
    if (i != 0) {
      print_per_kernel("[%d] wait segment %d copy done\n", i, segment_send);
      wait_ready(flags(rank).copy_done_ptr(segment_send));
    }
    if (i == kNumaWorldSize - 1) {  // reduce local only
      add_continous_kernel<T>(src, reduce_ptr(rank, segment_send), elems_per_rank);
    } else {
      if (i == 0) {  // copy to remote
        copy_continous_kernel(dst, src, elems_per_rank);
      } else {  // reduce to remote
        add_continous_kernel<T>(dst, src, reduce_ptr(rank, segment_send), elems_per_rank);
      }
      print_per_kernel("[%d] set %d segment %d copy done\n", i, rank_to, segment_send);
      set_ready(flags(rank_to).copy_done_ptr(segment_send));
    }
  }

  // reduce current NUMA node
  __const__ int slowest_ranks[2][4] = {{4, 5, 6, 7}, {3, 0, 1, 2}};
  __const__ int slowest_segments[2][4] = {{7, 4, 5, 6}, {0, 1, 2, 3}};
  int send_iter = 0;
  numa_iter = 1;
  int rank_next_numa_from = (rank_to + kNumaWorldSize) % kLocalWorldSize;
  segment_send_inter = (numa_id + numa_iter + 1) % kNumaNodes * kNumaWorldSize;
  for (int i = 0; i < kNumaWorldSize; i++) {
    bool send_cross_numa =
        ((ring_order_asc ? (i - 1 + kNumaWorldSize) % kNumaWorldSize : i) == rank_local);
    int segment_send = send_cross_numa ? (rank + kNumaWorldSize) % kLocalWorldSize
                                       : (segment_send_inter + get_send_segment_inner(send_iter));
    T *src = data_ptr(rank, segment_send);
    // remote_reduce_ptr(rank) is not copied to for intra copy, used as inter copy buffer.
    int segment_recv = send_cross_numa ? rank_next_numa_from : segment_send;
    T *dst = send_cross_numa ? reduce_ptr(rank_next_node, segment_recv)
                             : reduce_ptr(rank_to, segment_recv);

    if (!send_cross_numa) {
      print_per_kernel("[%d] wait segment %d gemm done\n", i, segment_send);
      wait_ready(flags(rank).gemm_done_ptr(segment_send));
      if (send_iter != 0) {
        print_per_kernel("[%d] wait segment %d copy done\n", i, segment_send);
        wait_ready(flags(rank).copy_done_ptr(segment_send));
        add_continous_kernel<T>(src, reduce_ptr(rank, segment_send), elems_per_rank);
      }
    }
    if (i != 0) {  // TODO(houqi.1993) this prevent this kernel from being general for all
                   // world_size & numa_world_size. to be fixed later
      // wait for last stage the slowest. otherwise NUMA and not cross NUMA interleaved
      wait_ready(
          flags(slowest_ranks[numa_id][i - 1]).copy_done_ptr(slowest_segments[numa_id][i - 1]));
    }

    copy_continous_kernel(dst, src, elems_per_rank);

    if (send_cross_numa) {
      int rank_to_inter = (rank + kNumaWorldSize) % kLocalWorldSize;
      print_per_kernel(
          "[%d] set %d segment %d copy done cross numa\n", i, rank_to_inter, segment_recv);
      set_ready(flags(rank_to_inter).copy_done_ptr(segment_recv));
    } else {
      print_per_kernel("[%d] set %d segment %d copy done\n", i, rank_to, segment_recv);
      set_ready(flags(rank_to).copy_done_ptr(segment_recv));
    }

    if (!send_cross_numa) {
      send_iter++;
    }
  }

  print_per_kernel("%d wait segment %d gemm done\n", rank, rank);
  wait_ready(flags(rank).gemm_done_ptr(rank));  // self gemm done
  print_per_kernel("%d wait segment %d copy done\n", rank, rank);
  wait_ready(flags(rank).copy_done_ptr(rank));  // intra NUMA last copy
  add_continous_kernel<T>(data_ptr(rank, rank), reduce_ptr(rank, rank), elems_per_rank);
  wait_ready(flags(rank).copy_done_ptr(rank_from));  // inter NUMA flag
  print_per_kernel("%d wait segment %d copy done\n", rank, rank_from);
  add_continous_kernel<T>(data_ptr(rank, rank), reduce_ptr(rank, rank_from), elems_per_rank);
}

template <typename T>
CUTLASS_GLOBAL void
run_per_segment_kernel_multinode(ReduceScatterParams params) {
  using Barrier = BarrierWithFallback;
  nanosleep(params.sleep_ns);  // first wave costs: 2.5e3 / 28 ~= 100us
  auto grid_group = cooperative_groups::this_grid();
  // false for 0 <- 1 <- 2 <- 3, true for 1 -> 2 -> 3
  int rank = params.rank;
  int local_world_size = params.world_size / params.nnodes;
  int rank_local = rank % local_world_size, node_id = rank / local_world_size;
  int rank_local_numa = rank_local % kNumaWorldSize, numa_id = rank_local / kNumaWorldSize;
  bool ring_order_asc = numa_id == 1;

  int rank_prev = (rank_local_numa + kNumaWorldSize - 1) % kNumaWorldSize +
                  numa_id * kNumaWorldSize + node_id * local_world_size;
  int rank_next = (rank_local_numa + 1) % kNumaWorldSize + numa_id * kNumaWorldSize +
                  node_id * local_world_size;
  int rank_to = ring_order_asc ? rank_next : rank_prev;
  int rank_from = ring_order_asc ? rank_prev : rank_next;
  int rank_to_next_numa = (rank + kNumaWorldSize) % kLocalWorldSize + node_id * local_world_size;
  int num_segments = params.n_split * params.world_size;
  auto flags = [&](int rank) {
    return PerRankFlagsWrapper(params.barrier_ptr[rank], num_segments);
  };

  int m_per_rank = params.m / params.world_size;
  int elems_per_rank = m_per_rank * params.n;
  auto data_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.scatter_ptr_aux[rank] + segment * elems_per_rank;
  };
  auto reduce_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.reduce_ptr[rank] + segment * elems_per_rank;
  };
  auto wait_ready = [&](int *ptr) {
#if defined(FLUX_DEBUG_RS)
    if (!params.do_wait)
      return;
#endif
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
      atomic_add_release_sys(ptr, 1);
    }
  };
  auto get_send_segment_inner_numa = [&](int iter) -> int {
    return ring_order_asc ? (rank_local_numa - 1 - iter + kNumaWorldSize) % kNumaWorldSize
                          : (rank_local_numa + iter + 1) % kNumaWorldSize;
  };
  for (int k = 0; k < params.nnodes; k++) {
    int node_iter = (k + node_id + 1) % params.nnodes;
    // calculate inner NUMA
    int segment_send_inter_node = node_iter * local_world_size;
    int segment_send_inter = (numa_id + 1) % kNumaNodes * kNumaWorldSize;
    for (int i = 0; i < kNumaWorldSize; i++) {
      int segment_send = segment_send_inter + get_send_segment_inner_numa(i);
      segment_send += segment_send_inter_node;  // add node offset
      // replace with printf
      print_per_kernel("[%d] wait segment %d gemm done\n", i, segment_send);
      wait_ready(flags(rank).gemm_done_ptr(segment_send));
      T *src = data_ptr(rank, segment_send);
      T *dst = reduce_ptr(rank_to, segment_send);
      if (i != 0) {
        print_per_kernel("[%d] wait segment %d copy done\n", i, segment_send);
        wait_ready(flags(rank).copy_done_ptr(segment_send));
        add_continous_kernel<T>(src, reduce_ptr(rank, segment_send), elems_per_rank);
      }
      if (i == kNumaWorldSize - 1) {
        break;  // reduce only
      }
#if defined(FLUX_DEBUG_RS)
      if (params.do_copy)
#endif
        copy_continous_kernel(dst, src, elems_per_rank);
      print_per_kernel("[%d] set %d segment %d copy done\n", i, rank_to, segment_send);
      set_ready(flags(rank_to).copy_done_ptr(segment_send));
    }

    int send_iter = 0;
    int rank_next_numa_from = (rank_to + kNumaWorldSize) % kLocalWorldSize;
    segment_send_inter = numa_id * kNumaWorldSize;
    for (int i = 0; i < kNumaWorldSize; i++) {
      bool send_cross_numa =
          ((ring_order_asc ? (i - 1 + kNumaWorldSize) % kNumaWorldSize : i) == rank_local_numa);
      int segment_send = send_cross_numa
                             ? (rank + kNumaWorldSize) % local_world_size
                             : (segment_send_inter + get_send_segment_inner_numa(send_iter));
      int segment_recv = send_cross_numa ? rank_next_numa_from : segment_send;
      segment_send += segment_send_inter_node;
      // remote_reduce_ptr(rank) is not copied to for intra copy, used as inter copy buffer.
      segment_recv += segment_send_inter_node;
      T *src = data_ptr(rank, segment_send);
      T *dst = send_cross_numa ? reduce_ptr(rank_to_next_numa, segment_recv)
                               : reduce_ptr(rank_to, segment_recv);

      if (!send_cross_numa) {
        print_per_kernel("[%d] wait segment %d gemm done\n", i, segment_send);
        wait_ready(flags(rank).gemm_done_ptr(segment_send));
        if (send_iter != 0) {
          print_per_kernel("[%d] wait segment %d copy done\n", i, segment_send);
          wait_ready(flags(rank).copy_done_ptr(segment_send));
          add_continous_kernel<T>(src, reduce_ptr(rank, segment_send), elems_per_rank);
        }
      }
      if (i != 0) {
      }

#if defined(FLUX_DEBUG_RS)
      if (params.do_copy)
#endif
        copy_continous_kernel(dst, src, elems_per_rank);

      if (send_cross_numa) {
        int rank_to_inter = rank_to_next_numa;
        print_per_kernel(
            "[%d] set %d segment %d copy done cross numa\n", i, rank_to_inter, segment_recv);
        set_ready(flags(rank_to_inter).copy_done_ptr(segment_recv));
      } else {
        print_per_kernel("[%d] set %d segment %d copy done\n", i, rank_to, segment_recv);
        set_ready(flags(rank_to).copy_done_ptr(segment_recv));
      }

      if (!send_cross_numa) {
        send_iter++;
      }
    }

    int rank_next_node = rank_local + segment_send_inter_node;
    print_per_kernel("%d wait segment %d gemm done\n", rank, rank_next_node);
    wait_ready(flags(rank).gemm_done_ptr(rank_next_node));  // self gemm done
    print_per_kernel("%d wait segment %d copy done\n", rank, rank_next_node);
    wait_ready(flags(rank).copy_done_ptr(rank_next_node));  // intra NUMA last copy
    add_continous_kernel<T>(
        data_ptr(rank, rank_next_node), reduce_ptr(rank, rank_next_node), elems_per_rank);
    int rank_from_curr_iter = rank_from % local_world_size + segment_send_inter_node;
    wait_ready(flags(rank).copy_done_ptr(rank_from_curr_iter));  // inter NUMA flag
    print_per_kernel("%d wait segment %d copy done\n", rank, rank_from_curr_iter);
    add_continous_kernel<T>(
        data_ptr(rank, rank_next_node), reduce_ptr(rank, rank_from_curr_iter), elems_per_rank);
    // set remote node ready

    if (k != params.nnodes - 1) {
      int *ptr = flags(rank).remote_copy_done_ptr(rank);  // symetric ptr
      grid_group.sync();
      if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef FLUX_SHM_USE_NVSHMEM
        nvshmem_int_atomic_set(ptr, 1, rank_next_node);
#endif
      }
    }
  }
#ifndef FLUX_REDUCE_SCATTERT_WITH_NCCL  // don't use this. too slow. don't know why
  for (int i = 1; i < params.nnodes; i++) {
    int segment = (rank + i * local_world_size) % params.world_size;
    print_per_kernel("[%d] %d wait segment %d remote copy\n", i, rank, segment);
    wait_ready(flags(rank).remote_copy_done_ptr(segment));
    print_per_kernel("[%d] %d wait segment %d remote copy done\n", i, rank, segment);
    int reduce_unused_segment =
        (rank_from + kNumaNodes + i * local_world_size) % params.world_size;
#ifdef FLUX_SHM_USE_NVSHMEM
    nvshmem_getmem(
        reduce_ptr(rank, reduce_unused_segment),
        data_ptr(rank, rank),
        elems_per_rank * sizeof(T),
        segment);
#endif
    add_continous_kernel(
        data_ptr(rank, rank), reduce_ptr(rank, reduce_unused_segment), elems_per_rank);
  }
#endif
}

struct LaunchProp {
  int max_num_blocks;
  int max_threads;
  int max_shared_memory_size;
};

LaunchProp
GetLaunchProp(void *func) {
  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int supports_cooplaunch = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&supports_cooplaunch, cudaDevAttrCooperativeLaunch, dev));
  FLUX_CHECK(supports_cooplaunch);

  cudaFuncAttributes attr;
  CUDA_CHECK(cudaFuncGetAttributes(&attr, func));

  /// This will launch a grid that can maximally fill the GPU, on the default stream with
  /// kernel arguments
  int num_blocks_per_sm = 0;
  // Number of threads my_kernel will be launched with
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks_per_sm, func, attr.maxThreadsPerBlock, 0);
  int max_num_blocks = device_prop.multiProcessorCount * num_blocks_per_sm;
  int max_threads = attr.maxThreadsPerBlock;
  int max_shared_memory_size = attr.maxDynamicSharedSizeBytes;
  fprintf(
      stderr,
      "launch property: %d x %d with %d\n",
      max_num_blocks,
      max_threads,
      max_shared_memory_size);
  // launch
  return {max_num_blocks, max_threads, max_shared_memory_size};
}

template <typename T>
cutlass::Status
run_per_segment_with_cudaMemcpyAsync(const ReduceScatterParams &params, cudaStream_t stream) {
  if (params.world_size == 1) {
    return cutlass::Status::kSuccess;
  }
  int rank_to = (params.rank + params.world_size - 1) % params.world_size;
  // int rank_from = (params.rank + 1) % params.world_size;
  int rank = params.rank;
  int m_per_rank = params.m / params.world_size;
  int num_segments = params.n_split * params.world_size;
  auto flags = [&](int rank) {
    return PerRankFlagsWrapper(params.barrier_ptr[rank], num_segments);
  };
  size_t nbytes = m_per_rank * params.n * sizeof(T);
  auto data_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.scatter_ptr_aux[rank] + segment * m_per_rank * params.n;
  };
  auto reduce_ptr = [&](int rank, int segment) -> T * {
    return (T *)params.reduce_ptr[rank] + segment * m_per_rank * params.n;
  };
  FLUX_CHECK(params.world_size <= 8);  // no NUMA trick
  auto wait_ready = [=](CUdeviceptr ptr) {
    CU_CHECK(cuda_stub().cuStreamWaitValue32_v2(stream, ptr, 1, CU_STREAM_WAIT_VALUE_EQ));
  };
  auto set_ready = [=](CUdeviceptr ptr) {
    CU_CHECK(cuda_stub().cuStreamWriteValue32_v2(stream, ptr, 1, CU_STREAM_WRITE_VALUE_DEFAULT));
  };

  // calculate
  for (int i = 0; i < params.world_size; i++) {
    int segment_send = (rank + i + 1) % params.world_size;
    wait_ready((CUdeviceptr)flags(rank).gemm_done_ptr(segment_send));
    T *src = data_ptr(rank, segment_send);
    T *dst = reduce_ptr(rank_to, segment_send);
    if (i != 0) {
      wait_ready((CUdeviceptr)flags(rank).copy_done_ptr(segment_send));
      // reduce to current segment
      add_continous<T><<<params.num_blocks, 1024, 0, stream>>>(
          src, reduce_ptr(rank, segment_send), m_per_rank * params.n);
      CUDA_CHECK(cudaGetLastError());
    }
    if (i == params.world_size - 1) {
      break;
    }
    CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, stream));
    set_ready((CUdeviceptr)flags(rank_to).copy_done_ptr(segment_send));
  }
  return cutlass::Status::kSuccess;
}

}  // namespace

template <typename T>
struct ReduceScatterOpBase {
  struct Arguments {
    int rank;
    int world_size;
    int nnodes;
    int m, n;
    int num_blocks = 12;
    T **output_scatter_ptrs;
    int **barrier_ptrs = nullptr;
    T *reduce_buffer_ptr = nullptr;
    void *rs_stream = nullptr;
    void *event = nullptr;
    bool use_gemmk = true;
    bool use_barrier_queue = false;
    bool per_tile_flags = true;
    bool use_cudaMemcpyAsync = false;
    int n_split = 1;
    int sub_world_size;
    void *opaque = nullptr;
    bool use_1d_ring = false;
    bool use_p2p_read = false;
    void *args_workspace = nullptr;
  };
};

template <typename T, int kM, int kN, bool kFlattenTile = false>
class ReduceScatterOp : public ReduceScatterOpBase<T> {
 public:
  using Arguments = typename ReduceScatterOpBase<T>::Arguments;

 private:
  static constexpr int kMN = kM * kN;
  static constexpr int kVecLen = sizeof(VecType) / sizeof(T);

 public:
  cutlass::Status
  initialize(const Arguments &args) {
    static int duration_ns = get_int_from_env("FLUX_RS_SLEEP_NS", 200000);
#ifdef FLUX_DEBUG_RS
    static bool do_copy = get_bool_from_env("FLUX_RS_DO_COPY", true);
    static bool do_wait = get_bool_from_env("FLUX_RS_DO_WAIT", true);
    static bool run_local = get_bool_from_env("FLUX_RS_RUN_LOCAL", true);
    static bool run_remote = get_bool_from_env("FLUX_RS_RUN_REMOTE", true);
#endif
    params_ = ReduceScatterParams{
        .rank = args.rank,
        .world_size = args.world_size,
        .nnodes = args.nnodes,
        .m = args.m,
        .n = args.n,
        .num_blocks = args.num_blocks,
        .sleep_ns = duration_ns,
#ifdef FLUX_DEBUG_RS
        .run_local = run_local,
        .run_remote = run_remote,
        .do_copy = do_copy,
        .do_wait = do_wait,
#endif
        .use_barrier_queue = args.use_barrier_queue,  // complex but no better
        .use_gemmk = args.use_gemmk,
        .per_tile_flags = args.per_tile_flags,
        .use_cudaMemcpyAsync = args.use_cudaMemcpyAsync,
        .n_split = args.n_split,
        .sub_world_size = args.sub_world_size,
        .opaque = args.opaque,
        .use_1d_ring = args.use_1d_ring,
        .use_p2p_read = args.use_p2p_read,
        .args_workspace = args.args_workspace,
    };
    for (int i = 0; i < params_.world_size; i++) {
#ifdef FLUX_SHM_USE_NVSHMEM
      // PCIE relays on nvshmem
      params_.reduce_ptr[i] = nvshmem_ptr(args.reduce_buffer_ptr, i);
#endif
      params_.scatter_ptr_aux[i] = args.output_scatter_ptrs[i];
      params_.barrier_ptr[i] = args.barrier_ptrs[i];
    }
    FLUX_CHECK(params_.world_size % params_.nnodes == 0);
    FLUX_CHECK(params_.m % params_.world_size == 0)
        << " m: " << params_.m << " world_size:" << params_.world_size;
    return cutlass::Status::kSuccess;
  }

  cutlass::Status
  update(const Arguments &args) {
    return initialize(args);
  }

  void
  operator()(void *stream) {
    run(stream);
  }

  cutlass::Status
  run_per_segment_with_cuda_core(cudaStream_t stream) {
    if (params_.world_size == 1) {
      return cutlass::Status::kSuccess;
    }

    void *args[1] = {(void *)&params_};
    static LaunchProp prop_intra_numa = GetLaunchProp((void *)run_per_segment_kernel<T>);
    static LaunchProp prop_intra_node = GetLaunchProp((void *)run_per_segment_kernel_tp8<T>);
    static LaunchProp prop_inter_node = GetLaunchProp((void *)run_per_segment_kernel_multinode<T>);
    void *func = params_.use_1d_ring
                     ? (void *)run_per_segment_kernel<T>
                     : (params_.nnodes == 1 ? (void *)run_per_segment_kernel_tp8<T>
                                            : (void *)run_per_segment_kernel_multinode<T>);
    auto &prop = params_.world_size <= kNumaWorldSize
                     ? prop_intra_numa
                     : (params_.nnodes == 1 ? prop_intra_node : prop_inter_node);

    CUDA_CHECK(cudaLaunchCooperativeKernel(
        func,
        dim3(min(prop.max_num_blocks, params_.num_blocks)),
        dim3(prop.max_threads),
        &args[0],
        prop.max_shared_memory_size,
        stream));
    auto wait_ready = [=](CUdeviceptr ptr) {
      CU_CHECK(cuda_stub().cuStreamWaitValue32_v2(stream, ptr, 1, CU_STREAM_WAIT_VALUE_EQ));
    };

#ifdef FLUX_REDUCE_SCATTERT_WITH_NCCL
    if (params_.opaque != nullptr && params_.nnodes > 1) {
      int rank = params_.rank;
      int world_size = params_.world_size;
      int local_world_size = world_size / params_.nnodes;
      int local_rank = rank % local_world_size;
      int node_id = rank / local_world_size;
      int numa_id = local_rank / kNumaWorldSize;
      int rank_numa_local = local_rank % kNumaWorldSize;
      int rank_prev = (rank_numa_local - 1 + kNumaWorldSize) % kNumaWorldSize;
      rank_prev += numa_id * kNumaWorldSize + node_id * local_world_size;
      int rank_next = (rank_numa_local + 1) % kNumaWorldSize;
      rank_next += numa_id * kNumaWorldSize + node_id * local_world_size;
      int rank_from = numa_id == 0 ? rank_next : rank_prev;

      int m_per_rank = params_.m / world_size;
      int num_segments = world_size * params_.n_split;
      int elems_per_rank = params_.n * m_per_rank;

      auto flag = PerRankFlagsWrapper(params_.barrier_ptr[rank], num_segments);

      auto data_ptr = [&](int rank, int segment) -> T * {
        return (T *)params_.scatter_ptr_aux[rank] + segment * m_per_rank * params_.n;
      };
      auto reduce_ptr = [&](int rank, int segment) -> T * {
        return (T *)params_.reduce_ptr[rank] + segment * m_per_rank * params_.n;
      };

      ncclComm_t nccl_comm = (ncclComm_t)params_.opaque;
      for (int i = 1; i < params_.nnodes; i++) {
        int send_segment = (rank + i * local_world_size) % world_size;
        int recv_segment = (rank - i * local_world_size + world_size) % world_size;
        int to_rank = send_segment;
        int from_rank = recv_segment;

        wait_ready((CUdeviceptr)flag.remote_copy_done_ptr(recv_segment));
        int reduce_unused_segment = (rank_from + kNumaNodes + i * local_world_size) % world_size;
        ncclGroupStart();
        NCCL_CHECK(ncclSend(
            data_ptr(rank, send_segment),
            elems_per_rank * sizeof(T),
            ncclInt8,
            to_rank,
            nccl_comm,
            stream));
        NCCL_CHECK(ncclRecv(
            reduce_ptr(rank, reduce_unused_segment),
            elems_per_rank * sizeof(T),
            ncclInt8,
            from_rank,
            nccl_comm,
            stream));
        ncclGroupEnd();
      }
    }
#endif
    return cutlass::Status::kSuccess;
  }

  cutlass::Status
  run(cudaStream_t stream) {
    if (params_.world_size == 1) {
      return cutlass::Status::kSuccess;
    }
    if (params_.per_tile_flags) {
      FLUX_CHECK(params_.nnodes == 1);
      if (params_.use_gemmk) {
        if (params_.use_barrier_queue) {
          static int counter = 0;
          if (counter++ == 0) {
            std::cerr << "[warning] use_barrier_queue is not supported for gemmk\n";
          }
        }
        // ignore use_cudaMemcpyAsync and n_spit arguments.
        if (params_.use_1d_ring) {
          if (params_.use_p2p_read) {
            using rskernel = ReduceScatterRing1dPullGemmk<T, kM, kN, kFlattenTile>;
            static std::pair<int, int> attr = get_func_attr((void *)cutlass::Kernel2<rskernel>);
            auto [num_threads, shmsize] = attr;  // wait for c++2a to support static unfold
            cutlass::Kernel2<rskernel>
                <<<params_.num_blocks, num_threads, shmsize, stream>>>(params_);
          } else {
            using rskernel = ReduceScatterRing1dPushGemmk<T, kM, kN, kFlattenTile>;
            static std::pair<int, int> attr = get_func_attr((void *)cutlass::Kernel2<rskernel>);
            auto [num_threads, shmsize] = attr;  // wait for c++2a to support static unfold
            cutlass::Kernel2<rskernel>
                <<<params_.num_blocks, num_threads, shmsize, stream>>>(params_);
          }
        } else {
          if (params_.use_p2p_read) {
            using rskernel = ReduceScatterRing2dPullGemmk<T, kM, kN, kFlattenTile>;
            static std::pair<int, int> attr = get_func_attr((void *)cutlass::Kernel2<rskernel>);
            auto [num_threads, shmsize] = attr;
            cutlass::Kernel2<rskernel>
                <<<params_.num_blocks, num_threads, shmsize, stream>>>(params_);
          } else {
            using rskernel = ReduceScatterRing2dPushGemmk<T, kM, kN, kFlattenTile>;
            static std::pair<int, int> attr = get_func_attr((void *)cutlass::Kernel2<rskernel>);
            auto [num_threads, shmsize] = attr;
            cutlass::Kernel2<rskernel>
                <<<params_.num_blocks, num_threads, shmsize, stream>>>(params_);
          }
        }
        CUDA_CHECK(cudaGetLastError());
        return cutlass::Status::kSuccess;
      } else {
        if (params_.world_size == 8) {
          if (params_.use_barrier_queue) {
            FLUX_CHECK(params_.m % (kM * params_.world_size) == 0);
            using rskernel = ReduceScatterRing2dPullWithQueue<T, kM, kN, kFlattenTile>;
            static std::pair<int, int> attr = get_func_attr((void *)cutlass::Kernel2<rskernel>);
            auto [num_threads, shmsize] = attr;
            cutlass::Kernel2<rskernel>
                <<<params_.num_blocks, num_threads, shmsize, stream>>>(params_);
          } else {
            using rskernel = ReduceScatterRing2dPull<T, kM, kN, kFlattenTile>;
            static std::pair<int, int> attr = get_func_attr((void *)cutlass::Kernel2<rskernel>);
            auto [num_threads, shmsize] = attr;
            cutlass::Kernel2<rskernel>
                <<<params_.num_blocks, num_threads, shmsize, stream>>>(params_);
          }
          CUDA_CHECK(cudaGetLastError());
          return cutlass::Status::kSuccess;
        }
      }
    } else {
      if (params_.use_cudaMemcpyAsync) {
        fprintf(stderr, "run with use_cudaMemcpyAsync\n");
        return run_per_segment_with_cudaMemcpyAsync<T>(params_, stream);
      } else {
        return run_per_segment_with_cuda_core(stream);
      }
    }
    return cutlass::Status::kErrorInternal;
  }

 public:
  static std::pair<int, int>
  get_func_attr(void *func) {
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, func));

    int max_blocks = -1;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &max_blocks, func, 1024, attr.maxDynamicSharedSizeBytes, cudaOccupancyDefault));
#if defined(FLUX_DEBUG)
    std::cout << " maxDynamicSharedSizeBytes: " << attr.maxDynamicSharedSizeBytes
              << " maxThreadsPerBlock: " << attr.maxThreadsPerBlock << " numRegs: " << attr.numRegs
              << " preferredShmemCarveout: " << attr.preferredShmemCarveout
              << " sharedSizeBytes: " << attr.sharedSizeBytes
              << " MaxActiveBlocksPerMultiprocessor: " << max_blocks << "\n";
#endif
    int num_threads = attr.maxThreadsPerBlock;
    if (kMN % (kVecLen * num_threads) != 0) {
      bool found = false;
      for (int i = kMN / (kVecLen * num_threads) + 1; i < kMN / kVecLen / 32; i++) {
        if (kMN % (kVecLen * i) == 0) {
          num_threads = kMN / (kVecLen * i);
          if (num_threads % (kM / kVecLen) == 0) {
            found = true;
            break;
          }
        }
      }
      FLUX_CHECK(found);
    }
#if defined(FLUX_DEBUG)
    std::cerr << "set num_threads to " << num_threads << "\n";
#endif
    return {num_threads, attr.maxDynamicSharedSizeBytes};
  };

 private:
  ReduceScatterParams params_;
};

}  // namespace bytedance::flux

#undef NextRank
#undef NextLocalRank
#undef PrevLocalRank
#undef NextNodeRank
