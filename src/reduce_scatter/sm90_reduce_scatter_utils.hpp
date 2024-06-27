//===- sm90_reduce_scatter_utils.hpp ------------------------------ C++ ---===//
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
#include "cute/arch/cluster_sm90.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/int.hpp"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/epilogue/collective/detail.hpp"
#include "flux/flux.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/memory_utils.hpp"
#ifdef FLUX_SHM_USE_NVSHMEM
#include "host/nvshmemx_api.h"
#endif
namespace bytedance::flux {

using namespace cute;

template <
    int Stages,
    class TileShape_,
    class EpilogueTile_,
    class SmemLayoutAtom_,
    class Element_,
    class StrideMNL_,
    CommKindEnum CommKind_>
struct Sm90ReduceScatterDma {
 public:
  // Type aliases
  using TileShape = TileShape_;
  using EpilogueTile = EpilogueTile_;
  using SmemLayoutAtom = SmemLayoutAtom_;
  using Element = Element_;
  using StrideMNL = StrideMNL_;
  static constexpr CommKindEnum CommKind = CommKind_;
  static constexpr int kAlignment = 128 / sizeof_bits_v<Element>;

  // Shared Mem
  constexpr static bool is_m_major =
      cutlass::epilogue::collective::detail::is_m_major<StrideMNL>();
  static_assert(not is_m_major, "only support n-major now");
  // Find the max contiguous layout usable by TMA (if EpilogueTile is a non-compact tiler)
  using SmemShapeTma = decltype(make_shape(
      max_common_vector(make_layout(get<0>(EpilogueTile{})), make_layout(get<0>(EpilogueTile{}))),
      max_common_vector(
          make_layout(get<1>(EpilogueTile{})), make_layout(get<1>(EpilogueTile{})))));
  using SmemLayoutTma = decltype(tile_to_shape(
      SmemLayoutAtom{},
      SmemShapeTma{},
      cute::conditional_t<is_m_major, Step<_2, _1>, Step<_1, _2>>{}));
  using SmemLayout = decltype(tile_to_shape(
      SmemLayoutTma{},
      make_shape(size<0>(shape(EpilogueTile{})), size<1>(shape(EpilogueTile{})), Int<Stages>{}),
      cute::conditional_t<is_m_major, Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  using StrideReduce = conditional_t<
      CommKind == _AcrossNode{},
      decltype(append<5>(StrideMNL{}, int64_t(0))),
      StrideMNL>;

  using ShapeReduce = conditional_t<
      CommKind == _AcrossNode{},
      decltype(make_shape(
          size<0>(TileShape{}), size<1>(TileShape{}), int32_t(0), int32_t(0), int32_t(0))),
      decltype(make_shape(int32_t(0), int32_t(0), int32_t(0)))>;

  // used for tma load copy
  using FetchPipeline = cutlass::PipelineTransactionAsync<Stages>;
  using PipelineState = cutlass::PipelineState<Stages>;
  using PipelineParams = typename FetchPipeline::Params;
  static constexpr int TmaTransactionBytes =
      size<0>(EpilogueTile{}) * size<1>(EpilogueTile{}) * sizeof(Element);

  static constexpr int ThreadCount = 32;

  struct Arguments {
    Element **output_scatter_ptrs;
    StrideMNL stride;
    int rank = 0;
    int world_size = 0;
    int nnodes = 1;
    void *local_reduce_buffer = nullptr;
    int **barrier_ptrs;
  };

  struct Params {
    using TMA_Fetch = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(
            static_cast<Element const *>(nullptr),
            repeat_like(StrideMNL{}, int32_t(0)),
            StrideMNL{}),
        SmemLayoutTma{}));

    tuple<int, int> problem_shape;
    int rank;
    int world_size;
    int local_rank;
    int local_world_size;
    int node_idx;
    int nnodes;
    int tile_m_perrank;

    StrideMNL stride;
    StrideReduce dReduce;
    ShapeReduce sReduce;

    Element *local_ptr[kMaxLocalWorldSize];
    TMA_Fetch tma_load_fetch[kMaxLocalWorldSize];
    Element *local_reduce_buffer;
    int *local_barrier_ptr[kMaxLocalWorldSize];
    Layout<Shape<int, int>> tile_layout;
  };

  struct TensorStorage {
    alignas(cutlass::detail::alignment_for_swizzle(SmemLayout{}))
        array_aligned<Element, size(SmemLayout{})> tensor;
  };

  using PipelineStorage = typename FetchPipeline::SharedStorage;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const &problem_shape, Arguments const &args) {
    Params params;
    auto [M, N, K] = problem_shape;
    constexpr int L = 1;

    params.rank = args.rank;
    params.world_size = args.world_size;
    params.nnodes = args.nnodes;

    FLUX_CHECK(params.nnodes <= kMaxWorldSize / kMaxLocalWorldSize);
    FLUX_CHECK(params.world_size % params.nnodes == 0);
    params.local_world_size = params.world_size / params.nnodes;
    FLUX_CHECK(params.nnodes <= params.local_world_size) << " not supported yet.";

    params.local_rank = params.rank % params.local_world_size;
    params.node_idx = params.rank / params.local_world_size;
    FLUX_CHECK(params.local_world_size <= kMaxLocalWorldSize);

    auto [tile_M, tile_N, tile_K] = TileShape{};

    FLUX_CHECK(M % tile_M == 0) << "M=" << M << " tile_M=" << tile_M;
    FLUX_CHECK(N % tile_N == 0) << "N=" << N << " tile_N=" << tile_N;
    FLUX_CHECK(M % (tile_M * params.world_size) == 0)
        << "M=" << M << " tile_M=" << tile_M << " world_size=" << params.world_size;

    FLUX_CHECK(args.barrier_ptrs != nullptr);
    FLUX_CHECK(args.local_reduce_buffer != nullptr);

    params.tile_m_perrank = M / (tile_M * params.world_size);
    params.stride = args.stride;

    int M_reduce = params.tile_m_perrank * params.nnodes * params.nnodes * tile_M;

    if constexpr (CommKind == _AcrossNode{}) {
      params.sReduce = make_shape(tile_M, tile_N, M_reduce / tile_M, N / tile_N, L);
      // NOTE: stride of L may not be correct if L>1
      params.dReduce = make_stride(
          int64_t(tile_N), Int<1>{}, tile_M * N, int64_t(tile_M * tile_N), get<2>(params.stride));
    } else {
      params.dReduce = args.stride;
      params.sReduce = make_shape(M_reduce, N, L);
    }

    params.local_reduce_buffer = static_cast<Element *>(args.local_reduce_buffer);
    for (int local_rank = 0; local_rank < params.local_world_size; ++local_rank) {
      int global_rank = params.node_idx * params.local_world_size + local_rank;
      Element *ptr = static_cast<Element *>(args.output_scatter_ptrs[global_rank]);
      int *barrier_ptr = reinterpret_cast<int **>(args.barrier_ptrs)[global_rank];
      FLUX_CHECK(barrier_ptr != nullptr);
      params.local_ptr[local_rank] = ptr;
      params.local_barrier_ptr[local_rank] = barrier_ptr;

      auto tensor_fetch = make_tensor(ptr, make_layout(make_shape(M, N, L), args.stride));
      auto tma = make_tma_copy(SM90_TMA_LOAD{}, tensor_fetch, SmemLayoutTma{});
      params.tma_load_fetch[local_rank] = cute::move(tma);
    }

    int m_tiles = ceil_div(M, size<0>(TileShape{}));
    int n_tiles = ceil_div(N, size<1>(TileShape{}));
    params.tile_layout = make_layout(make_shape(m_tiles, n_tiles));
    return params;
  }

  const Params *params_ptr;
  Element *smem_tensor;

  CUTLASS_HOST_DEVICE
  Sm90ReduceScatterDma() {}

  CUTLASS_HOST_DEVICE
  Sm90ReduceScatterDma(Params const &params, TensorStorage const &shared_tensor)
      : params_ptr(&params), smem_tensor(const_cast<Element *>(shared_tensor.tensor.data())) {}

  template <class T>
  CUTLASS_DEVICE void
  debug_print_v(T const &val, char const *name) {
    if (params_ptr->local_rank == 0 and block0() and threadIdx.x == 64) {
      print("%s:", name);
      print(val);
      print("\n");
    }
  }

  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  fetch(
      FetchPipeline fetch_pipeline,
      PipelineState fetch_write_state,
      ProblemShapeMNKL const &problem_shape,
      TileCoordMNKL const &tile_coord) {
    using namespace cute;
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      // early exit if out of bound
      return fetch_write_state;
    }

    int thread_idx = cutlass::canonical_lane_idx();
    int src_rank = m / params_ptr->tile_m_perrank;
    int local_src_rank = src_rank % params_ptr->local_world_size;
    int m_fetch = m + (params_ptr->local_rank - local_src_rank) * params_ptr->tile_m_perrank;

    /////////////////// Fetch Tensors ////////////////////
    Tensor mFetch =
        params_ptr->tma_load_fetch[local_src_rank].get_tma_tensor(make_shape(M, N, L));  // (M,N,L)
    Tensor gFetch =
        local_tile(mFetch, take<0, 2>(TileShape{}), make_coord(m_fetch, n, l));  // (TILE_M,TILE_N)
    Tensor gFetch_epi =
        flat_divide(gFetch, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
    Tensor sFetch_epi =
        make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});  // (EPI_TILE_M,EPI_TILE_N,PIPE)

    ThrCopy thrblk_g2s_fetch = params_ptr->tma_load_fetch[local_src_rank].get_slice(_0{});
    Tensor bGS_gFetch = thrblk_g2s_fetch.partition_S(gFetch_epi);
    Tensor bGS_sFetch = thrblk_g2s_fetch.partition_D(sFetch_epi);

    /////////////////// Process Loop ////////////////////
    // Predication for TMA load (one thread issues TMA load)
    bool issue_tma_load = cute::elect_one_sync();

    // wait for the tile to fetch ready before processing
    int fetch_tile_idx = params_ptr->tile_layout(m_fetch, n);

    using BarrierSync =
        cutlass::detail::NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::ReduceScatterFetch>;
    using Barrier = cutlass::detail::GenericSystemBarrier<BarrierSync>;

    Barrier::wait_eq_reset(
        params_ptr->local_barrier_ptr[local_src_rank], thread_idx, fetch_tile_idx * 2, 1);

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gFetch_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gFetch_epi); ++epi_m) {
        constexpr uint16_t mcast_mask = 0;
        uint64_t *tma_barrier = fetch_pipeline.producer_get_barrier(fetch_write_state);
        fetch_pipeline.producer_acquire(fetch_write_state);

        if (issue_tma_load) {
          copy(
              params_ptr->tma_load_fetch[local_src_rank].with(*tma_barrier, mcast_mask),
              bGS_gFetch(_, _, _, epi_m, epi_n),
              bGS_sFetch(_, _, _, fetch_write_state.index()));
          fetch_pipeline.producer_expect_transaction(fetch_write_state);
        }
        fetch_pipeline.producer_commit(fetch_write_state);
        ++fetch_write_state;
      }
    }
    return fetch_write_state;
  }

  CUTLASS_DEVICE auto
  fetch_tail(FetchPipeline fetch_pipeline, PipelineState fetch_write_state) {
    bool issue_tma_load = cute::elect_one_sync();
    if (issue_tma_load) {
      fetch_pipeline.producer_tail(fetch_write_state);
    }
  }

  template <class ProblemShapeMNKL, class TileCoordMNKL>
  CUTLASS_DEVICE auto
  reduce(
      FetchPipeline fetch_pipeline,
      PipelineState fetch_read_state,
      ProblemShapeMNKL const &problem_shape,
      TileCoordMNKL const &tile_coord) {
    auto [M, N, K, L] = problem_shape;
    auto [m, n, k, l] = tile_coord;

    if (m >= size<0>(params_ptr->tile_layout.shape()) or
        n >= size<1>(params_ptr->tile_layout.shape())) {
      // early exit if out of bound
      return fetch_read_state;
    }

    int thread_idx = cutlass::canonical_lane_idx();

    int dst_rank = m / params_ptr->tile_m_perrank;
    int local_dst_rank = dst_rank % params_ptr->local_world_size;
    int dst_node_idx = dst_rank / params_ptr->local_world_size;
    // the logical m coord of the reduction tile in the output buffer
    int m_reduce_in_output =
        m + (params_ptr->local_rank - local_dst_rank) * params_ptr->tile_m_perrank;
    // the actual m coord in reduce_buffer
    int m_reduce =
        (m % params_ptr->tile_m_perrank) +
        params_ptr->tile_m_perrank * (dst_node_idx * params_ptr->nnodes + params_ptr->node_idx);

    /////////////////// Reduce Tensors ////////////////////
    auto get_mReduce = [&]() {
      auto mReduce = make_tensor(
          params_ptr->local_reduce_buffer, make_layout(params_ptr->sReduce, params_ptr->dReduce));

      if constexpr (CommKind == _AcrossNode{}) {
        auto reduce_layout = mReduce.layout();
        auto new_reduce_layout = make_layout(
            select<0, 2>(reduce_layout), select<1, 3>(reduce_layout), select<4>(reduce_layout));
        Tensor mReduce_reshaped = make_tensor(mReduce.data(), new_reduce_layout);
        return mReduce_reshaped;
      } else {
        return mReduce;
      }
    };

    auto mReduce = get_mReduce();  // (M_reduce,N,L)
    Tensor gReduce = local_tile(
        mReduce, take<0, 2>(TileShape{}), make_coord(m_reduce, n, l));  // (TILE_M,TILE_N)
    Tensor gReduce_epi =
        flat_divide(gReduce, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

    Tensor sReduce_epi =
        make_tensor(make_smem_ptr(smem_tensor), SmemLayout{});  // (EPI_TILE_M,EPI_TILE_N,PIPE)

    // tiled copy for fetch from global memory from other rank to registers
    // each thread of the TiledMMA (256 threads for cooperative and 128 threads for pingpong
    // kernel) process contiguous Alignment values
    constexpr int ThreadLayoutN = size<1>(EpilogueTile{}) / kAlignment;
    constexpr int ThreadLayoutM = ThreadCount / ThreadLayoutN;

    auto tiled_copy = make_tiled_copy(
        Copy_Atom<DefaultCopy, Element>{},
        make_layout(
            make_shape(Int<ThreadLayoutM>{}, Int<ThreadLayoutN>{}),
            make_stride(Int<ThreadLayoutN>{}, _1{})),
        make_layout(make_shape(_1{}, Int<kAlignment>{}), make_stride(_0{}, _1{})));

    auto thread_copy = tiled_copy.get_slice(thread_idx);
    Tensor tsReduce = thread_copy.partition_S(sReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,PIPE)
    Tensor tgReduce =
        thread_copy.partition_D(gReduce_epi);  // ((Atom,AtomNum),ATOM_M,ATOM_N,EPI_M,EPI_N)

    using BarrierSync = cutlass::detail::
        NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::ReduceScatterReduce>;
    using Barrier = cutlass::detail::CustomizedGenericBarrier<BarrierSync>;

    int reduce_tile_idx = params_ptr->tile_layout(m_reduce_in_output, n);
    int *lock_ptr = params_ptr->local_barrier_ptr[params_ptr->local_rank];
    int flag_idx = reduce_tile_idx * 2 + 1;

    bool is_local_tile_reduce = local_dst_rank == params_ptr->local_rank;

    if (not is_local_tile_reduce) {
      // if this tile is fetched from other rank, wait for the local rank to reduce first
      Barrier::wait_lt(lock_ptr, thread_idx, flag_idx, 1);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int epi_n = 0; epi_n < size<3>(gReduce_epi); ++epi_n) {
      CUTLASS_PRAGMA_UNROLL
      for (int epi_m = 0; epi_m < size<2>(gReduce_epi); ++epi_m) {
        auto barrier_token = fetch_pipeline.consumer_try_wait(fetch_read_state);
        fetch_pipeline.consumer_wait(fetch_read_state, barrier_token);
        // do copy from smem to reg and reduce to gmem
        Tensor tsReduce_epi = tsReduce(_, _, _, fetch_read_state.index());
        Tensor tgReduce_epi = tgReduce(_, _, _, epi_m, epi_n);

        CUTLASS_PRAGMA_UNROLL
        for (int copy_m = 0; copy_m < size<1>(tgReduce_epi); ++copy_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int copy_n = 0; copy_n < size<2>(tgReduce_epi); ++copy_n) {
            Tensor trReduce = make_tensor<Element>(size<0>(tgReduce));
            // fetch from local_src_rank
            copy(tiled_copy, tsReduce_epi(_, copy_m, copy_n), trReduce);
            // write to reduce_buffer
            if (is_local_tile_reduce) {
              copy(tiled_copy, trReduce, tgReduce_epi(_, copy_m, copy_n));
            } else {
              using VecType = uint_byte_t<sizeof(trReduce)>;
              cutlass::arch::local_red<VecType, sizeof(Element) * kAlignment, Element>(
                  recast<VecType>(trReduce)(_0{}),
                  (void *)tgReduce_epi(_, copy_m, copy_n).data(),
                  true);
            }
          }
        }

        fetch_pipeline.consumer_release(fetch_read_state);
        ++fetch_read_state;
      }
    }

    int reduce_count = Barrier::arrive_inc_get(lock_ptr, thread_idx, flag_idx, 1);
    if (reduce_count == params_ptr->local_world_size) {
      Barrier::wait_eq_reset(lock_ptr, thread_idx, flag_idx, params_ptr->local_world_size, 0);
      if constexpr (CommKind == _AcrossNode{}) {
        if (dst_node_idx != params_ptr->node_idx) {
          int remote_rank = dst_node_idx * params_ptr->local_world_size + params_ptr->local_rank;
#ifdef FLUX_SHM_USE_NVSHMEM
          nvshmemx_putmem_nbi_warp(
              gReduce.data(), gReduce.data(), gReduce.size() * sizeof(Element), remote_rank);
#endif
        }
      }
    }
    return fetch_read_state;
  }
};

}  // namespace bytedance::flux
