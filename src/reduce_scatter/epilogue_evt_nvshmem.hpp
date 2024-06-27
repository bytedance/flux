//===- epilogue_evt_nvshmem.hpp ----------------------------------- C++ ---===//
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
// Some code from cutlass/epilogue/threadblock/fusion/visitor_store.hpp
// in NVIDIA cutlass project
// Original license as follows
/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
  \brief Visitor tree store operations for the CUTLASS 2x epilogue
*/

#pragma once

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "cutlass/barrier.h"
#include "flux/cuda/memory_utils.hpp"
#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;
using SystemBarrier = cutlass::detail::SystemBarrier;
using bytedance::flux::kMaxWorldSize;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ThreadMap,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape,
    bool FuseReduction>
struct VisitorAuxStoreScatterAccrossNode {
  struct Arguments {
    Element **scatter_ptr_aux;
    StrideMNL dAux = {};
    int64_t rank = 0;
    int64_t world_size = 1;
    SystemBarrier::T **barrier_ptr_aux;
    Element *remote_reduce_buffer;
    int64_t nnodes = 1;
  };

  struct Params {
    Element *scatter_ptr_aux[kMaxWorldSize];
    StrideMNL dAux;
    int64_t rank;
    int64_t world_size;
    SystemBarrier::T *barrier_ptr_aux[kMaxWorldSize];
    Element *remote_reduce_buffer;
    int64_t nnodes;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    Params params;
    params.dAux = args.dAux;
    params.rank = args.rank;
    params.world_size = args.world_size;
    params.remote_reduce_buffer = args.remote_reduce_buffer;
    params.nnodes = args.nnodes;
    for (int i = 0; i < args.world_size; i++) {
      params.scatter_ptr_aux[i] = args.scatter_ptr_aux[i];
      params.barrier_ptr_aux[i] = args.barrier_ptr_aux[i];
    }
    return params;
  }

  struct SharedStorage {};

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  static bool constexpr vec_bits_valid =
      (std::is_same_v<Element, cutlass::bfloat16_t> && FuseReduction && vec_bits == 16) or
      (!FuseReduction) or (!std::is_same_v<Element, cutlass::bfloat16_t>);
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static_assert(vec_bits_valid);
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatterAccrossNode() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatterAccrossNode(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params) {
    // if constexpr (std::is_same_v<Element, cutlass::bfloat16_t> and FuseReduction == true) {
    //   static_assert(vec_bits == 16);
    // }
  }

  Params const *params_ptr;

  /// Pad the given allocation size up to the nearest cache line
  static size_t
  cacheline_align_up(size_t size) {
    static const int CACHELINE_SIZE = 128;
    return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
  }

  /// Get the workspace size needed for barrier
  //   template <class ThreadblockShape>
  size_t
  get_barrier_workspace_size(int m, int n) const {
    // each threadblock tile needs a synchronization flag.
    int num_flags = ((m + ThreadblockShape::kM - 1) / ThreadblockShape::kM) *
                    ((n + ThreadblockShape::kN - 1) / ThreadblockShape::kN);

    return cacheline_align_up(sizeof(typename SystemBarrier::T) * num_flags);
  }

  template <class GTensor, class RTensor, class GTensor_R, class ProblemShape, class BlockShapeMNK>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tC_gAux,
        RTensor &&tC_rAux,
        GTensor_R &&tR_gAux,
        ProblemShape problem_shape,
        BlockShapeMNK blk_shape_mnk,
        Params const *params_ptr,
        SystemBarrier::T *barrier_ptr,
        void *ptr_local_dst,
        void *ptr_R,
        int thread_idx,
        int tile_idx,
        int remote_tile_idx,
        int local_rank,
        int local_dst_rank,
        int remote_dst_rank,
        int local_world_size,
        int m_coord,
        int n_coord,
        int new_m_coord)
        : tC_gAux(cute::forward<GTensor>(tC_gAux)),
          tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tR_gAux(cute::forward<GTensor_R>(tR_gAux)),
          problem_shape(problem_shape),
          blk_shape_mnk(blk_shape_mnk),
          params_ptr(params_ptr),
          barrier_ptr(barrier_ptr),
          ptr_local_dst(ptr_local_dst),
          ptr_R(ptr_R),
          thread_idx(thread_idx),
          tile_idx(tile_idx),
          remote_tile_idx(remote_tile_idx),
          local_rank(local_rank),
          local_dst_rank(local_dst_rank),
          remote_dst_rank(remote_dst_rank),
          local_world_size(local_world_size),
          m_coord(m_coord),
          n_coord(n_coord),
          new_m_coord(new_m_coord) {}

    GTensor tC_gAux;
    RTensor tC_rAux;
    GTensor_R tR_gAux;
    Params const *params_ptr;
    ProblemShape problem_shape;
    BlockShapeMNK blk_shape_mnk;
    SystemBarrier::T *barrier_ptr;
    int thread_idx;
    int tile_idx;
    int remote_tile_idx;
    int local_rank;
    int local_dst_rank;
    int remote_dst_rank;
    int local_world_size;
    int m_coord;
    int n_coord;
    int new_m_coord;
    void *ptr_R;
    void *ptr_local_dst;
    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tC_rAux);
    }

    template <class ElementAccumulator, class ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto  // returns an Array
    visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frg_idx,
        Array<ElementAccumulator, FragmentSize> const &frg_acc,
        Array<ElementInput, FragmentSize> const &frg_input) {
      using ConvertInput = NumericArrayConverter<Element, ElementInput, FragmentSize, RoundStyle>;
      ConvertInput convert_input{};

      Tensor tC_rAux_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rAux));
      tC_rAux_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }
    template <const int count, const int n_threads>
    CUTLASS_DEVICE void
    recast_backto_bf16(void *ptr) {
      // printf("blockDimx:%d\n", blockDim.x);
      assert(blockDim.x == n_threads);
      constexpr int n_per_access = 2;
      const int tid = threadIdx.x;
      constexpr int n_iter = count / n_per_access / n_threads;
      float2 *p_f = (float2 *)ptr;
      __nv_bfloat162 *p_b = (__nv_bfloat162 *)ptr;
      CUTLASS_PRAGMA_UNROLL
      for (int iter = 0; iter < n_iter; iter++) {
        float2 val = *(p_f + iter * n_threads + tid);
        __nv_bfloat162 new_v = __float22bfloat162_rn(val);
        __syncthreads();
        *(p_b + iter * n_threads + tid) = new_v;
      }
    }
    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tC_rAux);

      auto dst_v = filter(tC_gAux(_, _, _, step_idx));
      if constexpr (FuseReduction and std::is_same_v<Element, cutlass::half_t>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
              reinterpret_cast<VecType const &>(src_v(i)), (void *)&dst_v(i), 1);
        }
      } else if constexpr (FuseReduction and std::is_same_v<Element, cutlass::bfloat16_t>) {
        // use float to perform the reduction
        // Note: this need to double the size of the output buffer
        using Red_Vec = uint_bit_t<32>;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          __nv_bfloat16 val = *reinterpret_cast<__nv_bfloat16 *>((void *)&src_v(i));
          // convert to the float32
          float float32 = __bfloat162float(val);

          int offset = (char *)&dst_v(i) - (char *)ptr_local_dst;
          char *target_ptr = (char *)ptr_local_dst + 2 * offset;
          cutlass::arch::global_red<Red_Vec, sizeof(Red_Vec), float>(
              reinterpret_cast<Red_Vec const &>(float32), (void *)target_ptr, 1);
        }
      } else {
        // just copy to the corresponding position
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          dst_v[i] = src_v[i];
          // cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), (void*)&dst_v(i), 1);
        }
      }
    }

    CUTLASS_DEVICE void
    copy_remote(void *dst, void *src, int size, int dst_rank) {
#ifdef FLUX_SHM_USE_NVSHMEM
      nvshmemx_putmem_nbi_block(dst, src, size, dst_rank);
#endif
    }
    CUTLASS_DEVICE void
    end_epilogue() {
      if constexpr (FuseReduction) {
        // if fuse reduction, we use nvshmem directly send the reduced output
        using ReduceType =
            std::conditional_t<std::is_same_v<Element, cutlass::half_t>, half, float>;
        if (threadIdx.x == 0) {
          asm volatile(
              "fence.acq_rel.sys;\n");  // better performance compared to other kinds of fence
          int after_inc = atomicAdd_system(barrier_ptr + tile_idx, 1);
        }
        __syncthreads();
        // SystemBarrier::arrive_inc(barrier_ptr_local_dst, threadIdx.x, tile_idx);
        auto BM = get<0>(blk_shape_mnk);
        auto BN = get<1>(blk_shape_mnk);
        // wait for all local reduce done

        if (local_dst_rank == local_rank) {
          SystemBarrier::wait_eq(
              barrier_ptr,
              threadIdx.x,
              tile_idx,
              local_world_size);  // double check here
          void *src_ptr = static_cast<void *>(tile_idx * (BM * BN) + (ReduceType *)ptr_local_dst);
          void *dst_ptr = static_cast<void *>(remote_tile_idx * (BM * BN) + (half *)ptr_R);
          if constexpr (std::is_same_v<Element, cutlass::bfloat16_t>) {
            constexpr int N_COUNT = BM * BN;
            recast_backto_bf16<N_COUNT, 128>(src_ptr);
          }
          copy_remote(dst_ptr, src_ptr, BM * BN * 2, remote_dst_rank);
        }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    // TODO: support RMPad for the accross node RS in the future
    constexpr int local_world_size = 8;  // should be passed from parameters
    // constexpr int tile_size = ThreadblockShape::kM;
    constexpr int tile_size [[maybe_unused]] = ThreadblockShape::kM;
#ifdef FLUX_SHM_USE_NVSHMEM
    const int world_size = nvshmem_n_pes();
    const int rank = nvshmem_my_pe();
#else
    // only support InterNode when nvshmem is enabled
    const int world_size = -1;
    const int rank = -1;
#endif
    const int local_rank = rank % local_world_size;
    auto M = get<0>(problem_shape);
    auto N = get<1>(problem_shape);
    constexpr int BM = _128{};  // FIXME
    constexpr int BN = _128{};
    assert(BM == tile_size);
    using BlockShapeMNK = cute::Shape<_128, _128>;
    BlockShapeMNK blk_mnk = cute::make_shape(_128{}, _128{});

    const int m_coord = threadblock_tile_offset.m();
    const int n_coord = threadblock_tile_offset.n();
    const int coord_per_rank_local = (M / BM / local_world_size);
    const int coord_per_rank_global = (M / BM / world_size);
    const int local_m_block_idx = m_coord / coord_per_rank_global;
    const int local_rank_start = rank / local_world_size * local_world_size;
    int local_dst_rank = local_m_block_idx % local_world_size;
    const int n_nodes = world_size / local_world_size;
    const int nodeid = rank / local_world_size;
    const int remote_dst_node = local_m_block_idx / local_world_size;
    const int remote_dst_rank = remote_dst_node * local_world_size + local_rank;

    const int tile_idx =
        get<1>(problem_shape) / BN * threadblock_tile_offset.m() + threadblock_tile_offset.n();

    SystemBarrier::T *barrier_ptr =
        params_ptr->barrier_ptr_aux
            ? params_ptr->barrier_ptr_aux[local_dst_rank + local_rank_start]
            : nullptr;
    // assume we got a full dst tensor
    auto local_dst_offset = threadblock_tile_offset;
    auto remote_dst_offset = threadblock_tile_offset;
    const int new_m_coord = nodeid * coord_per_rank_global + m_coord % coord_per_rank_global;
    const int remote_tile_idx = new_m_coord * get<1>(problem_shape) / BN + n_coord;
    remote_dst_offset.m() = new_m_coord;
    auto ptr_R = params_ptr->remote_reduce_buffer;
    auto ptr_local_dst = params_ptr->scatter_ptr_aux[local_dst_rank + local_rank_start];

    Tensor mAux = make_tensor(
        make_gmem_ptr(ptr_local_dst),
        problem_shape,
        params_ptr->dAux);  // (M,N,L)

    Tensor rAux = make_tensor(make_gmem_ptr(ptr_R), problem_shape,
                              params_ptr->dAux);  // (M,N,L)

    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    static_assert(is_static<BlockShapeMNK>::value, "BlockShape MNK not static");
    Tensor tC_gAux = recast<VecType>(group_modes<3, 6>(ThreadMap::partition(
        mAux,
        problem_shape,
        blk_mnk,
        thread_idx,
        local_dst_offset,
        rank,
        world_size,
        n_nodes)));  // layout need to be updated
    Tensor tR_gAux = recast<VecType>(group_modes<3, 6>(ThreadMap::partition(
        rAux, problem_shape, blk_mnk, thread_idx, remote_dst_offset, rank, world_size, n_nodes)));

    Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));

    return Callbacks<
        decltype(tC_gAux),
        decltype(tC_rAux),
        decltype(tR_gAux),
        ProblemShape,
        BlockShapeMNK>(
        cute::move(tC_gAux),
        cute::move(tC_rAux),
        cute::move(tR_gAux),
        problem_shape,
        blk_mnk,
        params_ptr,
        barrier_ptr,
        ptr_local_dst,
        ptr_R,
        thread_idx,
        tile_idx,
        remote_tile_idx,
        local_rank,
        local_dst_rank,
        remote_dst_rank,
        local_world_size,
        m_coord,
        n_coord,
        new_m_coord);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
