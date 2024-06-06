//===- epilogue_evt.hpp ------------------------------------------- C++ ---===//
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

#include <cstddef>
#include <cstdio>
#include <device_atomic_functions.h>
#include "cute/algorithm/gemm.hpp"
#include "cute/int_tuple.hpp"
#include "cute/tensor.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/barrier.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm_coord.h"
#include "flux/cuda/memory_utils.hpp"
#include "flux/cuda/cuda_common_device.hpp"
#include "flux/flux.h"

#include "reduce_scatter/reduce_scatter_barrier_struct.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;
using SystemBarrier = cutlass::detail::SystemBarrier;
using bytedance::flux::kMaxWorldSize;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Store Scatter Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ThreadMap,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape,
    bool FuseReduction_,
    bool kPcieMode = false,  // kPcieMode controls write rank and write flags
    bool FlattenOutTile = false>
struct VisitorAuxStoreScatter {
  constexpr static bool FuseReduction = std::is_same_v<Element, half_t> ? FuseReduction_ : false;
  struct Arguments {
    Element **scatter_ptr_aux;
    StrideMNL dAux = {};
    int64_t rank = 0;
    int64_t world_size = 1;
    SystemBarrier::T **barrier_ptr_aux;
    bool use_barrier_queue = false;
    bool use_gemmk = false;
    bool per_tile_flags = false;
    int n_split = 1;
  };

  struct Params {
    Element *scatter_ptr_aux[kMaxWorldSize];
    StrideMNL dAux;
    int64_t rank;
    int64_t world_size;
    SystemBarrier::T *barrier_ptr_aux[kMaxWorldSize];
    bool has_barrier_ptr;
    bool use_barrier_queue;
    bool use_gemmk;
    bool per_tile_flags;
    int n_split;
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    Params params;
    params.dAux = args.dAux;
    params.rank = args.rank;
    params.world_size = args.world_size;
    for (int i = 0; i < args.world_size; ++i) {
      params.scatter_ptr_aux[i] = args.scatter_ptr_aux[i];
      if (args.barrier_ptr_aux != nullptr) {
        params.barrier_ptr_aux[i] = args.barrier_ptr_aux[i];
      }
    }
    params.has_barrier_ptr = args.barrier_ptr_aux != nullptr;
    params.use_barrier_queue = args.use_barrier_queue;  // only works for PCI-e mode
    params.use_gemmk = args.use_gemmk && false;         // only works for PCI-e mode
    params.n_split = args.n_split;                      // not used now.
    params.per_tile_flags = args.per_tile_flags;        // not used for PCI-e mode
    return params;
  }

  struct SharedStorage {
    cute::array_aligned<Element *, ThreadblockShape::kM> smem_rs_ptrs;
  };

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatter() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxStoreScatter(Params const &params_, SharedStorage const &shared_storage)
      : params(params_), storage_ptr(&shared_storage) {}

  Params const &params;
  SharedStorage const *storage_ptr;

  /// Pad the given allocation size up to the nearest cache line
  static std::size_t
  cacheline_align_up(std::size_t size) {
    static const int CACHELINE_SIZE = 128;
    return (size + CACHELINE_SIZE - 1) / CACHELINE_SIZE * CACHELINE_SIZE;
  }

  /// Get the workspace size needed for barrier
  std::size_t
  get_barrier_workspace_size(int m, int n) const {
    // each threadblock tile needs a synchronization flag.
    int num_flags = ((m + ThreadblockShape::kM - 1) / ThreadblockShape::kM) *
                    ((n + ThreadblockShape::kN - 1) / ThreadblockShape::kN);

    return cacheline_align_up(sizeof(typename SystemBarrier::T) * num_flags);
  }

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tC_gAux,
        RTensor &&tC_rAux,
        CTensor &&tC_cAux,
        ProblemShape problem_shape,
        Params const &params_,
        SystemBarrier::T *barrier_ptr,
        int thread_idx,
        int tile_idx,
        int dst_rank,
        int m_end,
        Element **smem_rs_ptrs,
        uint32_t row_start,
        uint32_t col_start)
        : tC_gAux(cute::forward<GTensor>(tC_gAux)),
          tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tC_cAux(cute::forward<CTensor>(tC_cAux)),
          problem_shape(problem_shape),
          params(params_),
          barrier_ptr(barrier_ptr),
          thread_idx(thread_idx),
          tile_idx(tile_idx),
          dst_rank(dst_rank),
          m_end(m_end),
          rs_ptrs(smem_rs_ptrs),
          row_start(row_start),
          col_start(col_start) {}

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    Params const &params;
    ProblemShape problem_shape;
    SystemBarrier::T *barrier_ptr;
    int thread_idx;
    int tile_idx;
    int dst_rank;
    int m_end;
    uint32_t row_start;
    uint32_t col_start;
    Element **rs_ptrs;

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

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tC_rAux);
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
      auto dst_v = filter(tC_gAux(_, _, _, step_idx));
      auto shape = cute::make_shape(m_end, size<1>(problem_shape), size<2>(problem_shape));

      {
        auto dst_flat = flatten(tC_cAux.layout().shape());
        int dst_step = size(take<3, 6>(dst_flat));

        if constexpr (FuseReduction) {
          CUTLASS_PRAGMA_UNROLL
          for (int p1 = 0; p1 < size(src_v) / size<0>(dst_v); p1++) {
            CUTLASS_PRAGMA_UNROLL
            for (int p0 = 0; p0 < size<0>(dst_v); p0++) {
              int row = row_start + p1 + dst_step * step_idx;
              int col = col_start + p0 * size(dst_flat);
              void *dst_ptr = rs_ptrs[row] + col;
              int pos = p0 + size<0>(dst_v) * p1;
              bool guard = elem_less(coord_v(pos), problem_shape);
              cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
                  src_v(pos), dst_ptr, guard);
            }
          }
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(src_v); ++i) {
            bool guard = elem_less(coord_v(i), problem_shape);
            int row = row_start + size<0>(src_v) * i + dst_step * step_idx;
            void *dst_ptr = rs_ptrs[row] + col_start;
            cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i), dst_ptr, guard);
          }
        }
      }
    }

    ////
    CUTLASS_DEVICE void
    end_epilogue() {
      using Barrier = cutlass::Barrier;
      char *barrier_ptr = (char *)params.barrier_ptr_aux[params.rank];
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape problem_shape) {
    const int M = get<0>(problem_shape);
    const int N = get<1>(problem_shape);
    constexpr int kM = ThreadblockShape::kM;
    constexpr int kN = ThreadblockShape::kN;

    int tiled_m = (M + kM - 1) / kM;
    int tiled_n = (N + kN - 1) / kN;
    int tiled_m_per_rank = tiled_m / params.world_size;
    int dst_rank = (threadblock_tile_offset.m() / tiled_m_per_rank);

    const int M_lines_per_rank = M / params.world_size;
    Element **smem_ptr = const_cast<Element **>(storage_ptr->smem_rs_ptrs.data());

    // assume we got a full dst tensor
    auto dst_offset = threadblock_tile_offset;

    // prepare the barrier args
    int tile_idx = tiled_n * threadblock_tile_offset.m() + threadblock_tile_offset.n();
    SystemBarrier::T *barrier_ptr =
        params.barrier_ptr_aux ? params.barrier_ptr_aux[dst_rank] : nullptr;

    /// PCIe related offset calculation
    int m_per_rank = size<0>(problem_shape) / params.world_size;
    int tiled_m_per_rank_ = (m_per_rank + kM - 1) / kM;
    int segment = threadblock_tile_offset.m() / tiled_m_per_rank_;
    int m_offset_per_seg = m_per_rank - tiled_m_per_rank_ * kM;
    int offset = params.use_gemmk ? (segment * m_offset_per_seg * size<1>(problem_shape)) : 0;
    int m_end = params.use_gemmk ? (segment * tiled_m_per_rank_ * kM + m_per_rank)
                                 : size<0>(problem_shape);

    /// To support small M
    const int m_offset_start = dst_offset.m() * kM;
    const int tid = threadIdx.x;

    {
      if constexpr (FuseReduction) {
        if (threadIdx.x < kM) {
          int cur_row = m_offset_start + tid;
          int cur_dst_rank = cur_row / M_lines_per_rank;
          smem_ptr[tid] = params.scatter_ptr_aux[cur_dst_rank] + N * (cur_row % M_lines_per_rank);
        }
        __syncthreads();
      } else {
        if (threadIdx.x < kM) {
          int cur_row = m_offset_start + tid;
          int cur_dst_rank = cur_row / M_lines_per_rank;
          int target_line = params.rank * M_lines_per_rank + cur_row % M_lines_per_rank;
          smem_ptr[tid] = params.scatter_ptr_aux[cur_dst_rank] + N * target_line;
        }
        __syncthreads();
      }
    }

    Tensor tC_gAux = [&]() {
      if constexpr (FlattenOutTile) {
        Element *ptr =
            params.scatter_ptr_aux[dst_rank] + tile_idx * size(typename ThreadMap::CtaShapeMNL{});
        using ThreadMapShapeFlatten = cute::Shape<
            // Column
            Int<ThreadMap::Base::kElementsPerAccess>,   // vector
            Int<ThreadMap::Base::kThreads>,             // Thread
            Int<ThreadMap::Base::Iterations::kColumn>,  // iteration::column
            // Row
            Int<ThreadMap::Base::Iterations::kRow>,  // iterations in row
            Int<ThreadMap::Count::kRow>,             // iteration::row
            Int<ThreadMap::Count::kGroup>,           // iteration::group
            Int<ThreadMap::Count::kCluster>          // iteration::cluster
            >;
        return recast<VecType>(group_modes<3, 6>(
            make_tensor(ptr, ThreadMapShapeFlatten{})(_, thread_idx, _, _, _, _, _)));
      } else {
        Tensor mAux = make_tensor(
            make_gmem_ptr(params.scatter_ptr_aux[params.rank]), problem_shape, params.dAux);
        return recast<VecType>(
            group_modes<3, 6>(ThreadMap::partition(mAux, thread_idx, dst_offset)));
      }
    }();

    Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(problem_shape);
    Tensor tC_cAux = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, dst_offset)),
        Shape<Int<VecLength>>{},
        (_0{}));

    uint32_t ptr_offset_start, row_offset_start, col_offset_start;
    if (params.use_gemmk) {
      ptr_offset_start = row_offset_start = col_offset_start = 0;
    } else {
      ptr_offset_start = reinterpret_cast<char *>(&tC_gAux(0)) -
                         reinterpret_cast<char *>(params.scatter_ptr_aux[params.rank]);
      row_offset_start = ptr_offset_start / sizeof(Element) / N - m_offset_start;
      col_offset_start = ptr_offset_start / sizeof(Element) % N;
    }

    return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_gAux),
        cute::move(tC_rAux),
        cute::move(tC_cAux),
        problem_shape,
        params,
        barrier_ptr,
        thread_idx,
        tile_idx,
        dst_rank,
        m_end,
        smem_ptr,
        row_offset_start,
        col_offset_start);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
