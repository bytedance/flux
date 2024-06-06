//===- gemmk_visitor_load.hpp ------------------------------------------- C++ ---===//
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
// Some code from cutlass/epilogue/threadblock/fusion/visitor_load.hpp
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

#pragma once
#include "cutlass/gemm_coord.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "cute/tensor.hpp"
namespace cutlass::epilogue::threadblock {
/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Elementwise Load Operations
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <class ThreadMap, class Element, class StrideMNL>
struct VisitorAuxLoadGemmk {
  struct Arguments {
    Element *ptr_aux = nullptr;
    Element null_default = Element(0);
    StrideMNL dAux = {};
    int world_size;
    bool use_gemmk;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const &problem_shape, Arguments const &args) {
    return 0;
  }

  // Software pipeline stages
  static const int Stages = ThreadMap::Stages;

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxLoadGemmk() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxLoadGemmk(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params) {}

  Params const *params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tC_gAux,
        RTensor &&tC_rAux,
        CTensor &&tC_cAux,
        ProblemShape problem_shape,
        Params const *params_ptr)
        : tC_gAux(cute::forward<GTensor>(tC_gAux)),
          tC_rAux(cute::forward<RTensor>(tC_rAux)),
          tC_cAux(cute::forward<CTensor>(tC_cAux)),
          problem_shape(problem_shape),
          params_ptr(params_ptr) {}

    GTensor tC_gAux;
    RTensor tC_rAux;
    CTensor tC_cAux;
    Params const *params_ptr;
    ProblemShape problem_shape;

    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tC_rAux(_, _, _, step_idx % Stages));
      auto src_v = filter(tC_gAux(_, _, _, step_idx));
      auto coord_v = filter(tC_cAux(_, _, _, step_idx));
      auto dst_v = filter(tC_rAux(_, _, _, step_idx % Stages));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        cutlass::arch::global_load<VecType, sizeof(VecType)>(
            dst_v(i), (void const *)&src_v(i), guard);
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto  // returns an Array
    visit(
        int iter_idx,
        int row_idx,
        int column_idx,
        int frg_idx,
        Array<ElementAccumulator, FragmentSize> const &frg_acc) {
      Tensor tC_rAux_frg =
          recast<Array<Element, FragmentSize>>(coalesce(tC_rAux(_, _, _, iter_idx % Stages)));
      return tC_rAux_frg(frg_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      cutlass::gemm::GemmCoord threadblock_tile_offset,
      int thread_idx,
      ProblemShape problem_shape) {
    // calculate offset
    constexpr int kM = cute::size<0>(typename ThreadMap::CtaShapeMNL{});
    constexpr int kN = cute::size<1>(typename ThreadMap::CtaShapeMNL{});
    int m_end = size<0>(problem_shape);
    int offset = 0;
    if (params_ptr->use_gemmk) {
      int m_per_rank = size<0>(problem_shape) / params_ptr->world_size;
      int tiled_m_per_rank_ = (m_per_rank + kM - 1) / kM;
      int segment = threadblock_tile_offset.m() / tiled_m_per_rank_;
      int m_offset_per_seg = m_per_rank - tiled_m_per_rank_ * kM;
      offset = segment * m_offset_per_seg * size<0>(params_ptr->dAux);
      m_end = segment * tiled_m_per_rank_ * kM + m_per_rank;
      // calculate offset and m_end done
    }

    Tensor mAux = make_tensor(
        make_gmem_ptr(params_ptr->ptr_aux + offset),  // with offset
        problem_shape,
        params_ptr->dAux);  // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gAux = recast<VecType>(
        group_modes<3, 6>(ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, Stages
    Tensor tC_rAux = make_tensor<VecType>(
        make_layout(flatten(make_shape(take<0, 3>(tC_gAux.shape()), Int<Stages>{}))));

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
        Shape<Int<VecLength>>{},
        (_0{}));

    return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_gAux),
        cute::move(tC_rAux),
        cute::move(tC_cAux),
        cute::make_shape(m_end, size<1>(problem_shape), size<2>(problem_shape)),  // m in submatrix
        params_ptr);
  }
};
}  // namespace cutlass::epilogue::threadblock
