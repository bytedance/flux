//===- visitor_2x_bsr.hpp ------------------------------ C++ ---===//
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
// Some code from cutlass/epilogue/threadblock/fusion/visitor_2x.hpp
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
  \brief Visitor tree operation base implementation to enable composable fusions
         for the CUTLASS 2x epilogue
*/

#pragma once

#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using cute::tuple;
using namespace detail;

template <
    typename ThreadblockShape_,
    typename WarpShape_,
    typename Element_,
    int ElementsPerAccess,
    int Stages_,
    bool FuseReduction>
struct OutputTileThreadLayoutBSR : DefaultThreadMapTensorOp<
                                       ThreadblockShape_,
                                       WarpShape_,
                                       ThreadblockShape_::kK / WarpShape_::kK,
                                       Element_,
                                       ElementsPerAccess>::Type {
  using Base = typename DefaultThreadMapTensorOp<
      ThreadblockShape_,
      WarpShape_,
      ThreadblockShape_::kK / WarpShape_::kK,
      Element_,
      ElementsPerAccess>::Type;
  using Base::Base;

  // Software pipeline stages in epilogue
  static_assert(Stages_ <= 2, "Sm80 EVT only support upto 2 Stages.");
  static const int Stages = Stages_;

  using ThreadShape = cute::Shape<
      cute::Int<Base::Detail::kAccessWidth>,            // lane col idx
      cute::Int<Base::Detail::kAccessRows>,             // lane row idx
      cute::Int<Base::Detail::kWarpsRemainingForRows>,  // warp row idx
      cute::Int<Base::Shape::kGroup>,                   // group idx
      cute::Int<Base::Shape::kCluster>                  // cluster idx
      >;

  using Shape = typename Base::Shape;
  using Count = typename Base::Count;

  using ThreadMapShape = cute::Shape<
      // Column
      Int<Base::kElementsPerAccess>,    // vector
      Int<Base::Detail::kAccessWidth>,  // lane_col_coord
      Int<Base::Iterations::kColumn>,   // iteration::column
      // Row
      Int<Base::Detail::kAccessRows>,             // lane_row_coord
      Int<Base::Iterations::kRow>,                // iterations in row
      Int<Base::Detail::kWarpsRemainingForRows>,  // warp_row_coord
      Int<Count::kRow>,                           // iteration::row
      Int<Count::kGroup>,                         // iteration::group
      Int<Shape::kGroup>,                         // group_coord
      Int<Count::kCluster>,                       // iteration::cluster
      Int<Shape::kCluster>                        // cluster_coord
      >;

  // The shape of CTA Tile
  using CtaShapeMNL = cute::Shape<
      Int<Shape::kRow * Count::kRow * Shape::kGroup * Count::kGroup * Shape::kCluster *
          Count::kCluster>,
      Int<Shape::kColumn * Count::kColumn>,
      _1>;

  static const int kElementsPerAccess = ElementsPerAccess;

  //
  // Methods
  //

  CUTLASS_DEVICE
  static auto
  tid2coord(int thread_idx) {
    return make_layout(ThreadShape{}).get_hier_coord(thread_idx);
  }

  template <class TensorInput, class ProblemShape, class BlockShapeMNK>
  CUTLASS_DEVICE static auto
  partition(
      TensorInput &&xT,
      ProblemShape problem_shape_mnkl,
      BlockShapeMNK blk_shape_mnk,
      int thread_idx,
      gemm::GemmCoord threadblock_tile_offset,
      int rank,
      int world_size,
      int nnodes) {
    // (BLK_M,BLK_N)
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = _1{};
    auto BM = get<0>(blk_shape_mnk);
    auto BN = get<1>(blk_shape_mnk);
    auto mD_mnl_shape = make_shape(make_shape(BM, M / BM), make_shape(BN, N / BN), L);
    auto mD_mnl_stride = make_stride(make_stride(BN, BM * N), make_stride(_1{}, BM * BN), _0{});
    auto mD_mnl_layout = make_layout(mD_mnl_shape, mD_mnl_stride);
    const int m_coord = threadblock_tile_offset.m();
    const int n_coord = threadblock_tile_offset.n();

    Tensor mD_mnl =
        make_tensor(make_gmem_ptr(xT.data().get()), mD_mnl_layout);  //             (m,n,l)
    int target_m_coord;
    if constexpr (FuseReduction) {
      target_m_coord = m_coord;
    } else {
      const int local_world_size = world_size / nnodes;
      const int local_rank = rank % local_world_size;
      const int coord_per_rank_local = (M / BM / local_world_size);
      const int coord_per_rank_global = (M / BM / world_size);
      const int local_m_block_idx = m_coord / coord_per_rank_global;
      // int local_dst_rank = local_m_block_idx % local_world_size;  // interleaved
      /*
      Take TP=16 nnodes=2 as an example
      after the local reduce without fuse reduction, the rank-0 should hold the 0-th and 8-th
      MBlocks partitioned based based on the global world size
      [
        M0,(from rank0)
        M8,(from rank0)
        M0,(from rank1)
        M8,(from rank1)
        M0,(from rank2)
        M8,(from rank2)
        M0,(from rank3)
        M8,(from rank3)
        M0,(from rank4)
        M8,(from rank4)
        M0,(from rank5)
        M8,(from rank5)
        M0,(from rank6)
        M8,(from rank6)
        M0,(from rank7)
        M8,(from rank7)
      ]

      */
      target_m_coord = local_rank * coord_per_rank_local +
                       local_m_block_idx / local_world_size * coord_per_rank_global +
                       m_coord % coord_per_rank_global;
    }

    auto Dcooperative_coord =
        make_coord(make_coord(_, target_m_coord), make_coord(_, n_coord), _1{});
    Tensor bCxT = mD_mnl(Dcooperative_coord);
    auto [lane_col_coord, lane_row_coord, warp_row_coord, group_coord, cluster_coord] =
        tid2coord(thread_idx);

    // transform to column-major
    Tensor bCxT_nm = make_tensor(
                         std::forward<decltype(bCxT)>(bCxT).data(),
                         make_layout(get<1>(bCxT.layout()), get<0>(bCxT.layout())))
                         .compose(make_layout(ThreadMapShape{}));

    auto re = bCxT_nm(
        _,
        lane_col_coord,
        _,
        lane_row_coord,
        _,
        warp_row_coord,
        _,
        _,
        group_coord,
        _,
        cluster_coord);

    return re;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
