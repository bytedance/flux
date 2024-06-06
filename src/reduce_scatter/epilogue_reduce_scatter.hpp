//===- epilogue_reduce_scatter.hpp -------------------------------- C++ ---===//
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
// Some code from cutlass/epilogue/collective/sm70_epilogue_vectorized.hpp
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
  \brief Functor performing elementwise operations used by epilogues. This is basically the same
  with default_epilogue.hpp except that the output pointer is calculated by an operator
*/

#pragma once

#include "cute/layout.hpp"
#include "cute/util/debug.hpp"
#include "cutlass/coord.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/int.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes them out to destination storage.
template <class StrideC_, class StrideD_, class ThreadEpilogueOp_, class EpilogueSchedule_>
class EpilogueReduceScatter {
 public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;

  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;

  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType =
      typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage {};

  // Host side epilogue arguments
  struct Arguments {
    int32_t rank;
    int32_t world_size;
    typename ThreadEpilogueOp::Params thread{};
    ElementC const *ptr_C = nullptr;
    StrideC dC{};
    ElementD **scatter_ptr_D = nullptr;
    StrideD dD{};
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const &_,
      Arguments const &args,
      [[maybe_unused]] void *workspace) {
    return args;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool
  can_implement(
      [[maybe_unused]] ProblemShape const &problem_shape, [[maybe_unused]] Arguments const &args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueReduceScatter(Params const &params_) : params(params_), epilogue_op(params_.thread) {}

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    return epilogue_op.is_source_needed();
  }

  template <
      class ProblemShapeMNKL,
      class BlockShapeMNK,
      class BlockCoordMNKL,
      class FrgEngine,
      class FrgLayout,
      class TiledMma,
      class ResidueMNK>
  CUTLASS_HOST_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const &accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      [[maybe_unused]] char *smem_buf) {
    using namespace cute;
    using X = Underscore;

    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    auto stride_c = detail::get_epilogue_stride<EpilogueSchedule>(params.dC);
    auto stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD);

    // Represent the full output tensor
    Tensor mC_mnl =
        make_tensor(make_gmem_ptr(params.ptr_C), make_shape(M, N, L), stride_c);  // (m,n,l)
    Tensor gC_mnl = local_tile(
        mC_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});  // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgC = thr_mma.partition_C(gC);  // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");

    // Make an identity coordinate tensor for predicating our output MN tile
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gC)), unwrap(shape<1>(gC))));
    Tensor tCcD = thr_mma.partition_C(cD);

    int tp_shard_M = M / this->params.world_size;

    // mapping element to dst rank
    auto scatter_layout = make_layout(
        make_shape(make_shape(tp_shard_M, this->params.world_size), N, L),
        make_stride(make_stride(Int<0>{}, Int<1>{}), Int<0>{}, Int<0>{}));
    // follow the same tiling procedure
    auto mScatter_mnl = make_counting_tensor(scatter_layout);
    auto gScatter_mnl =
        local_tile(mScatter_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});
    auto gScatter = gScatter_mnl(_, _, m_coord, n_coord, l_coord);

    auto get_tCgD = [&](int dst_rank) {
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
      // add the offset so that each rank write to different range of data
      ptrdiff_t offset = tp_shard_M * N * (this->params.rank - dst_rank);
      Tensor mD_mnl = make_tensor(
          make_gmem_ptr(params.scatter_ptr_D[dst_rank]) + offset, make_shape(M, N, L), stride_d);
      Tensor gD_mnl = local_tile(
          mD_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});  // (BLK_M,BLK_N,m,n,l)
      Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord);
      Tensor tCgD = thr_mma.partition_C(gD);
      CUTE_STATIC_ASSERT_V(
          size(tCgC) == size(tCgD),
          "Source and destination must have the same number of elements.");
      CUTE_STATIC_ASSERT_V(
          size(tCgD) == size(accumulators),
          "Accumulator count must have the same destination element count.");
      return tCgD;
    };

    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          int dst_rank = gScatter(tCcD(i));
          auto tCgD = get_tCgD(dst_rank);
          tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
        }
      }
    }
    // source is not needed, avoid load
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          int dst_rank = gScatter(tCcD(i));
          auto tCgD = get_tCgD(dst_rank);
          tCgD(i) = epilogue_op(accumulators(i));
        }
      }
    }
  }

 private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace collective
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
