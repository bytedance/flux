//===- scatter_epilogue_array.hpp ----------------------------- C++ -------===//
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

/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  \brief Functor performing elementwise operations used by epilogues.
*/
// Copied from cutlass/examples/52_hopper_gather_scatter_fusion/gather_tensor.hpp
// clang-format off
#pragma once

#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/detail.hpp"

#include "cute/tensor.hpp"
#include "cute/int_tuple.hpp"
#include "cute/numeric/int.hpp"

#include "./gather_tensor.hpp"
#include "flux/flux.h"

namespace cutlass::epilogue::collective {
using bytedance::flux::int32_t;

/// Applies an element wise operation to all elements within the fragment
/// and scatter-writes them out to destination storage.
/// GatherC and ScatterD are types of user-defined functions that apply the
/// transoformation of the strided coordinate (e.g. through an index array).
template <
  class StrideC_,
  class StrideD_,
  class ThreadEpilogueOp_,
  class EpilogueSchedule_,
  class ScatterD_
>
class EpilogueScatterArray {
public:
  //
  // Type Aliases
  //
  using EpilogueSchedule = EpilogueSchedule_;
  using DispatchPolicy = EpilogueSchedule_;

  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using InternalStrideC = cute::remove_pointer_t<StrideC>;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;
  using InternalStrideD = cute::remove_pointer_t<StrideD>;

  // Every epilogue needs these two GmemTiledCopy{C,D} aliases.
  // If you don't know what they should be, just use void.
  using GmemTiledCopyC = void;
  using GmemTiledCopyD = void;

  using ScatterD = ScatterD_;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType = typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(cute::is_same_v<EpilogueSchedule, PtrArrayNoSmemWarpSpecialized>, "Incompatible epilogue schedule.");
  static_assert(cute::rank(InternalStrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(cute::rank(InternalStrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage { };
  using TensorMapStorage = SharedStorage;

  // Host side epilogue arguments
  struct Arguments {
    typename ThreadEpilogueOp::Params thread{};
    ElementC const** ptr_C = nullptr;
    StrideC dC{};
    ElementD** ptr_D = nullptr;
    StrideD dD{};
    int32_t const** ptr_scatter_D = nullptr;
  };

  // Device side epilogue params
  using Params = Arguments;

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      [[maybe_unused]] ProblemShape const& _,
      Arguments const& args,
      [[maybe_unused]] void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args, int sm_count) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
      CudaHostAdapter *cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template<class ProblemShape>
  static bool
  can_implement(
      [[maybe_unused]] ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    return true;
  }

  CUTLASS_HOST_DEVICE
  EpilogueScatterArray(Params const& params_) : params(params_) { }

  CUTLASS_DEVICE
  bool
  is_source_needed() {
    // For Ptr-Array or Grouped Gemm we cannot determine if source is needed based on first beta.
    return true;
  }

  template<
    class ProblemShapeMNKL,
    class BlockShapeMNK,
    class BlockCoordMNKL,
    class FrgEngine, class FrgLayout,
    class TiledMma,
    class ResidueMNK
  >
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const& accumulators,
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char* smem_buf)
  {
    using namespace cute;
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    (void) smem_buf;

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    // If scalar alpha/beta are provided, i.e., same alpha/beta applies to all batches/groups.
    // If pointers to alpha/beta are provided, i.e., alpha/beta can differ between batches/groups,
    // we get the correct alpha/beta values for the current batch/group using group index.
    ThreadEpilogueOp epilogue_op = ThreadEpilogueOp(params.thread, l_coord);

    if (epilogue_op.is_source_needed() && params.dC == nullptr) {
      // Beta value is non-zero while pointer to C is a nullptr
      assert(0);
    }

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    // Batches are managed by using appropriate pointers to C and D matrices
    constexpr int32_t mock_L = 1;
    constexpr int32_t mock_l_coord = 0;

    InternalStrideC stride_c;
    InternalStrideD stride_d;
    if constexpr (!cute::is_same_v<InternalStrideC, StrideC>) {
      // If grouped gemm
      if (epilogue_op.is_source_needed()) {
        stride_c = detail::get_epilogue_stride<EpilogueSchedule>(params.dC[l_coord]);
      }
      stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD[l_coord]);
    } else {
      stride_c = detail::get_epilogue_stride<EpilogueSchedule>(params.dC);
      stride_d = detail::get_epilogue_stride<EpilogueSchedule>(params.dD);
    }

    // Represent the full output tensor
    ElementC const* ptr_C_l = nullptr;
    if (epilogue_op.is_source_needed()) {
      ptr_C_l = params.ptr_C[l_coord];
    }

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr(ptr_C_l), make_shape(M,N,mock_L), stride_c);      // (m,n,l)
    Tensor mD_mnl = make_gather_tensor(make_gmem_ptr(params.ptr_D[l_coord]), make_shape(M,N,mock_L), stride_d, ScatterD(params.ptr_scatter_D[l_coord]));      // (m,n,l)

    Tensor gC_mnl = local_tile(mC_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl, blk_shape_MNK, make_coord(_,_,_), Step<_1,_1, X>{});    // (BLK_M,BLK_N,m,n,l)

    Tensor gC = gC_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                 // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_,_,m_coord,n_coord,mock_l_coord);                                                 // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD);                                       // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC);                                       // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value, "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    // Make an identity coordinate tensor for predicating our output MN tile
    auto cD = make_identity_tensor(make_shape(unwrap(shape<0>(gD)), unwrap(shape<1>(gD))));
    Tensor tCcD = thr_mma.partition_C(cD);

    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          tCgD(i) = epilogue_op(accumulators(i), tCgC(i));
        }
      }
    }
    // source is not needed, avoid load
    else {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        if (elem_less(tCcD(i), make_coord(get<0>(residue_mnk), get<1>(residue_mnk)))) {
          tCgD(i) = epilogue_op(accumulators(i));
        }
      }
    }
  }

private:
  Params params;
};

} // namespace cutlass::epilogue::collective
// clang-format on
