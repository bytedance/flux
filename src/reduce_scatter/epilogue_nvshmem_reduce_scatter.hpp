//===- epilogue_nvshmem_reduce_scatter.hpp ------------------------ C++ ---===//
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
  \brief Functor performing elementwise operations used by epilogues.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/functional.h"
#include "cutlass/barrier.h"
#include "flux/cuda/memory_utils.hpp"
#ifdef FLUX_SHM_USE_NVSHMEM
#include <nvshmem.h>
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace collective {

template <class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
CUTLASS_DEVICE void
add_tensor(Tensor<SrcEngine, SrcLayout> const &src, Tensor<DstEngine, DstLayout> &&dst) {
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(src); i++) {
    dst[i] += src[i];
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Applies an element wise operation to all elements within the fragment
/// and writes it out to destination storage.
///
/// Ways to generalize this:
/// - CTA tile shape
/// - vectorization requirements (GMEM)
/// - vectoriz(able) transform()
///
template <
    class StrideC_,
    class StrideD_,
    class ThreadEpilogueOp_,
    class SmemLayout_,
    class CopyAtomR2S_,
    class TiledCopyS2R_,
    class CopyAtomR2G_>
class EpilogueReduceScatterVectorizedNvshmemLocalReduce {
 public:
  //
  // Type Aliases
  //
  // derived types of output thread level operator
  using ThreadEpilogueOp = ThreadEpilogueOp_;
  using ElementAccumulator = typename ThreadEpilogueOp::ElementAccumulator;
  using ElementCompute = typename ThreadEpilogueOp::ElementCompute;
  using ElementScalar = ElementCompute;
  using ElementOutput = typename ThreadEpilogueOp::ElementOutput;
  using ElementC = typename ThreadEpilogueOp::ElementC;
  using StrideC = StrideC_;
  using ElementD = typename ThreadEpilogueOp::ElementD;
  using StrideD = StrideD_;
  using SystemBarrier = cutlass::detail::SystemBarrier;
  using SmemLayout = SmemLayout_;
  using CopyAtomR2S = CopyAtomR2S_;
  using TiledCopyS2R = TiledCopyS2R_;
  using CopyAtomR2G = CopyAtomR2G_;

  static const int kOutputAlignment = ThreadEpilogueOp::kCount;
  using AlignmentType =
      typename cute::uint_bit<sizeof_bits<ElementOutput>::value * kOutputAlignment>::type;

  static_assert(rank(StrideC{}) == 3, "StrideCD must be rank-3: [M, N, L]");
  static_assert(rank(StrideD{}) == 3, "StrideCD must be rank-3: [M, N, L]");

  struct SharedStorage {
    cute::array_aligned<ElementAccumulator, cute::cosize_v<SmemLayout>> smem_epilogue;
  };

  // Host side epilogue arguments
  struct Arguments {
    int rank;
    int world_size;
    typename ThreadEpilogueOp::Params thread{};
    ElementC const *ptr_C = nullptr;
    StrideC dC{};
    ElementD **scatter_ptr_aux = nullptr;
    StrideD dD{};
    SystemBarrier::T **barrier_ptr_aux = nullptr;
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
  EpilogueReduceScatterVectorizedNvshmemLocalReduce(Params const &params_)
      : params(params_), epilogue_op(params_.thread) {}

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
  CUTLASS_DEVICE void
  operator()(
      ProblemShapeMNKL problem_shape_mnkl,
      BlockShapeMNK blk_shape_MNK,
      BlockCoordMNKL blk_coord_mnkl,
      cute::Tensor<FrgEngine, FrgLayout> const &accumulators,  // (MMA,MMA_M,MMA_N)
      TiledMma tiled_mma,
      ResidueMNK residue_mnk,
      int thread_idx,
      char *smem_buf) {
    using namespace cute;
    using X = Underscore;
    constexpr int kEpilogueThreads = size(tiled_mma);  // Number of threads
    static_assert(rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value, "ThreadBlock tile shape must be static");
    static_assert(rank(BlockShapeMNK{}) == 3, "BlockShapeMNK must be rank 3");
    static_assert(rank(BlockCoordMNKL{}) == 4, "BlockCoordMNKL must be rank 3");

    // synchronizing function for smem reads/writes
#if CUDA_BARRIER_ENABLED
    auto synchronize = []() {
      cutlass::arch::NamedBarrier::sync(typename TiledCopyS2R::TiledNumThr{}, 0);
    };
#else
    auto synchronize = []() { __syncthreads(); };
#endif

    const int local_world_size = 8;  // FIXME: should be given through the arguments
    const int world_size = this->params.world_size;
    int local_rank = this->params.rank % local_world_size;

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);
    auto BM = get<0>(blk_shape_MNK);
    auto BN = get<1>(blk_shape_MNK);
    assert(M % BM == 0);
    assert(N % BN == 0);

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    const int coord_per_rank_local = (M / BM / local_world_size);
    const int coord_per_rank_global = (M / BM / world_size);
    const int local_m_block_idx = m_coord / coord_per_rank_global;
    int local_dst_rank = local_m_block_idx % local_world_size;  // interleaved
    // const int n_nodes = world_size / local_world_size;
    const int nodeid = this->params.rank / local_world_size;
    const int remote_dst_node = local_m_block_idx / local_world_size;
    const int remote_dst_rank = remote_dst_node * local_world_size + local_rank;

    auto ptr_sym = params.scatter_ptr_aux[local_rank];  // the others are the ptrs from nvshmem_ptr
    auto ptr_local_dst = params.scatter_ptr_aux[local_dst_rank];
    auto ptr_R = params.scatter_ptr_aux[local_world_size];  //
    // if(threadIdx.x==0 && m_coord==0 && n_coord==0)
    //   printf("rank:%d PTR_R: %p \n", this->params.rank  , ptr_R);
    auto barrier_ptr_local_dst = params.barrier_ptr_aux[local_dst_rank];

    // refact the layout to use the pushmem_block
    auto mD_mnl_shape = make_shape(make_shape(BM, M / BM), make_shape(BN, N / BN), L);
    auto mD_mnl_stride =
        make_stride(make_stride(BN, BM * N), make_stride(1, BM * BN), get<2>(params.dD));
    auto mD_mnl_layout = make_layout(mD_mnl_shape, mD_mnl_stride);

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(
        make_gmem_ptr(params.ptr_C),
        make_shape(M, N, L),
        params.dC);  //             (m,n,l)
    Tensor mD_mnl =
        make_tensor(make_gmem_ptr(ptr_local_dst), mD_mnl_layout);      //             (m,n,l)
    Tensor mR_mnl = make_tensor(make_gmem_ptr(ptr_R), mD_mnl_layout);  //             (m,n,l)

    Tensor gC_mnl = local_tile(
        mC_mnl, blk_shape_MNK, make_coord(_, _, _), Step<_1, _1, X>{});  // (BLK_M,BLK_N,m,n,l)
    const int new_m_coord = nodeid * coord_per_rank_global + m_coord % coord_per_rank_global;
    auto Dcooperative_coord = make_coord(make_coord(_, m_coord), make_coord(_, n_coord), l_coord);
    auto Rcooperative_coord =
        make_coord(make_coord(_, new_m_coord), make_coord(_, n_coord), l_coord);
    Tensor gC = gC_mnl(_, _, m_coord, n_coord, l_coord);  // (BLK_M,BLK_N)
    Tensor gD = mD_mnl(Dcooperative_coord);
    Tensor gR = mR_mnl(Rcooperative_coord);

    // Construct a tensor in SMEM that we can partition for rearranging data
    SharedStorage &storage = *reinterpret_cast<SharedStorage *>(smem_buf);
    Tensor sC =
        make_tensor(make_smem_ptr(storage.smem_epilogue.data()), SmemLayout{});  // (SMEM_M,SMEM_N)

    // Partition sC to match the accumulator partitioning
    auto tiled_r2s = make_tiled_copy_C(CopyAtomR2S{}, tiled_mma);
    auto tC = tiled_r2s.get_thread_slice(thread_idx);
    Tensor tCaC = tC.retile_S(accumulators);  // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor tCsC = tC.partition_D(sC);         // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // Tile gD and gC by the shape of SmemLayout first
    auto tile = make_shape(size<0>(sC), size<1>(sC));
    Tensor gCt = flat_divide(gC, tile);  // (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor gDt = flat_divide(gD, tile);  // (SMEM_M,SMEM_N,TILE_M,TILE_N)

    // Partition sC, gC, and gD for the output
    auto tiled_s2r = TiledCopyS2R{};
    auto tD = tiled_s2r.get_thread_slice(thread_idx);
    Tensor tDsC = tD.partition_S(sC);   //               ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDgC = tD.partition_D(gCt);  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)
    Tensor tDgD = tD.partition_D(gDt);  // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    // Allocate intermediate registers on the dst tensors
    Tensor tDrC = make_tensor<ElementAccumulator>(
        take<0, 3>(shape(tDgC)));                           // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tDrD = make_tensor<ElementOutput>(shape(tDrC));  // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Repeat the D-partitioning for coordinates and predication
    Tensor cD = make_identity_tensor(
        make_shape(size<0>(gD), size<1>(gD)));  // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor cDt = flat_divide(cD, tile);         //                (SMEM_M,SMEM_N,TILE_M,TILE_N)
    Tensor tDcD = tD.partition_D(cDt);          // ((Atom,AtomNum),ATOM_M,ATOM_N,TILE_M,TILE_N)

    CUTE_STATIC_ASSERT(size<1>(tCaC) % size<3>(tDgC) == 0);  // TILE_M divides MMA_M
    CUTE_STATIC_ASSERT(size<2>(tCaC) % size<4>(tDgC) == 0);  // TILE_N divides MMA_N
    const int tile_idx = m_coord * (N / BN) + n_coord;       // FIXME : l_coord
    CUTE_STATIC_ASSERT(
        typename TiledCopyS2R::TiledNumThr{} == size<0>(typename TiledMma::AtomLayoutC_TV{}));

#if 0
    if (thread_idx == 0 && m_coord == 0 && n_coord == 0) {
      print("aC   : "); print(accumulators.layout()); print("\n");
      print("gC   : "); print(gC.layout()); print("\n");
      print("gD   : "); print(gD.layout()); print("\n");
      print("sC   : "); print(sC.layout()); print("\n");
      print("\n");
      print("tCsC : "); print(tCsC.layout()); print("\n");
      print("tCaC : "); print(tCaC.layout()); print("\n");
      print("\n");
      print("cDt  : "); print(cDt.layout()); print("\n");
      print("gDt  : "); print(gDt.layout()); print("\n");
      print("tDsC : "); print(tDsC.layout()); print("\n");
      print("tDrC : "); print(tDrC.layout()); print("\n");
      print("\n");
      print("tDrD : "); print(tDrD.layout()); print("\n");
      print("tDgC : "); print(tDgC.layout()); print("\n");
      print("tDgD : "); print(tDgD.layout()); print("\n");
      print("\n");
    }
#endif
    SystemBarrier::wait_lt(
        barrier_ptr_local_dst,
        threadIdx.x,
        tile_idx,
        -1);  // only remain the fence here
    // For each tiling needed for SmemLayout to cover shape(gD)
    CUTLASS_PRAGMA_UNROLL
    for (int step_m = 0; step_m < size<2>(cDt); ++step_m) {
      CUTLASS_PRAGMA_UNROLL
      for (int step_n = 0; step_n < size<3>(cDt); ++step_n) {
        // Step 1. Copy to SMEM
        CUTLASS_PRAGMA_UNROLL
        for (int pipe_m = 0; pipe_m < size<1>(tCsC); ++pipe_m) {
          CUTLASS_PRAGMA_UNROLL
          for (int pipe_n = 0; pipe_n < size<2>(tCsC); ++pipe_n) {
            int mma_m = step_m * size<1>(tCsC) + pipe_m;
            int mma_n = step_n * size<2>(tCsC) + pipe_n;

            copy(tiled_r2s, tCaC(_, mma_m, mma_n), tCsC(_, pipe_m, pipe_n));
          }
        }

        // Step 2. Wait for SMEM writes to complete
        using VecType = uint_bit_t<32>;
        synchronize();
        // Following code is performing the local reduce within the single node
        // The reason we abandon the TiledCopyS2R here is that the layout does not
        // works for the += operation. Previously each thread is responsible for a
        // 8x1 column vector. However, such layout will leads to heavy racing for
        // adjacent threads(two adjacent threads are += the same 4 bytes). As for direct
        // writing requests, this layout is fine, because the writing request is merged into
        // a coalesce request(I guess).
        // TODO: It's more elegant to solve the performance bug by passing the right layout
        // through parameters.
        if (epilogue_op.is_source_needed()) {
          // To be Done
          assert(0);  // FIXME: not implemented right now
        } else {
          // directly performe the reduce operation in an interleaved manner, becuase the address
          // of the dst block is alread continuous.
          // TODO use template to support bf16
          auto offset_src_shared = static_cast<int>(sC.layout()(cutlass::make_coord(0, 0)));
          float *src_ptr = reinterpret_cast<float *>((sC.data() + offset_src_shared).get());
          // half * src_ptr = reinterpret_cast<half*>(sC.data()+offset_src_shared.get());
          auto offset_dst_local = static_cast<int>(gD.layout()(cutlass::make_coord(0, 0)));
          half *dst_ptr = reinterpret_cast<half *>((gD.data() + offset_dst_local).get());
          constexpr int size_N = size<0>(cDt) * size<1>(cDt);

          CUTLASS_PRAGMA_UNROLL
          for (int iter = 0; iter < size_N / 2 / kEpilogueThreads; iter++) {
            int pos = (threadIdx.x + iter * kEpilogueThreads) * 2;
            if (pos < size_N) {
              float2 cur_val = *reinterpret_cast<float2 *>(src_ptr + pos);
              half2 re_val;

              re_val.x = epilogue_op(cur_val.x);
              re_val.y = epilogue_op(cur_val.y);
              // global_reduce
              // atomicAdd((half2*)(dst_ptr+pos), re_val);
              cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
                  reinterpret_cast<VecType const &>(re_val), (void *)(dst_ptr + pos), 1);
            }
          }
        }
      }
    }
    // Use the nvshmem primitives to write the gD to gR;
    synchronize();  // local reduce done

    if (threadIdx.x == 0) {
      // __threadfence_system(); // necessary
      // asm ("membar.sys;\n" ::);
      // asm ("membar.gl;\n" ::);
      // asm ("fence.sc;;\n" ::);

      asm volatile("fence.acq_rel.sys;\n");  // better performance compared to other kinds of fence

      int after_inc = atomicAdd_system(barrier_ptr_local_dst + tile_idx, 1);
    }
    __syncthreads();
    // SystemBarrier::arrive_inc(barrier_ptr_local_dst, threadIdx.x, tile_idx);

    // wait for all local reduce done
    if (local_dst_rank == local_rank) {
      SystemBarrier::wait_eq(
          barrier_ptr_local_dst,
          threadIdx.x,
          tile_idx,
          local_world_size);  // double check here

      auto offset_src = static_cast<int>(gD.layout()(cutlass::make_coord(0, 0)));
      auto offset_dst = static_cast<int>(gR.layout()(cutlass::make_coord(0, 0)));
#ifdef FLUX_SHM_USE_NVSHMEM
      nvshmemx_putmem_nbi_block(
          (gR.data() + offset_dst).get(),
          (gD.data() + offset_src).get(),
          BM * BN * sizeof(ElementD),
          remote_dst_rank);
#endif
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
