//===- epilogue_gather.hpp ----------------------------------------- C++ ---===//
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
#include "flux/flux.h"
#include "flux/cuda/memory_utils.hpp"
#include <nvshmem.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;
using X = Underscore;
using SystemBarrier = Barrier;
using bytedance::flux::kMaxWorldSize;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// GatherEpilogue
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ThreadMap,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape>
struct VisitorAuxGather {
  struct Arguments {
    Element *ptr_aux = nullptr;
    StrideMNL dAux = {};
    const int64_t *gather_index = nullptr;
    Element *gather_output = nullptr;
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    return args;
  }

  struct SharedStorage {};

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxGather() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxGather(Params const &params, SharedStorage const &shared_storage)
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
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        int64_t ptr_offset = reinterpret_cast<uint64_t>(&dst_v(i)) -
                             reinterpret_cast<uint64_t>(params_ptr->ptr_aux);
        int64_t row_offset = ptr_offset / sizeof(Element) / 12288;
        int64_t col_offset = ptr_offset / sizeof(Element) % 12288;
        int64_t gidx = params_ptr->gather_index[row_offset];
        // printf("store to %d, %ld\n", gidx, col_offset);
        void *dst_ptr = reinterpret_cast<char *>(params_ptr->gather_output) +
                        gidx * 12288 * sizeof(Element) + col_offset * sizeof(Element);
        // printf("step: %d, row_offset: %ld, col_offset: %ld, gidx: %d\n", step_idx, row_offset,
        // col_offset, gidx); cutlass::arch::global_store<VecType, sizeof(VecType)>(src_v(i),
        // (void*)&dst_v(i), guard); cutlass::arch::global_store<VecType,
        // sizeof(VecType)>(src_v(i), dst_ptr, guard);
        cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(src_v(i), dst_ptr, guard);
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    Tensor mAux = make_tensor(
        make_gmem_ptr(params_ptr->ptr_aux),
        problem_shape,
        params_ptr->dAux);  // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gAux = recast<VecType>(group_modes<3, 6>(
        ThreadMap::partition(mAux, thread_idx, threadblock_tile_offset)));  // global tensor
    Tensor tC_rAux = make_tensor_like(take<0, 3>(tC_gAux));                 // register tensor

    // Generate the pred tensor
    Tensor cAux = make_identity_tensor(mAux.shape());
    Tensor tC_cAux = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(cAux, thread_idx, threadblock_tile_offset)),
        Shape<Int<VecLength>>{},
        (_0{}));

    /*
    auto gout_shape = make_shape(get<0>(problem_shape), get<1>(problem_shape));
    auto gout_stride = make_stride(get<1>(problem_shape), Int<1>{});
    // Generate the gather output tensor
    Tensor mGOut = make_tensor(
      make_gmem_ptr(params_ptr->gather_output),
      gout_shape,
      gout_stride
    );
    */

    if (false && !threadIdx.x && !threadIdx.y && !threadIdx.z) {
      printf(
          "thread_idx:%d threadblock_tile_offset: %d %d %d problem_shape: %d %d "
          "threadblock_shape: %d %d %d\n",
          thread_idx,
          threadblock_tile_offset.m(),
          threadblock_tile_offset.n(),
          threadblock_tile_offset.k(),
          get<0>(problem_shape),
          get<1>(problem_shape),
          ThreadblockShape::kM,
          ThreadblockShape::kN,
          ThreadblockShape::kK);
    }

    return Callbacks<decltype(tC_gAux), decltype(tC_rAux), decltype(tC_cAux), ProblemShape>(
        cute::move(tC_gAux), cute::move(tC_rAux), cute::move(tC_cAux), problem_shape, params_ptr);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// GatherStoreEpilogue
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ThreadMap,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape>
struct VisitorAuxGatherStore {
  struct Arguments {
    int64_t rank;
    int64_t world_size;
    StrideMNL dAux;
    Element *gemm_out;
    const int64_t *gather_index;
    Element *gather_outputs;
    Element **rs_outputs_ptrs;
  };

  struct Params {
    int64_t rank;
    int64_t world_size;
    StrideMNL dAux;
    Element *gemm_out;
    const int64_t *gather_index;
    Element *gather_outputs;
    Element *rs_outputs_ptrs[kMaxWorldSize];
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    Params params;
    params.rank = args.rank;
    params.world_size = args.world_size;
    params.dAux = args.dAux;

    params.gemm_out = args.gemm_out;
    params.gather_index = args.gather_index;
    params.gather_outputs = args.gather_outputs;
    for (int i = 0; i < args.world_size; ++i) {
      params.rs_outputs_ptrs[i] = args.rs_outputs_ptrs[i];
    }
    return params;
  }

  struct SharedStorage {
    cute::array_aligned<int64_t, 640> smem_gather_index;
    cute::array_aligned<Element *, kMaxWorldSize> smem_rs_outputs_ptrs;
  };

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxGatherStore() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxGatherStore(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params), storage_ptr(&shared_storage) {}

  Params const *params_ptr;
  SharedStorage const *storage_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tCgC,
        RTensor &&tCrC,
        CTensor &&tCrP,
        ProblemShape problem_shape,
        Params const *params_ptr,
        SharedStorage const *storage_ptr)
        : tCgC(cute::forward<GTensor>(tCgC)),
          tCrC(cute::forward<RTensor>(tCrC)),
          tCrP(cute::forward<CTensor>(tCrP)),
          problem_shape(problem_shape),
          params_ptr(params_ptr),
          storage_ptr(storage_ptr) {
      this->tokens_per_rank = 4096 / params_ptr->world_size;
      this->gather_index_ptr = const_cast<int64_t *>(storage_ptr->smem_gather_index.data());
      this->rs_outputs_ptrs = const_cast<Element **>(storage_ptr->smem_rs_outputs_ptrs.data());
    }

    GTensor tCgC;
    RTensor tCrC;
    CTensor tCrP;
    Params const *params_ptr;
    SharedStorage const *storage_ptr;
    ProblemShape problem_shape;
    int64_t tokens_per_rank;
    int64_t *gather_index_ptr;
    Element **rs_outputs_ptrs;

    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tCrC);
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

      Tensor tCrC_frg = recast<Array<Element, FragmentSize>>(coalesce(tCrC));
      tCrC_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tCrC);
      auto coord_v = filter(tCrP(_, _, _, step_idx));
      auto dst_v = filter(tCgC(_, _, _, step_idx));
      // coord_v:ArithTuple(0,0,0) o (_2,_4):(_64@1,_1@0)

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        int64_t row_offset = get<0>(coord_v(i));
        int64_t col_offset = get<1>(coord_v(i));
        int64_t dst_idx = gather_index_ptr[row_offset];

        void *local_gather_outputs_ptr = reinterpret_cast<char *>(params_ptr->gather_outputs) +
                                         (dst_idx * 12288 + col_offset) * sizeof(Element);
        cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
            src_v(i), local_gather_outputs_ptr, guard);
        cutlass::arch::global_load<VecType, sizeof(VecType)>(
            src_v(i), local_gather_outputs_ptr, guard);

        int64_t dst_rank = dst_idx / this->tokens_per_rank;
        void *remote_rs_outputs_ptr =
            reinterpret_cast<char *>(rs_outputs_ptrs[dst_rank]) +
            ((params_ptr->rank * tokens_per_rank + dst_idx % tokens_per_rank) * 12288 +
             col_offset) *
                sizeof(Element);
        cutlass::arch::global_store<VecType, sizeof(VecType)>(
            src_v(i), remote_rs_outputs_ptr, guard);
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    Tensor mC = make_tensor(
        make_gmem_ptr(params_ptr->gemm_out),
        problem_shape,
        params_ptr->dAux);  // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tCgC = recast<VecType>(group_modes<3, 6>(
        ThreadMap::partition(mC, thread_idx, threadblock_tile_offset)));  // global tensor
    Tensor tCrC = make_tensor_like(take<0, 3>(tCgC));                     // register tensor

    // Generate the pred tensor
    Tensor mP = make_identity_tensor(mC.shape());
    Tensor tCrP = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(mP, thread_idx, threadblock_tile_offset)),
        Shape<Int<VecLength>>{},
        (_0{}));

    int nelem_per_thread = ceil_div(get<0>(problem_shape), blockDim.x);
    for (int i = 0; i < nelem_per_thread; ++i) {
      int pos = nelem_per_thread * threadIdx.x + i;
      int64_t *ptr = const_cast<int64_t *>(storage_ptr->smem_gather_index.data());
      if (pos < get<0>(problem_shape)) {
        ptr[pos] = params_ptr->gather_index[pos];
      }
    }
    if (threadIdx.x < params_ptr->world_size) {
      Element **ptr = const_cast<Element **>(storage_ptr->smem_rs_outputs_ptrs.data());
      ptr[threadIdx.x] = params_ptr->rs_outputs_ptrs[threadIdx.x];
    }

    __syncthreads();

    return Callbacks<decltype(tCgC), decltype(tCrC), decltype(tCrP), ProblemShape>(
        cute::move(tCgC),
        cute::move(tCrC),
        cute::move(tCrP),
        problem_shape,
        params_ptr,
        storage_ptr);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// GatherRsEpilogue
//
/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class ThreadMap,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class ThreadblockShape>
struct VisitorAuxGatherRs {
  struct Arguments {
    int64_t rank;
    int64_t world_size;
    StrideMNL dAux;
    Element *gemm_out;
    const int64_t *gather_index;
    Element *gather_outputs;
    const bool *finish_gather;
    Element **rs_outputs_ptrs;
  };

  struct Params {
    int64_t rank;
    int64_t world_size;
    StrideMNL dAux;
    Element *gemm_out;
    const int64_t *gather_index;
    Element *gather_outputs;
    const bool *finish_gather;
    Element *rs_outputs_ptrs[kMaxWorldSize];
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    Params params;
    params.rank = args.rank;
    params.world_size = args.world_size;
    params.dAux = args.dAux;

    params.gemm_out = args.gemm_out;
    params.gather_index = args.gather_index;
    params.gather_outputs = args.gather_outputs;
    params.finish_gather = args.finish_gather;
    for (int i = 0; i < args.world_size; ++i) {
      params.rs_outputs_ptrs[i] = args.rs_outputs_ptrs[i];
    }
    return params;
  }

  struct SharedStorage {
    cute::array_aligned<int64_t, 640> smem_gather_index;
    cute::array_aligned<bool, 640> smem_finish_gather;
    cute::array_aligned<Element *, kMaxWorldSize> smem_rs_outputs_ptrs;
  };

  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorAuxGatherRs() {}

  CUTLASS_HOST_DEVICE
  VisitorAuxGatherRs(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params), storage_ptr(&shared_storage) {}

  Params const *params_ptr;
  SharedStorage const *storage_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
        GTensor &&tCgC,
        RTensor &&tCrC,
        CTensor &&tCrP,
        ProblemShape problem_shape,
        Params const *params_ptr,
        SharedStorage const *storage_ptr)
        : tCgC(cute::forward<GTensor>(tCgC)),
          tCrC(cute::forward<RTensor>(tCrC)),
          tCrP(cute::forward<CTensor>(tCrP)),
          problem_shape(problem_shape),
          params_ptr(params_ptr),
          storage_ptr(storage_ptr) {
      this->tokens_per_rank = 4096 / params_ptr->world_size;
      this->gather_index_ptr = const_cast<int64_t *>(storage_ptr->smem_gather_index.data());
      this->finish_gather_ptr = const_cast<bool *>(storage_ptr->smem_finish_gather.data());
      this->rs_outputs_ptrs = const_cast<Element **>(storage_ptr->smem_rs_outputs_ptrs.data());
    }

    GTensor tCgC;
    RTensor tCrC;
    CTensor tCrP;
    Params const *params_ptr;
    SharedStorage const *storage_ptr;
    ProblemShape problem_shape;
    int64_t tokens_per_rank;
    int64_t *gather_index_ptr;
    bool *finish_gather_ptr;
    Element **rs_outputs_ptrs;

    CUTLASS_DEVICE void
    begin_step(int step_idx) {
      clear(tCrC);
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

      Tensor tCrC_frg = recast<Array<Element, FragmentSize>>(coalesce(tCrC));
      tCrC_frg(frg_idx) = convert_input(frg_input);

      return frg_input;
    }

    CUTLASS_DEVICE void
    end_step(int step_idx) {
      auto src_v = filter(tCrC);
      auto coord_v = filter(tCrP(_, _, _, step_idx));
      auto dst_v = filter(tCgC(_, _, _, step_idx));
      // coord_v:ArithTuple(0,0,0) o (_2,_4):(_64@1,_1@0)

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(src_v); ++i) {
        bool guard = elem_less(coord_v(i), problem_shape);
        int64_t row_offset = get<0>(coord_v(i));
        int64_t col_offset = get<1>(coord_v(i));
        int64_t dst_idx = gather_index_ptr[row_offset];

        void *local_gather_outputs_ptr = reinterpret_cast<char *>(params_ptr->gather_outputs) +
                                         (dst_idx * 12288 + col_offset) * sizeof(Element);
        cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
            src_v(i), local_gather_outputs_ptr, guard);

        if (finish_gather_ptr[row_offset]) {
          int64_t dst_rank = dst_idx / this->tokens_per_rank;
          cutlass::arch::global_load<VecType, sizeof(VecType)>(
              src_v(i), local_gather_outputs_ptr, guard);

          // Element** ptrs = const_cast<Element**>(storage_ptr->smem_rs_outputs_ptrs.data());
          // Element** ptrs = const_cast<Element**>(params_ptr->rs_outputs_ptrs);
          void *remote_rs_outputs_ptr =
              reinterpret_cast<char *>(rs_outputs_ptrs[dst_rank]) +
              (dst_idx % tokens_per_rank * 12288 + col_offset) * sizeof(Element);
          cutlass::arch::global_red<VecType, sizeof(VecType), half_t>(
              src_v(i), remote_rs_outputs_ptr, guard);
          // nvshmem_putmem_nbi(remote_rs_outputs_ptr, local_gather_outputs_ptr, sizeof(VecType),
          // dst_rank); nvshmem_putmem_nbi(rs_outputs_ptrs[params_ptr->rank],
          // rs_outputs_ptrs[params_ptr->rank], sizeof(VecType), params_ptr->rank);
        }
        // if (col_offset == 0) {
        //   printf("src_rank: %d, ptr_offset: %d, row_offset: %d, col_offset: %d, dst_idx: %d,
        //   dst_rank: %d, is_finished: %d, tokens_per_rank: %d\n",
        //          static_cast<int>(params_ptr->rank),
        //          static_cast<int>(ptr_offset),
        //          static_cast<int>(row_offset),
        //          static_cast<int>(col_offset),
        //          static_cast<int>(dst_idx),
        //          static_cast<int>(dst_rank),
        //          static_cast<int>(0),
        //          static_cast<int>(tokens_per_rank)
        //   );
        // }
      }
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
      gemm::GemmCoord threadblock_tile_offset, int thread_idx, ProblemShape problem_shape) {
    Tensor mC = make_tensor(
        make_gmem_ptr(params_ptr->gemm_out),
        problem_shape,
        params_ptr->dAux);  // (M,N,L)
    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tCgC = recast<VecType>(group_modes<3, 6>(
        ThreadMap::partition(mC, thread_idx, threadblock_tile_offset)));  // global tensor
    Tensor tCrC = make_tensor_like(take<0, 3>(tCgC));                     // register tensor

    // Generate the pred tensor
    Tensor mP = make_identity_tensor(mC.shape());
    Tensor tCrP = outer_partition(
        group_modes<3, 6>(ThreadMap::partition(mP, thread_idx, threadblock_tile_offset)),
        Shape<Int<VecLength>>{},
        (_0{}));

    int nelem_per_thread = ceil_div(get<0>(problem_shape), blockDim.x);
    for (int i = 0; i < nelem_per_thread; ++i) {
      int pos = nelem_per_thread * threadIdx.x + i;
      int64_t *ptr = const_cast<int64_t *>(storage_ptr->smem_gather_index.data());
      if (pos < get<0>(problem_shape)) {
        ptr[pos] = params_ptr->gather_index[pos];
      }
    }
    for (int i = 0; i < nelem_per_thread; ++i) {
      int pos = nelem_per_thread * threadIdx.x + i;
      bool *ptr = const_cast<bool *>(storage_ptr->smem_finish_gather.data());
      if (pos < get<0>(problem_shape)) {
        ptr[pos] = params_ptr->finish_gather[pos];
      }
    }
    if (threadIdx.x < params_ptr->world_size) {
      Element **ptr = const_cast<Element **>(storage_ptr->smem_rs_outputs_ptrs.data());
      ptr[threadIdx.x] = params_ptr->rs_outputs_ptrs[threadIdx.x];
    }

    __syncthreads();

    // __shared__ Element *smem_rs_outputs_ptrs[kMaxWorldSize];
    // if (threadIdx.x < params_ptr->world_size && threadIdx.y == 0 && threadIdx.z == 0) {
    //   smem_rs_outputs_ptrs[threadIdx.x] = params_ptr->rs_outputs_ptrs[threadIdx.x];
    //   // if (thread0()) {
    //   //   print("is shared memory?");
    //   //   print(isShared(storage_ptr->smem_rs_outputs_ptrs[0]));
    //   //   print("\n");
    //   // }
    // }
    // if (thread0()) {
    //   print("get<0>(problem_shape): ");
    //   print(get<0>(problem_shape));
    //   print("\n");
    // }
    // __shared__ int64_t smem_gather_index[800];
    // if (thread_idx < get<0>(problem_shape)) {
    //   smem_gather_index[thread_idx] = params_ptr->gather_index[thread_idx];
    //   // print("thread_idx: ");
    //   // print(thread_idx);
    //   // print("\n");
    //   // print("smem_gather_index[thread_idx]: ");
    //   // print(smem_gather_index[thread_idx]);
    //   // print("\n");
    // }
    // __syncthreads();

    // if (threadIdx.x < params_ptr->world_size && threadIdx.y == 0 && threadIdx.z == 0) {
    //   Element** ptrs = const_cast<Element**>(storage_ptr->smem_rs_outputs_ptrs.data());
    //   ptrs[threadIdx.x] = params_ptr->rs_outputs_ptrs[threadIdx.x];
    //   print("is shared memory?");
    //   print(isShared(ptrs[threadIdx.x]));
    // }
    // if (thread0()) {
    //   print("is shared memory?");
    //   print(isShared(storage_ptr->smem_rs_outputs_ptrs[0]));
    // }x
    // if (true) {
    //   printf("thread_idx:%d threadblock_tile_offset: %d %d %d problem_shape: %d %d
    //   threadblock_shape: %d %d %d\n",
    //     thread_idx,
    //     threadblock_tile_offset.m(),
    //     threadblock_tile_offset.n(),
    //     threadblock_tile_offset.k(),
    //     get<0>(problem_shape),
    //     get<1>(problem_shape),
    //     ThreadblockShape::kM,
    //     ThreadblockShape::kN,
    //     ThreadblockShape::kK
    //   );
    // }

    /*
    if (false && params_ptr->rank==LOG_RANK && !threadIdx.x && !threadIdx.y && !threadIdx.z) {
      for (size_t i = 0; i < params_ptr->world_size; ++i) {
        printf("scatter_out_ptr[%d]: %ld", i,
    reinterpret_cast<uint64_t>(params_ptr->scatter_ptr_aux[i]));
      }
      printf("gather_index: %ld\n", reinterpret_cast<uint64_t>(params_ptr->gather_index));

    }
    */

    return Callbacks<decltype(tCgC), decltype(tCrC), decltype(tCrP), ProblemShape>(
        cute::move(tCgC),
        cute::move(tCrC),
        cute::move(tCrP),
        problem_shape,
        params_ptr,
        storage_ptr);
  }
};

}  // namespace cutlass::epilogue::threadblock

/////////////////////////////////////////////////////////////////////////////////////////////////
