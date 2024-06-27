//===- sm90_epologue_evt.hpp -------------------------------------- C++ ---===//
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
// Some code from
// cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp
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
  \brief Visitor tree store operations for the sm90 TMA warp-specialized (ws) epilogue
*/

#pragma once

#include <cstddef>
#include "cute/arch/copy_sm90_tma.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"
#include "cute/container/tuple.hpp"
#include "cute/int_tuple.hpp"
#include "cute/layout.hpp"
#include "cute/pointer.hpp"
#include "cute/util/debug.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/array.h"
#include "cutlass/barrier.h"
#include "cutlass/cutlass.h"

#include "flux/flux.h"
#include "flux/cuda/memory_utils.hpp"
#include "cute/tensor.hpp"
#include "cutlass/detail/helper_macros.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;
using namespace bytedance::flux;

/////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////////
// Epilogue Store For _ReduceScatter{}
//   Update flag if tile has been written to global memory
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
    int Stages,
    class TileShape,
    class EpilogueTile,
    class Element,
    FloatRoundStyle RoundStyle,
    class StrideMNL,
    class SmemLayoutAtom,
    class CopyOpR2S,
    CommKindEnum CommKind,
    int Alignment = 128 / sizeof_bits_v<Element>>
struct Sm90AuxStoreReduceScatter {
  using ElementAux = Element;
  static_assert(
      Alignment * sizeof_bits_v<Element> % 128 == 0, "sub-16B alignment not supported yet");

  constexpr static bool is_m_major = epilogue::collective::detail::is_m_major<StrideMNL>();
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

  struct Arguments {
    int *barrier_ptr_aux;
  };

  struct Params {
    int *barrier_ptr;
  };

  struct SharedStorage {};

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(
      ProblemShape const &problem_shape, Arguments const &args, void *workspace) {
    // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
    auto problem_shape_mnkl = append<4>(problem_shape, 1);
    auto [M, N, K, L] = problem_shape_mnkl;
    Params params;

    params.barrier_ptr = args.barrier_ptr_aux;

    return params;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const &problem_shape, Arguments const &args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(
      ProblemShape const &problem_shape,
      Arguments const &args,
      void *workspace,
      cudaStream_t stream,
      CudaHostAdapter *cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90AuxStoreReduceScatter() {}

  CUTLASS_HOST_DEVICE
  Sm90AuxStoreReduceScatter(Params const &params, SharedStorage const &shared_storage)
      : params_ptr(&params) {}

  Params const *params_ptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const &args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class Barrier>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        Params const *params_ptr,
        int thread_idx,
        tuple<int32_t, int32_t, int32_t> problem_shape_mnl,
        tuple<int32_t, int32_t, int32_t> tile_coord_mnl,
        Barrier *barrier_mgr [[maybe_unused]])
        : params_ptr(params_ptr),
          thread_idx(thread_idx),
          problem_shape_mnl(cute::move(problem_shape_mnl)),
          tile_coord_mnl(cute::move(tile_coord_mnl)) {
      auto [M, N, L] = problem_shape_mnl;
      int m_tiles = cute::ceil_div(M, size<0>(TileShape{}));
      int n_tiles = cute::ceil_div(N, size<1>(TileShape{}));
      tile_layout = make_layout(make_shape(m_tiles, n_tiles));
    }

    Params const *params_ptr;
    const int thread_idx;
    const tuple<int32_t, int32_t, int32_t> problem_shape_mnl;
    const tuple<int32_t, int32_t, int32_t> tile_coord_mnl;
    Layout<Shape<int32_t, int32_t>> tile_layout;

    template <typename ElementAccumulator, typename ElementInput, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(
        Array<ElementAccumulator, FragmentSize> const &frg_acc,
        int epi_v,
        int epi_m,
        int epi_n,
        Array<ElementInput, FragmentSize> const &frg_input) {
      return frg_input;
    }

    CUTLASS_DEVICE void
    end() {
      auto [m, n, _] = tile_coord_mnl;
      if (m >= size<0>(tile_layout.shape()) or n >= size<1>(tile_layout.shape())) {
        // early exit if out of bound
        return;
      }
      int tile_idx = tile_layout(m, n);
      int flag_idx = tile_idx * 2;
      Barrier::wait_eq_reset(params_ptr->barrier_ptr, thread_idx, flag_idx, 0, 1);
    }

    template <class T>
    CUTLASS_DEVICE void
    debug_print_v(T const &val, char const *name) {
      if (params_ptr->local_rank == 0 and block0() and thread_idx == 0) {
        print("%s:", name);
        print(val);
        print("\n");
      }
    }
  };

  template <
      bool ReferenceSrc,  // do register tensors reference the src or dst layout of the
                          // tiled copy
      class... Args>
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const &args) {
    auto [tile_M, tile_N, tile_K] = args.tile_shape_mnk;
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    auto [epi_tile_m, epi_tile_n] = args.epi_tile;

    constexpr int ThreadCount = size(decltype(args.tiled_copy){});
    using BarrierSync = cutlass::detail::
        NamedBarrierSync<ThreadCount, (int)FluxNamedBarriers::ReduceScatterEpilogue>;
    using Barrier = cutlass::detail::GenericSystemBarrier<BarrierSync>;

    return ConsumerStoreCallbacks(
        params_ptr,
        args.thread_idx,
        make_tuple(int32_t(M), int32_t(N), int32_t(L)),
        make_tuple(int32_t(m), int32_t(n), int32_t(l)),
        static_cast<Barrier *>(nullptr));
  }
};

}  // namespace cutlass::epilogue::fusion

/////////////////////////////////////////////////////////////////////////////////////////////////
