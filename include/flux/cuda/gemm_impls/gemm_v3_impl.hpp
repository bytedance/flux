//===- gemm_v3_impl.hpp ------------------------------------------- C++ ---===//
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

// This file should be included before any other cutlass device headers.
// The order of cutlass headers is carefully adjusted
#pragma once
#include "flux/flux.h"
#include "flux/cuda/cutlass_v3_builder.hpp"
#include "flux/gemm_operator_base.h"
namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT, class DerivedImpl>
struct GemmV3Impl
    : public GemmOperatorBaseDefaultImplMixin<GemmV3Impl<GemmMetaT, GemmHParamsT, DerivedImpl>> {
 public:
  using Base = GemmOperatorBaseDefaultImplMixin<GemmV3Impl>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3Impl)

  using ProblemShape = cute::tuple<int, int, int>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});

  template <int EpiSmemSize = 0>
  auto
  default_collective_mma(cute::Int<EpiSmemSize> carveout_smem_size = cute::_0{}) const {
    using CollectiveEpilogue = identity_t<decltype(this->default_collective_epilogue())>;
    constexpr int epi_smem_size = carveout_smem_size == 0
                                      ? sizeof(typename CollectiveEpilogue::SharedStorage)
                                      : carveout_smem_size;
    auto params = cutlass_v3_builder::default_mainloop_params(
        meta, hparams, TypeWrapper<void>{}, cute::Int<epi_smem_size>{});
    return cutlass_v3_builder::build_collective_mainloop(params);
  }

  auto
  default_collective_epilogue() const {
    auto params = cutlass_v3_builder::default_epilogue_params(meta, hparams);
    return cutlass_v3_builder::build_collective_epilogue(params);
  }

  auto
  default_tile_scheduler() const {
    return make_declval<cutlass::gemm::PersistentScheduler>();
  }

  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    using GemmKernel = decltype(static_cast<DerivedImpl const *>(this)->gemm_kernel());
    return make_declval<cutlass::gemm::device::GemmUniversalAdapter<
        FluxGemmKernel<GemmMetaT, GemmHParamsT, GemmKernel>>>();
  }

  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return static_cast<DerivedImpl const *>(this)->to_gemm_args(args);
  }

  template <class PtrC, class StrideC, class PtrD, class StrideD, class Alpha, class Beta>
  auto
  default_get_epilogue_args(
      PtrC ptr_C, StrideC stride_C, PtrD ptr_D, StrideD stride_D, Alpha alpha, Beta beta) const {
    using Epilogue = identity_t<decltype(this->default_collective_epilogue())>;
    typename Epilogue::Arguments epilogue{{}, ptr_C, stride_C, ptr_D, stride_D};
    if constexpr (meta.arch() == _Sm90{}) {
      if constexpr (not cute::is_void_v<typename Epilogue::ElementC>) {
        epilogue.thread = {
            // ternary op : beta * C + (alpha * acc)
            {{beta}},  // leaf op+args : beta
            {},        // leaf op+args : C
            {
                // binary op : alpha * acc
                {{alpha}},  // leaf op+args : alpha
                {},         // leaf op+args : acc
                {}          // binary args : multiplies
            },              // end binary op
            {}              // ternary args : multiply_add
        };
      } else {
        epilogue.thread = {
            {{alpha}},  // leaf op+args : alpha
            {},         // leaf op+args : acc
            {}          // binary args : multiplies
        };
      }
    } else {
      epilogue.ptr_D = ptr_D;
      epilogue.thread = {alpha, beta};
    }
    return epilogue;
  }
};
}  // namespace flux
}  // namespace bytedance
