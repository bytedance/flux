//===- gemm_v3_impl.hpp ------------------------------------------- C++ ---===//
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

// This file should be included before any other cutlass device headers.
// The order of cutlass headers is carefully adjusted
#pragma once
#include "flux/flux.h"
#include "flux/cuda/cutlass_v3_builder.hpp"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmV3BaseKernel {
  using ProblemShape = cute::tuple<int, int, int>;
  using ProblemShapeMNKL = cute::tuple<int, int, int, int>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr bool is_s8_gemm = is_s8_dtype(dt_conf.a()) && is_s8_dtype(dt_conf.b());

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

  template <class... Ts>
  auto
  s8_gemm_evt_d(cutlass_v3_builder::Sm90EpilogueParams<Ts...> params) const {
    constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

    using ElementC = decltype(params.element_c());
    using ElementCNonVoid = decltype(params.element_c_unvoid());
    using ElementD = decltype(params.element_d());
    using ElementAccumulator = decltype(params.element_accumulator());
    using ElementScale = float;
    using TileShape = decltype(params.tile_shape());

    static constexpr int AlignmentScale = 128 / cute::sizeof_bits_v<ElementScale>;
    static constexpr int AlignmentC = 128 / cute::sizeof_bits_v<ElementCNonVoid>;

    auto select_evt_d = []() {
      using Accum = cutlass::epilogue::fusion::Sm90AccFetch;
      using ScaleA = cutlass::epilogue::fusion::Sm90ColBroadcast<
          0,
          TileShape,
          ElementScale,
          ElementScale,
          cute::Stride<cute::Int<1>, cute::Int<0>, cute::Int<0>>,
          AlignmentScale,
          false>;
      using ScaleB = cutlass::epilogue::fusion::Sm90RowBroadcast<
          0,
          TileShape,
          ElementScale,
          ElementScale,
          cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>,
          AlignmentScale,
          false>;

      using Compute0 = cutlass::epilogue::fusion::
          Sm90Compute<cutlass::multiplies, ElementScale, ElementScale, RoundStyle>;
      using EVTCompute0 =
          cutlass::epilogue::fusion::Sm90EVT<Compute0, ScaleA, ScaleB>;  // scale_a * scale_b

      using Compute1 = cutlass::epilogue::fusion::
          Sm90Compute<cutlass::multiplies, ElementCNonVoid, ElementScale, RoundStyle>;
      // scale_a * scale_b * acc
      using EVTCompute1 = cutlass::epilogue::fusion::Sm90EVT<Compute1, EVTCompute0, Accum>;

      if constexpr (cute::is_void_v<ElementC>) {  // no bias
        return make_declval<EVTCompute1>();
      } else {
        using C = cutlass::epilogue::fusion::Sm90RowBroadcast<
            0,
            TileShape,
            ElementC,
            ElementC,
            cute::Stride<cute::Int<0>, cute::Int<1>, cute::Int<0>>,
            AlignmentC,
            false>;
        using Compute2 = cutlass::epilogue::fusion::
            Sm90Compute<cutlass::multiply_add, ElementD, ElementC, RoundStyle>;
        using EVTCompute2 = cutlass::epilogue::fusion::Sm90EVT<
            Compute2,
            cutlass::epilogue::fusion::Sm90ScalarBroadcast<ElementC>,  // beta = 1.0
            C,                                                         // C
            EVTCompute1>;  // (ElementC)scale_a * scale_b * acc + beta * bias
        return make_declval<EVTCompute2>();
      }
    };
    return select_evt_d();
  }

  auto
  default_collective_epilogue() const {
    auto params = cutlass_v3_builder::default_epilogue_params(meta, hparams);
    if constexpr (
        this->is_s8_gemm &&
        not cute::is_same_v<decltype(to_cutlass_element(dt_conf.d())), int32_t>) {
      auto s8_gemm_callbacks = []() {
        using ElementAccumulator = decltype(params.element_accumulator());
        using EVTD = decltype(s8_gemm_evt_d(params));
        using FusionCallbacks = typename cutlass::epilogue::collective::detail::CallbacksBuilder<
            decltype(params.dispatch_policy()),
            EVTD,
            decltype(params.tile_shape()),
            decltype(params.epilogue_tile_mn()),
            ElementAccumulator>::Callbacks;
        return make_declval<FusionCallbacks>();
      };
      using FusionCallbacks = decltype(s8_gemm_callbacks());
      auto new_params = params.fusion_callbacks(TypeWrapper<FusionCallbacks>{});
      return cutlass_v3_builder::build_collective_epilogue(new_params);
    } else {
      return cutlass_v3_builder::build_collective_epilogue(params);
    }
  }

  auto
  default_tile_scheduler() const {
    return make_declval<cutlass::gemm::PersistentScheduler>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT, class DerivedImpl>
class GemmV3BaseDevice : public GemmOperatorBaseDefaultImplMixin<
                             GemmV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, DerivedImpl>> {
 public:
  using Base = GemmOperatorBaseDefaultImplMixin<GemmV3BaseDevice>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3BaseDevice)

  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  using KernelBuilder = GemmV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr bool is_s8_gemm = KernelBuilder::is_s8_gemm;

  //////////////////////////
  // CRTP functions
  //////////////////////////
  auto
  gemm_device() const {
    return make_declval<cutlass::gemm::device::GemmUniversalAdapter<GemmKernelT>>();
  }

  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    return static_cast<DerivedImpl const *>(this)->to_gemm_args(args);
  }
};
}  // namespace flux
}  // namespace bytedance
