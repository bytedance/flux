//===- gemm_v3_pre_attn_a2a_transpose_kernel.hpp --------------- C++ ------===//
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

#pragma once
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "flux/cuda/cutlass_v3_builder.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/gemm_a2a_transpose.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_v3_impl.hpp"

#include "gemm_a2a_transpose/sm90_evt.hpp"

namespace bytedance::flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmV3PreAttnAllToAllTranspose_Kernel : public GemmV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static_assert(meta.arch() == _Sm90{}, "requires _Sm90{}");
  static_assert(
      meta.comm_op() == _PreAttnAllToAllTranspose{} || meta.comm_op() == _PreAttnQKVPackAllToAll{},
      "requires _PreAttnAllToAllTranspose{} or _PreAttnQKVPackAllToAll{}");

  auto
  tile_scheduler() const {
    if constexpr (hparams.gemm_kind() == _GemmStreamK{}) {
      return make_declval<cutlass::gemm::StreamKScheduler>();
    } else {
      return make_declval<cutlass::gemm::PersistentScheduler>();
    }
  }

  auto
  sm90_collective_epilogue() const {
    cutlass_v3_builder::Sm90EpilogueParams params =
        cutlass_v3_builder::default_epilogue_params(meta, hparams);
    using OldDispatchPolicy = decltype(params.dispatch_policy());
    // reduce DispatchPolicy's stageD to 1
    using DispatchPolicy = cutlass::epilogue::Sm90TmaWarpSpecialized<
        /*StagesC=*/1,
        /*StagesD=*/1,
        OldDispatchPolicy::FragmentSize,
        OldDispatchPolicy::ReuseSmemC,
        OldDispatchPolicy::DelayTmaStore>;
    auto compose_fusion_callbacks = [this, &params]() {
      constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
      using namespace cutlass::epilogue::fusion;

      using ElementC = decltype(params.element_c());
      using ElementCNonVoid = decltype(params.element_c_unvoid());
      using ElementD = decltype(params.element_d());
      using ElementCompute = ElementD;
      using ElementAccumulator = decltype(params.element_accumulator());

      auto select_evt_d = [this, &params]() {
        using EVT_Compute0 = Sm90EVT<
            Sm90Compute<
                cutlass::multiplies,
                ElementD,
                ElementCompute,
                RoundStyle>,                          // alpha * acc
            Sm90ScalarBroadcast<ElementAccumulator>,  // alpha
            Sm90AccFetch                              // acc
            >;
        if constexpr (cute::is_void_v<ElementC>) {
          return make_declval<EVT_Compute0>();
        } else {
          using EVT_Compute1 = Sm90EVT<  // D
              Sm90Compute<
                  cutlass::multiply_add,
                  ElementD,
                  ElementCompute,
                  RoundStyle>,                          // beta * C + (alpha * acc)
              Sm90ScalarBroadcast<ElementAccumulator>,  // beta
              Sm90SrcFetch<ElementCNonVoid>,            // C
              EVT_Compute0>;
          return make_declval<EVT_Compute1>();
        }
      };

      using EVT_D = decltype(select_evt_d());
      auto select_evt_final = []() {
        using AuxSetReadyType = Sm90AuxSetReady<
            DispatchPolicy::StagesD,
            decltype(params.tile_shape()),
            decltype(params.epilogue_tile_mn()),
            ElementD,
            RoundStyle,
            decltype(params.stride_d()),
            decltype(params.smem_layout_atom_d()),
            decltype(params.copy_op_r2s())>;
        return make_declval<Sm90EVT<AuxSetReadyType, EVT_D>>();
      };

      using CustomEVT = decltype(select_evt_final());

      using FusionCallbacks = typename cutlass::epilogue::collective::detail::CallbacksBuilder<
          DispatchPolicy,
          CustomEVT,
          decltype(params.tile_shape()),
          decltype(params.epilogue_tile_mn()),
          ElementAccumulator>::Callbacks;
      return make_declval<FusionCallbacks>();
    };
    using FusionCallbacks = decltype(compose_fusion_callbacks());
    auto new_params = params.dispatch_policy(TypeWrapper<DispatchPolicy>{})
                          .fusion_callbacks(TypeWrapper<FusionCallbacks>{});
    return cutlass_v3_builder::build_collective_epilogue(new_params);
  }

  auto
  collective_epilogue() const {
    if constexpr (meta.arch() == _Sm90{}) {
      return this->sm90_collective_epilogue();
    } else {
      static_assert(cutlass::detail::dependent_false<decltype(meta.arch())>, "unsupported arch");
    }
  }

  auto
  gemm_kernel() const {
    using CollectiveEpilogue = identity_t<decltype(this->collective_epilogue())>;
    using CollectiveMma = decltype(this->default_collective_mma(
        cute::Int<sizeof(typename CollectiveEpilogue::SharedStorage)>{}));
    using TileScheduler = decltype(this->tile_scheduler());
    return make_declval<cutlass::gemm::kernel::GemmUniversal<
        typename Base::ProblemShape,
        CollectiveMma,
        CollectiveEpilogue,
        TileScheduler>>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV3PreAttnAllToAllTranspose_Device
    : public GemmV3BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmV3PreAttnAllToAllTranspose_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base = GemmV3BaseDevice<
      GemmMetaT,
      GemmHParamsT,
      GemmKernelT,
      GemmV3PreAttnAllToAllTranspose_Device>;
  using KernelBuilder = GemmV3PreAttnAllToAllTranspose_Kernel<GemmMetaT, GemmHParamsT>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3PreAttnAllToAllTranspose_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  auto
  to_gemm_args_impl(GemmAllToAllTransposeArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    using GemmKernel = GemmKernelT;
    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    constexpr bool has_bias = not cute::is_void_v<ElementC>;
    using ElementCNonVoid = cute::conditional_t<has_bias, ElementC, ElementD>;

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_C = static_cast<ElementCNonVoid const *>(args.bias);
    auto ptr_barriers = reinterpret_cast<int **>(args.barrier_ptrs);
    auto ptr_D = reinterpret_cast<ElementD *>(args.gemm_output);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, 1));

    auto get_epilogue_args = [&]() {
      typename GemmKernel::EpilogueArguments epilogue{
          {}, has_bias ? ptr_C : nullptr, stride_C, ptr_D, stride_D};
      if constexpr (has_bias) {
        epilogue.thread = {
            // unary op: aux store D
            {
                // ternary op : beta * C + (alpha * acc)
                {{args.beta}},  // leaf op+args : beta
                {},             // leaf op+args : C
                {
                    // binary op : alpha * acc
                    {{args.alpha}},  // leaf op+args : alpha
                    {},              // leaf op+args : acc
                    {}               // binary args : multiplies
                },                   // end binary op
                {}                   // ternary args : multiply_add
            },
            {.barrier_ptr_aux = ptr_barriers[args.rank]}  // unary args: aux store D
        };
      } else {
        epilogue.thread = {
            // unary op: aux store D
            {
                // binary op : alpha * acc
                {{args.alpha}},  // leaf op+args : alpha
                {},              // leaf op+args : acc
                {}               // binary args : multiplies
            },
            {.barrier_ptr_aux = ptr_barriers[args.rank]}  // unary args: aux store D
        };
      }
      return epilogue;
    };
    auto epilogue = get_epilogue_args();

    using TileScheduler = typename GemmKernel::TileScheduler;
    using TileSchedulerTag = decltype(KernelBuilder().tile_scheduler());
    auto scheduler = typename TileScheduler::Arguments{};

    if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    } else if (hparams.raster_order() == _RasterAlongM{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    }
    cutlass::KernelHardwareInfo hw_info{};
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0) - args.sm_margin;
    return GemmArguments{/*mode=*/cutlass::gemm::GemmUniversalMode::kGemm,
                         /*problem_shape=*/{args.m, args.n, args.k},
                         /*mainloop=*/
                         {ptr_A, stride_A, ptr_B, stride_B},
                         /*epilogue=*/epilogue,
                         /*hw_info=*/hw_info,
                         /*scheduler=*/scheduler};
  }

 public:
  auto
  to_gemm_args(std::any const &args) const {
    return to_gemm_args_impl(std::any_cast<GemmAllToAllTransposeArguments>(args));
  }

  std::size_t
  get_barrier_workspace_size(std::any const &var_args) const override {
    auto align_buffer = [](size_t size) { return (size + 127) / 128 * 128; };
    const auto &args = std::any_cast<GemmAllToAllTransposeArguments>(var_args);
    auto [tile_m, tile_n, tile_k] = hparams.tile_shape();
    // for each tile, one flag for finished writing to local
    std::size_t nflags = cute::ceil_div(args.m, tile_m) * cute::ceil_div(args.n, tile_n);
    return align_buffer(nflags * sizeof(int));
  }
};
}  // namespace bytedance::flux
