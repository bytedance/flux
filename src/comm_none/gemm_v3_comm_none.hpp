//===- gemm_v3_comm_none.hpp -------------------------------------- C++ ---===//
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
#include <utility>
#include "cute/layout.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_v3_impl.hpp"
#include "flux/args/comm_none.h"

namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmV3CommNone_Kernel : public GemmV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr bool is_fp8_gemm = is_fp8_dtype(dt_conf.a()) && is_fp8_dtype(dt_conf.b());
  static_assert(meta.comm_op() == _CommNone{}, "requires _CommNone{}");

  auto
  tile_scheduler() const {
    if constexpr (hparams.gemm_kind() == _GemmDefault{}) {
      return make_declval<cutlass::gemm::PersistentScheduler>();
    } else if constexpr (hparams.gemm_kind() == _GemmStreamK{}) {
      return make_declval<cutlass::gemm::StreamKScheduler>();
    } else {
      static_assert(cutlass::detail::dependent_false<GemmHParamsT>, "not supported");
    }
  }

  auto
  gemm_kernel() const {
    using CollectiveMma = decltype(this->default_collective_mma());
    using CollectiveEpilogue = decltype(this->default_collective_epilogue());
    // TODO(wenlei.bao): unify below kernels
    if constexpr (is_fp8_gemm and meta.impl_spec().block_scale()) {
      return make_declval<cute::conditional_t<
          hparams.gemm_kind() == _GemmStreamK{},
          cutlass::gemm::kernel::GemmUniversal<
              cute::Shape<int, int, int, int>,  // Indicates ProblemShape
              CollectiveMma,
              CollectiveEpilogue,
              cutlass::gemm::StreamKScheduler>,
          cutlass::gemm::kernel::GemmUniversal<
              cute::Shape<int, int, int, int>,  // Indicates ProblemShape
              CollectiveMma,
              CollectiveEpilogue>>>();
    } else {
      using TileScheduler = decltype(this->tile_scheduler());
      return make_declval<cutlass::gemm::kernel::GemmUniversal<
          typename Base::ProblemShape,
          CollectiveMma,
          CollectiveEpilogue,
          TileScheduler>>();
    }
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV3CommNone_Device : public GemmV3BaseDevice<
                                  GemmMetaT,
                                  GemmHParamsT,
                                  GemmKernelT,
                                  GemmV3CommNone_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base = GemmV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmV3CommNone_Device>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3CommNone_Device)

  using KernelBuilder = GemmV3CommNone_Kernel<GemmMetaT, GemmHParamsT>;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr bool is_s8_gemm = is_s8_dtype(dt_conf.a()) && is_s8_dtype(dt_conf.b());
  static constexpr bool is_fp8_gemm = is_fp8_dtype(dt_conf.a()) && is_fp8_dtype(dt_conf.b());
  static constexpr bool is_blockscale_gemm = meta.impl_spec().block_scale();

  auto
  to_s8_gemm_args_impl(S8GemmDequantArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = GemmKernelT;
    using GemmArguments = typename Gemm::Arguments;
    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    using ElementCNonVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
    using ElementScale = float;
    static constexpr bool has_bias = not cute::is_void_v<ElementC>;
    static constexpr bool s8_gemm_only = cute::is_same_v<ElementD, int32_t>;
    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_C = static_cast<ElementC const *>(args.bias);
    auto ptr_D = static_cast<ElementD *>(args.D);
    [[maybe_unused]] auto ptr_scale_A = static_cast<ElementScale const *>(args.scale_A);
    [[maybe_unused]] auto ptr_scale_B = static_cast<ElementScale const *>(args.scale_B);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
    // bias shape: [1, n], stride: [0, 1]
    cute::tuple<int64_t, cute::Int<1>, int64_t> stride_C = {
        s8_gemm_only ? args.n : cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}};
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, 1));

    using Epilogue = typename GemmKernel::CollectiveEpilogue;
    typename GemmKernel::EpilogueArguments epilogue{
        {}, has_bias ? ptr_C : nullptr, stride_C, ptr_D, stride_D};
    auto beta = static_cast<ElementCNonVoid>(args.beta);

    if constexpr (not s8_gemm_only) {
      if constexpr (has_bias) {
        epilogue.thread = {
            // ternary op : beta * C + (scale_a * scale_b * Acc)
            {beta},   // beta
            {ptr_C},  // bias
            {
                // binary op : (scale_a * scale_b) * Acc
                {
                    // scale_a * scale_b
                    {ptr_scale_A},  // scale_a
                    {ptr_scale_B},  // scale_b
                    {}              // binary args : multiplies
                },                  // end binary op
                {},                 // Acc
                {}                  // binary args : multiplies
            },                      // end binary op
            {}                      // ternary args : multiply_add
        };  // end ternary op
      } else {
        epilogue.thread = {
            // (scale_a * scale_b) * Acc
            {
                {ptr_scale_A},  // scale_a
                {ptr_scale_B},  // scale_b
                {}              // binary args : multiplies
            },                  // end binary op
            {},                 // Acc
            {}                  // binary args : multiplies
        };  // end binary op
      }
    } else {
      epilogue.thread = decltype(epilogue.thread){
          .alpha = static_cast<ElementD>(args.alpha), .beta = static_cast<ElementD>(args.beta)};
    }

    using TileScheduler = typename GemmKernel::TileScheduler;
    auto scheduler = typename TileScheduler::Arguments{};

    auto [m_tile_size, n_tile_size, _] = hparams.tile_shape();
    int min_tile_count =
        min(cute::ceil_div(args.m, m_tile_size), cute::ceil_div(args.n, n_tile_size));

    // don't set `max_swizzle_size` to avoid perf diff with cutlass profiler
    // if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
    //   scheduler.max_swizzle_size = min_tile_count;
    // }
    if constexpr (hparams.raster_order() == _RasterAlongM{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    } else if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    }
    return GemmArguments{
        /*mode=*/cutlass::gemm::GemmUniversalMode::kGemm,
        /*problem_shape=*/{args.m, args.n, args.k},
        /*mainloop=*/
        {ptr_A, stride_A, ptr_B, stride_B},
        /*epilogue=*/epilogue,
        /*hw_info=*/{},
        /*scheduler=*/scheduler};
  }

  auto
  to_blockscale_gemm_args_impl(BlockScaleGemmArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = GemmKernelT;
    using GemmArguments = typename Gemm::Arguments;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    using ElementCNonVoid = cute::conditional_t<cute::is_void_v<ElementC>, ElementD, ElementC>;
    using ElementBlockScale = float;

    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_C = static_cast<ElementC const *>(args.C);
    auto ptr_D = static_cast<ElementD *>(args.D);
    auto ptr_blockscale_A = static_cast<ElementBlockScale const *>(args.blockscale_A);
    auto ptr_blockscale_B = static_cast<ElementBlockScale const *>(args.blockscale_B);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, args.l));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, args.l));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(args.m, args.n, args.l));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, args.l));

    using Epilogue = typename GemmKernel::CollectiveEpilogue;
    typename Epilogue::Arguments epilogue{{}, ptr_C, stride_C, ptr_D, stride_D};
    epilogue.thread = decltype(epilogue.thread){
        .alpha = args.alpha,
        .beta = args.beta,
        .alpha_ptr = nullptr,
        .beta_ptr = nullptr,
        .scale_a = args.scale_a,
        .scale_b = args.scale_b,
        .scale_c = args.scale_c,
        .scale_d = args.scale_d,
        .scale_aux = args.scale_aux,
        .bias_ptr = nullptr,
    };

    GemmArguments gemm_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k, args.l},
        {ptr_A,
         stride_A,
         ptr_B,
         stride_B,
         args.mma_promotion_interval,
         ptr_blockscale_A,
         ptr_blockscale_B},
        epilogue};

    // TODO: maybe should add to flux search space
    using TileScheduler = typename GemmKernel::TileScheduler;

    auto [m_tile_size, n_tile_size, _] = hparams.tile_shape();
    int min_tile_count =
        min(cute::ceil_div(args.m, m_tile_size), cute::ceil_div(args.n, n_tile_size));
    if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
      gemm_args.scheduler.max_swizzle_size = min_tile_count;
    }

    if constexpr (hparams.raster_order() == _RasterAlongM{}) {
      gemm_args.scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    } else if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      gemm_args.scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    }
    return gemm_args;
  }

  auto
  to_gemm_args_impl(GemmOnlyArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = GemmKernelT;
    using GemmArguments = typename Gemm::Arguments;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_C = static_cast<ElementC const *>(args.bias);
    auto ptr_D = static_cast<ElementD *>(args.output);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, 1));

    using Epilogue = typename GemmKernel::CollectiveEpilogue;
    typename Epilogue::Arguments epilogue{{}, ptr_C, stride_C, ptr_D, stride_D};
    epilogue.thread = decltype(epilogue.thread){.alpha = args.alpha, .beta = args.beta};

    using TileScheduler = typename GemmKernel::TileScheduler;
    auto scheduler = typename TileScheduler::Arguments{};

    auto [m_tile_size, n_tile_size, _] = hparams.tile_shape();
    int min_tile_count =
        min(cute::ceil_div(args.m, m_tile_size), cute::ceil_div(args.n, n_tile_size));

    if constexpr (hparams.gemm_kind() != _GemmStreamK{}) {
      scheduler.max_swizzle_size = min_tile_count;
    }
    if constexpr (hparams.raster_order() == _RasterAlongM{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    } else if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    }
    return GemmArguments{
        /*mode=*/cutlass::gemm::GemmUniversalMode::kGemm,
        /*problem_shape=*/{args.m, args.n, args.k},
        /*mainloop=*/
        {ptr_A, stride_A, ptr_B, stride_B},
        /*epilogue=*/epilogue,
        /*hw_info=*/{},
        /*scheduler=*/scheduler};
  }

 public:
  auto
  to_gemm_args(std::any const &args) const {
    if constexpr (this->is_s8_gemm) {
      return to_s8_gemm_args_impl(std::any_cast<S8GemmDequantArguments>(args));
    } else if constexpr (this->is_fp8_gemm && this->is_blockscale_gemm) {
      return to_blockscale_gemm_args_impl(std::any_cast<BlockScaleGemmArguments>(args));
    } else {
      return to_gemm_args_impl(std::any_cast<GemmOnlyArguments>(args));
    }
  }
};
}  // namespace flux
}  // namespace bytedance
