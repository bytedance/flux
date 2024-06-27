//===- gemm_v3_comm_none.hpp -------------------------------------- C++ ---===//
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

#pragma once
#include <memory>
#include <utility>
#include "cutlass/conv/convolution.h"
#include "cutlass/detail/dependent_false.hpp"
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
class GemmV3CommNone
    : public GemmV3Impl<GemmMetaT, GemmHParamsT, GemmV3CommNone<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmV3Impl<GemmMetaT, GemmHParamsT, GemmV3CommNone>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3CommNone)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

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
    using TileScheduler = decltype(this->tile_scheduler());
    return make_declval<cutlass::gemm::kernel::GemmUniversal<
        typename Base::ProblemShape,
        CollectiveMma,
        CollectiveEpilogue,
        TileScheduler>>();
  }

  auto
  to_gemm_args_impl(GemmOnlyArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = identity_t<decltype(this->gemm_kernel())>;
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
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, cute::Int<1>{}));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, cute::Int<1>{}));
    auto stride_C = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, cute::make_shape(args.m, args.n, cute::Int<1>{}));
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, cute::Int<1>{}));

    auto epilogue =
        this->default_get_epilogue_args(ptr_C, stride_C, ptr_D, stride_D, args.alpha, args.beta);

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
    return to_gemm_args_impl(std::any_cast<GemmOnlyArguments>(args));
  }
};

using namespace cute;
struct GemmV3CommNone_Space : OpSpaceBase<GemmV3CommNone_Space> {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

  static constexpr auto AllGemmHParams = make_space_gemm_hparams(
      tuple_transform(
          tuple_cartesian_product(
              cute::make_tuple(Shape<_2, _1, _1>{}, Shape<_1, _2, _1>{}),
              cute::make_tuple(_Cooperative{}, _PingPong{})),
          [](auto tup) { return to_gemm_v3_hparams(tup); }),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(_GemmDefault{}));
};

}  // namespace flux
}  // namespace bytedance
