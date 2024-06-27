//===- gemm_v3_ag_kernel.hpp -------------------------------------- C++ ---===//
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
#include "cute/container/tuple.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/all_gather.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_v3_impl.hpp"
#include "all_gather/sm90_all_gather_gemm_tile_scheduler.hpp"
#include "all_gather/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp"

namespace cutlass::gemm {
struct AGKernelTileScheduler {};
namespace kernel::detail {
class Sm90AGKernelTileScheduler;
}  // namespace kernel::detail
}  // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail {
template <class ArchTag, class TileShape, class ClusterShape>
struct TileSchedulerSelector<AGKernelTileScheduler, ArchTag, TileShape, ClusterShape> {
  using Scheduler = Sm90AGKernelTileScheduler;
};
}  // namespace cutlass::gemm::kernel::detail

namespace bytedance::flux {
using SystemBarrier = cutlass::Barrier;
template <class GemmMetaT, class GemmHParamsT>
class GemmV3AGKernel
    : public GemmV3Impl<GemmMetaT, GemmHParamsT, GemmV3AGKernel<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmV3Impl<GemmMetaT, GemmHParamsT, GemmV3AGKernel>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3AGKernel)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static constexpr auto comm_spec = hparams.comm_spec();

  static_assert(meta.arch() == _Sm90{}, "requires _Sm90{}");
  static_assert(meta.comm_op() == _AGKernel{}, "requires _AGKernel{}");

  auto
  tile_scheduler() const {
    if constexpr (hparams.gemm_kind() == _GemmStreamK{}) {
      return make_declval<cutlass::gemm::StreamKScheduler>();
    } else {
      return make_declval<cutlass::gemm::AGKernelTileScheduler>();
    }
  }

  auto
  gemm_kernel() const {
    using CollectiveMma = decltype(this->default_collective_mma());
    using CollectiveEpilogue = decltype(this->default_collective_epilogue());
    using TileScheduler = decltype(this->tile_scheduler());
    return make_declval<cutlass::gemm::kernel::Sm90AGGemmUniversal<
        typename Base::ProblemShape,
        CollectiveMma,
        CollectiveEpilogue,
        TileScheduler>>();
  }

  auto
  to_gemm_args_impl(AGKernelArguments const &args) const {
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
    using TileSchedulerTag = decltype(this->tile_scheduler());
    auto scheduler = typename TileScheduler::Arguments{};

    if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelTileScheduler>) {
      if constexpr (hparams.raster_order() == _RasterAlongN{}) {
        scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
      } else {
        scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
      }

      scheduler.nnodes = args.nnodes;
      scheduler.rank = args.rank;
      scheduler.world_size = args.world_size;
      scheduler.local_world_size = args.world_size / args.nnodes;
      scheduler.local_rank = args.rank % scheduler.local_world_size;

      auto ptr_barrier = reinterpret_cast<typename SystemBarrier::T *>(args.barrier_buffer);
      scheduler.ptr_barrier = ptr_barrier;
    }

    auto ptr_full_A = static_cast<ElementA const *>(args.input_buffer);
    return GemmArguments{
        /*mode=*/cutlass::gemm::GemmUniversalMode::kGemm,
        /*problem_shape=*/{args.m, args.n, args.k},
        /*mainloop=*/
        {ptr_full_A, stride_A, ptr_B, stride_B},
        /*epilogue=*/epilogue,
        /*hw_info=*/{},
        /*scheduler=*/scheduler,
        args.barrier_buffer,
        args.rank,
        args.world_size};
  }

  auto
  to_gemm_args(std::any const &args) const {
    return to_gemm_args_impl(std::any_cast<AGKernelArguments>(args));
  }
};

using namespace cute;
struct GemmV3AGKernel_Space : OpSpaceBase<GemmV3AGKernel_Space> {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_AGKernel{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

  // [TODO]: epilogue scheduler: NoSmemWarpSpecialized ?
  static constexpr auto AllGemmHParams = make_space_gemm_hparams(
      cute::make_tuple(
          make_gemm_v3_hparams(Shape<_2, _1, _1>{}), make_gemm_v3_hparams(Shape<_1, _2, _1>{})),
      cute::make_tuple(Auto{}),
      cute::make_tuple(
          Shape<_128, _256, _64>{}, Shape<_256, _128, _64>{}, Shape<_128, _128, _64>{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(cute::_3{}, cute::_4{}),
      cute::make_tuple(_RasterAlongN{}, _RasterAlongM{}));
};
}  // namespace bytedance::flux
