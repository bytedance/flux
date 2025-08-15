//===- gemm_v3_ag_kernel.hpp -------------------------------------- C++ ---===//
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
#include "cute/container/tuple.hpp"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/ag_gemm.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_v3_impl.hpp"
#include "ag_gemm/sm90_all_gather_gemm_tile_scheduler.hpp"
#include "ag_gemm/sm90_all_gather_gemm_tile_scheduler_stream_k.hpp"
#include "ag_gemm/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp"
#include "ag_gemm/sm90_all_gather_gemm_tma_warpspecialized_pingpong.hpp"

namespace cutlass::gemm {
struct AGKernelTileScheduler {};
struct AGKernelStreamKScheduler {};
namespace kernel::detail {
class Sm90AGKernelTileScheduler;
}  // namespace kernel::detail
}  // namespace cutlass::gemm

namespace cutlass::gemm::kernel::detail {
template <class ArchTag, class TileShape, class ClusterShape>
struct TileSchedulerSelector<AGKernelTileScheduler, ArchTag, TileShape, ClusterShape> {
  using Scheduler = Sm90AGKernelTileScheduler;
};

template <class ArchTag, class TileShape, class ClusterShape>
struct TileSchedulerSelector<AGKernelStreamKScheduler, ArchTag, TileShape, ClusterShape> {
  using Scheduler = Sm90AGKernelPersistentTileSchedulerSm90StreamK<TileShape, ClusterShape>;
};
}  // namespace cutlass::gemm::kernel::detail

namespace bytedance::flux {
using SystemBarrier = cutlass::Barrier;

template <class GemmMetaT, class GemmHParamsT>
struct GemmV3AGKernel_Kernel : public GemmV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});

  static_assert(meta.arch() == _Sm90{}, "requires _Sm90{}");
  static_assert(meta.comm_op() == _AGKernel{}, "requires _AGKernel{}");

  auto
  tile_scheduler() const {
    if constexpr (hparams.gemm_kind() == _GemmStreamK{}) {
      return make_declval<cutlass::gemm::AGKernelStreamKScheduler>();
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
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV3AGKernel_Device : public GemmV3BaseDevice<
                                  GemmMetaT,
                                  GemmHParamsT,
                                  GemmKernelT,
                                  GemmV3AGKernel_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base = GemmV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmV3AGKernel_Device>;
  using KernelBuilder = GemmV3AGKernel_Kernel<GemmMetaT, GemmHParamsT>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3AGKernel_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  auto
  to_s8_gemm_args_impl(AGS8KernelArguments const &args) const {
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

    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_C = static_cast<ElementC const *>(args.bias);
    auto ptr_D = static_cast<ElementD *>(args.output);
    [[maybe_unused]] auto ptr_scale_A = static_cast<ElementScale const *>(args.scale_A);
    [[maybe_unused]] auto ptr_scale_B = static_cast<ElementScale const *>(args.scale_B);
    auto beta = static_cast<ElementCNonVoid>(args.beta);

    auto stride_A = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideA{}, cute::make_shape(args.m, args.k, 1));
    auto stride_B = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideB{}, cute::make_shape(args.n, args.k, 1));
    // bias shape: [1, n], stride: [0, 1]
    cute::tuple<int64_t, cute::Int<1>, int64_t> stride_C = {
        cute::Int<0>{}, cute::Int<1>{}, cute::Int<0>{}};
    auto stride_D = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideD{}, cute::make_shape(args.m, args.n, 1));
    using Epilogue = typename GemmKernel::CollectiveEpilogue;
    typename GemmKernel::EpilogueArguments epilogue{
        {}, has_bias ? ptr_C : nullptr, stride_C, ptr_D, stride_D};

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

    using TileScheduler = typename GemmKernel::TileScheduler;
    using TileSchedulerTag = decltype(KernelBuilder().tile_scheduler());
    auto scheduler = typename TileScheduler::Arguments{};

    if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelTileScheduler>) {
      auto ptr_barrier = reinterpret_cast<typename SystemBarrier::T *>(args.barrier_buffer);
      scheduler.ptr_barrier = ptr_barrier;
    }

    if constexpr (
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelTileScheduler> ||
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelStreamKScheduler>) {
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
    }

    return GemmArguments{/*mode=*/cutlass::gemm::GemmUniversalMode::kGemm,
                         /*problem_shape=*/{args.m, args.n, args.k},
                         /*mainloop=*/
                         {ptr_A, stride_A, ptr_B, stride_B},
                         /*epilogue=*/epilogue,
                         /*hw_info=*/{},
                         /*scheduler=*/scheduler,
                         args.barrier_buffer,
                         args.rank,
                         args.world_size};
  }

  auto
  to_gemm_args_impl(AGKernelArguments const &args) const {
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

    using Epilogue = typename GemmKernelT::CollectiveEpilogue;
    typename Epilogue::Arguments epilogue{{}, ptr_C, stride_C, ptr_D, stride_D};
    epilogue.thread = decltype(epilogue.thread){.alpha = args.alpha, .beta = args.beta};

    using TileScheduler = typename GemmKernel::TileScheduler;
    using TileSchedulerTag = decltype(KernelBuilder().tile_scheduler());
    auto scheduler = typename TileScheduler::Arguments{};

    if constexpr (cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelTileScheduler>) {
      auto ptr_barrier = reinterpret_cast<typename SystemBarrier::T *>(args.barrier_buffer);
      scheduler.ptr_barrier = ptr_barrier;
    }

    if constexpr (
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelTileScheduler> ||
        cute::is_same_v<TileSchedulerTag, cutlass::gemm::AGKernelStreamKScheduler>) {
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
    }
    return GemmArguments{/*mode=*/cutlass::gemm::GemmUniversalMode::kGemm,
                         /*problem_shape=*/{args.m, args.n, args.k},
                         /*mainloop=*/
                         {ptr_A, stride_A, ptr_B, stride_B},
                         /*epilogue=*/epilogue,
                         /*hw_info=*/{},
                         /*scheduler=*/scheduler,
                         args.barrier_buffer,
                         args.rank,
                         args.world_size};
  }

  auto
  to_gemm_args(std::any const &args) const {
    if constexpr (this->is_s8_gemm) {
      return to_s8_gemm_args_impl(std::any_cast<AGS8KernelArguments>(args));
    } else {
      return to_gemm_args_impl(std::any_cast<AGKernelArguments>(args));
    }
  }
};
}  // namespace bytedance::flux
