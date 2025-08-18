//===- gemm_v3_post_attn_a2a_transpose_kernel.hpp ----------------- C++ ---===//
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
#include "flux/args/a2a_transpose_gemm.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_v3_impl.hpp"
#include "a2a_transpose_gemm/sm90_consumer_gemm_tma_warpspecialized_cooperative.hpp"

#include "cutlass/kernel_hardware_info.h"

namespace bytedance::flux {
using SystemBarrier = cutlass::Barrier;

template <class GemmMetaT, class GemmHParamsT>
struct GemmV3PostAttnAllToAllTranspose_Kernel : public GemmV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});

  static_assert(meta.arch() == _Sm90{}, "requires _Sm90{}");
  static_assert(
      meta.comm_op() == _PostAttnAllToAllTranspose{} or meta.comm_op() == _PostAttnAllToAllOnly{},
      "requires _PostAttnAllToAllTranspose{} or _PostAttnAllToAllOnly{}");

  auto
  tile_scheduler() const {
    if constexpr (hparams.gemm_kind() == _GemmStreamK{}) {
      return make_declval<cutlass::gemm::StreamKScheduler>();
    } else {
      return make_declval<cutlass::gemm::PersistentScheduler>();
    }
  }

  auto
  gemm_kernel() const {
    using CollectiveMma = decltype(this->default_collective_mma());
    using CollectiveEpilogue = decltype(this->default_collective_epilogue());
    using TileScheduler = decltype(this->tile_scheduler());
    return make_declval<cutlass::gemm::kernel::Sm90ConsumerGemmUniversal<
        typename Base::ProblemShape,
        CollectiveMma,
        CollectiveEpilogue,
        TileScheduler>>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV3PostAttnAllToAllTranspose_Device
    : public GemmV3BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmV3PostAttnAllToAllTranspose_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base = GemmV3BaseDevice<
      GemmMetaT,
      GemmHParamsT,
      GemmKernelT,
      GemmV3PostAttnAllToAllTranspose_Device>;
  using KernelBuilder = GemmV3PostAttnAllToAllTranspose_Kernel<GemmMetaT, GemmHParamsT>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV3PostAttnAllToAllTranspose_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  auto
  to_gemm_args_impl(A2ATransposeGemmKernelArguments const &args) const {
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
                         /*scheduler=*/scheduler,
                         args.barrier_buffer,
                         args.rank,
                         args.world_size,
                         args.m_per_barrier};
  }

  auto
  to_gemm_args(std::any const &args) const {
    return to_gemm_args_impl(std::any_cast<A2ATransposeGemmKernelArguments>(args));
  }
};
}  // namespace bytedance::flux
