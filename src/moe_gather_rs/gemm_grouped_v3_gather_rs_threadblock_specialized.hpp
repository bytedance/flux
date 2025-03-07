//===- gemm_grouped_v3_gather_rs_threadblock_specialized.hpp ------ C++ ---===//
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

// Note: ts represents the threadblock specialized

#pragma once
#include <memory>
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_grouped_v3_impl.hpp"
#include "moe_gather_rs/sm90_gemm_array_threadblock_specialized.hpp"

namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV3GatherRSTS_Kernel : public GemmGroupedV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmGroupedV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr int32_t GATHER_RS_N_CTAS = hparams.comm_spec().gather_rs_ctas();
  static constexpr int32_t TOPK = meta.comm_spec().topk();
  static constexpr int32_t GATHER_RS_BLOCK_M = 24;
  static constexpr int32_t GATHER_RS_BLOCK_N = hparams.comm_spec().n_dim_per_split();
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  static_assert(meta.comm_op() == _GatherRS{}, "requires _GatherRS{}");

  auto
  gemm_kernel() const {
    using CollectiveMma = decltype(this->default_collective_mma());
    using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;
    using CollectiveEpilogue =
        identity_t<decltype(this->template default_collective_epilogue<EpilogueSchedule>())>;
    return make_declval<cutlass::gemm::kernel::GroupGemmUniversalRSTS<
        typename Base::ProblemShape,
        CollectiveMma,
        CollectiveEpilogue,
        void,
        TOPK,
        GATHER_RS_BLOCK_M,
        GATHER_RS_BLOCK_N,
        GATHER_RS_N_CTAS>>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmGroupedV3GatherRSTS_Device
    : public GemmGroupedV3BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmGroupedV3GatherRSTS_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using KernelBuilder = GemmGroupedV3GatherRSTS_Kernel<GemmMetaT, GemmHParamsT>;
  using Base = GemmGroupedV3BaseDevice<
      GemmMetaT,
      GemmHParamsT,
      GemmKernelT,
      GemmGroupedV3GatherRSTS_Device>;

  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV3GatherRSTS_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

 public:
  auto
  to_gemm_args(std::any const &unified_args, void *args_workspace) const {
    auto args = std::any_cast<GemmGroupedV3GatherRSArguments>(unified_args);
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmKernel = typename Gemm::GemmKernel;
    using TileScheduler = typename GemmKernel::TileScheduler;
    using GemmArguments = typename Gemm::Arguments;
    using StrideA = cute::remove_pointer_t<typename Gemm::GemmKernel::StrideA>;
    using StrideB = cute::remove_pointer_t<typename Gemm::GemmKernel::StrideB>;
    using StrideC = cute::remove_pointer_t<typename Gemm::GemmKernel::StrideC>;
    using StrideD = cute::remove_pointer_t<typename Gemm::GemmKernel::StrideD>;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using ElementD = typename Base::ElementD;

    using UnderlyingProblemShape = typename Base::ProblemShape::UnderlyingProblemShape;
    using ElementAccumulator = typename Base::ElementAccumulator;

    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and
    // wish to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id) -
        KernelBuilder::GATHER_RS_N_CTAS - args.sm_margin;
    auto problem_sizes_device =
        reinterpret_cast<UnderlyingProblemShape *>(args.problem_sizes_device);
    // problem_sizes on host can be None
    auto problem_sizes_host = reinterpret_cast<UnderlyingProblemShape *>(args.problem_sizes_host);
    // parameters for the epilogue
    bool has_input_scale = args.input_scale_ptr != nullptr;

    using EpilogueParams = typename Gemm::EpilogueOutputOp::Params;
    EpilogueParams epi_params;
    if (has_input_scale) {
      FLUX_CHECK(args.beta == 0.0f) << args.beta << "must be 0.0 if ptr_alpha is set";
      epi_params = EpilogueParams(args.weight_scale_ptr_array);
    } else {
      epi_params = EpilogueParams(ElementAccumulator(args.alpha), ElementAccumulator(args.beta));
    }
    auto ptr_A = reinterpret_cast<ElementA const **>(args.ptr_A);
    auto ptr_B = reinterpret_cast<ElementB const **>(args.ptr_B);
    auto ptr_C = reinterpret_cast<ElementC const **>(args.ptr_C);
    auto ptr_D = reinterpret_cast<ElementD **>(args.ptr_D);
    auto ptr_Stride_A = reinterpret_cast<StrideA *>(args.lda);
    auto ptr_Stride_B = reinterpret_cast<StrideB *>(args.ldb);
    auto ptr_Stride_C = reinterpret_cast<StrideC *>(args.ldc);
    auto ptr_Stride_D = reinterpret_cast<StrideD *>(args.ldd);
    auto ptr_inter_D = reinterpret_cast<ElementD **>(args.inter_Ds);
    auto ptr_routing_idx = reinterpret_cast<int32_t *>(args.routing_idx);
    auto scheduler_args = typename TileScheduler::Arguments{};
    auto output_scatter_ptrs = reinterpret_cast<ElementD **>(args.output_scatter_ptrs);

    if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      scheduler_args.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    } else {
      scheduler_args.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    }
    FLUX_CHECK(args.n_dim % args.SPLITS == 0);
    // FLUX_CHECK((args.n_dim / args.SPLITS) % GATHER_RS_BLOCK_N == 0);
    FLUX_CHECK(args.topk == KernelBuilder::TOPK);
    GemmArguments gemm_args{
        /*mode=*/cutlass::gemm::GemmUniversalMode::kGrouped,
        {args.problem_count, problem_sizes_device, problem_sizes_host},
        /*MMA*/
        {ptr_A, ptr_Stride_A, ptr_B, ptr_Stride_B},
        /*Epilogue*/
        {epi_params, ptr_C, ptr_Stride_C, ptr_D, ptr_Stride_D},
        hw_info,
        scheduler_args,
        args.rank,
        args.world_size,
        args.topk,
        output_scatter_ptrs,
        ptr_inter_D,
        ptr_routing_idx,
        args.barrier,
        args.SPLITS,
        args.totalM,
        args.n_dim,
        args.tp_world_size,
        args.ep_world_size,
        args.globalM,
        args.ep_m_start,
        args.ep_m_end,
        args.input_scale_ptr,
        args.output_vec_scale_ptr,
        args.input_groups,
        args.ep_pos_filtered,
        args.ep_token_idx_filtered,
        args.ep_total_token_acc};

    return gemm_args;
  }
};
}  // namespace flux
}  // namespace bytedance
