//===- gemm_grouped_v3_comm_none.hpp ------------------------------ C++ ---===//
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
#include <any>
#include <memory>
#include "flux/flux.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/cuda/cuda_common.h"
#include "flux/cuda/gemm_impls/gemm_grouped_v3_impl.hpp"

namespace bytedance {
namespace flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV3CommNone_Kernel : public GemmGroupedV3BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmGroupedV3BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static_assert(meta.comm_op() == _CommNone{}, "requires _CommNone{}");

  auto
  gemm_kernel() const {
    using CollectiveMma = decltype(this->default_collective_mma());
    using CollectiveEpilogue = decltype(this->default_collective_epilogue());
    return make_declval<
        cutlass::gemm::kernel::
            GemmUniversal<typename Base::ProblemShape, CollectiveMma, CollectiveEpilogue>>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmGroupedV3CommNone_Device
    : public GemmGroupedV3BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmGroupedV3CommNone_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base =
      GemmGroupedV3BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmGroupedV3CommNone_Device>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV3CommNone_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static_assert(meta.comm_op() == _CommNone{}, "requires _CommNone{}");

  // Preparing args from host to device workspace
  std::size_t
  get_args_workspace_size(std::any const &unify_args) const override {
    const auto &args = std::any_cast<GemmGroupedV3Arguments>(unify_args);
    return this->get_args_workspace_size_impl(args);
  }

  void
  initialize_args_workspace(
      std::any const &unify_args, void *args_workspace, void *stream) const override {
    const auto &args = std::any_cast<GemmGroupedV3Arguments>(unify_args);
    this->initialize_args_workspace_impl(args, args_workspace, stream);
  }

  auto
  to_gemm_args_impl(GemmGroupedV3Arguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    using UnderlyingProblemShape = typename Base::ProblemShape::UnderlyingProblemShape;
    using ElementAccumulator = typename Base::ElementAccumulator;

    cutlass::KernelHardwareInfo hw_info;
    // Change device_id to another value if you are running on a machine with multiple GPUs and
    // wish to use a GPU other than that with device ID 0.
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);
    // problem_sizes on host can be None
    auto problem_sizes_host = static_cast<UnderlyingProblemShape *>(args.problem_sizes);

    auto
        [problem_sizes_device,
         ptr_A,
         ptr_Stride_A,
         ptr_B,
         ptr_Stride_B,
         ptr_C,
         ptr_Stride_C,
         ptr_D,
         ptr_Stride_D,
         ptr_alpha] = this->parse_common_gemm_args_from_workspace(args, args_workspace);
    // parameters for the epilogue
    auto epi_params = decltype(GemmArguments{}.epilogue.thread){
        .alpha = args.alpha,
        .beta = args.beta,
        .alpha_ptr = nullptr,
        .beta_ptr = nullptr,
        .alpha_ptr_array = ptr_alpha,
        .beta_ptr_array = nullptr};

    if (ptr_alpha != nullptr) {
      FLUX_CHECK(args.beta == 0.0f) << args.beta << "must be 0.0 if ptr_alpha is set";
    }

    using TileScheduler = typename Gemm::GemmKernel::TileScheduler;
    auto scheduler = typename TileScheduler::Arguments{};
    if constexpr (hparams.raster_order() == _RasterAlongM{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongM;
    } else if constexpr (hparams.raster_order() == _RasterAlongN{}) {
      scheduler.raster_order = TileScheduler::RasterOrderOptions::AlongN;
    }

    return GemmArguments{
        /*mode=*/cutlass::gemm::GemmUniversalMode::kGrouped,
        {args.problem_count, problem_sizes_device, problem_sizes_host},
        /*MMA*/
        {ptr_A, ptr_Stride_A, ptr_B, ptr_Stride_B},
        /*Epilogue*/
        {epi_params, ptr_C, ptr_Stride_C, ptr_D, ptr_Stride_D},
        hw_info,
        scheduler};
  }

 public:
  auto
  to_gemm_args(std::any const &unified_args, void *args_workspace) const {
    auto const &args = std::any_cast<GemmGroupedV3Arguments>(unified_args);
    return to_gemm_args_impl(args, args_workspace);
  }
};

}  // namespace flux
}  // namespace bytedance
