//===- gemm_grouped_v2_comm_none.hpp ------------------------------ C++ ---===//
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
#include "cutlass/kernel_hardware_info.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/args/comm_none.h"
#include "flux/gemm_operator_base.h"
#include "flux/cuda/gemm_impls/gemm_grouped_v2_impl.hpp"

namespace bytedance::flux {
template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV2CommNone_Kernel : public GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static_assert(meta.comm_op() == _CommNone{}, "requires _CommNone{}");

  auto
  gemm_kernel() const {
    return this->default_gemm_kernel();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmGroupedV2CommNone_Device
    : public GemmGroupedV2BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmGroupedV2CommNone_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using Base =
      GemmGroupedV2BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmGroupedV2CommNone_Device>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedV2CommNone_Device)

  using KernelBuilder = GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT>;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  // Preparing args from host to device workspace
  std::size_t
  get_args_workspace_size(std::any const &unify_args) const override {
    const auto &args = std::any_cast<GemmGroupedV2Arguments>(unify_args);
    return this->get_args_workspace_size_impl(args);
  }

  void
  initialize_args_workspace(
      std::any const &unify_args, void *args_workspace, void *stream) const override {
    const auto &args = std::any_cast<GemmGroupedV2Arguments>(unify_args);
    this->initialize_args_workspace_impl(args, args_workspace, stream);
  }

  auto
  to_gemm_args_impl(GemmGroupedV2Arguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;

    int threadblock_count = Gemm::sufficient(
        reinterpret_cast<cutlass::gemm::GemmCoord *>(args.problem_sizes),
        args.problem_count,
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0) - args.sm_margin);
    // Early exit
    assert(
        threadblock_count &&
        "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");

    auto problem_sizes_host = static_cast<cutlass::gemm::GemmCoord *>(args.problem_sizes);
    auto
        [problem_sizes_device,
         ptr_A,
         ptr_B,
         ptr_C,
         ptr_D,
         ptr_Aux,
         ptr_Vector,
         ptr_scaleA,
         ptr_scaleB,
         lda,
         ldb,
         ldc,
         ldd,
         ldr] = this->parse_group_gemm_args_from_workspace(args, args_workspace);

    using EpilogueOutputOpParams = typename Gemm::EpilogueOutputOp::Params;
    if constexpr (this->is_fp8_gemm) {
      using ElementD = decltype(to_cutlass_element(dt_conf.d()));
      EpilogueOutputOpParams epilogue_params{
          {ElementD(args.alpha), ElementD(args.beta)},
          (float *)ptr_scaleA,
          (float *)ptr_scaleB,  // here is a magic: pass a `void **` as a `float *`
          args.scaleC,
          args.scaleD,
          args.scaleAux,
          args.abs_max_Aux,
          args.abs_max_D};
      typename Gemm::Arguments gemm_args{
          problem_sizes_device,                                        // GemmCoord*
          args.problem_count,                                          // int
          threadblock_count,                                           // int
          epilogue_params,                                             // EpilogueOutputOp::Params
          reinterpret_cast<typename Base::ElementA **>(ptr_A),         // ElementA**
          reinterpret_cast<typename Base::ElementB **>(ptr_B),         // ElementB**
          reinterpret_cast<typename Base::ElementCNonVoid **>(ptr_C),  // ElementC**
          reinterpret_cast<typename Base::ElementD **>(ptr_D),         // ElementD**
          ptr_Aux,
          ptr_Vector,
          lda,  // LayoutA::Stride::LongIndex*
          ldb,  // LayoutB::Stride::LongIndex*
          ldc,  // LayoutC::Stride::LongIndex*
          ldd,  // LayoutD::Stride::LongIndex*
          ldr,
          ldd,
          problem_sizes_host  // GemmCoord*
      };
      return gemm_args;
    } else {
      EpilogueOutputOpParams epilogue_params{args.alpha, args.beta};
      typename Gemm::Arguments gemm_args{
          problem_sizes_device,                                        // GemmCoord*
          args.problem_count,                                          // int
          threadblock_count,                                           // int
          epilogue_params,                                             // EpilogueOutputOp::Params
          reinterpret_cast<typename Base::ElementA **>(ptr_A),         // ElementA**
          reinterpret_cast<typename Base::ElementB **>(ptr_B),         // ElementB**
          reinterpret_cast<typename Base::ElementCNonVoid **>(ptr_C),  // ElementC**
          reinterpret_cast<typename Base::ElementD **>(ptr_D),         // ElementD**
          lda,                // LayoutA::Stride::LongIndex*
          ldb,                // LayoutB::Stride::LongIndex*
          ldc,                // LayoutC::Stride::LongIndex*
          ldd,                // LayoutD::Stride::LongIndex*
          problem_sizes_host  // GemmCoord*
      };
      return gemm_args;
    }
  }

 public:
  auto
  to_gemm_args(std::any const &unified_args, void *args_workspace) const {
    return to_gemm_args_impl(std::any_cast<GemmGroupedV2Arguments>(unified_args), args_workspace);
  }
};

}  // namespace bytedance::flux
