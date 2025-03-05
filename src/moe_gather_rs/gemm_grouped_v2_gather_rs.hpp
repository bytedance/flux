//===- gemm_grouped_v2_gather_rs.hpp ----------------------------- C++ ---===//
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
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/gemm_impls/gemm_grouped_v2_impl.hpp"
#include "cutlass_impls/default_gather_rs_gemm_grouped_with_absmax.h"
#include <cute/numeric/integral_constant.hpp>
#include <cute/layout.hpp>

namespace bytedance::flux {
template <class GemmMetaT, class GemmHParamsT>
struct GemmGroupedV2GatherRS_Kernel : public GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT> {
  using Base = GemmGroupedV2BaseKernel<GemmMetaT, GemmHParamsT>;
  using typename Base::ArchTag;
  using typename Base::OpClass;

  using Base::kAlignmentA, Base::kAlignmentB, Base::NumStages, Base::is_fp8_gemm;
  using typename Base::ElementA;
  using typename Base::ElementAccumulator;
  using typename Base::ElementB;
  using typename Base::ElementC;
  using typename Base::ElementCNonVoid;
  using typename Base::ElementD;
  using typename Base::GmemLayoutA;
  using typename Base::GmemLayoutB;
  using typename Base::GmemLayoutC;
  using typename Base::GmemLayoutD;
  using typename Base::InstructionShape;
  using typename Base::ThreadblockShape;
  using typename Base::WarpShape;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));
  static_assert(meta.comm_op() == _GatherRS{}, "requires _GatherRS{}");

  auto
  epilogue() const {
    using EpilogueOp = cutlass::epilogue::thread::LinearCombinationGenericWithScalingAndAbsMax<
        cutlass::epilogue::thread::Identity,  // maybe not need this, so use Identity
        ElementCNonVoid,
        ElementCNonVoid,
        cute::Int<128 / cutlass::sizeof_bits<ElementCNonVoid>::value>{},
        ElementAccumulator,
        ElementAccumulator>;
    return make_declval<EpilogueOp>();
  }

  auto
  gemm_kernel() const {
    using Epilogue = decltype(this->epilogue());

    // NOTE: Threadblock swizzling is currently not supported by CUTLASS's grouped kernels.
    // This parameter is passed in at present to match the APIs of other kernels. The parameter
    // is unused within the kernel.
    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;

    using GroupScheduleMode = cutlass::gemm::kernel::GroupScheduleMode;

    using Operator = cute::conditional_t<
        to_gemm_v2_meta(meta.impl_spec()).fast_accum(),
        cutlass::arch::OpMultiplyAddFastAccum,
        cutlass::arch::OpMultiplyAdd>;
    using SM89Impl = typename cutlass::gemm::kernel::DefaultGatherRSGemmGroupedWithAbsMax<
        ElementA,
        GmemLayoutA,
        cutlass::ComplexTransform::kNone,
        kAlignmentA,
        ElementB,
        GmemLayoutB,
        cutlass::ComplexTransform::kNone,
        kAlignmentB,
        ElementD,
        GmemLayoutD,
        ElementAccumulator,
        OpClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Epilogue,
        SwizzleThreadBlock,
        NumStages,
        GroupScheduleMode::kDeviceOnly,
        Operator>;
    return make_declval<typename SM89Impl::GemmKernel>();
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmGroupedV2GatherRS_Device
    : public GemmGroupedV2BaseDevice<
          GemmMetaT,
          GemmHParamsT,
          GemmKernelT,
          GemmGroupedV2GatherRS_Device<GemmMetaT, GemmHParamsT, GemmKernelT>> {
 public:
  using KernelBuilder = GemmGroupedV2GatherRS_Kernel<GemmMetaT, GemmHParamsT>;
  using Base =
      GemmGroupedV2BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmGroupedV2GatherRS_Device>;

  using typename Base::ElementA;
  using typename Base::ElementAccumulator;
  using typename Base::ElementB;
  using typename Base::ElementC;
  using typename Base::ElementCNonVoid;
  using typename Base::ElementD;
  using typename Base::GmemLayoutA;
  using typename Base::GmemLayoutB;
  using typename Base::GmemLayoutC;
  using typename Base::GmemLayoutD;
  using typename Base::ThreadblockShape;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  auto
  to_gemm_args_impl(GemmGroupedV2GatherRSArguments const &args, void *args_workspace) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;

    int threadblock_count = get_threadblock_count(args.sm_margin);

    using EpilogueOutputOpParams = typename Gemm::EpilogueOutputOp::Params;
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));
    EpilogueOutputOpParams epilogue_params{
        {ElementD(args.alpha), ElementD(args.beta)},
        (float *)args.scaleA,
        (float *)args.scaleB,  // here is a magic: pass a `void **` as a `float *`
        nullptr,               // scaleC
        nullptr,               // scaleD
        nullptr,               // scaleAux
        nullptr,               // abs_max_Aux
        nullptr};              // abs_max_D
    typename Gemm::Arguments gemm_args{
        (cutlass::gemm::GemmCoord *)args.problem_sizes,            // GemmCoord*
        args.problem_count,                                        // int
        threadblock_count,                                         // int
        epilogue_params,                                           // EpilogueOutputOp::Params
        reinterpret_cast<typename Base::ElementA **>(args.ptr_A),  // ElementA**
        reinterpret_cast<typename Base::ElementB **>(args.ptr_B),  // ElementB**
        reinterpret_cast<typename Base::ElementCNonVoid **>(args.ptr_C),  // ElementC**
        reinterpret_cast<typename Base::ElementD **>(args.ptr_D),         // ElementD**
        nullptr,                                                          // ptr_Aux
        nullptr,                                                          // ptr_Vector
        (typename GmemLayoutA::Stride::LongIndex *)args.lda,  // LayoutA::Stride::LongIndex*
        (typename GmemLayoutA::Stride::LongIndex *)args.ldb,  // LayoutB::Stride::LongIndex*
        (typename GmemLayoutC::Stride::LongIndex *)args.ldc,  // LayoutC::Stride::LongIndex*
        (typename GmemLayoutC::Stride::LongIndex *)args.ldd,  // LayoutD::Stride::LongIndex*
        (typename GmemLayoutC::Stride::LongIndex *)args.ldr,
        (typename GmemLayoutC::Stride::LongIndex *)args.ldd,
        (cutlass::gemm::GemmCoord *)nullptr,  // host_problem_sizes
        args.n_split,
        args.barrier,
        args.non_empty_problem_count};
    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &unified_args, void *args_workspace) const {
    return to_gemm_args_impl(
        std::any_cast<GemmGroupedV2GatherRSArguments>(unified_args), args_workspace);
  }

 private:
  int
  get_threadblock_count(int sm_margin) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    int num_multiprocessor = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
    num_multiprocessor = (sm_margin < 0 || sm_margin >= num_multiprocessor)
                             ? num_multiprocessor
                             : (num_multiprocessor - sm_margin);
    int threadblock_count = Gemm::maximum_active_blocks() * num_multiprocessor;

    FLUX_CHECK(
        threadblock_count &&
        "Active CUDA device lacks hardware resources to run CUTLASS Grouped GEMM kernel.");
    return threadblock_count;
  }
};
}  // namespace bytedance::flux
