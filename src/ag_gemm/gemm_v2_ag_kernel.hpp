//===- gemm_v2_ag_kernel.hpp -------------------------------------- C++ ---===//
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
#include "cute/container/tuple.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/arch/arch.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/args/ag_gemm.h"
#include "ag_gemm/sm80_gemm_universal_with_visitor.hpp"
#include "ag_gemm/sm80_all_gather_gemm_threadblock_swizzle.hpp"

// import sm89 gemm kernel files
#include "cutlass/gemm/kernel/default_gemm_with_absmax.h"
#include "ag_gemm/sm89_all_gather_fp8_gemm_threadblock_swizzle.hpp"
#include "ag_gemm/sm89_gemm_with_absmax.hpp"

namespace bytedance::flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmV2AGKernel_Kernel : public GemmV2BaseKernel<
                                   GemmMetaT,
                                   GemmHParamsT,
                                   GemmV2AGKernel_Kernel<GemmMetaT, GemmHParamsT>> {
  using Base = GemmV2BaseKernel<GemmMetaT, GemmHParamsT, GemmV2AGKernel_Kernel>;

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto comm_spec = hparams.comm_spec();

  static_assert(meta.comm_op() == _AGKernel{}, "requires _AGKernel{}");
  static constexpr bool raster_alongN = hparams.raster_order() == _RasterAlongN{};

  auto
  tb_swizzle() const {
    if constexpr (
        cute::is_same_v<typename Base::ArchTag, cutlass::arch::Sm89> && this->is_fp8_gemm) {
      if constexpr (raster_alongN) {
        return make_declval<cutlass::gemm::threadblock::AGGemmHorizontalThreadblockSwizzle>();
      } else {
        // Default swizzle order along N=8 first, then along M.
        return make_declval<cutlass::gemm::threadblock::AGGemmIdentityThreadblockSwizzle<8>>();
      }
    } else {
      return make_declval<cutlass::gemm::threadblock::AGThreadblockSwizzleStreamKRankOffset>();
    }
  }

  auto
  kernel_params() const {
    auto default_params = this->default_kernel_params();
    // modify tb_swizzle
    return gemm_v2_impl::KernelParams<
        decltype(this->tb_swizzle()),
        cute::Int<default_params.alignment_c()>,
        decltype(default_params.evt())>();
  }

  auto
  gemm_kernel() const {
    auto params = this->kernel_params();

    using Operator = cute::conditional_t<
        to_gemm_v2_meta(meta.impl_spec()).fast_accum(),
        cutlass::arch::OpMultiplyAddFastAccum,
        cutlass::arch::OpMultiplyAdd>;

    if constexpr (
        cute::is_same_v<typename Base::ArchTag, cutlass::arch::Sm89> && this->is_fp8_gemm) {
      if constexpr (cute::is_same_v<typename Base::GmemLayoutC, cutlass::layout::ColumnMajor>) {
        using Sm89AGColImpl = cutlass::gemm::kernel::Sm89GemmWithAbsMax<
            typename Base::ElementB,
            typename cutlass::layout::LayoutTranspose<typename Base::GmemLayoutB>::type,
            cutlass::ComplexTransform::kNone,
            Base::AlignmentB,
            typename Base::ElementA,
            typename cutlass::layout::LayoutTranspose<typename Base::GmemLayoutA>::type,
            cutlass::ComplexTransform::kNone,
            Base::AlignmentA,
            typename Base::ElementCNonVoid,
            cutlass::layout::RowMajor,
            typename Base::ElementAccumulator,
            typename Base::OpClass,
            typename Base::ArchTag,
            typename Base::ThreadblockShape,
            typename Base::WarpShape,
            typename Base::InstructionShape,
            decltype(params.evt()),
            decltype(params.tb_swizzle()),
            hparams.mainloop_stage(),
            Operator>;
        return make_declval<typename Sm89AGColImpl::GemmKernel>();
      } else {
        using Sm89AGImpl = cutlass::gemm::kernel::Sm89GemmWithAbsMax<
            typename Base::ElementA,
            typename Base::GmemLayoutA,
            cutlass::ComplexTransform::kNone,
            Base::AlignmentA,
            typename Base::ElementB,
            typename Base::GmemLayoutB,
            cutlass::ComplexTransform::kNone,
            Base::AlignmentB,
            typename Base::ElementCNonVoid,
            typename Base::GmemLayoutC,
            typename Base::ElementAccumulator,
            typename Base::OpClass,
            typename Base::ArchTag,
            typename Base::ThreadblockShape,
            typename Base::WarpShape,
            typename Base::InstructionShape,
            decltype(params.evt()),
            decltype(params.tb_swizzle()),
            hparams.mainloop_stage(),
            Operator>;
        return make_declval<typename Sm89AGImpl::GemmKernel>();
      }
    } else if constexpr (this->is_s8_gemm) {
      using ElementEpilogueCompute = typename Base::ElementScale;
      using Sm80S8GemmDequantAGImpl = cutlass::gemm::kernel::Sm80GemmWithVisitor<
          typename Base::ElementA,
          typename Base::GmemLayoutA,
          cutlass::ComplexTransform::kNone,
          Base::AlignmentA,
          typename Base::ElementB,
          typename Base::GmemLayoutB,
          cutlass::ComplexTransform::kNone,
          Base::AlignmentB,
          typename Base::ElementCNonVoid,
          typename Base::GmemLayoutC,
          params.alignment_c(),
          typename Base::ElementAccumulator,
          ElementEpilogueCompute,
          typename Base::OpClass,
          typename Base::ArchTag,
          typename Base::ThreadblockShape,
          typename Base::WarpShape,
          typename Base::InstructionShape,
          decltype(params.evt()),
          decltype(params.tb_swizzle()),
          hparams.mainloop_stage(),
          cutlass::arch::OpMultiplyAddSaturate,
          Base::EVTEpilogueStages>;
      return make_declval<typename Sm80S8GemmDequantAGImpl::GemmKernel>();
    } else {
      using ElementCompute = typename Base::ElementD;
      using Impl = cutlass::gemm::kernel::Sm80GemmWithVisitor<
          typename Base::ElementA,
          typename Base::GmemLayoutA,
          cutlass::ComplexTransform::kNone,
          Base::AlignmentA,
          typename Base::ElementB,
          typename Base::GmemLayoutB,
          cutlass::ComplexTransform::kNone,
          Base::AlignmentB,
          typename Base::ElementCNonVoid,
          typename Base::GmemLayoutC,
          params.alignment_c(),
          typename Base::ElementAccumulator,
          ElementCompute,
          typename Base::OpClass,
          typename Base::ArchTag,
          typename Base::ThreadblockShape,
          typename Base::WarpShape,
          typename Base::InstructionShape,
          decltype(params.evt()),
          decltype(params.tb_swizzle()),
          hparams.mainloop_stage(),
          cutlass::arch::OpMultiplyAdd,
          Base::EVTEpilogueStages>;
      return make_declval<typename Impl::GemmKernel>();
    }
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV2AGKernel_Device : public GemmV2BaseDevice<
                                  GemmMetaT,
                                  GemmHParamsT,
                                  GemmKernelT,
                                  GemmV2AGKernel_Device<GemmMetaT, GemmHParamsT, GemmKernelT>,
                                  GemmV2AGKernel_Kernel<GemmMetaT, GemmHParamsT>> {
 public:
  using KernelBuilder = GemmV2AGKernel_Kernel<GemmMetaT, GemmHParamsT>;
  using Base =
      GemmV2BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmV2AGKernel_Device, KernelBuilder>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2AGKernel_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto comm_spec = hparams.comm_spec();

  static_assert(meta.comm_op() == _AGKernel{}, "requires _AGKernel{}");

  auto
  to_s8_gemm_args_impl(AGS8KernelArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    using EVT = identity_t<decltype(KernelBuilder().kernel_params().evt())>;

    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementBias = typename Base::ElementCNonVoid;
    using ElementD = typename Base::ElementD;
    using ElementScale = typename Base::ElementScale;

    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_bias = static_cast<ElementBias const *>(args.bias);
    auto ptr_D = static_cast<ElementD *>(args.output);
    auto ptr_scale_A = static_cast<ElementScale const *>(args.scale_A);
    auto ptr_scale_B = static_cast<ElementScale const *>(args.scale_B);

    auto ptr_barrier = reinterpret_cast<typename SystemBarrier::T *>(args.barrier_buffer);
    auto beta = static_cast<ElementBias>(args.beta);

    int stride_a = args.k;
    int stride_b = this->get_stride_b(args.n, args.k);
    // output's layout is same with bias
    int stride_c = this->get_stride_c(args.m, args.n);
    int stride_d = stride_c;
    auto callback_args = this->s8gemm_callback_args(
        args.m, args.n, beta, ptr_bias, ptr_D, stride_d, ptr_scale_A, ptr_scale_B);
    auto const &v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
    int avail_sms = -1;
    if (hparams.gemm_kind() == _GemmStreamK{} and v2_hparams.streamk_mode() == _StreamkDP{}) {
      avail_sms = 1;
    }
    auto gemm_args = GemmArguments{
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,
        callback_args,
        ptr_A,
        ptr_B,
        nullptr,
        nullptr,
        args.m * args.k,  // batch stride A
        args.n * args.k,  // batch stride B
        0,                // batch stride C(unused)
        0,                // batch stride D(unused)
        stride_a,         // stride A
        stride_b,         // stride B
        0,                // stride C(unused)
        0,                // stride D(unused)
        avail_sms,
        ptr_barrier};

    gemm_args.nnodes = args.nnodes;
    gemm_args.rank = args.rank;
    gemm_args.world_size = args.world_size;
    gemm_args.local_world_size = args.world_size / args.nnodes;
    gemm_args.local_rank = args.rank % gemm_args.local_world_size;

    gemm_args.raster_order = hparams.raster_order() == _RasterAlongN{} ? 0 : 1;
    return gemm_args;
  }

  auto
  to_fp8_gemm_args_impl(AGFP8KernelArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using EVT = identity_t<decltype(KernelBuilder().kernel_params().evt())>;

    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using ElementD = typename Base::ElementD;

    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_C = static_cast<ElementC *>(const_cast<void *>(args.C));
    auto ptr_D = static_cast<ElementD *>(args.D);
    auto ptr_Aux = static_cast<ElementD *>(args.Aux);
    auto ptr_Vector = static_cast<ElementC *>(args.Vector);

    auto ptr_barrier = reinterpret_cast<typename SystemBarrier::T *>(args.barrier_buffer);

    int stride_b = this->get_stride_b(args.n, args.k);
    int stride_c = this->get_stride_c(args.m, args.n);
    int stride_d = stride_c;

    typename EVT::Params epilogue_params{
        {ElementD(args.alpha), ElementD(args.beta)},
        args.scaleA,
        args.scaleB,
        args.scaleC,
        args.scaleD,
        args.scaleAux,
        args.abs_max_Aux,
        args.abs_max_D};

    typename Gemm::Arguments gemm_args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k},
        /* batch_count = */ 1,
        epilogue_params,
        ptr_A,  // full A
        ptr_B,
        ptr_C,
        ptr_D,
        ptr_Aux,
        ptr_Vector,
        args.m * args.k,
        args.n * args.k,
        args.m * args.n,
        args.m * args.n,
        (int)args.m,  // Batch stride vector
        args.k,
        stride_b,
        stride_c,
        stride_d,
        (int64_t)0,  // Leading dimension of vector. This must be 0
    };

    gemm_args.ptr_barrier = ptr_barrier;

    gemm_args.nnodes = args.nnodes;
    gemm_args.rank = args.rank;
    gemm_args.world_size = args.world_size;
    gemm_args.local_world_size = args.world_size / args.nnodes;
    gemm_args.local_rank = args.rank % gemm_args.local_world_size;

    gemm_args.raster_order = hparams.raster_order() == _RasterAlongN{} ? 0 : 1;

    return gemm_args;
  }

  auto
  to_gemm_args_impl(AGKernelArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    auto ptr_B = static_cast<typename Base::ElementB const *>(args.weight);
    auto ptr_C = static_cast<typename Base::ElementC *>(const_cast<void *>(args.bias));
    auto ptr_D = static_cast<typename Base::ElementD *>(args.output);
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = stride_C;

    auto ptr_A = static_cast<typename Base::ElementA const *>(args.input);
    auto ptr_barrier = reinterpret_cast<typename SystemBarrier::T *>(args.barrier_buffer);

    auto callback_args =
        this->default_get_callback_args(ptr_C, stride_C, ptr_D, stride_D, args.alpha, args.beta);

    int stride_b = this->get_stride_b(args.n, args.k);
    auto const &v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
    int avail_sms = -1;
    if (hparams.gemm_kind() == _GemmStreamK{} and v2_hparams.streamk_mode() == _StreamkDP{}) {
      avail_sms = 1;
    }

    auto gemm_args = GemmArguments{
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,                                        // split_k factors
        callback_args,
        ptr_A,            // ptr_A
        ptr_B,            // ptr_B
        nullptr,          // ptr_C (unused)
        nullptr,          // ptr_D (unused)
        args.m * args.k,  // batch_stride_A
        args.n * args.k,  // batch_stride_B
        args.m * args.n,  // batch_stride_C (unused)
        args.m * args.n,  // batch_stride_D
        args.k,           // stride_a
        stride_b,         // stride_b
        args.n,           // stride_c (unused)
        args.n,           // stride_d
        avail_sms,        // avail_sms
        ptr_barrier       // barrier
    };

    gemm_args.nnodes = args.nnodes;
    gemm_args.rank = args.rank;
    gemm_args.world_size = args.world_size;
    gemm_args.local_world_size = args.world_size / args.nnodes;
    gemm_args.local_rank = args.rank % gemm_args.local_world_size;

    gemm_args.raster_order = hparams.raster_order() == _RasterAlongN{} ? 0 : 1;
    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    if constexpr (this->is_sm89 && this->is_fp8_gemm) {
      return to_fp8_gemm_args_impl(std::any_cast<AGFP8KernelArguments>(args));
    } else if constexpr (this->is_s8_gemm) {
      return to_s8_gemm_args_impl(std::any_cast<AGS8KernelArguments>(args));
    } else {
      return to_gemm_args_impl(std::any_cast<AGKernelArguments>(args));
    }
  }
};
}  // namespace bytedance::flux
