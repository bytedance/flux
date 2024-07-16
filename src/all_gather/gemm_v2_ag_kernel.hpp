//===- gemm_v2_ag_kernel.hpp -------------------------------------- C++ ---===//
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

#include <any>
#include "cute/container/tuple.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/arch/arch.h"
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/args/all_gather.h"
#include "all_gather/sm80_gemm_universal_with_visitor.hpp"
#include "all_gather/sm80_all_gather_gemm_threadblock_swizzle.hpp"

namespace bytedance::flux {

template <class GemmMetaT, class GemmHParamsT>
class GemmV2AGKernel
    : public GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2AGKernel<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2AGKernel>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2AGKernel)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto comm_spec = hparams.comm_spec();

  static_assert(meta.comm_op() == _AGKernel{}, "requires _AGKernel{}");
  static constexpr bool raster_alongN = hparams.raster_order() == _RasterAlongN{};

  auto
  tb_swizzle() const {
    return make_declval<cutlass::gemm::threadblock::AGThreadblockSwizzleStreamKRankOffset>();
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
        typename Base::ElementCNoVoid,
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

  auto
  to_gemm_args_impl(AGKernelArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    auto ptr_A = static_cast<typename Base::ElementA const *>(args.input);
    auto ptr_B = static_cast<typename Base::ElementB const *>(args.weight);
    auto ptr_C = static_cast<typename Base::ElementC *>(const_cast<void *>(args.bias));
    auto ptr_D = static_cast<typename Base::ElementD *>(args.output);
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, cute::Int<1>{}));
    auto stride_D = stride_C;

    auto ptr_full_A = static_cast<typename Base::ElementA const *>(args.input_buffer);
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
        ptr_full_A,       // ptr_A
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
    return to_gemm_args_impl(std::any_cast<AGKernelArguments>(args));
  }
};

using namespace cute;
using namespace bytedance::flux;
struct GemmV2AGKernel_Space : OpSpaceBase<GemmV2AGKernel_Space> {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}),
      cute::make_tuple(_AGKernel{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV2{}));

  static constexpr auto AllGemmHParams = make_space_gemm_hparams(
      cute::make_tuple(
          make_gemm_v2_hparams(Shape<_64, _64, _32>{}, Shape<_16, _8, _16>{}, _StreamkSK{}),
          make_gemm_v2_hparams(Shape<_64, _64, _32>{}, Shape<_16, _8, _16>{}, _StreamkDP{})),
      cute::make_tuple(Auto{}),
      cute::make_tuple(
          Shape<_128, _128, _32>{},
          Shape<_128, _128, _64>{},
          Shape<_64, _128, _32>{},
          Shape<_64, _128, _64>{},
          Shape<_64, _256, _32>{},
          Shape<_64, _256, _64>{},
          Shape<_128, _256, _32>{},
          Shape<_256, _128, _32>{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(cute::_3{}, cute::_4{}),
      cute::make_tuple(_RasterAlongM{}, _RasterAlongN{}));
};

}  // namespace bytedance::flux
