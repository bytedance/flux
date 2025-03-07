//===- gemm_v2_comm_none.hpp -------------------------------------- C++ ---===//
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
#include <type_traits>
#include "cute/container/tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/util/packed_stride.hpp"
#include "flux/flux.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/comm_none.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"

namespace bytedance::flux {

template <class GemmMetaT, class GemmHParamsT>
struct GemmV2CommNone_Kernel : public GemmV2BaseKernel<
                                   GemmMetaT,
                                   GemmHParamsT,
                                   GemmV2CommNone_Kernel<GemmMetaT, GemmHParamsT>> {
  using Base = GemmV2BaseKernel<GemmMetaT, GemmHParamsT, GemmV2CommNone_Kernel>;
  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static_assert(meta.comm_op() == _CommNone{}, "requires _CommNone{}");

  auto
  gemm_kernel() const {
    auto params = this->default_kernel_params();
    return this->default_gemm_kernel(params);
  }
};

template <class GemmMetaT, class GemmHParamsT, class GemmKernelT>
class GemmV2CommNone_Device : public GemmV2BaseDevice<
                                  GemmMetaT,
                                  GemmHParamsT,
                                  GemmKernelT,
                                  GemmV2CommNone_Device<GemmMetaT, GemmHParamsT, GemmKernelT>,
                                  GemmV2CommNone_Kernel<GemmMetaT, GemmHParamsT>> {
 public:
  using KernelBuilder = GemmV2CommNone_Kernel<GemmMetaT, GemmHParamsT>;
  using Base =
      GemmV2BaseDevice<GemmMetaT, GemmHParamsT, GemmKernelT, GemmV2CommNone_Device, KernelBuilder>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2CommNone_Device)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  auto
  to_s8_gemm_dequant_args_impl(S8GemmDequantArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;
    using EVT = identity_t<decltype(KernelBuilder().default_kernel_params().evt())>;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementBias = typename Base::ElementCNonVoid;
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));

    using ElementScale = typename Base::ElementScale;

    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_bias = static_cast<ElementBias const *>(args.bias);
    auto ptr_D = static_cast<ElementD *>(args.D);
    auto ptr_scale_A = static_cast<ElementScale const *>(args.scale_A);
    auto ptr_scale_B = static_cast<ElementScale const *>(args.scale_B);
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

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {args.m, args.n, args.k},  // problem_size
        1,                         // batch count
        callback_args,             // EVT args
        ptr_A,                     // ptr_A
        ptr_B,                     // ptr_B
        nullptr,                   // ptr_C (unused)
        nullptr,                   // ptr_D (unused)
        args.m * args.k,           // batch stride A
        args.n * args.k,           // batch stride B
        0,                         // batch stride C(unused)
        0,                         // batch stride D(unused)
        stride_a,                  // stride A
        stride_b,                  // stride B
        0,                         // stride C(unused)
        0,                         // stride D(unused)
        /*avail_sms=*/avail_sms);

    return gemm_args;
  }

  auto
  to_fp8_gemm_args_impl(GemmFP8Arguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    using EVT = identity_t<decltype(KernelBuilder().default_kernel_params().evt())>;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));

    auto ptr_A = static_cast<ElementA const *>(args.A);
    auto ptr_B = static_cast<ElementB const *>(args.B);
    auto ptr_C = static_cast<ElementC *>(const_cast<void *>(args.C));
    auto ptr_D = static_cast<ElementD *>(args.D);
    auto ptr_Aux = static_cast<ElementD *>(args.Aux);
    auto ptr_Vector = static_cast<ElementC *>(args.Vector);

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
        ptr_A,
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
        (int64_t)0  // Leading dimension of vector. This must be 0
    };

    return gemm_args;
  }

  auto
  to_gemm_args_impl(GemmOnlyArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    using ElementA = decltype(to_cutlass_element(dt_conf.a()));
    using ElementB = decltype(to_cutlass_element(dt_conf.b()));
    using ElementC = decltype(to_cutlass_element(dt_conf.c()));
    using ElementD = decltype(to_cutlass_element(dt_conf.d()));

    auto ptr_A = static_cast<ElementA const *>(args.input);
    auto ptr_B = static_cast<ElementB const *>(args.weight);
    auto ptr_C = static_cast<ElementC *>(const_cast<void *>(args.bias));
    auto ptr_D = static_cast<ElementD *>(args.output);

    auto stride_C = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, 1));
    auto stride_D = stride_C;

    auto callback_args =
        this->default_get_callback_args(ptr_C, stride_C, ptr_D, stride_D, args.alpha, args.beta);
    int stride_b = this->get_stride_b(args.n, args.k);
    auto const &v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
    int avail_sms = -1;
    if (hparams.gemm_kind() == _GemmStreamK{} and v2_hparams.streamk_mode() == _StreamkDP{}) {
      avail_sms = 1;
    }

    auto gemm_args = GemmArguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,                                        // split_k factors
        callback_args,
        ptr_A,            // ptr_A
        ptr_B,            // ptr_B
        nullptr,          // ptr_C (unused)
        nullptr,          // ptr_D
        args.m * args.k,  // batch_stride_A
        args.n * args.k,  // batch_stride_B
        args.m * args.n,  // batch_stride_C (unused)
        args.m * args.n,  // batch_stride_D (unused)
        args.k,           // stride_a
        stride_b,         // stride_b
        args.n,           // stride_c
        args.n,           // stride_d
        avail_sms);       // avail_sms
    return gemm_args;
  }

 public:
  auto
  to_gemm_args(std::any const &args, void *args_workspace) const {
    if constexpr (this->is_sm89 && this->is_fp8_gemm) {
      return to_fp8_gemm_args_impl(std::any_cast<GemmFP8Arguments>(args));
    } else if constexpr (this->is_s8_gemm) {
      return to_s8_gemm_dequant_args_impl(std::any_cast<S8GemmDequantArguments>(args));
    } else {
      return to_gemm_args_impl(std::any_cast<GemmOnlyArguments>(args));
    }
  }
};
}  // namespace bytedance::flux
