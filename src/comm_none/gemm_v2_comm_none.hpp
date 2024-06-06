//===- gemm_v2_comm_none.hpp -------------------------------------- C++ ---===//
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
#include "cute/container/tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cutlass/gemm/gemm_enumerated_types.h"

#include "flux/flux.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/comm_none.h"

namespace bytedance::flux {
template <class GemmMetaT, class GemmHParamsT>
class GemmV2CommNone
    : public GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2CommNone<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2CommNone>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2CommNone)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static constexpr auto dt_conf = to_gemm_dtype_config(make_gemm_dtype_config(meta.dtype()));

  static_assert(meta.comm_op() == _CommNone{}, "requires _CommNone{}");

  auto
  gemm_kernel() const {
    auto params = this->default_kernel_params();
    return this->default_gemm_kernel(params);
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

    // auto stride_A = cutlass::make_cute_packed_stride()
    auto stride_C = cutlass::make_cute_packed_stride(
        typename Base::StrideC{}, cute::make_shape(args.m, args.n, cute::Int<1>{}));
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
  to_gemm_args(std::any const &args) const {
    return to_gemm_args_impl(std::any_cast<GemmOnlyArguments>(args));
  }
};

struct GemmV2CommNone_Space : OpSpaceBase<GemmV2CommNone_Space> {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV2{}));

  static constexpr auto AllGemmHParams = make_space_gemm_hparams();

  // static constexpr auto AllGemmHParams = make_space_gemm_hparams();
};

}  // namespace bytedance::flux
