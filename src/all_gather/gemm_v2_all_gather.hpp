//===- gemm_v2_all_gather.hpp ------------------------------------- C++ ---===//
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
#include <memory>
#include "flux/flux.h"
#include "flux/gemm_meta.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/op_registry.h"
#include "flux/args/all_gather.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"

namespace bytedance::flux {
template <class GemmMetaT, class GemmHParamsT>
class GemmV2AllGather
    : public GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2AllGather<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmV2Impl<GemmMetaT, GemmHParamsT, GemmV2AllGather>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmV2AllGather)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});

  static_assert(meta.comm_op() == _AllGather{}, "requires _AllGather{}");

  auto
  gemm_kernel() const {
    auto params = this->default_kernel_params();
    return this->default_gemm_kernel(params);
  }

  auto
  to_gemm_args_impl(AllGatherGemmArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    auto ptr_A = static_cast<typename Base::ElementA const *>(args.input);
    auto ptr_B = static_cast<typename Base::ElementB const *>(args.weight);
    auto ptr_C = static_cast<typename Base::ElementC *>(const_cast<void *>(args.bias));
    auto ptr_D = static_cast<typename Base::ElementD *>(args.output);
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

    return GemmArguments(
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
        avail_sms);       // avail_sms
  }

 public:
  auto
  to_gemm_args(std::any const &args) const {
    return to_gemm_args_impl(std::any_cast<AllGatherGemmArguments>(args));
  }
};

struct GemmV2AllGather_Space : OpSpaceBase<GemmV2AllGather_Space> {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}),
      cute::make_tuple(_AllGather{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV2{}));

  static constexpr auto AllGemmHParams = make_space_gemm_hparams();
};
}  // namespace bytedance::flux
