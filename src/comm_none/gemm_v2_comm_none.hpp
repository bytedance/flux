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
#include <type_traits>
#include "cute/container/tuple.hpp"
#include "cute/layout.hpp"
#include "cute/numeric/integral_constant.hpp"
#include "cutlass/gemm/gemm_enumerated_types.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/conv/convolution.h"
#include "cutlass/util/packed_stride.hpp"
#include "flux/flux.h"
#include "flux/cuda/gemm_impls/gemm_v2_impl.hpp"
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/comm_none.h"
#include "cutlass/gemm/device/gemm_universal_with_absmax.h"

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
  to_fp8_gemm_args_impl(GemmFP8Arguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using GemmArguments = typename Gemm::Arguments;

    using EVT = identity_t<decltype(this->default_kernel_params().evt())>;

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
  to_gemm_args(std::any const &args, void *args_workspace) const {
    if constexpr (this->is_sm89 && this->is_fp8_gemm) {
      return to_fp8_gemm_args_impl(std::any_cast<GemmFP8Arguments>(args));
    } else {
      return to_gemm_args_impl(std::any_cast<GemmOnlyArguments>(args));
    }
  }
};

struct GemmV2CommNone_Space : OpSpaceBase<GemmV2CommNone_Space> {
  static constexpr auto AllGemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          _FP16{},
          _BF16{},
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}, _Sm89{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV2{}));

  static constexpr auto AllGemmHParams_FP16 = make_space_gemm_hparams();

  static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm89{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}),  // Only register RCR layout for FP8 GEMM
      cute::make_tuple(_GemmV2{}),
      cute::make_tuple(make_gemm_v2_meta(_True{}), make_gemm_v2_meta(_False{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}));

  static constexpr auto AllGemmMeta = tuple_cat(AllGemmMeta_FP16, AllGemmMeta_FP8);

  template <int SplitIdx, int NSplits, int ArchFilter = 0>
  static constexpr auto
  enumerate_split_meta_hparams_pairs() {
    auto meta_split = split_slice_meta<SplitIdx, NSplits, ArchFilter>();
    return tuple_unpack_cat(tuple_transform(meta_split, [](auto meta) {
      if constexpr (tuple_has_elem(AllGemmMeta_FP16, meta)) {
        return tuple_enumerate(
            make_space_meta_hparams_pair(cute::make_tuple(meta), AllGemmHParams_FP16));
      } else {
        return tuple_enumerate(
            make_space_meta_hparams_pair(cute::make_tuple(meta), AllGemmHParams_FP8));
      }
    }));
  }
};

}  // namespace bytedance::flux
