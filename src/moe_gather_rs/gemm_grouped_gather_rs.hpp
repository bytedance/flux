//===- gemm_grouped_gather_rs.hpp --------------------------------- C++ ---===//
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
#include "flux/gemm_hparams.h"
#include "flux/gemm_operator_base.h"
#include "flux/args/moe_gather_rs.h"
#include "flux/cuda/gemm_impls/gemm_gather_impl.hpp"
#include "moe_gather_rs/epilogue_gather.hpp"

namespace bytedance::flux {

template <class GemmMetaT, class GemmHParamsT>
class GemmGroupedGatherRS : public GemmGatherImpl<
                                GemmMetaT,
                                GemmHParamsT,
                                GemmGroupedGatherRS<GemmMetaT, GemmHParamsT>> {
 public:
  using Base = GemmGatherImpl<GemmMetaT, GemmHParamsT, GemmGroupedGatherRS>;
  FLUX_DEFINE_DEFAULT_SPECIAL_FUNCS(GemmGroupedGatherRS)

  static constexpr auto meta = to_gemm_meta(GemmMetaT{});
  static constexpr auto hparams = to_gemm_hparams(GemmHParamsT{});
  static_assert(meta.comm_op() == _GatherRS{}, "requires _GatherRS{}");

  auto
  kernel_params() const {
    using namespace cutlass::epilogue::threadblock;
    using EVT_D = decltype(this->default_evt_d());
    using EVT_GatherRS = Sm80EVT<
        VisitorAuxGatherRs<
            typename Base::OutputTileThreadMap,
            typename Base::ElementD,
            cutlass::FloatRoundStyle::round_to_nearest,
            cute::Stride<int64_t, cute::_1, int64_t>,
            typename Base::ThreadblockShape>,
        EVT_D>;
    return gemm_gather_impl::KernelParams<EVT_GatherRS>();
  }

  auto
  gemm_kernel() const {
    auto params = this->kernel_params();
    return this->default_gemm_kernel(params);
  }

  auto
  to_gemm_args_impl(GemmGatherRsArguments const &args) const {
    using Gemm = identity_t<decltype(this->gemm_device())>;
    using EVT = identity_t<decltype(this->kernel_params().evt())>;

    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using ElementD = typename Base::ElementD;
    auto ptr_A = static_cast<const ElementA *>(args.input);
    auto ptr_B = static_cast<const ElementB *>(args.weight);
    auto ptr_C = static_cast<const ElementC *>(args.bias);
    auto gemm_out = static_cast<ElementD *>(args.gemm_output);
    auto gather_index = args.gather_index;
    auto gather_weight = static_cast<const ElementD *>(args.gather_weight);
    auto gather_outputs = static_cast<ElementD *>(args.gather_outputs);
    auto finish_gather = args.finish_gather;
    auto rs_outputs_ptrs = reinterpret_cast<ElementD **>(args.rs_outputs_ptrs);
    typename EVT::Arguments callback_args{
        {
            {
                {},                                                              // Accum
                {args.alpha},                                                    // Alpha
                {}                                                               // Compute0
            },                                                                   // EVTCompute0
            {gather_weight, ElementC(0), {cute::_1{}, cute::_0{}, cute::_0{}}},  // C1
            {}                                                                   // Compute1
        },                                                                       // EVTCompute1
        {args.rank,
         args.world_size,
         {args.n, cute::_1{}, args.m * args.n},
         gemm_out,
         gather_index,
         gather_outputs,
         finish_gather,
         rs_outputs_ptrs}  // D
    };

    auto const &v2_hparams = to_gemm_v2_hparams(hparams.impl_spec());
    int avail_sms = -1;
    if (hparams.gemm_kind() == _GemmStreamK{} and v2_hparams.streamk_mode() == _StreamkDP{}) {
      avail_sms = 1;
    }

    return typename Gemm::Arguments(
        cutlass::gemm::GemmUniversalMode::kGemm,  // universal mode
        {args.m, args.n, args.k},                 // problem_size
        1,                                        // batch count / splitk slices
        callback_args,                            // argument of EVT callbacks
        ptr_A,                                    // ptr_A
        ptr_B,                                    // ptr_B
        ptr_C,                                    // ptr_C (unused)
        nullptr,                                  // ptr_D (unused)
        args.m * args.k,                          // batch_stride_A
        args.n * args.k,                          // batch_stride_B
        0,                                        // batch_stride_C (unused)
        0,                                        // batch_stride_D (unused)
        args.k,                                   // stride_a
        args.n,                                   // stride_b
        0,                                        // stride_c (unused)
        0,                                        // stride_d (unused)
        avail_sms);                               // avail_sms
  }

 public:
  auto
  to_gemm_args(std::any const &args) const {
    return to_gemm_args_impl(std::any_cast<GemmGatherRsArguments>(args));
  }
};
}  // namespace bytedance::flux
