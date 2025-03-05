//===- gen_moe_gather_rs.cc -------------------------------------- C++ ---===//
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
#include "flux/flux.h"
#include "./generator_utils.h"
#include "flux/gemm_hparams.h"
#include "flux/gemm_meta.h"

namespace bytedance::flux::generator {
using namespace cute;
struct GemmGroupedV2GatherRS_Space {
  static constexpr auto AllGemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_FP16{}),
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}, _Sm89{}),
      cute::make_tuple(_GatherRS{}),
      cute::make_tuple(_RCR{}),  // TODO(houqi.1993) only RCR is supported
      cute::make_tuple(_GemmGroupedV2{}),
      cute::make_tuple(make_gemm_v2_meta(_False{})),
      cute::make_tuple(
          // make_gather_rs_meta(cute::Int<10>{}),
          // make_gather_rs_meta(cute::Int<8>{}),
          // make_gather_rs_meta(cute::Int<6>{}),
          // make_gather_rs_meta(cute::Int<5>{}),
          // make_gather_rs_meta(cute::Int<4>{}),
          make_gather_rs_meta(cute::Int<1>{})));

  static constexpr auto AllGemmHParams_FP16 = make_space_gemm_hparams();

  static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm89{}),
      cute::make_tuple(_GatherRS{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmGroupedV2{}),
      cute::make_tuple(make_gemm_v2_meta(_True{}), make_gemm_v2_meta(_False{})),
      cute::make_tuple(
          // make_gather_rs_meta(cute::Int<10>{}),
          // make_gather_rs_meta(cute::Int<8>{}),
          // make_gather_rs_meta(cute::Int<6>{}),
          // make_gather_rs_meta(cute::Int<5>{}),
          // make_gather_rs_meta(cute::Int<4>{}),
          make_gather_rs_meta(cute::Int<1>{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(cute::Shape<cute::_128, cute::_128, cute::_64>{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta_FP16, AllGemmHParams_FP16),
        build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
    });
  }
};

struct GemmGroupedV3GatherRSTS_Space {
  static constexpr auto AllGemmMeta_FP16 = tuple_filter(
      make_space_gemm_meta(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(_Sm90{}),
          cute::make_tuple(_GatherRS{}),
          cute::make_tuple(_RCR{}, _RRR{}, _RCC{}),
          cute::make_tuple(_GemmGroupedV3{}),
          cute::make_tuple(make_gemm_v3_meta(_False{})),
          cute::make_tuple(
              make_gather_rs_meta(cute::Int<5>{}),
              make_gather_rs_meta(cute::Int<4>{}),
              make_gather_rs_meta(cute::Int<3>{}),
              make_gather_rs_meta(cute::Int<2>{}),
              make_gather_rs_meta(cute::Int<1>{}))),
      [](auto meta_tuple) { return true; });

  static constexpr auto AllGemmHParams_FP16 = make_space_gemm_hparams(
      cute::make_tuple(make_gemm_v3_hparams(Shape<_1, _1, _1>{})),
      cute::make_tuple(
          make_gather_rs_hparams(cute::Int<28>{}, cute::Int<1024>{}),
          make_gather_rs_hparams(cute::Int<28>{}, cute::Int<768>{}),
          make_gather_rs_hparams(cute::Int<28>{}, cute::Int<640>{}),
          make_gather_rs_hparams(cute::Int<28>{}, cute::Int<512>{}),
          make_gather_rs_hparams(cute::Int<28>{}, cute::Int<384>{}),
          make_gather_rs_hparams(cute::Int<28>{}, cute::Int<256>{})),
      cute::make_tuple(Shape<_128, _256, _64>{}));

  static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
      cute::make_tuple(make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_GatherRS{}),
      cute::make_tuple(_RCR{}, _RRR{}, _RCC{}),
      cute::make_tuple(_GemmGroupedV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})),
      cute::make_tuple(
            make_gather_rs_meta(cute::Int<4>{}),
            make_gather_rs_meta(cute::Int<3>{}),
            make_gather_rs_meta(cute::Int<2>{}),
            make_gather_rs_meta(cute::Int<1>{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{})),
      make_tuple(
          make_gather_rs_hparams(cute::Int<32>{}, cute::Int<1024>{}),
          make_gather_rs_hparams(cute::Int<32>{}, cute::Int<512>{}),
          make_gather_rs_hparams(cute::Int<32>{}, cute::Int<256>{})));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta_FP16, AllGemmHParams_FP16),
        build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
    });
  }
};

}  // namespace bytedance::flux::generator

int
main(int argc, char const **args) {
  using namespace bytedance::flux::generator;
  Options options;
  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }
  std::cout << "Running moe_gather_rs generator...\n";
  return main_template(
      options,
      {
          cute::make_tuple(
              GemmGroupedV2GatherRS_Space::get_space(),
              std::string("moe_gather_rs/gemm_grouped_v2_gather_rs.hpp"),
              std::string("GemmGroupedV2GatherRS")),
          cute::make_tuple(
              GemmGroupedV3GatherRSTS_Space::get_space(),
              std::string("moe_gather_rs/gemm_grouped_v3_gather_rs_threadblock_specialized.hpp"),
              std::string("GemmGroupedV3GatherRSTS")),
      });
}
