//===- gen_moe_ag_scatter.cc -------------------------------------- C++ ---===//
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

struct GemmGroupedV2AGScatter_Space {
  static constexpr auto AllGemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_FP16{}),
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}, _Sm89{}),
      cute::make_tuple(_A100{}, _L20{}),
      cute::make_tuple(_AGScatter{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmGroupedV2{}),
      cute::make_tuple(make_gemm_v2_meta(_False{})));

  static constexpr auto AllGemmHParams_FP16 = make_space_gemm_hparams();

  static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm89{}),
      cute::make_tuple(_L20{}),
      cute::make_tuple(_AGScatter{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmGroupedV2{}),
      cute::make_tuple(make_gemm_v2_meta(_True{}), make_gemm_v2_meta(_False{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      cute::make_tuple(Auto{}),
      make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(_RasterAlongN{}));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta_FP16, AllGemmHParams_FP16),
        build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
    });
  }
};

struct GemmGroupedV3AGScatter_Space {
  static constexpr auto AllGemmMeta = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_FP16{}),
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H800{}, _H20{}),
      cute::make_tuple(_AGScatter{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmGroupedV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

  static constexpr auto AllGemmHParams = make_space_gemm_hparams(
      tuple_transform(
          tuple_cartesian_product(
              cute::make_tuple(Shape<_2, _1, _1>{}, Shape<_1, _1, _1>{}),
              cute::make_tuple(_Cooperative{}, _PingPong{}),
              cute::make_tuple(_BlockScaleMPerRow{}),
              cute::make_tuple(_BlockScaleNPerCol{}, _BlockScaleNPerBlock{})),
          [](auto tup) { return to_gemm_v3_hparams(tup); }),
      make_tuple(Auto{}),
      cute::make_tuple(Shape<Auto, _256, Auto>{}, Shape<Auto, _128, Auto>{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(_RasterAlongM{}, _RasterAlongN{}));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta, AllGemmHParams),
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
  std::cout << "Running moe_ag_scatter generator...\n";
  return main_template(
      options,
      {
          cute::make_tuple(
              GemmGroupedV2AGScatter_Space::get_space(),
              std::string("moe_ag_scatter/gemm_grouped_v2_ag_scatter.hpp"),
              std::string("GemmGroupedV2AGScatter")),
          cute::make_tuple(
              GemmGroupedV3AGScatter_Space::get_space(),
              std::string("moe_ag_scatter/gemm_grouped_v3_ag_scatter.hpp"),
              std::string("GemmGroupedV3AGScatter")),
      });
}
