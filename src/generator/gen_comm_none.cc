//===- gen_comm_none.cc ------------------------------------------ C++ ---===//
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

struct GemmV2CommNone_Space {
  static constexpr auto AllGemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_FP16{}),
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}, _Sm89{}),
      cute::make_tuple(_A100{}, _L20{}),
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
      cute::make_tuple(_L20{}),
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

  static constexpr auto AllGemmMeta_S8 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_S8{}, _S8{}, _BF16{}, _BF16{}, _S32{}),
          make_gemm_dtype_config(_S8{}, _S8{}, _Void{}, _BF16{}, _S32{})),
      cute::make_tuple(_Sm80{}, _Sm89{}),
      cute::make_tuple(_A100{}, _L20{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmV2{}));

  static constexpr auto AllGemmHParams_S8 = tuple_cat(
      make_space_gemm_hparams(
          cute::make_tuple(
              make_gemm_v2_hparams(Shape<_64, _32, _128>{}, Shape<_16, _8, _32>{}, _StreamkSK{}),
              make_gemm_v2_hparams(Shape<_64, _32, _128>{}, Shape<_16, _8, _32>{}, _StreamkDP{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(
              Shape<_128, _64, _128>{}, Shape<_128, _128, _128>{}, Shape<_64, _64, _128>{}),
          cute::make_tuple(Auto{}),
          cute::make_tuple(cute::_3{}, cute::_4{}),
          cute::make_tuple(_RasterAlongM{}, _RasterAlongN{}, _RasterHeuristic{})),
      make_space_gemm_hparams(
          cute::make_tuple(
              make_gemm_v2_hparams(Shape<_64, _64, _64>{}, Shape<_16, _8, _32>{}, _StreamkSK{}),
              make_gemm_v2_hparams(Shape<_64, _64, _64>{}, Shape<_16, _8, _32>{}, _StreamkDP{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(cute::Shape<cute::_128, cute::_128, cute::_64>{}),
          cute::make_tuple(Auto{}),
          cute::make_tuple(cute::_5{}),
          cute::make_tuple(_RasterAlongM{}, _RasterAlongN{}, _RasterHeuristic{})));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta_FP16, AllGemmHParams_FP16),
        build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
        build_gen_space(AllGemmMeta_S8, AllGemmHParams_S8),
    });
  }
};

struct GemmV3CommNone_Space {
  static constexpr auto AllGemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_FP16{}),
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H20{}, _H800{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RRR{}),
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

  static constexpr auto AllGemmHParams_FP16 = tuple_filter(
      make_space_gemm_hparams(
          tuple_transform(
              tuple_cartesian_product(
                  cute::make_tuple(Shape<_2, _1, _1>{}, Shape<_1, _2, _1>{}),
                  cute::make_tuple(_Cooperative{}, _PingPong{})),
              [](auto tup) {
                auto [cluster_shape, kernel_schedule] = tup;
                return make_gemm_v3_hparams(cluster_shape, kernel_schedule);
              }),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Auto{}),
          cute::make_tuple(_GemmDefault{}, _GemmStreamK{})),
      [](auto hparams_tuple) {
        auto hparams = to_gemm_hparams(hparams_tuple);
        auto v3_hparams = to_gemm_v3_hparams(hparams.impl_spec());
        return not(
            v3_hparams.kernel_schedule() == _PingPong{} and hparams.gemm_kind() == _GemmStreamK{});
      });

  static constexpr auto AllGemmMeta_FP8_BlockScale = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _FP32{}),
          make_gemm_dtype_config(_E5M2{}, _E4M3{}, _Void{}, _FP32{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H20{}, _H800{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCC{}, _RCR{}),
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_False{}, _True{})));

  static constexpr auto AllGEMMHParam_FP8_BlockScale = make_space_gemm_hparams(
      tuple_transform(
          tuple_cartesian_product(
              cute::make_tuple(Shape<_1, _2, _1>{}),
              cute::make_tuple(_Cooperative{}),
              cute::make_tuple(_BlockScaleMPerRow{}),
              cute::make_tuple(_BlockScaleNPerBlock{}, _BlockScaleNPerCol{})),
          [](auto tup) { return to_gemm_v3_hparams(tup); }),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Shape<_128, _128, _128>{}),
      cute::make_tuple(_GemmDefault{}, _GemmStreamK{}),
      cute::make_tuple(cute::_0{}),  // auto calculate
      cute::make_tuple(_RasterAlongM{}));

  static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E4M3{}, _Void{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E4M3{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H20{}, _H800{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}),  // Only register RCR layout for FP8 GEMM
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_False{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      tuple_transform(
          tuple_cartesian_product(
              cute::make_tuple(Shape<_2, _1, _1>{}), cute::make_tuple(_Cooperative{})),
          [](auto tup) {
            auto [cluster_shape, kernel_schedule] = tup;
            return make_gemm_v3_hparams(cluster_shape, kernel_schedule);
          }),
      cute::make_tuple(None{}),                     // comm_specs
      cute::make_tuple(Shape<_128, _128, _128>{}),  // Tile Shape
      cute::make_tuple(_GemmDefault{}),             // GemmKinds
      cute::make_tuple(Auto{}),                     // auto calculate
      cute::make_tuple(_RasterAlongM{}));

  static constexpr auto AllGemmMeta_S8 = make_space_gemm_meta(
      cute::make_tuple(
          // S8 gemm with dequant
          make_gemm_dtype_config(_S8{}, _S8{}, _BF16{}, _BF16{}, _S32{}),
          make_gemm_dtype_config(_S8{}, _S8{}, _Void{}, _BF16{}, _S32{}),
          // S8 gemm only
          make_gemm_dtype_config(_S8{}, _S8{}, _S32{}, _S32{}, _S32{}),
          make_gemm_dtype_config(_S8{}, _S8{}, _Void{}, _S32{}, _S32{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H20{}, _H800{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

  static constexpr auto AllGemmHParams_S8 = tuple_cat(
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _Cooperative{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_128, _256, _128>{}),
          cute::make_tuple(_GemmStreamK{}),
          cute::make_tuple(cute::_4{}),
          cute::make_tuple(_RasterHeuristic{})),
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _Cooperative{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_128, _128, _128>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_3{}),
          cute::make_tuple(_RasterAlongN{})),
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _Cooperative{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_128, _128, _128>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_3{}),
          cute::make_tuple(_RasterHeuristic{})),
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _PingPong{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_64, _128, _128>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_8{}),
          cute::make_tuple(_RasterHeuristic{})),
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _PingPong{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_64, _128, _128>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_2{}, cute::_3{}, cute::_4{}),
          cute::make_tuple(_RasterAlongM{}, _RasterAlongN{}, _RasterHeuristic{})),
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _PingPong{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_64, _128, _128>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_8{}),
          cute::make_tuple(_RasterHeuristic{})));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta_FP16, AllGemmHParams_FP16),
        build_gen_space(AllGemmMeta_S8, AllGemmHParams_S8),
        build_gen_space(AllGemmMeta_FP8_BlockScale, AllGEMMHParam_FP8_BlockScale),
        build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
    });
  }
};

struct GemmGroupedV2CommNone_Space {
  static constexpr auto AllGemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_FP16{}),
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_FP16{}, _FP16{}, _Void{}, _FP16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm80{}, _Sm89{}),
      cute::make_tuple(_A100{}, _L20{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RCC{}),
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
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RCC{}),
      cute::make_tuple(_GemmGroupedV2{}),
      cute::make_tuple(make_gemm_v2_meta(_True{}), make_gemm_v2_meta(_False{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Auto{}),
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

struct GemmGroupedV3CommNone_Space {
  static constexpr auto AllGemmMeta_FP16 = tuple_filter(
      make_space_gemm_meta(
          cute::make_tuple(_FP16{}, _BF16{}),
          cute::make_tuple(_Sm90{}),
          cute::make_tuple(_H20{}, _H800{}),
          cute::make_tuple(_CommNone{}),
          cute::make_tuple(_RCR{}, _RRR{}, _RCC{}),
          cute::make_tuple(_GemmGroupedV3{}),
          cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{}))),
      [](auto meta_tuple) { return true; });

  static constexpr auto AllGemmHParams_FP16 = make_space_gemm_hparams(
      cute::make_tuple(
          make_gemm_v3_hparams(Shape<_2, _1, _1>{}),
          make_gemm_v3_hparams(Shape<_1, _2, _1>{}),
          make_gemm_v3_hparams(Shape<_1, _1, _1>{})),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Shape<_128, _256, _64>{}, Shape<_256, _128, _64>{}));

  static constexpr auto AllGemmMeta_FP8 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E5M2{}, _BF16{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H20{}, _H800{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}, _RCC{}),
      cute::make_tuple(_GemmGroupedV3{}),
      cute::make_tuple(make_gemm_v3_meta(_True{}), make_gemm_v3_meta(_False{})));

  static constexpr auto AllGemmHParams_FP8 = make_space_gemm_hparams(
      cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{})), cute::make_tuple(Auto{}));

  static constexpr auto AllGemmMeta_FP8_BlockScale = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E5M2{}, _E4M3{}, _BF16{}, _BF16{}),
          make_gemm_dtype_config(_E4M3{}, _E4M3{}, _FP32{}, _FP32{}),
          make_gemm_dtype_config(_E5M2{}, _E4M3{}, _FP32{}, _FP32{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_H20{}, _H800{}),
      cute::make_tuple(_CommNone{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmGroupedV3{}),
      cute::make_tuple(make_gemm_v3_meta(_False{}, _True{})));

  static constexpr auto AllGEMMHParam_FP8_BlockScale = make_space_gemm_hparams(
      tuple_transform(
          tuple_cartesian_product(
              cute::make_tuple(Shape<_1, _2, _1>{}),
              cute::make_tuple(_Cooperative{}),
              cute::make_tuple(_BlockScaleMPerRow{}),
              cute::make_tuple(_BlockScaleNPerBlock{}, _BlockScaleNPerCol{})),
          [](auto tup) { return to_gemm_v3_hparams(tup); }),
      cute::make_tuple(Auto{}),
      cute::make_tuple(Shape<_128, _128, _128>{}),
      cute::make_tuple(_GemmDefault{}, _GemmStreamK{}),
      cute::make_tuple(cute::_0{}),  // auto calculate
      cute::make_tuple(_RasterAlongM{}));

  static auto
  get_space() {
    return merge_gen_space({
        build_gen_space(AllGemmMeta_FP16, AllGemmHParams_FP16),
        build_gen_space(AllGemmMeta_FP8, AllGemmHParams_FP8),
        build_gen_space(AllGemmMeta_FP8_BlockScale, AllGEMMHParam_FP8_BlockScale),
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
  std::cout << "Running comm_none generator...\n";
  return main_template(
      options,
      {
          cute::make_tuple(
              GemmV2CommNone_Space::get_space(),
              std::string("comm_none/gemm_v2_comm_none.hpp"),
              std::string("GemmV2CommNone")),
          cute::make_tuple(
              GemmV3CommNone_Space::get_space(),
              std::string("comm_none/gemm_v3_comm_none.hpp"),
              std::string("GemmV3CommNone")),
          cute::make_tuple(
              GemmGroupedV2CommNone_Space::get_space(),
              std::string("comm_none/gemm_grouped_v2_comm_none.hpp"),
              std::string("GemmGroupedV2CommNone")),
          cute::make_tuple(
              GemmGroupedV3CommNone_Space::get_space(),
              std::string("comm_none/gemm_grouped_v3_comm_none.hpp"),
              std::string("GemmGroupedV3CommNone")),
      });
}
