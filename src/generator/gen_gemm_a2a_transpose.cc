//===- gen_gemm_a2a_transpose.cc ------------------------------- C++ ------===//
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

namespace bytedance::flux::generator {
using namespace cute;

struct GemmV3PreAttnAllToAllTranspose_Space {
  static constexpr auto GemmMeta_FP16 = make_space_gemm_meta(
      cute::make_tuple(
          make_gemm_dtype_config(_BF16{}),
          make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{})),
      cute::make_tuple(_Sm90{}),
      cute::make_tuple(_PreAttnAllToAllTranspose{}, _PreAttnQKVPackAllToAll{}),
      cute::make_tuple(_RCR{}),
      cute::make_tuple(_GemmV3{}),
      cute::make_tuple(make_gemm_v3_meta(_False{})));

  static constexpr auto GemmHParams_FP16 = tuple_cat(
      make_space_gemm_hparams(
          cute::make_tuple(
              make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _Cooperative{}),
              make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _Cooperative{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_128, _256, _64>{}),
          cute::make_tuple(_GemmDefault{}, _GemmStreamK{}),
          cute::make_tuple(cute::_4{}),
          cute::make_tuple(_RasterAlongN{})),
      make_space_gemm_hparams(
          cute::make_tuple(
              make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _Cooperative{}),
              make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _Cooperative{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_128, _128, _64>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_0{}),
          cute::make_tuple(_RasterAlongN{})),
      make_space_gemm_hparams(
          cute::make_tuple(
              make_gemm_v3_hparams(Shape<_1, _2, _1>{}, _Cooperative{}),
              make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _Cooperative{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_256, _128, _64>{}),
          cute::make_tuple(_GemmDefault{}, _GemmStreamK{}),
          cute::make_tuple(cute::_0{}),
          cute::make_tuple(_RasterAlongN{})),
      make_space_gemm_hparams(
          cute::make_tuple(make_gemm_v3_hparams(Shape<_2, _1, _1>{}, _PingPong{})),
          cute::make_tuple(Auto{}),
          cute::make_tuple(Shape<_128, _128, _64>{}),
          cute::make_tuple(_GemmDefault{}),
          cute::make_tuple(cute::_0{}),
          cute::make_tuple(_RasterAlongN{})));

  static auto
  get_space() {
    return build_gen_space(GemmMeta_FP16, GemmHParams_FP16);
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

  std::cout << "Running gemm_a2a_transpose generator...\n";
  return main_template(
      options,
      {
          cute::make_tuple(
              GemmV3PreAttnAllToAllTranspose_Space::get_space(),
              std::string("gemm_a2a_transpose/gemm_v3_pre_attn_a2a_transpose_kernel.hpp"),
              std::string("GemmV3PreAttnAllToAllTranspose")),
      });
  return 0;
}