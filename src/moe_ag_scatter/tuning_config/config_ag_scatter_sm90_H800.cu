//===- config_ag_scatter_sm90_H800.cu ---------------------------------- C++ ---===//
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

// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_ag_scatter_sm90_h800 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_BF16{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_AGScatter{}(),_RCR{}(),_GemmGroupedV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(768,2048,5120,None{}),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l),_PingPong{}()),None{},cute::make_tuple(64l,256l,64l),_GemmDefault{}(),0,_RasterAlongM{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_E4M3{}(),_E4M3{}(),_BF16{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_AGScatter{}(),_RCR{}(),_GemmGroupedV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(6144,288,6144,None{}),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,1l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,128l,128l),_GemmDefault{}(),0,_RasterAlongN{}()));
  return 0;
}();
}
// clang-format on
