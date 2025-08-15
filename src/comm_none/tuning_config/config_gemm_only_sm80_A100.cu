//===- config_gemm_only_sm80_A100.cu ----------------------------------- C++ ---===//
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

static int config_gemm_only_sm80_a100 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(make_gemm_dtype_config(_S8{}(),_S8{}(),_Void{}(),_BF16{}(), _S32{}()),_Sm80{}(),_A100{}(),_CommNone{}(),_RCR{}(),_GemmV2{}()),make_runtime_config(512,8192,1024),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64,64),cute::make_tuple(16l,8l,32l),_StreamkSK{}()),None{},cute::make_tuple(128l,128l,64l),_GemmStreamK{}(),5,_RasterAlongM{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_S8{}(),_S8{}(),_BF16{}(),_BF16{}(), _S32{}()),_Sm80{}(),_A100{}(),_CommNone{}(),_RCR{}(),_GemmV2{}()),make_runtime_config(512,8192,1024),make_gemm_hparams(make_gemm_v2_hparams(cute::make_tuple(64l,64,64),cute::make_tuple(16l,8l,32l),_StreamkSK{}()),None{},cute::make_tuple(128l,128l,64l),_GemmStreamK{}(),5,_RasterAlongM{}()));
  return 0;
}();
}
// clang-format on
