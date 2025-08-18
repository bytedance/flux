//===- config_gemm_a2a_transpose_kernel_sm90_H800_tp8_nnodes1.cu ------- C++ ---===//
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

static int config_gemm_a2a_transpose_kernel_sm90_h800_tp8_nnodes1 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(2048,8192,4096,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),4,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(1024,4096,4096,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,128l,64l),_GemmDefault{}(),0,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(1024,8192,8192,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,256l,64l),_GemmStreamK{}(),4,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(1024,6144,6144,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,128ll,64l),_GemmDefault{}(),0,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(1024,6400,6400,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l),_PingPong{}()),None{},cute::make_tuple(128l,128l,64l),_GemmDefault{}(),0,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(1024,2048,2048,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,128l,64l),_GemmDefault{}(),0,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnAllToAllTranspose{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(1024,1280,1280,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l),_PingPong{}()),None{},cute::make_tuple(128l,128l,64l),_GemmDefault{}(),0,_RasterAlongN{}()));
  inst.add(make_gemm_meta(make_gemm_dtype_config(_BF16{}(),_BF16{}(),_Void{}(),_BF16{}(),_FP32{}()),_Sm90{}(),_H800{}(),_PreAttnQKVPackAllToAll{}(),_RCR{}(),_GemmV3{}(),make_gemm_v3_meta(false),None{}),make_runtime_config(2048,10240,4096,make_all_to_all_transpose_runtime_config(8,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l),_Cooperative{}()),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),4,_RasterAlongN{}()));

  return 0;
}();
}
// clang-format on
