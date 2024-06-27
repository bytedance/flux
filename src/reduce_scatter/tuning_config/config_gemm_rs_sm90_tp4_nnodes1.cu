//===- config_gemm_rs_sm90_tp4_nnodes1.cu ------------------------- C++ ---===//
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

// clang-format off
#include "flux/op_registry.h"
namespace bytedance::flux {
using namespace cute;

static int config_gemm_rs_sm90_tp4_nnodes1 = []() {
  auto &inst = TuningConfigRegistry::instance();
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(1024,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(1l,2l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(2048,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(4096,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,1536,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,3072,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,6144,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_BF16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  inst.add(make_gemm_meta(_FP16{}(),_Sm90{}(),_ReduceScatter{}(),_RCR{}(),_GemmV3{}(),None{},make_reduce_scatter_meta(false,_IntraNode{}())),make_runtime_config(8192,12288,12288,make_reduce_scatter_runtime_config(4,1)),make_gemm_hparams(make_gemm_v3_hparams(cute::make_tuple(2l,1l,1l)),None{},cute::make_tuple(128l,256l,64l),_GemmDefault{}(),0,_RasterHeuristic{}()));
  return 0;
}();
}
// clang-format on
