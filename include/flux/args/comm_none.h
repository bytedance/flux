//===- comm_none.h ------------------------------------------------ C++ ---===//
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
#include "cute/int_tuple.hpp"

namespace bytedance::flux {

struct GemmOnlyArguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void *output;
};

// FP8 GEMM
// Aux = ((alpha * scale_a * scale_b) * accumulator) + ((beta * scale_c) * source) + bias
// D = activation(Aux)
// if Aux is fp8:
//   abs_max_output = max( abs(aux) | (for every aux in Aux) )
//   Aux = scale_aux * Aux
// if D is fp8 type:
//   abs_max_output = max( abs(d) | (for every d in D) )
//   D = scale_d * D
struct GemmFP8Arguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *A;  // m * k
  void const *B;  // k * n
  void const *C;  // m * n
  void *Aux;      // m * n
  void *D;        // output: m * n
  void *Vector;   // bias: 1 * n
  float *abs_max_Aux;
  float *abs_max_D;
  // scaling tensors
  float const *scaleA;
  float const *scaleB;
  float const *scaleC;
  float const *scaleD;    // require if D is fp8
  float const *scaleAux;  // require if Aux is fp8
};

struct GemmGroupedOpArguments {
  void *problem_sizes_device;
  int problem_count;
  float alpha;
  float beta;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  int64_t *lda;
  int64_t *ldb;
  int64_t *ldc;
  int64_t *ldd;
  void *problem_sizes_host;
};

struct GemmGroupedV3Arguments {
  int problem_count;
  float alpha;
  float beta;
  cute::tuple<int, int, int> *problem_sizes;
  void const **ptr_A;
  void const **ptr_B;
  void const **ptr_C;
  void **ptr_D;
  float **ptr_alpha = nullptr;
};

}  // namespace bytedance::flux
