//===- comm_none.h ------------------------------------------------ C++ ---===//
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

// GEMM with dequantization
// Dequant[i, j] = scale_a[i] * scale_b[j] * accumulator[i, j]
// D = Dequant + bias
// Accumulator dtype can be different from scale, Dequant calculation is based on scale dtype,
// D/bias dtype can be different from Dequant.
// For example:
//    Dequant and scale_a/scale_b: fp32
//    accumulator: s32
//    D and bias: bf16
struct S8GemmDequantArguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *A;        // m * k
  void const *B;        // k * n
  void const *bias;     // bias, 1 * n
  void const *scale_A;  // m * 1
  void const *scale_B;  // 1 * n
  void *D;              // output: m * n
};

// Block/Group-wise scaling GEMM
// Aux = ((alpha * scale_a * scale_b) * (blockscale_A * blockscale_B * A @ B) + ((beta * scale_c) *
// C) + bias D = (Aux) if Aux is fp8:
//   abs_max_output = max( abs(aux) | (for every aux in Aux) )
//   Aux = scale_aux * Aux
// if D is fp8 type:
//   abs_max_output = max( abs(d) | (for every d in D) )
//   D = scale_d * D
struct BlockScaleGemmArguments {
  // gemm args
  int m;
  int n;
  int k;
  int l;
  void const *A;
  void const *B;
  uint32_t mma_promotion_interval = 4;
  void const *blockscale_A;
  void const *blockscale_B;
  void const *C;
  void *D;
  // epilogue args
  float alpha = 1.0f;
  float beta = 0.0f;

  float scale_a = 1.0f;
  float scale_b = 1.0f;
  float scale_c = 1.0f;
  float scale_d = 1.0f;
  float scale_aux = 1.0f;

  void const *bias = nullptr;
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

struct GemmGroupedV2Arguments {
  void *problem_sizes;  // cutlass::gemm::GemmCoord*
  int problem_count;
  float alpha;
  float beta;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  // for FP8 arguments
  void **ptr_Aux = nullptr;     // m * n
  void **ptr_Vector = nullptr;  // bias: 1 * n
  float *abs_max_Aux = nullptr;
  float *abs_max_D = nullptr;
  // scaling tensors
  float const **scaleA = nullptr;
  float const **scaleB = nullptr;
  float const *scaleC = nullptr;
  float const *scaleD = nullptr;    // require if D is fp8
  float const *scaleAux = nullptr;  // require if Aux is fp8
  int sm_margin = 0;
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
  int sm_margin = 0;
};

}  // namespace bytedance::flux
