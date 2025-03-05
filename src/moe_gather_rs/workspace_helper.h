
//===- workspace_helper.h -------------------------------------- C++ ------===//
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
#include "flux/args/moe_gather_rs.h"
#include <cutlass/gemm_coord.h>
#include <cutlass/layout/matrix.h>
namespace bytedance::flux {

struct MoeGatherRSWorkspaceArgs {
  int num_groups;
  int N_split;
  int ep_start;
  int ep_nexperts;
  int N, K;
  int32_t *splits_gpu;
  void *input[kMaxNumGroups];
  void *weights[kMaxNumGroups];
  void *output[kMaxNumGroups];
  float *input_scales[kMaxNumGroups];
  float *weight_scales[kMaxNumGroups];
};

/**

 workspace architecture

problem_sizes, cutlass::gemm::GemmCoord *, problem_count
ptr_A, void *, problem_count
ptr_B, void *, problem_count
ptr_C, void *, problem_count
ptr_D, void *, problem_count
lda, int64_t, problem_count
ldb, int64_t, problem_count
ldc, int64_t, problem_count
ldd, int64_t, problem_count
scale_A, float *, problem_count
scale_B, float *, problem_count
non_empty_problem_count, int, 1
 */

void make_workspace(
    const MoeGatherRSWorkspaceArgs &args,
    GemmLayoutEnum layout,
    int input_elem_size,
    int output_elem_size,
    void *workspace,
    cudaStream_t stream);
}  // namespace bytedance::flux
