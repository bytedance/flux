//===- moe_ag_scatter.h ------------------------------------------- C++ ---===//
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
#include "./comm_none.h"
#include "flux/utils.h"

namespace bytedance::flux {
constexpr int kMaxNumGroups = 2;

struct GemmGroupedAgScatterArguments : GemmGroupedV3Arguments {
  DistEnv dist_env;
  int ntokens;
  int h;
  void *nvshmem_input_buffer;
  int32_t const **gather_A;
  int32_t const **scatter_D;
  void const *problem_schedule;
  void *barrier_ptr = nullptr;
  int sm_margin = 0;
};

struct GemmGroupedV2AGScatterArguments {
  int rank;
  int world_size;
  int sm_margin;

  int num_groups;  // make sure num_groups <= kMaxNumGroups
  int ep_start;
  int ep_nexperts;
  void *input;                  // before gather_A
  void *weight[kMaxNumGroups];  // with groups
  void *output[kMaxNumGroups];  // with groups
  // FP8 arguments
  float *scaleD[kMaxNumGroups];  // with groups
  int M_this_ep, N, K;
  int lda, ldb, ldc, ldd;
  int *splits;
  // calculated on prepare workspace
  int32_t *gather_A;   // on device memory expected
  int32_t *scatter_D;  // on device memory expected
  void *problem_schedules;
  int num_problem_schedules;
  int *accum_per_rank_ptr;  // on device memory expected
  int tile_size_m, tile_size_n;
  int *barrier_ptr;
  // fill inside op. only Op has the information
  float alpha = 1.f;
  float beta = 0.f;
};

}  // namespace bytedance::flux
