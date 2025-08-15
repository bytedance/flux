//===- moe_gather_rs.h -------------------------------------------- C++ ---===//
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

#pragma once
namespace bytedance::flux {

constexpr int kMaxNumGroups = 2;
constexpr int kMaxExpertCount = 1024;

struct GemmGatherArguments {
  int m;
  int n;
  int k;
  float alpha;
  float beta;
  void const *input;
  void const *weight;
  void const *bias;
  void *output;
  void const *gather_index;
  void const *gather_weight;
  void *gather_output;
};

struct GemmGatherStoreArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  float alpha;
  float beta;
  const void *input;
  const void *weight;
  const void *bias;
  void *gemm_output;
  const index_t *gather_index;
  const void *gather_weight;
  void *gather_outputs;
  void **rs_outputs_ptrs;
};

struct GemmGatherRsArguments {
  int m;
  int n;
  int k;
  int rank;
  int world_size;
  float alpha;
  float beta;
  const void *input;
  const void *weight;
  const void *bias;
  void *gemm_output;
  const index_t *gather_index;
  const void *gather_weight;
  void *gather_outputs;
  const bool *finish_gather;
  void **rs_outputs_ptrs;
};

struct GemmGroupedV3GatherRSArguments {
  void *problem_sizes_device;
  int problem_count;
  float alpha;
  float beta;
  const void **ptr_A;
  const void **ptr_B;
  const void **ptr_C;
  void **ptr_D;
  void *lda;
  void *ldb;
  void *ldc;
  void *ldd;
  void *problem_sizes_host;
  int32_t rank;
  int32_t world_size;
  void **output_scatter_ptrs;
  void **inter_Ds;
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t SPLITS;
  int32_t totalM;
  int32_t n_dim;
  // following args are for expert parallel
  int32_t tp_world_size;
  int32_t ep_world_size;
  int32_t globalM;
  int32_t max_token_per_rank;
  int32_t ep_m_start;
  int32_t ep_m_end;
  float **input_scale_ptr;
  float **weight_scale_ptr_array;
  float **output_vec_scale_ptr;
  int32_t sm_margin;
  int32_t input_groups;
  int32_t *ep_pos_filtered;
  int32_t *ep_token_idx_filtered;
  int32_t *ep_total_token_acc;
};

struct GemmGroupedV2GatherRSArguments {
  void *problem_sizes;  // cutlass::gemm::GemmCoord*
  int problem_count;
  int *non_empty_problem_count;  // a pointer in GPU memory
  float alpha;
  float beta;
  void **ptr_A;
  void **ptr_B;
  void **ptr_C;
  void **ptr_D;
  void *lda;  // to support split_n
  void *ldb;
  void *ldc;
  void *ldd;
  void *ldr;
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
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t n_split;
  // following args are for expert parallel
  int sm_margin;
};

struct TopKReduceGatherRSArguments {
  int32_t rank;
  int32_t world_size;
  void **output_scatter_ptrs;
  void *inter_D;
  int32_t topk;
  int32_t *barrier;
  int32_t *routing_idx;
  int32_t SPLITS;
  int32_t totalM;  // M for the current group gemm
  int32_t n_dim;
  int32_t n_tb_blocks;
  int32_t tp_world_size;
  int32_t ep_world_size;
  // M for all the experts, should be equal to totalM when expert parallel is not enabled
  int32_t globalM;
  float *input_scale_ptr;
  float *output_vec_scale_ptr;
};

struct TopKReduceGatherRSV2Arguments {
  void *input_ptrs[kMaxNumGroups];  // [input_groups, m_this_ep * n] of output_dtype
  void *output_ptr;                 // [world_size, ntokens * n] of output_dtype
  float *output_vec_scale_ptrs[kMaxNumGroups];
  int *splits;
  int *routing_idx;  // [m_full = ntokens * topk]
  int m_full;  // M for all the experts, m_full = ntokens * topk; m_full == m_this_ep for EP=1
  int n;
  int nexperts;
  int topk;
  int input_groups;
  bool do_all_reduce = false;
  bool use_read_mode = false;

  int threadblock_count;
  int tile_size_m;  // tile shape m. 128 by default.
  int tile_size_n;  // tile shape n. 1024 by default. supported 1024/1280/640/384. (n /
                    // n_split) % tiled_n expected.
  // for reduce_scatter
  int rank;
  int world_size;
  int n_split;
  int **barrier;            // [world_size, n_split * 2]
  void **reduce_ptrs;       // [world_size, ntokens * n] of output_dtype
  int **tile_barrier_ptrs;  // [world_size * num_tiles]
};

}  // namespace bytedance::flux
