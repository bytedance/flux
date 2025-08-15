//===- post_attn_a2a_transpose_impls.hpp -------------------------- C++ ---===//
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
#include "flux/flux.h"

namespace bytedance {
namespace flux {
struct PostAttnAll2AllParams {
  void *input_ptr;
  void **output_ptrs;
  void **barrier_ptrs;
  int32_t **sync_barriers;
  int32_t bs;
  int32_t nheads;
  int32_t seq_len;
  int32_t head_dim;
  int32_t rank;
  int32_t world_size;
  int32_t TILE_M;

  int32_t num_comm_sm;
  int32_t *a2a_signal;

  int32_t cusum_seq_lens[kMaxWorldSize + 1];  // [world_size + 1, ]
  bool skip_barrier;
};

enum class SyncMethod : int32_t { SyncNone = 0, SyncAtomic };

void post_attn_a2a_transpose_impl(
    const PostAttnAll2AllParams &param,
    DataTypeEnum input_dtype,
    SyncMethod sync_method,
    cudaStream_t stream);

void post_attn_a2a_impl(
    const PostAttnAll2AllParams &params,
    DataTypeEnum input_dtype,
    SyncMethod sync_method,
    cudaStream_t stream);

void post_attn_a2a_dyn_impl(
    const PostAttnAll2AllParams &params,
    DataTypeEnum input_dtype,
    SyncMethod sync_method,
    cudaStream_t stream);

// num block of post_attn_a2a_impl is equal to post_attn_a2a_transpose_impl
int32_t get_post_attn_all2all_transpose_block_num(int32_t bs, int32_t seq_len, int32_t tile_m);
}  // namespace flux
}  // namespace bytedance
